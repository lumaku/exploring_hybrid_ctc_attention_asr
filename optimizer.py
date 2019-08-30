#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Gaussian Process optimization for Attention-based ASR
Copyright 2019 Ludwig Kürzinger, TU München
Licensed under the MIT License


Known Issues:
    - None


TABLE OF CONTENTS


"""


import os
import sys
import subprocess
import datetime
import time
import random
import numpy as np
import glob
import pickle
from parse import parse
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

try:
    from espnet.nets.pytorch_backend.e2e_asr import E2E
    ESPNET_AVAILABLE = True
except:
    ESPNET_AVAILABLE = False

try:
    import pandas as pd
    import seaborn as sns
    SEABORN_AVAILABLE = True
except:
    SEABORN_AVAILABLE = False

def isclose(a, b, abs_tol=0.1):
    return abs(a-b) <= abs_tol

# PRIVATE PATHS (redacted)
EXPPATH_LETO = None
EXPPATH_PLUTO = None
EXPPATH_REMOTE = None


# formatting string to parse parameters, taken from the run.sh script
AMPARAMSTR = "{train_set}_pytorch_{etype}_e{elayers}_subsample{subsample}_unit{eunits}_proj{eprojs}_d{dlayers}_unit{dunits}_{atype}_adim{adim}_aconvc{aconv_chans}_aconvf{aconv_filts}_mtlalpha{mtlalpha}_{opt}_sampprob{samp_prob}_bs{batchsize}_mli{maxlen_in}_mlo{maxlen_out}"
WERPARAMSTR = "decode_{rtask}_beam{beam_size}_e{recog_model}_p{penalty}_len{minlenratio}-{maxlenratio}_ctcw{ctc_weight}_rnnlm{lm_weight}_{lmtag}"
LMPARAMSTR = "train_rnnlm_pytorch_{lm_layers}layer_unit{lm_units}_{lm_opt}_bs{lm_batchsize}"
LMTAGSTR = "{lm_layers}layer_unit{lm_units}_{lm_opt}_bs{lm_batchsize}"

# ---------------------------- OPTIMIZER


class HyperparamOptimizer:
    """
    Gaussian Process optimizer.

    The code in this class is heavily inspired by (copied from):
        http://krasserm.github.io/2018/03/21/bayesian-optimization/
        https://github.com/krasserm/bayesian-machine-learning/blob/af6882305d9d65dbbf60fd29b117697ef250d4aa/gaussian_processes_util.py#L7
        https://docs.scipy.org/doc/scipy-0.18.1/reference/optimize.html
    """
    def __init__(self, X, Y, params, minimizeTarget=True, kernel='Matern', xi=0.01, n_restarts=25):
        self.params = params
        self.bounds = paramsToBounds(self.params)
        if kernel == 'Matern':
            self.kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5)
        elif kernel == 'RBF':
            self.kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        else:
            raise Exception("unknown kernel: " + kernel)
        self.minimizeTarget = minimizeTarget
        self.xi = xi
        self.n_restarts = n_restarts
        self.gp = GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=20, alpha=1e-4)
        self.update(X,Y)

    def update(self, X,Y):
        self.X = X
        self.Y = Y
        if self.minimizeTarget:
            self.Y *= -1
        self.gp.fit(X,Y)

    def expected_improvement(self, newX):
        mu, sigma = self.gp.predict(newX, return_std=True)
        mu_sample = self.gp.predict(self.X)
        sigma = sigma.reshape(-1, 1 ) # self.Y.shape[1])
        mu_sample_opt = np.min(mu_sample) ## <- min oder max?
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def propose_location(self):
        dim = self.X.shape[1]
        min_val = None
        min_x = None
        def min_obj(X):
            # Minimization objective is the negative acquisition function
            return -1 * self.expected_improvement(X.reshape(-1, dim))
        # Find the best optimum by starting from n_restart different random points.
        for x0 in rndPointsFromParams(self.params, self.n_restarts):
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')
            if (min_val == None) or (res.fun < min_val):
                min_val = res.fun[0]
                min_x = res.x
        return min_x.reshape(-1, 1)

    def nextX(self):
        self.gp.fit(self.X, self.Y)
        self.newPoint = self.propose_location()
        return self.newPoint


# ------------------- ESPnet plug

amParams = [ # encoder
            {'name': 'elayers',     'type': int, 'lower_bound': 1, 'upper_bound': 10},
            {'name': 'eunits',      'type': int, 'lower_bound': 25, 'upper_bound': 500},
            {'name': 'eprojs',      'type': int, 'lower_bound': 25, 'upper_bound': 400},
            # decoder
            {'name': 'dlayers',     'type': int, 'lower_bound': 1, 'upper_bound': 10},
            {'name': 'dunits',      'type': int, 'lower_bound': 25, 'upper_bound': 400},
            # attention related
            {'name': 'adim',        'type': int, 'lower_bound': 25, 'upper_bound': 400},
            {'name': 'aconv_chans', 'type': int, 'lower_bound': 1, 'upper_bound': 20},
            {'name': 'aconv_filts', 'type': int, 'lower_bound': 30, 'upper_bound': 150},
            # hybrid attention
            {'name': 'mtlalpha',    'type': float, 'lower_bound': 0.0, 'upper_bound': 1.0}]


class amPlug():
    def __init__(self, params: list, target="main/acc", minimizeTarget=False,
                 egsPath="/xxx/delme", runCmd="./test.sh", cmdOutFile="./amlog.txt",
                 dryRun=False, verbose=1, fillValue=0.5):
        self.params = params
        self.target = target
        self.minimizeTarget = minimizeTarget
        self.expPath = egsPath
        self.runCmd = runCmd
        self.dryRun = dryRun
        self.verbose = verbose
        self.cmdOutFile = cmdOutFile
        self.fillValue = fillValue # bad value of the target parameter (e.g. 0.1 for main/acc)
        self.worker = None
        self.readState()

    def readState(self):
        self.trace = getAcousticModelResults(self.expPath, verbose=self.verbose, excludeUnfinished=False)
        self.X, self.Y = traceToXY(self.trace, self.params, self.target)

    def run(self, x):
        # run program
        command = self.runCmd + pointToCmdStr(x, self.params) + " >> " + self.cmdOutFile + " 2>&1"
        if self.verbose > 0:
            statusstr = datetime.datetime.now().strftime("%Y%m%d-%H%M") + " {:03}.".format(len(self.trace) + 1)
            print(statusstr, " --> ", command)
        fullCommand = "cd {};".format(self.expPath) + command
        if not self.dryRun:
            self.worker = subprocess.run(fullCommand, shell=True, check=True)

    def step(self):
        optimizer = HyperparamOptimizer(self.X, self.Y, self.params, minimizeTarget=self.minimizeTarget)
        newX = optimizer.nextX()
        self.run(newX)
        self.readState()


def amtest():
    a = amPlug(amParams, target="main/acc", minimizeTarget=False, dryRun=False)
    a.step()


def amRun(steps, dryRun=False, cer=False):
    if cer:
        plug = amPlug(amParams, target="main/cer", egsPath=EXPPATH_LETO, runCmd="./am.sh", dryRun=dryRun, minimizeTarget=True, fillValue=100.0)
    else:
        plug = amPlug(amParams, target="main/acc", egsPath=EXPPATH_LETO, runCmd="./am.sh", dryRun=dryRun, minimizeTarget=False, fillValue=.1)
    print("running {} steps starting from {}.".format(steps, len(plug.trace)))
    for i in range(steps):
        plug.step()



# ------------------- LM + AM + decoder plug

decParams = [ # encoder
            {'name': 'lm_weight',     'type': float, 'lower_bound': 0.0, 'upper_bound': 1.0},
            {'name': 'ctc_weight',     'type': float, 'lower_bound': 0.0, 'upper_bound': 1.0}]


class decPlug():
    """
    hierzu muss schon bekannt sein, welche Langugae Model und welches Acoustic Model man nimmt...
    """
    def __init__(self, params: list, target="CER-Mean", minimizeTarget=True,
                 egsPath="/xxx/delme", runCmd="./wer.sh", cmdOutFile="./declog.txt",
                 dryRun=True, verbose=1, filterBadAM=False, gpu=False):
        self.startParams = params
        self.target = target
        self.minimizeTarget = minimizeTarget
        self.egsPath = egsPath
        self.runCmd = runCmd
        if gpu:
            self.runCmd += " --nj 1 --ngpu 1"
        self.dryRun = dryRun
        self.verbose = verbose
        self.filterBadAM = filterBadAM
        self.cmdOutFile = cmdOutFile
        self.worker = None
        # get decoding results
        self.readState()

    def readState(self):
        self.decResults = DecodingResultHandle(expDir=self.egsPath, verbose=self.verbose, skipInvalid=True)
        if self.filterBadAM:
            self.decResults.filterThresholdAMacc = 0.75
            self.decResults.applyFilter()
        self.params = self.startParams + [self.decResults.getParamEntryAM(), self.decResults.getParamEntryLM()]
        # put only results of the dev set into the trace, so that the test set does not leak into the optimization
        self.trace = []
        for item in self.decResults.DRs:
            if self.target in item:
                if item['rtask'] == 'dev':
                    self.trace.append(item)
            else:
                print("ERROR: item does not have target value: exp/"+item['expName'] +'/'+item['pathname'])
        if self.target:
            self.X, self.Y = traceToXY(self.trace, self.params, self.target)

    def run(self, x):
        # run program
        command = self.runCmd
        command += self.decResults.getCmdStr(x, self.params)
        command += " >> " + self.cmdOutFile + " 2>&1"
        if self.verbose > 0:
            statusstr = datetime.datetime.now().strftime("%Y%m%d-%H%M") + " {:03}. ".format(len(self.trace) + 1)
            xstr = np.array2string(x.reshape(-1), suppress_small=True, separator=',', precision=3).replace('\n', '')
            print(statusstr, xstr, " --> ", command)
        fullCommand = "cd {};".format(self.egsPath) + command
        if not self.dryRun:
            self.worker = subprocess.run(fullCommand, shell=True, check=True)

    def step(self):
        optimizer = HyperparamOptimizer(self.X, self.Y, self.params, minimizeTarget=self.minimizeTarget)
        newX = optimizer.nextX()
        self.run(newX)
        self.readState()


def decRun(steps, dryRun=True, filterBadAM=False, gpu=False):
    plug = decPlug(decParams, egsPath=EXPPATH_LETO, target="CER-Sum/Avg", dryRun=dryRun, filterBadAM=filterBadAM, gpu=gpu)
    print("running {} steps starting from {}.".format(steps, len(plug.trace)))
    for i in range(steps):
        plug.step()


def dectest():
    a = decPlug(decParams, egsPath=EXPPATH_LETO, target="CER-Mean", minimizeTarget=False, dryRun=True)
    print (a.Y)
    #a.step()


def decBaseModels(egsPath=EXPPATH_LETO, dryRun=True, idxRNNLM=2, filterBadAM=False, gpu=False):
    plug = decPlug(decParams, egsPath=egsPath, target=None, dryRun=dryRun, filterBadAM=filterBadAM, gpu=gpu)
    aufgaben =[]
    for lmMode in [0.0, 1.0]:
        for ctcMode in [0.0, 1.0, None]:
            for idxAM, AM in enumerate(plug.decResults.AMs):
                if ctcMode is None:
                    ctcVal = plug.decResults.AMs[idxAM]['mtlalpha']
                else:
                    ctcVal = ctcMode
                # check if m<odel alreadey exists
                xdict = {'lm_weight': lmMode, 'ctc_weight': ctcVal, 'AM-Index': idxAM}
                if not plug.decResults.checkDuplicate(xdict):
                    xdict['LM-Index'] = idxRNNLM
                    x = np.array([float(lmMode), float(ctcVal), float(idxAM), float(idxRNNLM)])
                    aufgaben.append(x)
                    #plug.readState()
                else:
                    print("already calculated: ", xdict)
    print('There are {} Jobs'.format(len(aufgaben)))
    random.shuffle(aufgaben)
    for x in aufgaben:
        plug.run(x)


# ------------------- unit tests

class generalTest():
    def __init__(self, noise=0.2, verbose=1):
        self.noise=noise
        self.verbose = verbose
        self.sampleInit()

    def run(self, x):
        self.trace.append({'x': x, 'y':self.myFkt(x, self.noise)})

    def readState(self):
        self.sampleInit()


    def step(self):
        optimizer = HyperparamOptimizer(self.X, self.Y, self.params, minimizeTarget=False)
        newX = optimizer.nextX()
        self.run(newX)
        self.X, self.Y = traceToXY(self.trace, self.params, 'y')
        if self.verbose > 0:
            print(self.trace[-1])

    def myFkt(self, X, noise=0.0):
        X = np.array(X)
        return 1 * (-np.sin(3 * X) - X ** 2 + 0.7 * X + noise * np.random.randn(*X.shape))

    def sampleInit(self):
        self.params = []
        self.params.append({'name': 'x', 'value': 0.5, 'type': float, 'lower_bound': -1.0, 'upper_bound': 2.0})
        self.bounds = paramsToBounds(self.params)
        self.trace = []
        self.trace.append({'x': -0.9, 'y':self.myFkt(-0.9, self.noise)})
        self.trace.append({'x':  1.1, 'y':self.myFkt( 1.1, self.noise)})
        self.X, self.Y =  traceToXY(self.trace, self.params, 'y')
        if self.verbose > 0:
            print("trace ", self.trace)
            print("params ", self.params)
            print("X ", self.X)
            print("Y ", self.Y)

    def plotSamples(self):
        # Dense grid of points within bounds
        X = np.arange(self.bounds[:, 0], self.bounds[:, 1], 0.01).reshape(-1, 1)
        # Noise-free objective function values at X
        Y = self.myFkt(X, 0)
        Yn = self.myFkt(X, self.noise)
        # Plot optimization objective with noise level
        plt.plot(X, Y, 'y--', lw=2, label='Noise-free objective')
        plt.plot(X, Yn, 'bx', lw=1, alpha=0.1, label='Noisy samples')
        plt.plot(self.X, self.Y, 'kx', mew=3, label='Initial samples')
        plt.legend()
        plt.show()


def test():
    a = generalTest()
    for i in range(20):
        a.step()
    a.plotSamples()

# --------------------------------- Plots und Analyser

class decPlotter():
    def __init__(self, egsPath="/xxx/delme", outDir="./plots", resultFile="./results.txt", autoLoad=False):
        self.egsPath = egsPath
        self.outDir = outDir + '/'
        self.resultFile = resultFile
        if autoLoad:
            self.loadFromFile()

    def readState(self):
        self.decResults = DecodingResultHandle(expDir=self.egsPath, verbose=1, skipInvalid=True)
        # calculate the model sizes of all E2E models.
        for item in self.decResults.AMs:
            for netPart in ['att', 'locconv', 'dec','ctc', '']:
                item[netPart+'param_count'] = getE2EmodelSize(item, netPart)

    def getTraceXY(self, xTarget="param_count", yTarget="CER-Mean", rtask=None):
        # rtask: put only results of the dev set into the trace, so that the test set does not leak into the optimization
        if rtask is None:
            rtask = ['test'] # ['dev']
        trace = []
        for item in self.decResults.DRs:
            if item['rtask'] in rtask:
                trace.append(item)
        params = [ {'name': yTarget, 'type': float, 'lower_bound': 0.0, 'upper_bound': 1.0} ]
        X, Y = traceToXY(trace, params, xTarget)
        return X, Y

    def saveToFile(self):
        with open(self.resultFile, 'wb') as handle:
            state = self.decResults
            pickle.dump(state, handle)

    def loadFromFile(self):
        try:
            with open(self.resultFile, 'rb') as handle:
                state = pickle.load(handle)
                self.decResults = state
                numAMs, numLMs, numDRs = len(self.decResults.AMs), len(self.decResults.LMs), len(self.decResults.DRs)
                print("Loaded successfully from file {}, we have {} Acoustic Models, ".format(self.resultFile, numAMs),
                      "{} Language Models and {} decoding results".format(numLMs, numDRs))
            return True
        except:
            return False


def initdecPlotter(egsPath=EXPPATH_LETO):
    results = decPlotter(egsPath=egsPath)
    results.readState()
    results.saveToFile()


def filterDecodingResults(results: decPlotter, filter=None, filterAcc=None, filterCER=None):
    #TODO: sollte auch die Eignenschafen des Acousic Models hinzügen
    DRs = []
    if filterAcc or filterCER:
        results.decResults.filterThresholdAMacc = filterAcc
        results.decResults.filterThresholdCER   = filterCER
        results.decResults.applyFilter()
    if filter is None or filter == '':
        DRs = results.decResults.DRs
    elif filter == 'CTC-only':
        # best CTC-only Model
        for item in results.decResults.DRs:
            idxAM = item['AM-Index']
            if results.decResults.AMs[idxAM]['mtlalpha'] == 1.0:
                DRs.append(item)
    elif filter == 'Att-only':
        # best CTC-only Model
        for item in results.decResults.DRs:
            idxAM = item['AM-Index']
            if results.decResults.AMs[idxAM]['mtlalpha'] == 0.0:
                DRs.append(item)
    elif filter == 'no-LM':
        # best CTC-only Model
        for item in results.decResults.DRs:
            if item['lm_weight'] == 0.0:
                DRs.append(item)
    elif filter == 'no-LM-Att':
        # best CTC-only Model
        for item in results.decResults.DRs:
            idxAM = item['AM-Index']
            if results.decResults.AMs[idxAM]['mtlalpha'] == 0.0 and item['lm_weight'] == 0.0:
                DRs.append(item)
    elif filter == 'no-LM-CTC':
        # best CTC-only Model
        for item in results.decResults.DRs:
            idxAM = item['AM-Index']
            if results.decResults.AMs[idxAM]['mtlalpha'] == 1.0 and item['lm_weight'] == 0.0:
                DRs.append(item)
    elif 'lambda=kappa' in filter:
        # only models that have the same lambda and kappa
        for item in results.decResults.DRs:
            idxAM = item['AM-Index']
            a = results.decResults.AMs[idxAM]['mtlalpha']
            b = item['ctc_weight']
            if isclose(a, b, abs_tol=0.1):
                DRs.append(item)
    elif 'att-ctc-LM-' in filter:
        # 0: == 0   # att-only
        # 1: == 1   # CTC-only
        # 2: keines von beiden
        # 3: ist egal
        # 4: ist nicht null
        # 5: kleiner als .5
        # 6: größer als .5
        # 7: .2 .. .8
        # 9: eine weichere Version von 2, die nicht funktioniert hat
        # z.B. 'att-ctc-LM-010'
        att = filter[-3]
        ctc = filter[-2]
        lm  = filter[-1]
        for item in results.decResults.DRs:
            idxAM = item['AM-Index']
            vetoAtt = False
            vetoCTC = False
            vetoLM = False
            # mtlalpha
            if att == '0' and not results.decResults.AMs[idxAM]['mtlalpha']  < 0.01: # == 0.0:
                vetoAtt = True
            if att == '1' and not results.decResults.AMs[idxAM]['mtlalpha'] == 1.0:
                vetoAtt = True
            if att == '2' and not (0.01 <= results.decResults.AMs[idxAM]['mtlalpha'] <= 0.99):
                vetoAtt = True
            if att == '4' and (results.decResults.AMs[idxAM]['mtlalpha'] <= 0.01):
                vetoAtt = True
            if att == '5' and (results.decResults.AMs[idxAM]['mtlalpha'] > 0.5):
                vetoAtt = True
            if att == '6' and (results.decResults.AMs[idxAM]['mtlalpha'] <= 0.5):
                vetoAtt = True
            if att == '7' and not (0.2 <= results.decResults.AMs[idxAM]['mtlalpha'] <= 0.8):
                vetoAtt = True
            if att == '8' and not (0.2 <= results.decResults.AMs[idxAM]['mtlalpha'] <= 0.3):
                vetoAtt = True
            if att == '9' and not (results.decResults.AMs[idxAM]['mtlalpha'] == 1.0 or results.decResults.AMs[idxAM]['mtlalpha'] == 0.0):
                vetoAtt = True
            # CTC weight
            if ctc == '0' and not item['ctc_weight']  < 0.01: # == 0.0:
                vetoCTC = True
            if ctc == '1' and not item['ctc_weight'] == 1.0:
                vetoCTC = True
            if ctc == '2' and not (0.01 <= item['ctc_weight'] <= 0.99):
                vetoCTC = True
            if ctc == '4' and item['ctc_weight'] <= 0.01:
                vetoCTC = True
            if ctc == '5' and item['ctc_weight'] > 0.5:
                vetoCTC = True
            if ctc == '6' and item['ctc_weight'] <= 0.5:
                vetoCTC = True
            if ctc == '7' and not (0.2 <= item['ctc_weight'] <= 0.8):
                vetoCTC = True
            if ctc == '8' and not (0.2 <= item['ctc_weight'] <= 0.6):
                vetoCTC = True
            if ctc == '9' and not (item['ctc_weight'] == 1.0 or item['ctc_weight'] == 0.0):
                vetoCTC = True
            # LM
            if lm == '0' and not item['lm_weight'] == 0.0:
                vetoLM = True
            if lm == '1' and not item['lm_weight'] == 1.0:
                vetoLM = True
            if lm == '2' and not (0.01 <= item['lm_weight'] <= 0.99):
                vetoLM = True
            if lm == '4' and (item['lm_weight'] <= 0.01):
                vetoLM = True
            if lm == '5' and (item['lm_weight'] > 0.5):
                vetoLM = True
            if lm == '6' and (item['lm_weight'] <= 0.5):
                vetoLM = True
            if lm == '7' and not (0.2 <= item['lm_weight'] <= 0.8):
                vetoLM = True
            if lm == '8' and not (0.2 <= item['lm_weight'] <= 0.8):
                vetoLM = True
            if lm == '9' and not (item['lm_weight'] == 1.0 or item['lm_weight'] == 0.0):
                vetoLM = True
            if not (vetoAtt or vetoCTC or vetoLM):
                DRs.append(item)
    return DRs


def appendAMdata(results: decPlotter, DRs):
    entriesAM = ["elayers", "eunits", "eprojs", "dlayers", "dunits", "adim", "aconv_chans", "aconv_filts", "mtlalpha",
                 "param_count"]
    for item in DRs:
        idxAM = item['AM-Index']
        for entry in entriesAM:
            item[entry] = results.decResults.AMs[idxAM][entry]
    return DRs



def showBestDecodingResult(resultFile="./results.txt", target='CER-Sum/Avg', rtask="test", filter=None):
    results = decPlotter(resultFile=resultFile, autoLoad=True)
    bestDR = {target: 100.0, "name": "None found."}
    DRs = filterDecodingResults(results, filter)
    # find best model
    rtaskDevList = []
    for item in DRs:
        if item['rtask'] == rtask:
            if not target in item:
                print(item)
                raise
            if item[target] < bestDR[target]:
                bestDR = item
        else:
            # will need later
            rtaskDevList.append(item)
    # find corresponding entry in rtaskDevList, get CER and WER and append to bestDR..... naaahh
    bestDR['test-WER'] = 0.0
    bestDR['test-CER'] = 0.0
    bestDR['dev-WER'] = 0.0
    bestDR['dev-CER'] = 0.0
    if rtask == 'dev':
        othertask = 'test'
    else:
        othertask = 'dev'
    bestDR[rtask + '-WER'] = bestDR['WER-Sum/Avg']
    bestDR[rtask + '-CER'] = bestDR['CER-Sum/Avg']
    myDevDRpath = bestDR['pathname'].replace(rtask, othertask)
    for item in rtaskDevList:
        if item['pathname'] == myDevDRpath:
            bestDR[othertask + '-CER'] = item['CER-Sum/Avg']
            bestDR[othertask + '-WER'] = item['WER-Sum/Avg']
            break
    # extract data from AM
    DRs =  appendAMdata(results, DRs)
    print("Best decoding result with {} = {}. (from {} out of {})".format(target, bestDR[target], len(DRs), len(results.decResults.DRs)))
    return bestDR


def printDecResultTable(latex=False):
    #tableString = '\nelayers & $6$ & {} & {} & {}  \\\\ \neunits & $320$ & {} & {} & {}  \\\\ \neprojs &  $320$ & {} & {} & {}  \\\\ \ndlayers &  $1$ & {} & {} & {}  \\\\ \ndunits &  $300$ & {} & {} & {}  \\\\ \nadim &  $-$ & {} & {} & {}  \\\\ \naconv\\_chans & $10$ & {} & {} & {}   \\\\ \naconv\\_filts &  $100$ & {} & {} & {}  \\\\ \nmtlalpha &  $0.5$ & {} & {} & {}  \\\\ \nlm\\_weight &  $1.0$ & {} & {} & {}  \\\\ \nctc\\_weight & $0.3$ & {} & {} & {}   \\\\ \n\t\\hline \n\t\\hline \ndev/CER & $10.8$ & {} & {} & {}  \\\\ \ndev/WER & $19.8$ & {} & {} & {}   \\\\ \ntest/CER & $10.1$ & {} & {} & {}  \\\\ \ntest/WER & $18.6$ & {} & {} & {}   \\\\ \n'
    filterList = [None, 'att-ctc-LM-034', 'att-ctc-LM-134', 'att-ctc-LM-330', 'att-ctc-LM-030', 'att-ctc-LM-130']
    if latex:
        outDict = {    # komprimierte Version
         'elayers':    'Encoder layers                   & $6$       ',
         'eunits':     'Encoder BLSTM cells              & $320$     ',
         'eprojs':     'Projection units                 &  $320$    ',
         'dlayers':    'Decoder Layers                   &  $1$      ',
         'dunits':     'Decoder LSTM cells               &  $300$    ',
         'adim':       'Attention neurons                &  $320$    ',
         'aconv_chans':'Att. channels in $K$             & $10$      ',
         'aconv_filts':'Att. filters in $K$              &  $100$    ',
         'mtlalpha':  r'Multi-obj. (training) $\kappa$   &  $0.5$    ',
         'param_count':'Model size ($1e6$)               & $18.7$    ',
         'lm_weight': r'RNNLM weight $\beta$             &  $1.0$    ',
         'ctc_weight':r'Multi-obj. (beam) $\lambda$      & $0.3$     ',
         'dev-CER':    'TEDlium dev/CER                  & $10.8$    ',
         'dev-WER':    'TEDlium dev/WER                  & $19.8$    ',
         'test-CER':   'TEDlium test/CER                 & $10.1$    ',
         'test-WER':   'TEDlium test/WER                 & $18.6$    '}
        print("Filters: ", filterList)
        for filter in filterList:
            rtask = 'dev'
            if filter == 'att-ctc-LM-034':
                rtask = 'dev'
            myModel = showBestDecodingResult(filter=filter, rtask=rtask)
            for datapoint in outDict:
                if datapoint in ['param_count']:
                    # abbrev formatiing
                    outDict[datapoint] += '& ${:.1f}$ '.format(myModel[datapoint] / 1000000)
                elif datapoint in ['mtlalpha', "lm_weight", "ctc_weight"]:
                    #print(myModel[datapoint])
                    outDict[datapoint] += '& ${:.2f}$ '.format(myModel[datapoint])
                elif datapoint in ['dev-CER', 'dev-WER', 'test-CER', 'test-WER'] and filter is None:
                    outDict[datapoint] += '& $\\mathbf{' + '{}'.format(myModel[datapoint]) + '}$ '
                else:
                    outDict[datapoint] += '& ${}$ '.format(myModel[datapoint])
        for item in outDict:
            print(outDict[item], r' \\ \hline ')
    else:
        testDataList = ["elayers", "eunits", "eprojs", "dlayers", "dunits", "adim", "aconv_chans", "aconv_filts",
                        "mtlalpha", "lm_weight", "ctc_weight",
                        "CER-Sum/Avg", "WER-Sum/Avg", "param_count",
                        'dev-CER', 'dev-WER']  # added dev results
        for filter in filterList:
            myModel = showBestDecodingResult(filter=filter)
            for datapoint in testDataList:
                print(filter, "  -  ", datapoint, ': ', myModel[datapoint])



def showBestAcousticResult(resultFile="./results.txt", target='main/acc', minimize=False):
    """
    examples:     showBestAcousticResult(target="main/cer", minimize=True)
                  showBestAcousticResult(target="main/acc", minimize=False)
                  showBestAcousticResult(target="main/loss_ctc", minimize=True)
    :param resultFile:
    :param target:
    :param minimize:
    :return:
    """
    results = decPlotter(resultFile=resultFile, autoLoad=True)
    bestAM = {target: 100.0, "name": "None found."}
    if not minimize:
        bestAM[target] = 0.0
    numSkipped = 0
    for item in results.decResults.AMs:
        if (not target in item) or (item[target] == 0.0):
            numSkipped += 1
            #print("skipped ", item["pathname"])
        elif minimize and (item[target] < bestAM[target]):
            bestAM = item
        elif (not minimize) and (item[target] > bestAM[target]):
            bestAM = item
    resStr = "Best decoding result with {} {}. (skipped: {}/{})"
    print(resStr.format(bestAM[target], target, numSkipped, len(results.decResults.AMs)))
    return bestAM


def plotXtoYfromDRs(resultFile="./results.txt", Xname="CER-Sum/Avg", Yname="WER-Sum/Avg", pltName="CERtoWER", filterBad=False, ending="png"):
    results = decPlotter(resultFile=resultFile, autoLoad=True)
    faillist = []
    plt.clf()
    plt.xlabel(Xname)
    plt.ylabel(Yname)
    if filterBad:
        results.decResults.filterThresholdAMacc = 0.75
        results.decResults.filterThresholdCER   = 35.0
        results.decResults.applyFilter()
    for rtask in ['test', 'dev']:
        Xlist = []
        Ylist = []
        for item in results.decResults.DRs:
            if item['rtask'] == rtask:
                # check if key exists!
                if (Xname in item) and (Yname in item):
                    Xlist.append(item[Xname])
                    Ylist.append(item[Yname])
                else:
                    faillist.append(item['expName'] + ' - ' + item['pathname'])
        print("Number of missed models: {}/{}".format(len(faillist), len(results.decResults.DRs)))
        X = np.asarray(Xlist)
        Y = np.asarray(Ylist)
        trend = np.poly1d(np.polyfit(X, Y, 1))
        if rtask == 'test':
            plt.plot(X,Y, 'x', X, trend(X), '-')
        else:
            plt.plot(X, Y, '+')
    plt.savefig(results.outDir + "{}_{}_f{}_DRs.{}".format(pltName, rtask, filterBad, ending))


def plotXtoYfromAMandDR(resultFile="./results.txt", Xname="main/acc", Yname="CER-Sum/Avg", pltName="accToCER", rtask='test', filterBad=False, ending="png"):
    results = decPlotter(resultFile=resultFile, autoLoad=True)
    faillist = []
    plt.clf()
    plt.xlabel(Xname)
    plt.ylabel(rtask +'/'+ Yname)
    Xlist = []
    Ylist = []
    if filterBad:
        results.decResults.filterThresholdAMacc = 0.75
        results.decResults.filterThresholdCER   = 35.0
        results.decResults.applyFilter()
    for item in results.decResults.DRs:
        if item['rtask'] == rtask:
            # check if key exists!
            if (Yname in item) and (Xname in results.decResults.AMs[item['AM-Index']]):
                Xlist.append(results.decResults.AMs[item['AM-Index']][Xname])
                Ylist.append(item[Yname])
            else:
                print("missed: ", item['expName'] + ' - ' + item['pathname'])
                faillist.append(item['expName'] + ' - ' + item['pathname'])
    print("Number of missed models: {}/{} (because of key mismatches)".format(len(faillist), len(results.decResults.DRs)))
    if len(faillist) >= 1:
        print(faillist[0])
    X = np.asarray(Xlist)
    Y = np.asarray(Ylist)
    trend = np.poly1d(np.polyfit(X, Y, 1))
    plt.plot(X,Y, 'x', X, trend(X), '-')
    plt.savefig(results.outDir + "{}_{}_f{}_AMsDRs.{}".format(pltName, rtask, filterBad, ending))


def plotXtoYfromAMs(resultFile="./results.txt", Xname="param_count", Yname="main/acc", pltName="paramcntToAcc", filterBad=True, ending="png"):
    results = decPlotter(resultFile=resultFile, autoLoad=True)
    faillist = []
    plt.clf()
    plt.xlabel(Xname)
    plt.ylabel(Yname)
    Xlist = []
    Ylist = []
    if filterBad:
        results.decResults.filterThresholdAMacc = 0.75
        results.decResults.filterThresholdCER   = 35.0
        results.decResults.applyFilter()
    AMlist = results.decResults.AMs
    for item in AMlist:
        if (Yname in item) and (Xname in item):
            Xlist.append(item[Xname])
            Ylist.append(item[Yname])
        else:
            print("missed: ", item['pathname'])
            faillist.append(item['pathname'])
    print("Ploitting {} to {}".format(Xname, Yname), " - Number of missed models: {}/{}".format(len(faillist), len(AMlist)))
    if len(faillist) >= 1:
        print(faillist[0])
    X = np.asarray(Xlist)
    Y = np.asarray(Ylist)
    trend = np.poly1d(np.polyfit(X, Y, 1))
    plt.plot(X,Y, 'x', X, trend(X), '-')
    plt.savefig(results.outDir + "{}_filter{}_AMsDRs.{}".format(pltName, filterBad, ending))


def genPlots():
    for filterBad in [True, False]:
        plotXtoYfromDRs(Xname="CER-Sum/Avg", Yname="WER-Sum/Avg", pltName="CERtoWER", filterBad=filterBad)
        plotXtoYfromDRs(Xname="ctc_weight", Yname="CER-Sum/Avg", pltName="CTCtoCER", filterBad=filterBad)
        plotXtoYfromDRs(Xname="lm_weight", Yname="CER-Sum/Avg", pltName="LMWtoCER", filterBad=filterBad)
        plotXtoYfromDRs(Xname="ctc_weight", Yname="CER-Mean", pltName="CTCtoCERmean", filterBad=filterBad)
        plotXtoYfromDRs(Xname="lm_weight", Yname="CER-Mean", pltName="LMWtoCERmean", filterBad=filterBad)
        plotXtoYfromAMandDR(Xname="main/acc", Yname="CER-Sum/Avg", pltName="accToCER", rtask='test', filterBad=filterBad)
        plotXtoYfromAMandDR(Xname="mtlalpha", Yname="CER-Sum/Avg", pltName="mtlalphaToCER", rtask='test',filterBad=filterBad)
        plotXtoYfromAMandDR(Xname="elayers", Yname="CER-Sum/Avg", pltName="elayersToCER", rtask='test', filterBad=filterBad)
        plotXtoYfromAMandDR(Xname="dlayers", Yname="CER-Sum/Avg", pltName="dlayersToCER", rtask='test', filterBad=filterBad)
        plotXtoYfromAMandDR(Xname="dunits", Yname="CER-Sum/Avg", pltName="dunitsToCER", rtask='test', filterBad=filterBad)
        plotXtoYfromAMandDR(Xname="epoch", Yname="CER-Sum/Avg", pltName="epochsToCER", rtask='test',filterBad=filterBad)
        plotXtoYfromAMandDR(Xname="adim", Yname="CER-Sum/Avg", pltName="adimToCER", rtask='test', filterBad=filterBad)
        plotXtoYfromAMandDR(Xname="eunits", Yname="CER-Sum/Avg", pltName="enunitsToCER", rtask='test', filterBad=filterBad)
        plotXtoYfromAMandDR(Xname="aconv_chans", Yname="CER-Sum/Avg", pltName="aconvchansToCER", rtask='test', filterBad=filterBad)
        plotXtoYfromAMandDR(Xname="aconv_filts", Yname="CER-Sum/Avg", pltName="aconvfiltsToCER", rtask='test', filterBad=filterBad)
        plotXtoYfromAMs(Xname="mtlalpha", Yname="main/acc", pltName="attToacc", filterBad=filterBad)
        for netPart in ['att', 'locconv', 'dec','ctc', '']:
            plotXtoYfromAMs(Xname=netPart + "param_count", Yname="main/acc", pltName="paramcnt_"+ netPart +"ToAcc", filterBad=filterBad)
            plotXtoYfromAMandDR(Xname=netPart + "param_count", Yname="CER-Sum/Avg", pltName="paramcnt_"+ netPart +"ToCER", rtask='test', filterBad=filterBad)


def delme(resultFile="./results.txt"):
    Xname = "CER-Sum/Avg"
    Yname = "WER-Sum/Avg"
    results = decPlotter(resultFile=resultFile, autoLoad=True)
    out = {}
    out['rtask'] = []
    out[Xname] = []
    out[Yname] = []
    faillist = []
    for rtask in ['test', 'dev']:
        for item in results.decResults.DRs:
            if item['rtask'] == rtask:
                # check if key exists!
                if (Xname in item) and (Yname in item):
                    out['rtask'].append(rtask == 'test')
                    out[Xname].append(item[Xname])
                    out[Yname].append(item[Yname])
                else:
                    faillist.append(item['expName'] + ' - ' + item['pathname'])
        print("Number of missed models: {}/{}".format(len(faillist), len(results.decResults.DRs)))
    return out

def dataSbCER2WER_alt(ax=plt, resultFile="./results.txt"):
    Xname = "CER-Sum/Avg"
    Yname = "WER-Sum/Avg"
    results = decPlotter(resultFile=resultFile, autoLoad=True)
    faillist = []
    plt.xlabel(Xname)
    plt.ylabel(Yname)
    for rtask in ['test', 'dev']:
        Xlist = []
        Ylist = []
        for item in results.decResults.DRs:
            if not Xname in item or item[Xname] > 60.0:
                ignore = True
            else:
                ignore = False
            if item['rtask'] == rtask and not ignore:
                # check if key exists!
                if (Xname in item) and (Yname in item):
                    Xlist.append(item[Xname])
                    Ylist.append(item[Yname])
                else:
                    faillist.append(item['expName'] + ' - ' + item['pathname'])
        print("Number of missed models: {}/{}".format(len(faillist), len(results.decResults.DRs)))
        X = np.asarray(Xlist)
        Y = np.asarray(Ylist)
        if rtask == 'test':
            ax.plot(X,Y, 'x')
        else:
            ax.plot(X, Y, '+')

def dataSbCER2WER(ax=plt, resultFile="./results.txt"):
    Xname = "CER-Sum/Avg"
    Yname = "WER-Sum/Avg"
    results = decPlotter(resultFile=resultFile, autoLoad=True)
    faillist = []
    plt.xlabel(Xname)
    plt.ylabel(Yname)
    for rtask in ['test', 'dev']:
        Xlist = []
        Ylist = []
        for item in results.decResults.DRs:
            if not Xname in item or item[Xname] > 60.0:
                ignore = True
            else:
                ignore = False
            if item['rtask'] == rtask and not ignore:
                # check if key exists!
                if (Xname in item) and (Yname in item):
                    Xlist.append(item[Xname])
                    Ylist.append(item[Yname])
                else:
                    faillist.append(item['expName'] + ' - ' + item['pathname'])
        print("Number of missed models: {}/{}".format(len(faillist), len(results.decResults.DRs)))
        X = np.asarray(Xlist)
        Y = np.asarray(Ylist)
        if rtask == 'test':
            ax.plot(X,Y, 'x')
        else:
            ax.plot(X, Y, '+')

def dataSbCER2WER_beta(ax=plt, resultFile="./results.txt"):
    Xname = "CER-Sum/Avg"
    Yname = "WER-Sum/Avg"
    results = decPlotter(resultFile=resultFile, autoLoad=True)
    faillist = []
    plt.xlabel("a) CER-Sum/Avg (test and dev)")
    plt.ylabel(Yname)
    Xlist = {}
    Ylist = {}
    keys = [ 'β <= 0.4' , 'β > 0.4' ]
    keys = [ 'other beam\n search results' , 'attention-only\n with RNNLM', 'CTC-only\n without RNNLM' ]
    for nk in keys:
        Xlist[nk] = []
        Ylist[nk] = []
    for item in results.decResults.DRs:
        if (Xname in item) and (Yname in item):
            nk = 0
            if item['ctc_weight'] > 0.8 and item['lm_weight'] < 0.1:
                nk = 2
            elif item['ctc_weight'] < 0.05 and item['lm_weight'] > 0.3:
                nk = 1
            Xlist[keys[nk]].append(item[Xname])
            Ylist[keys[nk]].append(item[Yname])
        else:
            faillist.append(item['expName'] + ' - ' + item['pathname'])
        print("Number of missed models: {}/{}".format(len(faillist), len(results.decResults.DRs)))
    ax.set_ylim(bottom=15.0, top=90.0)
    ax.set_xlim(left=5.0, right=70.0)
    mkers = ['+', 'x', '1', '2']
    for nk in keys:
        X = np.array(Xlist[nk])
        Y = np.array(Ylist[nk])
        # print(X, Y)
        ax.plot(X, Y, mkers.pop(), label= nk)
    ax.legend(loc=4)

def dataSbCER2WER_gamma(ax=plt, resultFile="./results.txt"):
    "wie beta, nur die vergrößerung der Spitze"
    Xname = "CER-Sum/Avg"
    Yname = "WER-Sum/Avg"
    results = decPlotter(resultFile=resultFile, autoLoad=True)
    faillist = []
    plt.xlabel("a) CER-Sum/Avg (test and dev)")
    plt.ylabel(Yname)
    Xlist = {}
    Ylist = {}
    keys = [ 'other beam\n search results' , 'less RNNLM', 'more RNNLM' ]
    for nk in keys:
        Xlist[nk] = []
        Ylist[nk] = []
    for item in results.decResults.DRs:
        if (Xname in item) and (Yname in item):
            nk = 0
            if False:
                if item['ctc_weight'] < 0.3 and item['lm_weight'] < 0.5:
                    nk = 2
                elif item['ctc_weight'] > 0.5 and item['lm_weight'] > 0.5:
                    nk = 1
            else:
                if item['ctc_weight'] > 0.7 and item['lm_weight'] < 0.5:
                    nk = 2
                elif item['ctc_weight'] < 0.05 and item['lm_weight'] > 0.3:
                    nk = 1
            Xlist[keys[nk]].append(item[Xname])
            Ylist[keys[nk]].append(item[Yname])
        else:
            faillist.append(item['expName'] + ' - ' + item['pathname'])
        print("Number of missed models: {}/{}".format(len(faillist), len(results.decResults.DRs)))
    ax.set_ylim(bottom=17.5, top=22.0)
    ax.set_xlim(left=8.5, right=12.0)
    mkers = ['+', 'x', '1', '2']
    for nk in keys:
        X = np.array(Xlist[nk])
        Y = np.array(Ylist[nk])
        # print(X, Y)
        ax.plot(X, Y, mkers.pop(), label= nk)
    ax.legend(loc=4)



def dataElayersCER(ax=plt, resultFile="./results.txt"):
    Xname = "elayers"
    Yname = "CER-Sum/Avg"
    ax.set_xlabel("Encoder layers")
    ax.set_ylabel("CER-Sum/Avg (test)")
    rtask = "test"
    filterBad = False
    results = decPlotter(resultFile=resultFile, autoLoad=True)
    faillist = []
    Xlist = {}
    if filterBad:
        results.decResults.filterThresholdCER   = 25.0
        results.decResults.applyFilter()
    for item in results.decResults.DRs:
        if item['rtask'] == rtask:
            # check if key exists!
            if (Yname in item) and (Xname in results.decResults.AMs[item['AM-Index']]):
                XVal = results.decResults.AMs[item['AM-Index']][Xname]
                if XVal not in Xlist:
                    Xlist[XVal] = []
                Xlist[XVal].append(item[Yname])
            else:
                #print("missed: ", item['expName'] + ' - ' + item['pathname'])
                faillist.append(item['expName'] + ' - ' + item['pathname'])
    print("Number of missed models: {}/{} (because of key mismatches)".format(len(faillist), len(results.decResults.DRs)))
    Ylist = []
    for i in range(10):
        if i in Xlist:
            Ylist.append(Xlist[i])
    print(Ylist)
    #
    # ax.boxplot(Ylist)
    ax.set_ylim(bottom=8.0, top=25.0)
    ax.violinplot(Ylist, showmedians=True)



def dataHyperHyper(ax=plt, resultFile="./results.txt"):
    from scipy.stats.kde import gaussian_kde
    Xname = "CER-Sum/Avg"
    results = decPlotter(resultFile=resultFile, autoLoad=True)
    faillist = []
    plt.xlabel("a) CER-Sum/Avg (test and dev)")
    plt.ylabel('CER-Sum/Avg')
    Xlist = []
    Ylist = []
    Zlist = []
    keys = [ 'lm_weight' , 'ctc_weight' ]
    DRs = filterDecodingResults(results, filter='att-ctc-LM-773')
    for item in DRs:
        if (Xname in item):
            Xlist.append(item['lm_weight'])
            Ylist.append(item['ctc_weight'])
            Zlist.append(item[Xname])
        else:
            faillist.append(item['expName'] + ' - ' + item['pathname'])
        print("Number of missed models: {}/{}".format(len(faillist), len(results.decResults.DRs)))
    #ax.set_ylim(bottom=6.0, top=90.0)
    #ax.set_xlim(left=5.0, right=70.0)
    x = np.array(Xlist)
    y = np.array(Ylist)
    z = np.array(Zlist)
    #ax.plot(x, y, 'x')
    #heatmap, xedges, yedges = np.histogram2d(x, y, bins=100)
    #heatmap = gaussian_filter(heatmap, sigma=15)
    #extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #ax.imshow(heatmap.T, extent=extent, origin='lower')
    ax.plot(y, z, 'x')



def dataPC2CER(ax=plt, resultFile="./results.txt"):
    Xname = "param_count"
    XTwinName = 'main/acc'
    Yname = "CER-Sum/Avg"
    ax.set_xlabel("b) Network model size")
    ax.set_ylabel("CER-Sum/Avg (test)")
    rtask = "test"
    filterBad = False
    results = decPlotter(resultFile=resultFile, autoLoad=True)
    faillist = []
    Xlist = []
    XTwin = []
    Ylist = []
    if filterBad:
        results.decResults.filterThresholdCER   = 20.0
        results.decResults.applyFilter()
    for item in results.decResults.DRs:
        if item['rtask'] == rtask:
            # check if key exists!
            if (Yname in item) and (Xname in results.decResults.AMs[item['AM-Index']]) and (XTwinName in results.decResults.AMs[item['AM-Index']]):
                Xlist.append(results.decResults.AMs[item['AM-Index']][Xname])
                XTwin.append(results.decResults.AMs[item['AM-Index']][XTwinName])
                Ylist.append(item[Yname])
            else:
                print("missed: ", item['expName'] + ' - ' + item['pathname'])
                faillist.append(item['expName'] + ' - ' + item['pathname'])
    print("Number of missed models: {}/{} (because of key mismatches)".format(len(faillist), len(results.decResults.DRs)))
    if len(faillist) >= 1:
        print(faillist[0])
    X = np.asarray(Xlist)
    Y = np.asarray(Ylist)
    ax.tick_params()
    axtwin = ax.twinx()
    ax.set_xlim(right=38000000.0)
    ax.set_ylim(bottom=9.0, top=25.0)
    if False:
        axtwin.set_yticks(np.array([1.0, 0.9, 0.8, 0.7]))
        axtwin.set_yticklabels(['1.', '.9', '.8', '.7'])
    else:
        axtwin.set_yticks(np.array([1.0, 0.9, 0.8, 0.7, 0.95, 0.85, 0.75]))
        axtwin.set_yticklabels(['1.0', '.90', '.80', '.70', '.95', '.85', '.75'])
    axtwin.set_ylabel("Attention decoder accuracy")
    axtwin.set_ylim(bottom=0.8, top=1.0)
    axtwin.invert_yaxis()
    #ax.set_xlim(left=5.0, right=60.0)
    ax.plot(X, Y, '+', label='CER')
    ax.plot(np.array([]),np.array([]), 'rx', label='acc')
    #sns.scatterplot(x=X, y=Y, marker='x', ax=ax)
    Y = np.asarray(XTwin)
    axtwin.plot(X, Y, 'rx')
    #axtwin.plot(X, Y, '+', label='acc')
    #sns.scatterplot(x=X, y=Y, marker='+', ax=axtwin)
    #sns.scatterplot(x=X, y=Y, marker='x', ax=ax)
    ax.legend(loc=1)



def dataAcc2CER(ax=plt, resultFile="./results.txt"):
    Xname = "main/acc"
    Yname = "CER-Sum/Avg"
    ax.set_xlabel("Attention accuracy")
    ax.set_ylabel("CER-Sum/Avg (test)")
    rtask = "test"
    filterBad = True
    results = decPlotter(resultFile=resultFile, autoLoad=True)
    faillist = []
    Xlist = []
    Ylist = []
    if filterBad:
        #results.decResults.filterThresholdAMacc = 0.75
        results.decResults.filterThresholdCER   = 40.0
        results.decResults.applyFilter()
    for item in results.decResults.DRs:
        if item['rtask'] == rtask:
            # check if key exists!
            if (Yname in item) and (Xname in results.decResults.AMs[item['AM-Index']]):
                Xlist.append(results.decResults.AMs[item['AM-Index']][Xname])
                Ylist.append(item[Yname])
            else:
                print("missed: ", item['expName'] + ' - ' + item['pathname'])
                faillist.append(item['expName'] + ' - ' + item['pathname'])
    print("Number of missed models: {}/{} (because of key mismatches)".format(len(faillist), len(results.decResults.DRs)))
    if len(faillist) >= 1:
        print(faillist[0])
    X = np.asarray(Xlist)
    Y = np.asarray(Ylist)
    sns.scatterplot(x=X, y=Y, marker='+', ax=ax)


def dataCERoverview(ax=plt, resultFile="./results.txt"):
    Yname = "CER-Sum/Avg"
    ax.set_ylim(bottom=7.0, top=85.0)
    ax.set_xlabel("c) Selected categories of results over the TEDlium v2 test set.")
    ax.set_ylabel("CER-Sum/Avg (test)")
    filterList = [None, 'Att-only', 'CTC-only', 'no-LM', 'no-LM-Att', 'no-LM-CTC']

    #Params = 'κ λ β' 0<κ<1
    #              hybrid,  att only   dec             |     att                          |                                   |
    filterList = [None, #'att-ctc-LM-888',
                  'att-ctc-LM-720',  'att-ctc-LM-726',  # hybrid model, hybrid decoding
                  'att-ctc-LM-700',  'att-ctc-LM-701',  # hybrid model, att decoding
                  'att-ctc-LM-000', 'att-ctc-LM-006', # att model ,att decoding
                  'att-ctc-LM-710',   'att-ctc-LM-711', # hybrid model, ctc decoding
                  'att-ctc-LM-110', 'att-ctc-LM-111'] # ctc model + decoding
    labelList =  ['overall results', #'best configuration',
                  'hybrid model/beam w/o LM', 'hybrid model/beam with LM' ,
                  'hybrid model, att. beam w/o LM', 'hybrid model, att. beam with LM' ,
                  'attention model/beam w/o LM' , 'attention model/beam with LM' ,
                  'hybrid model, CTC beam w/o LM' , 'hybrid model, CTC beam with LM' ,
                  'CTC model/beam w/o LM' , 'CTC model/beam with LM']
    #addLabels =  ['hybr., att.bs., w/o LM']
    #labelList = ()
    #for i in range(len(labelList)):
    #    labelList[i] = "{}".format("ABCDEFGHIJKLMNOPQ"[i])# + labelList[i]

    filterBad = False
    if filterBad:
        filterAcc = 0.8
        filterCER = 25.0
    else:
        filterAcc = None
        filterCER = None
    Xlist = {}
    for i, filter in enumerate(filterList):
        Xlist[filter] = []
        results = decPlotter(resultFile=resultFile, autoLoad=True)
        DRs = filterDecodingResults(results, filter, filterAcc=filterAcc, filterCER=filterCER)
        if False:
            # frisiere Ergebnisse
            if filter == 'att-ctc-LM-311':
                DRs = [x for x in DRs if x['CER-Sum/Avg'] < 27.0]
            if filter == 'att-ctc-LM-111':
                DRs = [x for x in DRs if x['CER-Sum/Avg'] < 24.0]
        print(filter, ' # ', len(DRs), '/ ', len(results.decResults.DRs))
        for item in DRs:
            if Yname in item and item['rtask'] == 'test':
                Xlist[filter].append(item[Yname])
            #else:
            #    print ("WTF")
            #    print(item)
            #    raise
    Ylist = []
    numClasses = len(labelList)
    numFillUp = 7 # 7 ..12
    #plt.xticks(rotation=90)
    for i in range(numClasses):
        labelList[i] = "{}: N={}; ".format(i+1, len(Xlist[filterList[i]])) + labelList[i]
    for key in Xlist:
        Ylist.append(Xlist[key])
    for i in range(numFillUp):
        Ylist.append(np.array([]))
    plotlabels = []
    for i in range(8):
        plotlabels += ['{}'.format(i+1)]
    plotlabels += ['', '', '']
    #Ylist += [, , ]
    ax.boxplot(Ylist)# ), labels=labelList)
    han = []
    for i in range(len(labelList)):
        han += [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]
    #ax.annotate((0,0), labelList)
    if True:
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        #ax.set_position([box.x0, box.y0, box.width, box.height])
        ax.set_xticks(list(range(1, numClasses+1)))
        ax.set_xticklabels(list(range(1, numClasses+1)))
        ax.legend(handles=han, handlelength=0, handletextpad=0, labels=labelList, fancybox=True, loc=1)
    else:
        ax.legend(handles=han, handlelength=0, handletextpad=0, labels=labelList, fancybox=True,
                  loc=1)  # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html



def genSeabornPlots():
    if not SEABORN_AVAILABLE:
        print("No Seaborn available. Abbruch.")
        return False
    print("Generatign Seaborn Tables for Paper!!!!!!!!!!!!!!!!!!!!")
    # ------  part 1. CER to WER, unfiltered
    sns.set(style="whitegrid")
    # https://stackoverflow.com/questions/37576160/how-do-i-add-category-names-to-my-seaborn-boxplot-when-my-data-is-from-a-python
    if False:
        fig = plt.figure(figsize=(8.27,11.69))
        gs = GridSpec(3, 2, figure=fig)
        # CER zu WER
        ax0 = fig.add_subplot(gs[0, 0])
        dataSbCER2WER_beta(ax=ax0)
        # elayers zu CER
        ax1 = fig.add_subplot(gs[0, 1])
        dataSbCER2WER_beta(ax=ax1)
        # paramcount zu CER
        ax2 = fig.add_subplot(gs[1, 0])
        dataPC2CER(ax=ax2)
        # paramcount zu CER
        ax3 = fig.add_subplot(gs[1, 1])
        dataAcc2CER(ax=ax3)
        # paramcount zu CER
        ax4 = fig.add_subplot(gs[2, :])
        dataCERoverview(ax=ax4)
    else:
        fig = plt.figure(figsize=(8.27,9.5))
        gs = GridSpec(2, 2, figure=fig)
        ax0 = fig.add_subplot(gs[0, 0])
        dataSbCER2WER_beta(ax=ax0)
        # paramcount zu CER
        sns.set_style("white")
        ax2 = fig.add_subplot(gs[0, 1])
        if False:
            dataSbCER2WER_gamma(ax=ax2)
        else:
            dataPC2CER(ax=ax2)
        # paramcount zu CER
        sns.set_style("whitegrid")
        ax4 = fig.add_subplot(gs[1:, :])
        dataCERoverview(ax=ax4)
    plt.tight_layout()
    plt.savefig("plots/seaborn.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("plots/seaborn.eps", format='eps', bbox_inches='tight')
    #plt.show()


def genPresentationPlots():
    if not SEABORN_AVAILABLE:
        print("No Seaborn available. Abbruch.")
        return False
    print("Generatign Presentaiton Plots")
    # ------  part 1. CER to WER, unfiltered
    sns.set(style="whitegrid")
    # https://stackoverflow.com/questions/37576160/how-do-i-add-category-names-to-my-seaborn-boxplot-when-my-data-is-from-a-python

    # WER-CER
    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    dataSbCER2WER_beta(ax=ax0)
    plt.tight_layout()
    plt.savefig("plots/presWER.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("plots/presWER.eps", format='eps', bbox_inches='tight')

    # WER-CER enlarged
    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    dataSbCER2WER_gamma(ax=ax0)
    plt.tight_layout()
    plt.savefig("plots/presWERtip.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("plots/presWERtip.eps", format='eps', bbox_inches='tight')

    # CER acc model size
    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    dataPC2CER(ax=ax0)
    plt.tight_layout()
    plt.savefig("plots/presSize.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("plots/presSize.eps", format='eps', bbox_inches='tight')

    # CER overview violin
    fig = plt.figure(figsize=(10.0,4.0))
    gs = GridSpec(1, 1, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    dataCERoverview(ax=ax0)
    plt.tight_layout()
    plt.savefig("plots/presViolin.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("plots/presViolin.eps", format='eps', bbox_inches='tight')


# --------------------------------- Parsers and helper functions

def extractParamsFromPath(path, parseStr=AMPARAMSTR):
    substrlist = path.split('/')
    for substr in substrlist[::-1]:
        extracted = parse(parseStr, substr)
        if extracted is not None:
            extracted = extracted.named # convert to dict
            extracted['fullstr'] = substr
            return extracted
    return None

def extractScliteResults(resultFile):
    results = {}
    with open(resultFile, 'r') as myfile:
        data = []
        readTrigger = 0
        for line in myfile:
            if line.startswith("|===========") or line.startswith("`----------"):
                readTrigger += 1
            elif readTrigger in [1, 2]:
                data.append(line)
    for line in data:
        tableEntry = line.replace("|", '').split()
        results[tableEntry[0]] = float(tableEntry[7])
        results[tableEntry[0] + "-NumSnt"] = float(tableEntry[1])
        results[tableEntry[0] + "-NumWrd"] = float(tableEntry[2])
        results[tableEntry[0] + "-Corr"] = float(tableEntry[3])
        results[tableEntry[0] + "-Sub"] = float(tableEntry[4])
        results[tableEntry[0] + "-Del"] = float(tableEntry[5])

        results[tableEntry[0] + "-Ins"] = float(tableEntry[6])
        results[tableEntry[0] + "-Err"] = float(tableEntry[7])
        results[tableEntry[0] + "-S.Err"] = float(tableEntry[8])
    return results


def getAcousticModelResults(expDir="/xxx/delme", verbose=2, excludeUnfinished=True):
    # list all files that fit pattern "train*/results/log"
    filelist = [y for x in os.walk(expDir) for y in glob.glob(os.path.join(x[0], 'results/log'))]
    results = []
    # filter the fresh models so that training of AM can be done in parallel (the log file should be older than at least 2h)
    if excludeUnfinished:
        for i, item in enumerate(filelist):
            age = time.time() - os.path.getctime(item)
            if age < (60*60*2):
                if verbose >= 2:
                    print("Filtered unfinished audio model, age ", age / 3600.0, ": ", item)
                del filelist[i]
    # for each entry,
    for item in filelist:
        if verbose >= 2:
            print("parsing: ", item)
        singleResult = {}
        # parse path and get params.
        extracted = extractParamsFromPath(item)
        singleResult['pathname'] = extracted['fullstr']
        singleResult['train_set'] = extracted['train_set']
        singleResult['etype'] = extracted['etype']
        singleResult['elayers'] = int(extracted['elayers'])
        singleResult['subsample'] = extracted['subsample']
        singleResult['eunits'] = int(extracted['eunits'])
        singleResult['eprojs'] = int(extracted['eprojs'])
        singleResult['dlayers'] = int(extracted['dlayers'])
        singleResult['dunits'] = int(extracted['dunits'])
        singleResult['atype'] = extracted['atype']
        singleResult['adim'] = int(extracted['adim'])
        singleResult['aconv_chans'] = int(extracted['aconv_chans'])
        singleResult['aconv_filts'] = int(extracted['aconv_filts'])
        singleResult['mtlalpha'] = float(extracted['mtlalpha'])
        singleResult['opt'] = extracted['opt']
        singleResult['samp_prob'] = float(extracted['samp_prob'])
        singleResult['batchsize'] = int(extracted['batchsize'])
        singleResult['maxlen_in'] = int(extracted['maxlen_in'])
        singleResult['maxlen_out'] = int(extracted['maxlen_out'])
        # get last log entry
        with open(item, 'r') as myfile:
            data = myfile.read().replace('\n', '').replace('Infinity', '9999999999')
        x = eval(data)
        lastEntry = x[-1]
        for key in lastEntry:
            singleResult[key] = lastEntry[key]
        results.append(singleResult)
    return results


def getLanguageModelResults(expDir="/xxx/delme", verbose=2, skipInvalid=True):
    filelist = [y for x in os.walk(expDir) for y in glob.glob(os.path.join(x[0], 'train_rnnlm_pytorch*'))]
    results = []
    # for each entry,
    for item in filelist:
        if verbose >= 2:
            print("parsing: ", item)
        singleResult = {}
        # parse path and get params.
        extracted = extractParamsFromPath(item, parseStr=LMPARAMSTR)
        singleResult['pathname'] = extracted['fullstr']
        singleResult['lm_layers'] = int(extracted['lm_layers'])
        singleResult['lm_units'] = int(extracted['lm_units'])
        singleResult['lm_opt'] = extracted['lm_opt']
        singleResult['lm_batchsize'] = int(extracted['lm_batchsize'])
        singleResult['lmtag'] = LMTAGSTR.format(**extracted)
        # read
        with open(item + "/log", 'r') as myfile:
            data = myfile.read().replace('\n', '').replace('Infinity', '9999999999')
            if "NaN" in data:
                if verbose >= 2:
                    print ("ERROR: Item not usable, NaN found in ", item)
                data = data.replace('NaN', '9999999999')
                if skipInvalid:
                    continue
        x = eval(data)
        lastEntry = x[-1]
        for key in lastEntry:
            singleResult[key] = lastEntry[key]
        if verbose >= 3:
            print(singleResult)
        results.append(singleResult)
    return results



def getDecodingResults(expDir="/xxx/delme", verbose=2, excludeUnfinished=True):
    filelist = [y for x in os.walk(expDir) for y in glob.glob(os.path.join(x[0], 'decode_*'))]
    results = []
    # for each entry,
    for item in filelist:
        if verbose >= 2:
            print("parsing: ", item)
        singleResult = {}
        # parse path and get params.
        extracted = extractParamsFromPath(item, parseStr=WERPARAMSTR)
        if extracted is None:
            print("ERROR: Could not parse Decoding Result of ", item)
            continue
        singleResult['pathname'] = extracted['fullstr']
        singleResult['rtask'] = extracted['rtask']
        singleResult['beam_size'] = int(extracted['beam_size'])
        singleResult['recog_model'] = extracted['recog_model']
        singleResult['penalty'] = float(extracted['penalty'])
        singleResult['minlenratio'] = float(extracted['minlenratio'])
        singleResult['maxlenratio'] = float(extracted['maxlenratio'])
        singleResult['ctc_weight'] = float(extracted['ctc_weight'])
        singleResult['lm_weight'] = float(extracted['lm_weight'])
        singleResult['lmtag'] = extracted['lmtag']
        # get expdir
        singleResult["expName"] = extractParamsFromPath(item, AMPARAMSTR)['fullstr']
        singleResult["lmName"] = "train_rnnlm_pytorch_" + singleResult['lmtag']
        # read results
        if excludeUnfinished:
            if (not os.path.isfile(item + "/result.txt")) or (not os.path.isfile(item + "/result.wrd.txt")):
                if verbose >= 2:
                    print("FILE NOT FOUND: ", item, "   - (maybe unfinished)")
                continue
        CERresults = extractScliteResults(item + "/result.txt")
        WERresults = extractScliteResults(item + "/result.wrd.txt")
        for key in CERresults:
            singleResult["CER-" + key] = CERresults[key]
        for key in WERresults:
            singleResult["WER-" + key] = WERresults[key]
        if verbose >= 3:
            print(singleResult)
        results.append(singleResult)
    return results


class DecodingResultHandle:
    def __init__(self, expDir="/xxx/delme", verbose=1, skipInvalid=True, autoUpdate=True,
                 filterThresholdAMacc=None, filterThresholdCER=None):
        """
        :param filterThresholdAMacc:  0.75 is a good value
        :param filterThresholdCER:    35.0 is a good value
        """
        self.expDir = expDir
        self.verbose = verbose
        self.skipInvalid = skipInvalid
        self.filterThresholdAMacc = filterThresholdAMacc
        self.filterThresholdCER = filterThresholdCER
        if autoUpdate:
            self.update()

    def update(self):
        self.AMs, self.LMs, self.DRs = getAllResults(self.expDir, self.verbose, self.skipInvalid)
        self.applyFilter()

    def getAMexpdir(self, idx):
        return "exp/" + self.AMs[idx]["pathname"]

    def getAMmtlalpha(self, idx):
        return self.AMs[idx]["mtlalpha"]

    def getLMtag(self, idx):
        return self.LMs[idx]["lmtag"]

    def getParamEntryAM(self):
        return {'name': 'AM-Index', 'type': int, 'lower_bound': 0, 'upper_bound': len(self.AMs) - 1}

    def getParamEntryLM(self):
        return {'name': 'LM-Index', 'type': int, 'lower_bound': 0, 'upper_bound': len(self.LMs) - 1}

    def getCmdStr(self, x, params: list):
        # catch AM-Index and LM-Index
        command = pointToCmdStr(x, params).split()
        amIdx = command.index("--AM-Index")
        command[amIdx] = "--expdir"
        amID = int(command[amIdx + 1])
        command[amIdx + 1] = self.getAMexpdir( amID )
        ctcIdx = command.index("--ctc_weight")
        # CTC mode
        if self.getAMmtlalpha(amID) == 1.0:
            command[ctcIdx + 1] = "1.0"
            command += ["--recog_model", "model.loss.best"]
        # attention mode
        if self.getAMmtlalpha(amID) == 0.0:
            command[ctcIdx + 1] = "0.0"
        lmIdx = command.index("--LM-Index")
        command[lmIdx] = "--lmtag"
        command[lmIdx + 1] = self.getLMtag( int(command[lmIdx + 1]) )
        return " " + " ".join(command)

    def applyFilter(self):
        if self.filterThresholdAMacc:
            AMlist = []
            for item in self.AMs:
                isAtt = False if "mtlalpha" in item and item["mtlalpha"] == 1.0 else True
                if "main/acc" in item and item["main/acc"] <= self.filterThresholdAMacc and isAtt:
                    if self.verbose >= 2:
                        print("fitlered AM: ", item["pathname"])
                else:
                    AMlist.append(item)
            self.AMs = AMlist[:]
        if self.filterThresholdCER:
            DRlist = []
            for item in self.DRs:
                if "CER-Sum/Avg" in item and item["CER-Sum/Avg"] >= self.filterThresholdCER:
                    if self.verbose >= 2:
                        print("fitlered DR: ", item["pathname"])
                else:
                    DRlist.append(item)
            self.DRs= DRlist[:]
        # update the keys LM-index and AM-Index
        self.DRs = relateDRsToAMs(self.AMs, self.LMs, self.DRs, fallbackLM=False)

    def checkDuplicate(self, x):
        requiredHits = len(x)
        hitList = [] # debug
        almost = []
        for y in self.DRs:
            hits = 0
            for item in x:
                if item in y:
                    if isclose(y[item], x[item]):
                        hits += 1
            hitList.append(hits) # debug
            if hits == 3:
                almost.append(y)
            if hits == requiredHits:
                if self.verbose >= 2:
                    print("Duplicate  found: ", x)
                return True
        return False


def getAllResults(expDir="/xxx/delme", verbose=1, skipInvalid=True, fallbackLM=True):
    AMresults = getAcousticModelResults(expDir, verbose)
    LMresults = getLanguageModelResults(expDir, verbose, skipInvalid)
    DecodingResults = getDecodingResults(expDir, verbose)
    # relate Decoding results to LMs and AMs
    DecodingResults = relateDRsToAMs(AMresults, LMresults, DecodingResults, fallbackLM=fallbackLM)
    return AMresults, LMresults, DecodingResults


def relateDRsToAMs(AMresults, LMresults, inputDRs, fallbackLM=True, verbose=1):
    """
    This function updates the keys ['AM-Index'] and ['LM-Index'] in the decoding results
    """
    outputDRs = []
    cntMissedAMs = 0
    for DecRes in inputDRs:
        valid = True
        AMfound = 0
        LMfound = 0
        for i, am in enumerate(AMresults):
            if DecRes['expName'] == am['pathname']:
                DecRes['AM-Index'] = i
                AMfound += 1
        for i, lm in enumerate(LMresults):
            if DecRes['lmName'] == lm['pathname']:
                DecRes['LM-Index'] = i
                LMfound += 1
        if AMfound != 1:
            cntMissedAMs += 1
            if verbose >= 2:
                print("ERROR: found ", AMfound, " Acoustic Models for ", DecRes['pathname'])
            valid = False
        if LMfound != 1:
            print("ERROR: found ", LMfound, " Language Models for ", DecRes['pathname'])
            if fallbackLM:
                DecRes['LM-Index'] = 2
            else:
                valid = False
        if valid:
            outputDRs.append(DecRes)
    if cntMissedAMs and verbose >= 1:
        print("ERROR: missed out on {} AMs, {} of {} decoding results left".format(cntMissedAMs, len(outputDRs), len(inputDRs)))
    return outputDRs


def rndPointsFromParams(params: list, numDraws):
    rndList = []
    for i in range(numDraws):
        pplist = []
        for item in params:
            if item['type'] == int:
                val = np.random.randint(item['lower_bound'], item['upper_bound'] + 1)
            elif item['type'] == float:
                val = np.random.uniform(item['lower_bound'], item['upper_bound'])
            else:
                raise Exception("type of value unknown: " + item['name'] + ' is ' + str(item['type']) )
            pplist.append(float(val))
        rndList.append(pplist)
    # shape should be (numDraws, num_params)
    return np.array(rndList)


def pointToCmdStr(x, params: list):
    x = x.reshape(-1)
    paramCount = len(x)
    cmdstr = ''
    for i in range(paramCount):
        name = params[i]['name']
        # fix strange off-by-one errors for integers
        if params[i]['type'] == int:
            value = np.round(x[i])
            value = params[i]['type'](value)
        else:
            value = params[i]['type'](x[i])
        # check bounds again
        value = max(value, params[i]['lower_bound'])
        value = min(value, params[i]['upper_bound'])
        if params[i]['type'] == float:
            valstr = "{:f}".format(value)
        else:
            valstr = "{}".format(value)
        cmdstr += " --{} {}".format(name, valstr)
    return cmdstr


def traceToXY(trace:list, paramList:list, target:str, fillValue=1.0, verbose=1):
    # iterate over trace for X and Y
    Ylist = []
    Xlist = []
    missedCounter = 0
    for sample in trace:
        if not target in sample:
            missedCounter += 1
            yval = fillValue
        else:
            yval = sample[target]
        Ylist.append(yval)
        Xset = []
        for param in paramList:
            name = param['name']
            Xset.append(sample[name])
        currentRow = np.array(Xset)
        Xlist.append(currentRow)
    Y = np.array(Ylist)
    X = np.array(Xlist)
    if missedCounter != 0 and (verbose >= 2):
        print("Warning: Samples have been filled with a dafault value")
    return X, Y


def paramsToBounds(paramList: list):
    boundList = []
    for param in paramList:
        boundList.append(np.array( [  param['lower_bound'], param['upper_bound']  ]  ))
    myBounds = np.array(boundList)
    return Bounds(myBounds[:,0], myBounds[:,1])

# ----------------------------------- ESPNET model specific

class ArgsForE2E():
    def __init__(self, mtlalpha=0.5, elayers=1, etype='vggblstmp',
                 verbose=0, outdir="/tmp", subsample='1_2_2_1_1',
                 dunits=320,  dlayers=1, lsm_weight=0.0,
                 eunits=300, eprojs=320, atype='dot', adim=320,
                 dtype='lstm', aheads=4, awin=5, lsm_type='',
                 opt="adadelta", eps=1e-8, dropout_rate=0.0,
                 dropout_rate_decoder=0.0, sampling_probability=0.0,
                 aconv_chans=-1, aconv_filts=100):
        self.idims = 83
        self.odims = 34
        self.mtlalpha = mtlalpha
        self.etype = etype
        self.verbose = verbose
        self.char_list = None
        self.outdir = outdir
        self.elayers = elayers
        self.subsample = subsample
        self.eprojs = eprojs
        self.atype = atype
        self.adim = adim
        self.dunits = dunits
        self.dlayers = dlayers
        self.dtype = dtype
        self.aheads = aheads
        self.awin = awin
        self.lsm_type = lsm_type
        self.opt = opt
        self.eps = eps
        self.dropout_rate = dropout_rate
        self.dropout_rate_decoder = dropout_rate_decoder
        self.report_wer = False
        self.report_cer = False
        self.eunits = eunits
        self.lsm_weight = lsm_weight
        self.sampling_probability = sampling_probability
        self.aconv_chans = aconv_chans
        self.aconv_filts = aconv_filts


def getE2EmodelSize(acousticModel: dict, net=None):
    # netList = ['att', 'locconv', 'dec','ctc', '']
    if not ESPNET_AVAILABLE:
        return -1
    args = ArgsForE2E()
    for key, value in  acousticModel.items():
        if hasattr(args, key):
            setattr(args, key, value)
    model = E2E(args.idims, args.odims, args)
    if net == 'att':
        x = sum(p.numel() for p in model.att.parameters())
    elif net == 'locconv':
        x = sum(p.numel() for p in model.att.loc_conv.parameters())
    elif net == 'dec':
        x = sum(p.numel() for p in model.dec.parameters())
    elif net == 'enc':
        x = sum(p.numel() for p in model.enc.parameters())
    elif net == 'ctc':
        x = sum(p.numel() for p in model.att.loc_conv.parameters())
    else:
        x = sum(p.numel() for p in model.parameters())
    return x


def get_baseline_model_size():
    # 18.749716
    if not ESPNET_AVAILABLE:
        return -1
    args = ArgsForE2E()
    args.mtlalpha = 0.5
    args.elayers = 6
    args.eprojs = 320
    args.adim = 320
    args.dunits = 300
    args.dlayers = 1
    args.eunits = 320
    args.aconv_chans = 10
    args.aconv_filts = 100
    model = E2E(args.idims, args.odims, args)
    x = sum(p.numel() for p in model.parameters())
    return x


# ----------------------------------- MAIN
def main(arguments):
    print("!")
    genPresentationPlots()
    # paper plots
    print("Eingelesen.")
    print("genPlots")
    #genPlots()
    print("genSeabornPlots")
    genSeabornPlots()
    print("printDecResultTable")
    printDecResultTable(True)
    print("Done.")
    showBestDecodingResult()

    # To optimize, use this command:
    # plug = decPlug(decParams, egsPath=EXPPATH_REMOTE, target="CER-Mean", dryRun=True, verbose=2)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

