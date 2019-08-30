# Exploring Hybrid CTC/Attention End-to-End Speech Recognition with Gaussian Processes


Abstract:

> Hybrid CTC/attention end-to-end speech recognition combines two powerful concepts. Given a speech feature sequence, the attention mechanism directly outputs a sequence of letters. Connectionist Temporal Classification (CTC) helps to bind the attention mechanism to sequential alignments. This hybrid architecture also gives more degrees of freedom in choosing parameter configurations. We applied Gaussian process optimization to estimate the impact of network parameters and language model weight in decoding towards Character Error Rate (CER), as well as attention accuracy. In total, we trained 70 hybrid CTC/attention networks and performed 590 beam search runs with an RNNLM as language model on the TEDlium v2 test set.
> To our surprise, the results challenge the assumption that CTC primarily regularizes the attention mechanism. We argue in an evidence-based manner that CTC instead regularizes the impact of language model feedback in a one-pass beam search, as letter hypotheses are fed back into the attention mechanism.


See the [presentation](/2019_SPECOM_Exploring.pdf) for more details.


## Contents of the Repository

This repository contains the source code for the paper [Exploring Hybrid CTC/Attention End-to-End Speech Recognition with Gaussian Processes](https://link.springer.com/chapter/10.1007/978-3-030-26061-3_27).

* Gaussian Process (GP) optimizer as described in the paper
* ESPnet plugin for the GP optimizer
* Result parser for ESPnet results
* Routine for calculating ESPnet model sizes
* Visualization functions of results

Experimental results can be found in `results.txt` and read in using `optimizer.py` to generate the plots (see `plots/`).
In this work, ESPnet commit [716ff54](https://github.com/espnet/espnet/commit/716ff548ed013c052a1f4596e4a291449412d21b) was used.
