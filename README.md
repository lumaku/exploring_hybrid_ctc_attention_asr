# Exploring Hybrid CTC/Attention End-to-End Speech Recognition with Gaussian Processes

This repository contains the source code for the paper [Exploring Hybrid CTC/Attention End-to-End Speech Recognition with Gaussian Processes](https://link.springer.com/chapter/10.1007/978-3-030-26061-3_27).

We analyzed the multi-objective training approach from ESPnet that combines CTC and location-aware attention using a Gaussian Process hyperparameter optimizer. The analyzed model consists of two parts.

1. The encoder is a pyramid BLSTM encoder that uses projection layers:

![The Encoder Architecture](/plots/enc.png)


2. The decoding network has three parts. (1) location-aware attention that can be seen on the left part, (2) frame-based classification netowrk that was trained using the CTC loss function, and (3) the beam search that combines two network components with the RNNLM language model into one single probability indicator for generating the transcription.

![The CTC and Attention Networks](/plots/dec.png)

Using Gaussian Process optimization, certain parameter groups were identified. (a) shows the general overview over these groups. (b) shows a certain correlation of network depth and accuracy or reduced CER, respectively. (c) lists the performance of these parameter groups. Notice that attention-only decoding without any language model (6) has already acceptable performance, whereas combining it with a language model deteriorates results (7).

![Result Overview](/plots/seaborn.png)

See the [presentation](/2019_SPECOM_Exploring.pdf) for a short explanation of results or the paper for more details.

If you use these results or a part of the source code in your work, you may cite:

```
@incollection{Kuerzinger_2019,
	doi = {10.1007/978-3-030-26061-3_27},
	url = {https://doi.org/10.1007%2F978-3-030-26061-3_27},
	year = 2019,
	publisher = {Springer International Publishing},
	pages = {258--269},
	author = {Ludwig Kürzinger and Tobias Watzel and Lujun Li and Robert Baumgartner and Gerhard Rigoll},
	title = {Exploring Hybrid {CTC}/Attention End-to-End Speech Recognition with Gaussian Processes},
	booktitle = {Speech and Computer}
}
```


## Contents of the Repository

* Gaussian Process (GP) optimizer as described in the paper
* ESPnet plugin for the GP optimizer
* Result parser for ESPnet results
* Routine for calculating ESPnet model sizes
* Visualization functions of results

Experimental results can be found in `results.txt` and read in using `optimizer.py` to generate the plots (see `plots/`).
In this work, ESPnet commit [716ff54](https://github.com/espnet/espnet/commit/716ff548ed013c052a1f4596e4a291449412d21b) was used.
