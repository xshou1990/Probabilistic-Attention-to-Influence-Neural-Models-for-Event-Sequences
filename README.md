# Probabilistic Attention-to-Influence Neural Models for Event Sequences

Source code for Probabilistic Attention-to-Influence Neural Models for Event Sequences ICML 2023

# Run the code for  for Probabilistic Attention-to-Influence Neural Models

### Dependencies
* Python 3.7.
* [Anaconda](https://www.anaconda.com/) contains all the required packages.
* [PyTorch](https://pytorch.org/) version 1.8.0. above

### Instructions
1. folders:
a. data. The **data** folder is already placed inside the root folder. Each dataset contains train dev test pickled files. Event registry only contains train and dev.
b.code. We used summs_utils_v3.py to generate topology based event sequences.
c.preprocess. We used to convert data into standard inpt for our model.
d.transformer. This folder contains our main models, modules that supports the training of our models.
e.prior. This folder contains prior distribution for our experiments.
f.predictive_result. This folder gives binary prediction results for our models:uniform-tau and sparse-tau.
g.transformer_baseline_prediction. This folder contains Transformer for event sequences (TES)as a baseline for our binary prediction experiments.
h. ablations_sampling. This fold contains runs and results for binary prediction using uniform-2 and sparse-2.
i. graph_recovery. This folder contains runs  and results of Probabilistic Attention-to-Influence Neural Models for identifying influencing parents for generated datasets. We also include TES results on this experiment.
j.qualitative_result.This folder contains qualitative results on stack-overflow, linkedin and event_registry with our model. The columns are threshold, event label, dev loglikehood, posterior of train, posterior of dev and posterior of test.

2. **bash run.sh** to run the code (i.e.  ./so_run.sh ). If permission denied, put chmod u+x so_run.sh on commandline. and then ./so_run.sh.  See discussion https://stackoverflow.com/questions/18960689/ubuntu-says-bash-program-permission-denied

3. All _run.sh are in batch mode which means we are running for many event of interests. We will update it with single event of interest in the future. 

4. Neural_SuMMs_appendix.pdf. This is appendix for our paper.
