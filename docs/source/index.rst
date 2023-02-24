.. hw6-hmm documentation master file, created by
   sphinx-quickstart on Sat Feb 11 16:27:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Lab 6: Inferring CRE Selection Strategies from Chromatin Regulatory State Observations using a Hidden Markov Model and the Viterbi Algorithm
============================================================================================================================================

The aim of hw6 is to implement the Viterbi algorithm, a dynamic program that is a common decoder for Hidden Markov Models (HMMs). The lab is structured by training objective, project deliverables, and experimental deliverables:

**Training Objective**: Learn how to design reusable Python packages with automated code documentation and develop testable (user case) hypotheses using the Viterbi algorithm to decode the best path of hidden states for a sequence of observations.

**Project Deliverable**: Produce a simple report for functional characterization inferred from a binary regulatory observation state pattern across cardiac developmental timepoints.

**Experimental Deliverable**: Construct a positive control library for massively parallel reporter assays (MPRAs) and CRISPRi/a experiments in primitive and progenitor cardiomyocytes (i.e., cardiogenomics).

Key Words
==========
Chromatin; histones; nucleosomes; genomic element; accessible chromatin; chromatin states; genomic annotation; candidate cis-regulatory element (cCRE); Hidden Markov Model (HMM); ENCODE; ChromHMM; cardio-genomics; congenital heart disease(CHD); TBX5


Functional Characterization Report
===================================

Please evaluate the project deliverable and briefly answer the following speculative question, with an eye to the project's limitations as related to the theory, model design, experimental data (i.e., biology and technology). We recommend answers between 2-6 sentences. It is OK if you are not familiar already with this biological user case; you can receive full points for your best-effort answer.

1. Speculate how the progenitor cardiomyocyte Hidden Markov Model and primitive cardiomyocyte regulatory observations and inferred hidden states might change if the model design's sliding window (default set to 60 kilobases) were to increase or decrease?

If the sliding window is decreased, then the observed sequence is longer. More detailed information would be obtained. There will likely be a general trend of improved hidden state sequence prediction and observation measurements when the window decreases and window bounds become within the TAD bounds. However, too much detail could be extraneous and cause no improvement or maybe even worsening of hidden state prediction accuracy and observation measurements.

2. How would you recommend integrating additional genomics data (i.e., histone and transcription factor ChIP-seq data) to update or revise the progenitor cardiomyocyte Hidden Markov Model? In your updated/revised model, how would you define the observation and hidden states, and the prior, transition, and emission probabilities? Using the updated/revised design, what new testable hypotheses would you be able to evaluate and/or disprove?

Whether a part of the sequence is bound to histones and/or transcription factors can be added to the atac and encode-atac hidden states. Each of these could have a defined correlation with the rate of whether each window is regulatory or regulatory potential (emission). Whether one window has atac, histones, and/or a transcription factor might influence whether neighboring windows have such features (transition). I could set the starting rate of these features to be whatever is measured for the first windows of sequences (prior). I could compare this to the model that only includes the atac/atac-encode feature to determine whether the added complexity of the model improved hidden state prediction. I hypothesize that this added complexity would improve hidden state prediction if the proportions are set well. 

3. Following functional characterization (i.e., MPRA or CRISPRi/a) of progenitor and primitive cardiomyocytes, consider all possible scenarios for recommending how to update or revise our genomic annotation for *cis*-candidate regulatory elements (cCREs) and candidate regulatory elements (CREs)?

Analysis could be rerun with a variety of window sizes and overlap, and some optimization could be run on the results of this analysis to determine the best or most accurate hidden section labels for cCREs vs CREs. Instead of applying constant rates for categorical states, linear or nonlinear models to represent continuous states to enable better customization of thresholds may improve CRE vs cCRE labels. If continuous states can’t be measured biochemically, then a computer might be able to infer continuous measurements through estimations from other computational models. 

Models Package 
======================
.. toctree::
   :maxdepth: 2
   
   modules
