# BIOS 404: Time series analysis for neuroscience data using state space models

This course will present the basics of state space modeling to analyze time series
data that are frequently encountered in neuroscience problems. The course
lectures will cover basics of time series analysis, Markov chains, linear state space
models, switching state space models, and algorithms for learning and inference.
Students and instructors will work through practical data analysis exercises in
Python in weekly labs.

## Objectives
1. Students will learn the basics of frequency domain analysis methods (i.e.,
spectral estimation) and understand their limitation.
2. Students will appreciate the complexity of analyzing neural oscillations in
electrophysiological recordings.
3. Students will have a working understanding of the nuts and bolts of fitting
state space models, including Kalman filtering and smoothing, the EM
algorithm, and applications to neuroscience.
4. Students will be able to use the SOMATA python package that stream-
lines analysis of neural oscillation using the abovementioned time-domain
analysis tools.

## Environment setup instructions
You can install the requirered environment using the `env-lab.yml` file.
If you run into any error during installtions, that's due to conda's failure of dependency resolution.
In that case, we recommend to use mamba. To do so, simply download miniforge from their github page
[https://github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge), and install.
With Miniforge in your path, you can use `mamba` to create the environment from the `env-lab.yml`
file 