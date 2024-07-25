# Environment setup instructions
You can install the requirered environment using the `env-lab.yml` file.
If you run into any error during installtions, that's due to conda's failure of dependency resolution.
In that case, we recommend to use mamba. To do so, simply download miniforge from their github page
[https://github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge), and install.
With Miniforge in your path, you can use `mamba` to create the environment from the `env-lab.yml`
file 


# Objectives
By the end of the lab, you should have:
- implemented a periodogram, tapered periodogram and multitaper spectrum computation, built from basic scipy functions
- applied them in three different examples
  - white noise
  - AR(2) process
  - AR(4) process
- realized how to control Bias and Variance in spectrum estimates
- computed multitaper spectrum from segments of a real EEG data under general anesthesia
- created a spectrogram representation of a real EEG data under general anesthesia.

