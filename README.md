## patchyoptimizer
[![Build Status](https://travis-ci.org/uwescience/patchyoptimizer.svg?branch=master)](https://travis-ci.org/uwescience/patchyoptimizer)
### Project Overview

This is a small project intended to wrap around Bayesian optimization, based on the Gaussian Process Regressor in scikit-learn.

The basic functionality is that it reads in specified data (or allows user input), uses a GPR to predict the means and stddev of the data, and then produces an estimate of the best next point to sample based on the Expected Improvement Acquisition function described in [1]

### Organization of the  project

The project has the following structure:

    patchyoptimizer/
      |- README.md
      |- patchyoptimizer/
         |- __init__.py
         |- patchyoptimizer.py
         |- due.py
         |- data/
            |- ...
         |- tests/
            |- ...
      |- doc/
         |- Makefile
         |- conf.py
         |- sphinxext/
            |- ...
         |- _static/
            |- ...
      |- setup.py
      |- .travis.yml
      |- .mailmap
      |- appveyor.yml
      |- LICENSE
      |- Makefile
      |- ipynb/
         |- ...

### Citations

[1] Brochu, Eric, Vlad M. Cora, and Nando De Freitas. "A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning." arXiv preprint arXiv:1012.2599 (2010).
