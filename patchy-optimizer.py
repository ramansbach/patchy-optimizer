from __future__ import absolute_import, division, print_function
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

__all__ = ["BayesOptimizer"]

class BayesOptimizer:
    """
    Class that implements a Bayesian optimizer with a Gaussian Process 
    Regressor
    
    ----------
    Attributes
    ----------
    regressor: GaussianProcessRegressor object
    data: data to regress on
    
    ---------
    Functions
    ---------
    regress: regress on the data
    addpoint: adds a data point to the data
    readData: reads a set of data in
    sampleNext: returns the optimal point to be sampled next
        this is based on exploitation/exploration tradeoff 
    acquisition: acquisition function to predict next sampling point
    """
    def __init__(kernel,nrestarts,alpha):
        
        regressor = GaussianProcessRegressor(kernel=kernel,
                                             n_restarts_optimizer=nrestarts,
                                             alpha=alpha)
        data = np.array()
    
    def readData(fname):
        """
        Read in data from the given file name
        
        ----------
        Parameters
        ----------
        fname: string
            name of file where data is stored
        
        """
    


