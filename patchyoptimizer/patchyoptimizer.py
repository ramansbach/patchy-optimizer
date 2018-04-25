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
    inputData: numpy array
        input data to regress on
    outputData: numpy array
        values (f(x)) of data
        
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
    def __init__(self,kernel,nrestarts,alpha):
        
        self.regressor = GaussianProcessRegressor(kernel=kernel,
                                             n_restarts_optimizer=nrestarts,
                                             alpha=alpha)
        self.inputData = np.array()
        self.outputData = np.array()
    
    def readData(self,fname,nparams):
        """
        Read in data from the given file name
        
        ----------
        Parameters
        ----------
        fname: string
            name of file where data is stored
        nparams: int
            number of columns in the file that are input parameters rather than
            output values
        
        """
        fid = open(fname)
        data = fid.readlines()
        fid.close()
        spline1 = data[0].split()
        ncols = len(spline1)
        self.inputData = np.zeros((len(data),nparams))
        self.outputData = np.zeros((len(data),ncols-nparams))
        
        lind = 0
        for line in data:
            spline = line.split()
            self.inputData[lind] = np.array([float(p) for p in \
                                            spline[0:nparams]])
            self.outPutData[lind] = np.array([float(op) for op in \
                                             spline[nparams:ncols]])            
            
            lind += 1

