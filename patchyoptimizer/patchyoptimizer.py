from __future__ import absolute_import, division, print_function
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
__all__ = ["BayesOptimizer"]

class BayesOptimizer:
    """
    Class that implements a Bayesian optimizer with a Gaussian Process 
    Regressor
    
    Currently assumes a 1D vector of target values
    ----------
    Attributes
    ----------
    regressor: GaussianProcessRegressor object
    inputData: numpy array
        input data to regress on
    outputData: numpy array
        values (f(x)) of data
    outputSigmas: numpy array
        errors in f(x) values
        
    ---------
    Functions
    ---------
    addpoint: adds a data point to the data
    readData: reads a set of data in
    renewRegressor: creates a new regressor based on the current data
    sampleNext: returns the optimal point to be sampled next
        this is based on exploitation/exploration tradeoff 
    acquisition: acquisition function to predict next sampling point
    """
    def __init__(self,kernel,nrestarts,alpha):
        
        self.regressor = GaussianProcessRegressor(kernel=kernel,
                                             n_restarts_optimizer=nrestarts,
                                             alpha=alpha)
        self.inputData = np.array([])
        self.outputData = np.array([])
        self.outputError = np.array([])
        
 
        
    def readData(self,fname,nparams,wlist):
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
        wlist: numpy array
            Weights to give the separate values when summing to 1D
            
        ------
        Raises
        ------
        ValueError: 
            if the number of parameters is <1
            if the number of parameters is > the number of columns - 1
            if the weights do not sum to 1
            if ncols - nparams is not an even number
        -----
        Notes
        -----
        Assumes that the target columns are of the structure
        t1 err(t1) t2 err(t2) ... tn err(tn)
        """
        if nparams < 1:
            raise ValueError("There must be at least one parameter.")
        if np.sum(wlist) != 1.0:
            raise ValueError("Weights must sum to one.")
        fid = open(fname)
        data = fid.readlines()
        fid.close()
        spline1 = data[0].split()
        ncols = len(spline1)
        if nparams > (ncols - 2):
            raise ValueError("There must be at least one target value.")
        if (ncols - nparams) % 2 != 0:
            raise ValueError("There must be an even number of target columns.")
        self.inputData = np.zeros((len(data),nparams))
        self.outputData = np.zeros((len(data),ncols-nparams))
        
        lind = 0
        for line in data:
            spline = line.split()
            self.inputData[lind] = np.array([float(p) for p in \
                                            spline[0:nparams]])
            
            self.outputData[lind] = np.array([float(op) for op in \
                                             spline[nparams:ncols]])            
            
            lind += 1
            
    def sampleNext(self,x,xi=0.01):
        """
        Find the optimal point to be sampled next 
        
        ----------
        Parameters
        ----------
        x: m x n numpy array
            range of values over which to look for sampling points
            m is the number of potential sampling points
            n is the number of parameters
            
        xi: float
            parameter for the acquisition function that may be played with
            default = 0.01
        -------
        Returns
        -------
        xnext: q x n numpy vector
            point or points to be sampled next
            
        -----
        Notes
        -----
            
        This function currently deals with one-dimensional output data
        """
        self.regressor.fit(self.inputData,self.outputData)
        y_pred,sigma = self.regressor.predict(x,return_std=True)
        fmax = np.max(self.outputData)
        ei = self.acquisition(y_pred,sigma,fmax,xi)
        eiinds = np.argmax(ei)
        xnext = x[eiinds,:]
        return (xnext,eiinds)   
        
    def learnLoop(self,initInput,initOutput,initError,x,yact=None,vis=True):
        """
        Loop that updates data and predictions at each step and chooses the 
        next place to evaluate the function
        
        ----------
        Parameters
        ----------
        initInput: m x n numpy array
            Initial set of data with m data points and n features
        initOutput: m x 1 numpy array
            Initial set of objective function values, currently must be 1D
        initError: m x 1 numpy array
            error in the objective function
        x: k x n numpy array
            set of points available for measurements to be taken at each step
        xplot: l x n numpy array
            set of points at which to visualize the function, does not have
            to be the same as x
        yact: the actual functional values at x, used for debugging
        vis: bool
            whether or not to show plots at each step, default is true   
            sets to false if n > 2
        
        -----
        Notes
        -----
        Can turn off the visualization at each step if so desired
        For now, define a specific type of kernel to work with
        Also the error is just a mean over all the separate errors of each
        function value--is there a better way to do this?
        """
        self.inputData = initInput
        self.outputData = initOutput
        self.outputError = initError
        alpha = 0.
        nrestarts = int(9)
        nl = np.mean(initError)
        snl = np.std(initError)
        kernel = 1.0*RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
            + WhiteKernel(noise_level=nl,
                    noise_level_bounds=(1e-10, nl+1.96*snl))
        self.regressor = GaussianProcessRegressor(kernel=kernel,
                                             n_restarts_optimizer=nrestarts,
                                             alpha=alpha)
        self.regressor.fit(self.inputData,self.outputData)
        mn = np.shape(initInput)
        n = mn[1]
        if n > 2:
            vis = False
        while True:
            
            if vis:
                
                fig = plt.figure()
                if n == 1:
                    y_pred,y_cov = self.regressor.predict(x,return_cov=True)
                    plt.plot(x,y_pred)
                    plt.fill_between(x[:,0],y_pred-np.sqrt(np.diag(y_cov)),
                                     y_pred+np.sqrt(np.diag(y_cov)),alpha=0.3)
                    plt.errorbar(self.inputData,self.outputData,
                                 self.outputError,fmt='r.')
                    if yact is not None:
                        plt.plot(x,yact,'k--')
                    plt.pause(0.05)
                else:
                    ax = fig.gca(projection='3d')
                    pgridX,pgridY = np.meshgrid(x[:,0],x[:,1])
                    pgrid = np.concatenate((pgridX.reshape(-1,1),
                                            pgridY.reshape(-1,1)),axis=1)
                    sz = np.shape(pgridX)
                    y_pred,y_cov = self.regressor.predict(pgrid,
                                                          return_cov=True)
                    
                    surf = ax.plot_surface(pgridX,pgridY,y_pred.reshape(sz),
                                           cmap=cm.coolwarm,
                                           rstride=1,cstride=1,linewidth=0)
                    ax.scatter(self.inputData,self.outputData)
                    ax.plot_surface(pgridX,pgridY,
                                (y_pred+np.sqrt(np.diag(y_cov))).reshape(sz),
                                alpha=0.2,
                                linewidth=0,cmap=cm.coolwarm,cstride=1,
                                rstride=1)
                    ax.plot_surface(pgridX,pgridY,
                                (y_pred-np.sqrt(np.diag(y_cov))).reshape(sz),
                                alpha=0.2,
                                linewidth=0,cmap=cm.coolwarm,cstride=1,
                                rstride=1)
                    fig.colorbar(surf,shrink=0.5,aspect=5)
            xnew,xnewinds = self.sampleNext(x)
            x = np.delete(x,np.argwhere(x==xnew))
            print("Perform sampling at x = {0}\n".format(xnew))
            xval = raw_input("Objective value of next point? q to quit> ")
            
            if xval == 'q':
                return
            xval = np.array([float(xval)])
            xerr = raw_input("Error in objective value?> ")
            xerr = np.array([float(xerr)])
            self.inputData = np.concatenate((self.inputData,xnew.reshape(1,n)))
            self.outputData = np.concatenate((self.outputData,xval))
            self.outputError = np.concatenate((self.outputError,xerr))
            nl = np.mean(initError)
            snl = np.std(initError)
            kernel = 1.0*RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
            + WhiteKernel(noise_level=nl,
                    noise_level_bounds=(1e-10, nl+1.96*snl))
            self.regressor = GaussianProcessRegressor(kernel=kernel,
                                             n_restarts_optimizer=nrestarts,
                                             alpha=alpha)
            self.regressor.fit(self.inputData,self.outputData)
        
        
    def acquisition(self,mu,sigma,fmax,xi):
        """
        Acquisition function, based on EI(x) function that trades off 
        exploration and exploitation
                
        ----------
        Parameters
        ----------
        mu: m x 1 numpy array
            range of predicted means (returned by regressor)
        sigma: m x 1 numpy array
            range of predicted sigmas (returned by regressor)
        fmax: float
            observed maximum value of function
        xi: float
            parameter for the acquisition function that may be played with
            
        -------
        Returns
        -------
        ei: m x 1 numpy array
            predicted values of acquisition function at x values

        -----
        Notes
        -----
        This is a one-dimensional function.
        It is implemented based on Eqn (4) in 
        [Brochu, Eric, Vlad M. Cora, and Nando De Freitas. 
        "A tutorial on Bayesian optimization of expensive cost functions, 
        with application to active user modeling and hierarchical reinforcement
        learning." arXiv preprint arXiv:1012.2599 (2010).]   
                 
        """
        Z = (mu - fmax - xi)/sigma
        ei = (mu - fmax - xi)*norm.cdf(Z) + sigma*norm.pdf(Z)
        ei[np.where(sigma==0.)] = 0.
        return ei