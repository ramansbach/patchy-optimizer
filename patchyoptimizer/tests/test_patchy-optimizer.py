from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import patchyoptimizer as po
import pytest

data_path = op.join(po.__path__[0], 'data')


#op.join(data_path, 'ortho.csv')

def test_readData2col():
    """
    test reading data into Bayes Optimizer with 2 parameter columns
    """
    wlist = np.array([1.])

    BO = po.BayesOptimizer(None,0,0)
    BO.readData(op.join(data_path,'minitest.dat'),2,wlist)
    p1 = np.array([[1.,2.],[2.,4.],[3.,6.],[4.,8.],[5.,10.]])
    o1 = np.array([[3.,4.],[6.,8.],[9.,12.],[12.,16.],[15.,20.]])
    npt.assert_array_equal(p1,BO.inputData)
    npt.assert_array_equal(o1,BO.outputData)
    
def test_readData3col():
    """
    test reading data into Bayes Optimizer with 3 parameter columns
    """
    wlist = np.array([1.])
    BO = po.BayesOptimizer(None,0,0)
    BO.readData(op.join(data_path,'minitest.dat'),3,wlist)
    p1 = np.array([[1.,2.,3.],[2.,4.,6.],[3.,6.,9.],[4.,8.,12.],[5.,10.,15.]])
    o1 = np.array([[4.],[8.],[12.],[16.],[20.]])
    
    npt.assert_array_equal(p1,BO.inputData)
    npt.assert_array_equal(o1,BO.outputData)
    
def test_readData0colfail():
    """
    test failure case for reading data with 0 parameter columns
    """
    BO = po.BayesOptimizer(None,0,0)
    wlist = np.array([1.])
    with pytest.raises(ValueError):
        BO.readData(op.join(data_path,'minitest.dat'),0,wlist)
    
def test_readDataNegcolfail():
    """
    test failure case for reading data with negative parameter columns
    """
    wlist = np.array([1.])
    BO = po.BayesOptimizer(None,0,0)
    with pytest.raises(ValueError):
        BO.readData(op.join(data_path,'minitest.dat'),-1,wlist)
        
def test_readDataTooManyParamsfail():
    BO = po.BayesOptimizer(None,0,0)
    wlist = np.array([1.])
    with pytest.raises(ValueError):
        BO.readData(op.join(data_path,'minitest.dat'),4,wlist)
        
def test_readDataTooManyParamsfail2():
    BO = po.BayesOptimizer(None,0,0)
    wlist = np.array([1.])
    with pytest.raises(ValueError):
        BO.readData(op.join(data_path,'minitest.dat'),7,wlist)