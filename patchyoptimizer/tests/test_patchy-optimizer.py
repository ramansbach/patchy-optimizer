from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import patchyoptimizer as po

data_path = op.join(po.__path__[0], 'data')


#op.join(data_path, 'ortho.csv')

def test_readData2col():
    """
    test reading data into Bayes Optimizer with 2 parameter columns
    """
    BO = po.BayesOptimizer(None,0,0)
    BO.readData(op.join(data_path,'minitest.dat'),2)
    p1 = np.array([[1.,2.],[2.,4.],[3.,6.],[4.,8.],[5.,10.]])
    o1 = np.array([[3.,4.],[6.,8.],[9.,12.],[12.,16.],[15.,20.]])
    
    npt.assert_array_equal(p1,BO.inputData)
    npt.assert_array_equal(o1,BO.outputData)
    
