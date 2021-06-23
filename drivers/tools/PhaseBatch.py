# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:44:38 2020

@author: shi9
"""

# This script run RA.Phase for a number of user-specified cases in batch
import os
os.chdir('../../')
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')


#import SE
#import QP
import RA
#import Auxiliary

#import matplotlib.pyplot as plt
import numpy as np
#from IPython import get_ipython
#from datetime import datetime

## redirect stdout
#default_stdout = sys.stdout
#sys.stdout = open('filename', 'w+')
#sys.stdout = default_stdout

#from decimal import getcontext

######################################
# parameters 
logAmin = -2
NA = 16

# for SolveUntil
Ntmax = 2**24
epsilon = 1e-3
 
# decimal precision
#getcontext().prec = 128 # default is 28
# for QP convergence during refinement
#epsilon_bisection_Q = 1e-125

# R array 
RL = 10**np.linspace(-1,2,25) # large scale
RS = 10**np.linspace(-2,-1.25,4) # small scale

# R array for 1D
R1D = np.concatenate((RS, RL))
# skip values that have already been run by setting R to np.nan
skip1D = [] # index

# R array for 2D
R2DF = 10**np.linspace(np.log10(1.5),np.log10(3.2),33) # intermediate scale
R2D = np.concatenate((RS, RL, R2DF))
R2D = np.sort(R2D)
#R2D = np.linspace(0.8,1.2,4)
# skip values that have already been run by setting R to np.nan
#skip2D = [4, 12, 53, 61] # index
skip2D = []

# R array for 3D
R3DF = 10**np.linspace(np.log10(3.2),np.log10(7),33) # intermediate scale
R3D = np.concatenate((RS, RL, R3DF))
R3D = np.sort(R3D)
# skip values that have already been run by setting R to np.nan
#skip3D = [4, 12, 53, 61] # index
skip3D = []

# small scale test
para = {'1D':{'dim':1, 'R':R1D, 'offset':0, 'fname':'../RA1D.out','skip':skip1D},
        '2D':{'dim':2, 'R':R2D, 'offset':0, 'fname':'../RA2D.out','skip':skip2D},       
        '3D':{'dim':3, 'R':R3D, 'offset':0, 'fname':'../RA3D.out','skip':skip3D},
}

# precision scheme
sdict1 = None # for 1D, sufficient to use default low precision 
sdictn = {'10':28, '100':125} # 2D and 3D need higher resolution for R>10 

########################
# run all cases
default_stdout = sys.stdout
for key in para:
    # report progress
    print(para[key])
    
    # unload parameters
    dim = para[key]['dim']
    
    # set precision
    if dim==1: sdict = sdict1
    else: sdict = sdictn
    
    R =para[key]['R']
    offset = para[key]['offset']
    fname =  para[key]['fname']
    skip = para[key]['skip']
    # default path name for raw output
    rpath = f'../data/{dim}D/batch_e{epsilon}/raw/'
    
    # skip R values
    for ind in skip:
        print(f'skip R[{ind}] = {R[ind]}')
        R[ind] = np.nan        
    
    # direct output file
    #sys.stdout = open(fname, 'w+') 
    # instantiate new phase object
    phase = RA.Phase(dim,R,NA=NA,logAmin=logAmin,Ntmax=Ntmax,epsilon=epsilon,\
                     offset=offset,rpath=rpath,sdict=sdict)    
    phase.debug = True
    # compute phase diagram and save data
    phase.Diagram()
    # restore output file
    #sys.stdout = default_stdout