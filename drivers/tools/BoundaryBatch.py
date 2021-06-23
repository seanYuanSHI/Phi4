# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:44:38 2020

@author: shi9
"""

# This script is used to generate QP island boundary/separatrix points
# by mannual specifying sample for a number of selected cases in batch

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')

import SE
import QP
#import RA
#import Auxiliary

#import matplotlib.pyplot as plt
#import numpy as np
#from IPython import get_ipython
from datetime import datetime

######################################
# parameters needed by all modes 

# dimension of the problem    
dim = 1 

# for SolveUntil
Ntmax = 2**24
epsilon = 1e-3  

# file name head
fhead = f'../../../data/{dim}D/manual/e{epsilon}/'
# fname tail
ftail = f'_N{Ntmax}_e{epsilon}.txt'

########################
# high sampling number
Nx = 16
Ny = 16
# low sampling number
nx = 8
ny = 8

# list of parameters
# R, A, sample parameters
# first sample on big scale, the zoom in interesting features
para = [[0.1, 0.2, [{'Lx':3, 'Ly':2, 'q0D':0, 'p0D':0, 'theta':0, 'Nx':Nx, 'Ny':Ny},
                    {'Lx':1, 'Ly':1, 'q0D':-0.9, 'p0D':0, 'theta':0, 'Nx':nx, 'Ny':ny},
                    {'Lx':1, 'Ly':1, 'q0D':1, 'p0D':0, 'theta':0, 'Nx':nx, 'Ny':ny}]],
        [0.1, 0.6, [{'Lx':3, 'Ly':2, 'q0D':0, 'p0D':-0.3, 'theta':0, 'Nx':Nx, 'Ny':Ny},
                    {'Lx':1, 'Ly':1, 'q0D':-0.9, 'p0D':-0.3, 'theta':0, 'Nx':nx, 'Ny':ny},
                    {'Lx':1, 'Ly':1, 'q0D':1, 'p0D':-0.3, 'theta':0, 'Nx':nx, 'Ny':ny}]],
        [0.1, 1.2, [{'Lx':3, 'Ly':2, 'q0D':0, 'p0D':-0.6, 'theta':0, 'Nx':Nx, 'Ny':Ny},
                    {'Lx':1, 'Ly':1, 'q0D':-0.9, 'p0D':-0.6, 'theta':0, 'Nx':nx, 'Ny':ny},
                    {'Lx':1, 'Ly':1, 'q0D':1, 'p0D':-0.6, 'theta':0, 'Nx':nx, 'Ny':ny}]],
        [2.371373705661655, 0.6, [{'Lx':4, 'Ly':2, 'q0D':0.5, 'p0D':-0.5, 'theta':0, 'Nx':Nx, 'Ny':Ny},
                  {'Lx':2, 'Ly':1, 'q0D':2.3, 'p0D':-2.2, 'theta':-60, 'Nx':nx, 'Ny':ny},
                  {'Lx':1, 'Ly':1, 'q0D':-0.6, 'p0D':-0.4, 'theta':0, 'Nx':nx, 'Ny':ny}]],
        [2.371373705661655, 1.35, [{'Lx':4, 'Ly':3, 'q0D':1, 'p0D':-1, 'theta':-50, 'Nx':Nx, 'Ny':Ny},
                  {'Lx':3, 'Ly':1, 'q0D':2.6, 'p0D':-3, 'theta':-70, 'Nx':nx, 'Ny':ny},
                  {'Lx':1, 'Ly':1, 'q0D':-0.4, 'p0D':-0.6, 'theta':0, 'Nx':nx, 'Ny':ny}]],
        [2.371373705661655, 2.0, [{'Lx':4, 'Ly':3, 'q0D':0.8, 'p0D':-0.8, 'theta':-50, 'Nx':Nx, 'Ny':Ny},
                  {'Lx':3, 'Ly':1, 'q0D':2.6, 'p0D':-3.4, 'theta':-70, 'Nx':nx, 'Ny':ny},
                  {'Lx':1, 'Ly':1, 'q0D':0.2, 'p0D':-1.1, 'theta':0, 'Nx':nx, 'Ny':ny}]],
        [10, 1.0, [{'Lx':4, 'Ly':3, 'q0D':0.8, 'p0D':-0.3, 'theta':-50, 'Nx':Nx, 'Ny':Ny},
                  {'Lx':3, 'Ly':1, 'q0D':2.5, 'p0D':-2.7, 'theta':-65, 'Nx':nx, 'Ny':ny},
                  {'Lx':1, 'Ly':1, 'q0D':-0.6, 'p0D':0, 'theta':0, 'Nx':nx, 'Ny':ny}]],
        [10, 3.2, [{'Lx':4, 'Ly':3, 'q0D':0.8, 'p0D':-0.3, 'theta':-50, 'Nx':Nx, 'Ny':Ny},
                  {'Lx':3, 'Ly':1, 'q0D':2.5, 'p0D':-3, 'theta':-65, 'Nx':nx, 'Ny':ny},
                  {'Lx':3, 'Ly':1, 'q0D':1.1, 'p0D':-0.45, 'theta':-40, 'Nx':nx, 'Ny':ny}]],
        [10, 4.6, [{'Lx':4, 'Ly':3, 'q0D':0.9, 'p0D':-0.4, 'theta':-50, 'Nx':Nx, 'Ny':Ny},
                  {'Lx':3, 'Ly':0.5, 'q0D':1.53, 'p0D':-1.04, 'theta':-50, 'Nx':nx, 'Ny':ny},
                  {'Lx':3, 'Ly':0.5, 'q0D':2.5, 'p0D':-2.8, 'theta':-65, 'Nx':nx, 'Ny':ny}]]
      ]


# print start time 
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print('Compute 1D phase space started on ', dt_string)
print('Data will be saved in', fhead)

# compute all cases
N = len(para)
for i in range(N):
    # current list
    lp = para[i]
    
    # R value
    R = lp[0]
    # A value
    A = lp[1]
    # specify the problem
    ode = SE.ODE(R, A, dim)
    # instantiate island
    island = QP.Island(ode, Ntmax=Ntmax, epsilon=epsilon)
    print(f'i/N = {i}/{N}')
    print(ode)
    
    # filename body
    fbody = f'R{R}_A{A}'
    # filename for island boundary
    fnameQPb = fhead + 'Boundary1D_' + fbody + ftail
    # filename for separatrix
    fnameQPs = fhead + 'Separatrix1D_' +fbody + ftail      
    
    # sample parameters
    ps = lp[2]    
    # seach using all sample boxes
    ns = len(ps)
    for j in range(ns):
        # current sample dictionary
        ds = ps[j]
        # sample box
        sample = QP.Sample(**ds) #kwargs
        # island.Boundary
        island.Boundary(sample)         

        # save data
        island.save(fnameQPb, sample,'boundary')
        island.save(fnameQPs, sample,'separatrix')
        
# print finish time 
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print('Compute 1D phase space finished on ', dt_string)

