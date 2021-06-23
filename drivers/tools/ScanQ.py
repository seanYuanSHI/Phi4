# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:44:38 2020

@author: shi9
"""

# This script is used to zoom in critical points by manually running q scan
# in order to determine whether the jump is a separatrix or a narrow island

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')


import SE
import QP
import Auxiliary

import matplotlib.pyplot as plt
#import sys
#import numpy as np
#from IPython import get_ipython
#from datetime import datetime

from decimal import Decimal, getcontext

######################################
# parameters needed by all modes 

# dimension of the problem    
dim = 3 

# parameters of ODE 
A = 10 # source amplitude
R = 31.6 # source size

# for SolveUntil
Ntmax = 2**24
epsilon = 1e-2  
# decimal precision
getcontext().prec = 28 # default is 28

# initial conditions & center of search box
# specify high precision using string   
q0D=Decimal('-1.0486423863499996')
# box size
Lx = 1.2
# number of grid points
Nx = 101 

fontsize=8      

# specify the problem
ode = SE.ODE(R, A, dim)
# instantiate island
island = QP.Island(ode, Ntmax=Ntmax, epsilon=epsilon)
island.fontsize=fontsize
xsample = QP.Sample(Lx=Lx, q0D=q0D, Nx=Nx, Ny=1)
print(f'R={R}, A={A}')
print(f'q0={q0D}, Lx={Lx}, epsilon={epsilon}')
print(f'xstep = {xsample.xstep()}')

    
plt.figure()    
xqD, _, xflags = island.Scan(xsample, ifplot=True)

# locate jump after which jumps occur
index, _ = Auxiliary.Jump(xflags, 0.5)
for ind in index:
    # mid point
    mpD = (xqD[ind] + xqD[ind+1])/2
    print(f'index = {ind}, midpoint q ={mpD}')       
