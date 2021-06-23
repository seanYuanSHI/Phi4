# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:52:49 2020

@author: shi9
"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

import SE
import matplotlib.pyplot as plt
#import numpy as np
#from IPython import get_ipython

from decimal import Decimal, getcontext

# This script is use to drive member functions in SE
# to solve the ODE in various modes and plot results.
######################################
# run mode 
# mode =  0 : run all of the following
#         1 : solve initial value problem
#         2 : compare terms in the ODE
#         3 : test for convergence
#         4 : test termination reason

mode = 0

######################################
# frequently used parameters 
# dimension of the problem    
dim = 3
# parameters of ODE 
R = 17.78279410038923 # source size
A = 7005.114758138365    # source amplitude

# initial conditions for the ODE, decimal precision
# specify high precision using string   
q0D=Decimal('0.47632850233853380740427480329473631729455547518137932933532972060717048165080249596468574158844043017505670643745405184359478778510776919574709609150886536') 
p0D=Decimal(0)

# final time of simulation 
tf = 5*max(1,R)

# parameter needed by mode 4 only
if mode in [0,4]:
    # termination criteria for SolveUntil
    criteria = 'type'
    #criteria = 'cross'

######################################
# infrequently used parameteres 
# precision of Decimal
getcontext().prec = 155 # default is 28

# for SolveUntil
Ntmax = 2**24
epsilon = 1e-2

# initial time
t0 = 0
# solve forward in time 1, backward in time-1
sign = 1

# rescaling q = f + a*w, t = u*z
# defaults are f=0, a=1, and u=1
f = 0
a = 1
u = 1

# parameters for making plots
fontsize = 12 # fontsize of labels
color = 'r' # color of scatter plot   
    
##############################################################################
##############################################################################
# specify the problem
ode = SE.ODE(R, A, dim, f=f, a=a, u=u)

# time step
dt = sign*abs(epsilon*min(1,R))  # dt = abs(epsilon)*min(1,R)
# number of time steps
Nt = int(tf/dt)

# specify initial conditions 
cds = SE.Initial(Nt=Nt, dt=dt, q0D=q0D, p0D=p0D, t0=t0)  
  
# instantiate the solution
sol = SE.Solution(ode, cds, Ntmax=Ntmax, epsilon=epsilon)
sol.fontsize=fontsize
# print the problem
print(sol)   
 
###########################
if mode in [0,1]:
    # numerical solution, starting from t0
    plt.figure()
    t, yn = sol.Solve(ifplot=True)     
    
if mode in [0,2]:
    # compute and plot terms in the ODE
    dterms = sol.Terms(ifplot=True)   
    it = dterms['it']
    print(f'Energy estimate it={it}, E={dterms["E"][it]}, t={dterms["t"][it]}')
 

if mode in [0,3]:
    # test convergence, starting from t0
    norm = sol.TestCvg(ifnorm=True) 
    print(norm)
 
if mode in [0,4]:
    # test termination reason, starting from t0
    tu, yu, flag = sol.SolveUntil(criteria=criteria, save='plot')       
    # report reason
    sol.terminatMsg(flag)