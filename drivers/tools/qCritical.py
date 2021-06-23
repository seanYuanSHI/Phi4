# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:23:26 2020

@author: Yuan Shi
"""
# This script is used to find the qc values use bisection
# for given R and A. This is a driver function for QP.Island.xBisect

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')

import SE
import QP
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext

########################################
# problem setup
# dimension of the problem    
dim = 1
# parameters of ODE 
R = 10 # source size
A = 3.2 # source amplitude

# for SolveUntil
epsilon = 1e-2
Ntmax = 2**24
# precision of Decimal
getcontext().prec = 28 # default is 28
# max number of iterations determined automatically according to prec


########################################
# specify high precision using string   
# initial guess left bound
qlD = Decimal('-1.15')
# initial guess right bound
qrD = Decimal('-1.13')

# final target time
#tf = 5*max(R,1)
tf = 150

########################################
# specify the problem
ode = SE.ODE(R, A, dim)
# instantiate island, by default dt = epsilon*min(1,R) 
island = QP.Island(ode, Ntmax=Ntmax, epsilon=epsilon)
# find q0 using bisection, return none if error
q0D, t, ql, qr = island.xBisect(qlD, qrD, tf, debug=True)
# print final q0
print('Refined q0 = ',q0D)
print(f'Last search interval ql={ql}, qr={qr}')

# plot final solution
if q0D != None: # if no error
    #plt.figure()
    # specify initial conditions 
    cds = SE.Initial(dt=epsilon*min(1,R), q0D=q0D)    
    # instantiate the solution
    sol = SE.Solution(ode, cds, epsilon=epsilon, Ntmax=Ntmax)
    sol.SolveUntil(save='plot')
    plt.xlim((0,tf))
    plt.ylim((-2,2))
    plt.axhline(y=0, color='grey') 
    plt.axhline(y=1, color='grey') 
    plt.axhline(y=-1, color='grey') 
