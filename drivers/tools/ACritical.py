# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:23:26 2020

@author: Yuan Shi
"""
# This script is used to find the Ac values for given q0 use bisection
# Should test the boundary cases with driver_SE.py first

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')

#import numpy as np
import SE
import matplotlib.pyplot as plt

from decimal import Decimal, getcontext

########################################
# problem setup

# dimension of the problem    
dim = 1
# parameters of ODE 
R = 1 # source size
# initial value, specify high precision using string   
q0D = Decimal('0')

# final time in unit of source size
tfR = 1e3
# resolution for ODE
epsilon = 1e-2
# decimal precision
getcontext().prec = 28 # default is 28


########################################
# initial guess left bound
Al=1
# initial guess right bound
Ar=10

# method for determine boundary conditions
#ifflag = True
ifflag = False

if ifflag: # determine boundary by asymptotic flag
    # list of asymptotics on the left
    fl = [2,-2]
    # list of asymptotics on the right
    fr = [1]
else: # determine boundary of final q value, float
    # lower bound of q
    ql = 0.9999988
    # upper bound of q
    qr = 1.0


# max number of iterations
Niter = 64

########################################
# time step size
dt = epsilon*min(1,R) 
# specify initial conditions 
cds = SE.Initial(dt=dt, q0D=q0D)   

# final target time
tf = tfR*R
# initial time
t = 0
# initial flag
flag = 0
# iteration counter
ii = 0
# search for qc
while (t < tf or flag==999) and ii < Niter:
    # count iterations
    ii += 1
    
    # mid point
    A = (Al + Ar)/2

    # specify the problem
    ode = SE.ODE(R, A, dim)
    # instantiate the solution
    sol = SE.Solution(ode, cds, epsilon=epsilon, Ntmax=2**24)

    # solve until end point
    t, q, flag = sol.SolveUntil(tfmax=tf)   
    print(f'{ii}/{Niter}: t/tf={t}/{tf}, Al={Al}, Ar={Ar}, flag={flag}, q={q}')

    # # update bisection points
    if ifflag: # determine by asymptotic flag
        if flag in fr: # move right boundary left
            Ar = A
        elif flag in fl: # move left boundary right
            Al = A
        else:
            print('Unexpected situation!')  
            break
    else: # determine by boundary q values        
        if q>qr: # move right boundary left
            Ar = A
        elif q<=ql: # move left boundary right
            Al = A
        else:
            print('q within targeted range!')   
            break


# print final q0
print('Refined Ac = ',A)

# plot final solution
#plt.figure()
sol.SolveUntil(save='plot')
plt.xlim((0,tf))
plt.ylim((-2,2))
plt.axhline(y=0, color='grey') 
plt.axhline(y=1, color='grey') 
plt.axhline(y=-1, color='grey') 
