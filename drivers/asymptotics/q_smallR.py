# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 23:00:48 2020

@author: water
"""

# This script investigate asymptotic solution when the 
# source size is small and strength is large 

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')

import SE
import matplotlib.pyplot as plt
#import numpy as np

# asymptotic functions
import Asympt_1D
import Asympt_nD

from decimal import Decimal, getcontext

###############
# parameters users are expected to change
# dimension of the problem    
dim = 3 # dim = 1, 2, 3

# final time in unit of source size
tf = 5

# for plotting asymptotic solution
na = 1000 # time grid points = na
# y axis limit
ymax = 2
# linewidth
lw = 3


###############
# parameters users are not expected to change
assert(dim in [1,2,3])
    
# for SolveUntil
Ntmax = 2**24
#epsilon = 1e-2
epsilon = 1e-3

# precision of Decimal
getcontext().prec = 28 # default is 28
    
R = 0.01 # source size
# Data table, cases[dim-1] example parameters for that dimension
# q0p: initial condition for q->1 perturbative solution
# q0l: initial condition for q->1 nonperturbative solution
# q0n: initial condition for q->-1 adiabatic solution
cases = [{'dim': 1, 'A': 0.8, 
          'q0n': Decimal('-1.3392340388149027'),   
          'q0l': Decimal('-0.4454676974564791'),     
          'q0p': Decimal('0.4499770700931549')},
         #########################
           {'dim': 2, 'A': 12000, # far above 0010 phase boundary
            'q0n': Decimal('-402.6097824949496'),    
            'q0l': None,
            'q0p': Decimal('-402.6097823156759')},
          # {'dim': 2, 'A': 1000, # far above 0010 phase boundary
          #  'q0n': Decimal('-142.73270229182845'),    
          #  'q0l': None,
          #  'q0p': Decimal('-142.73262042068558')},
         #########################
         {'dim': 3, 'A': 360,
          'q0n': Decimal('-477.3260537495611'),      
          'q0l': None,  
          'q0p': Decimal('-477.3259850935679')},]

 
# auxilliary tables
initials = ['q0n', 'q0l', 'q0p']   
flist = [-1, 1, 1]
messages = ['Calculating q -> -1 separatrix solution...',
            'Calculating q -> 1 nonperturbative solution...',
            'Calculating q -> 1 perturbative solution...'] 
ncolors = ['b', 'g', 'r']
acolors = ['cyan', 'y', 'm']
    

######################################################
######################################################
# unpack parameters
cd = cases[dim-1]
A = cd['A']
# specify the problem
ode = SE.ODE(R, A, dim)

# time step
dt = abs(epsilon*min(1,R))  # dt = abs(epsilon)*min(1,R)
# number of time steps
Nt = int(tf/dt)
    
# initialize new figure
plt.figure()
for ic in range(3):
    # initial condition
    q0D = cd[initials[ic]]
    if q0D!=None:
        print(messages[ic])
        
        # specify initial conditions 
        cds = SE.Initial(Nt=Nt, dt=dt, q0D=q0D) 
        # instantiate the solution
        sol = SE.Solution(ode, cds, Ntmax=Ntmax, epsilon=epsilon)
        # print the problem
        print(sol)            
        # numerical solution
        t, yn, f = sol.SolveUntil(save='trace', tfmax=tf) 
        # numerical solution
        plt.plot(t, yn,color=ncolors[ic],linestyle=':',linewidth=lw+ic/5)
        
        if dim==1:
            # compute asymptotic solution
            ya = Asympt_1D.qDelta(t, float(q0D), f=flist[ic])            
            # plot asymptotic solution
            plt.plot(t, ya, color=acolors[ic])    


if dim!=1: # plot inner asymptotic solution
    # asymptotic inner solution
    qin = Asympt_nD.qin_RsAl(t, R, A, dim=dim, ifplot=True)
    # asymptotic middle solution
    qmid = Asympt_nD.qmid_RsAl(t, R, A, dim=dim, ifplot=True)

 

# label the figure
plt.xlim((0,tf))
plt.ylim((1.1*float(cd['q0n']),ymax))
plt.title(f'{dim}D, R={R}, A={A}')
plt.xlabel('t')
plt.ylabel('q')

# mark special lines
plt.axvline(x=0, color='grey')
plt.axhline(y=1, color='grey') 
plt.axhline(y=0, color='grey',linestyle='--') 
plt.axhline(y=-1, color='grey') 
plt.show()
