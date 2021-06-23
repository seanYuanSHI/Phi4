# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 23:00:48 2020

@author: water
"""

# This script investigate approximations to the critical solution
# The critical solution satisfies q(t=0)=q0 and q(t=infty)=1

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')

import matplotlib.pyplot as plt
import numpy as np
#import scipy.special as ss

import SE
import Asympt_1D
import Asympt_nD

from decimal import Decimal, getcontext

###############
# parameters users are expected to change
# source size choose only from 0.001, 0.01, 100, and 1000
R=0.01
# dimension of the problem    
dim = 3 # dim = 1, 2, 3

# final time in unit of source size
if R<1: # small source
    if dim==1: tf = 5
    else: tf = 1
else: # R>1, large source
    tf = 0.5*R
    
# for plotting asymptotic solution
na = 1000 # time grid points = na
# y axis limit
ymax = 1.1

    
###############
# parameters users are not expected to change
# initial value
if R<1: # small source
    q0 = 0
    #q0 = -1
else: # large source
    q0 = 1/np.sqrt(3) 


# table of known critical A values at R values
# obtained from existing RA runs at q0=p0=0
Ac= {# asymptotics at small R
     '0.001':{'1D': 1-q0**2, 
              '2D': 0.84496581278766,
              '3D': 0.01114638494873047,      
             },
     '0.01':{# 1D delta function source Ac=1-q0^2
             '1D': 1-q0**2, 
             # 2D when R->0, Ac~2*pi*(1-q0)/(log(2/R)-gamma/2)
             # When R=0.01, q0=0, Ac~1.2542
             '2D':1.2105536062960118,
             #'2D':2.336996358925145, #q0=-1
             # 3D when R->0, Ac~2*R*pi^(3/2)*(1-q0)/(1-sqrt(pi)R/2). 
             # When R=0.01, q0=0, Ac~0.11236
             '3D': 0.11231639461666715
             #'3D':0.22457129328804298 #q0=-1
             },
     # asymptotics at large R
     # A/(R^D*pi^(D/2))~1/3/sqrt(3)
     '100':{'1D':34.27451910093105,
            '2D':6097.529004647392,
            '3D':1084168.9428428863},
     '1000':{'1D':341.18541159127165,
             '2D':604840.5432346498,
             '3D':1072210893.2310699},
     }

# for SolveUntil
Ntmax = 2**24
epsilon = 1e-3
# precision of Decimal
getcontext().prec = 28 # default is 28

######################################################
######################################################
# source amplitude
A = Ac[str(R)][str(dim)+'D']
# specify the problem
ode = SE.ODE(R, A, dim)

# time step
dt = abs(epsilon*min(1,R))  # dt = abs(epsilon)*min(1,R)
# number of time steps
Nt = int(tf/dt)

# specify initial conditions 
cds = SE.Initial(Nt=Nt, dt=dt, q0D=Decimal(q0)) # for critical solution
# instantiate the solution
sol = SE.Solution(ode, cds, Ntmax=Ntmax, epsilon=epsilon)
# print the problem
print(sol)   
 
# numerical solution
t, yn = sol.Solve(ifplot=False) 
# initialize new figure to plot solution
plt.figure()
# plot numerical solution
plt.plot(t, yn,'k:', linewidth=4)    

###########################
# asymptotic solutions
# time step size
dta = tf/na
# all time, removing zero time to aviod possible singularity
ta = np.linspace(dta, tf, int(na))

        
if R<1: # small R
    # initialize ode already assert that dim=1,2,3
    if dim==1:         
        # exact solution for delta function source
        ya = Asympt_1D.qDelta(ta, q0, ifplot=True)
        
    else:       
        # asymptotic solution in inner and outer regions
        qin, qout = Asympt_nD.q_RsAc(ta, R, A, dim=dim, ifplot=True)
        
        # asymptotc Ac for given q0
        Ac = Asympt_nD.Ac_Rs(R, q0=q0, dim=dim)
        # asymptotic qc for given A
        qc = Asympt_nD.qc_RsAc(R, A, dim=dim)        
    
        # report asymptotic matching
        print(10*'#'+f'\nFor q0={q0}\nasymptotic Ac={Ac} \nactual A={A}\n')
        print(10*'#'+f'\nFor A={A}\nasymptotic qc={qc} \nactual q0={q0}\n')

else: # large R
    # source term
    #S = ode.source(ta)
    S = np.exp(-(ta/R)**2)/3/np.sqrt(3)
    # larget one of the cubic root
    ya = SE.Fixed(S, n=2)
    # plot solution
    plt.plot(ta,ya,'r',linestyle='-')


###########################
# mark figure
plt.xlim((0,tf))
plt.ylim((0,ymax))

plt.title(f'{dim}D, R={R}')
plt.xlabel('t')
plt.ylabel('q')

# mark special lines
plt.axvline(x=0, color='grey')
#if tf>Rm: plt.axvline(x=Rm, color='red',linestyle=':')
#plt.annotate(r'$R/\sqrt{2}$', xy =(Rm, 0), xytext =(Rm, -ymax/6)) 

plt.axhline(y=1, color='grey') 
#plt.axhline(y=0, color='grey',linestyle='--') 
#plt.axhline(y=-1, color='grey') 
plt.show()
