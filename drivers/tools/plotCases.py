# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:33:18 2020

@author: water
"""

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

import SE
import matplotlib.pyplot as plt
#import numpy as np
#from IPython import get_ipython

from decimal import Decimal, getcontext


# This script plot solutions of the ODE for a list of initial conditions
######################################
# dimension of the problem    
dim = 2
# parameters of ODE 
R = 100    # source size
A = 6000    # source amplitude

# final time of simulation 
tf = 5*max(1,R)

# initial conditions for the ODE, decimal precision
# specify high precision using string   
q0Dlist=[Decimal('-1.1536892286039285550043709709936696647629962931209679188087863050519063190401632318853475139403274386348472880316487110496486'),
         Decimal('-1.1536892286039285550043709709936696647629962931209679188087863050519063190401632318853475139403274386346982892444375644864414'),
         Decimal('0.35324420628252212748962800224583237769399347209928689848696115917099502055150898951376368454716267129180001496609287343045522'),
         Decimal('0.6267393730286948431295429250963999380684762293925522540009039604144144119484785854338363422345399030676560547874157080684521')]

# precision of Decimal
getcontext().prec = 125 # default is 28

# for SolveUntil
Ntmax = 2**24
epsilon = 1e-2

##############################################################################
# specify the problem
ode = SE.ODE(R, A, dim)
# time step
dt = abs(epsilon*min(1,R))  # dt = abs(epsilon)*min(1,R)
# number of time steps
Nt = int(tf/dt)

plt.figure()
for q0D in q0Dlist:
    # specify initial conditions 
    cds = SE.Initial(Nt=Nt, dt=dt, q0D=q0D)  
      
    # instantiate the solution
    sol = SE.Solution(ode, cds, Ntmax=Ntmax, epsilon=epsilon)
    # print the problem
    print(sol)   
     
    # numerical solution        
    t, yn = sol.Solve()   
    plt.plot(t, yn)
    
#plt.legend(loc='best', fontsize=fs)
plt.xlabel('t=m*x')
plt.ylabel('q=phi/v')
plt.axhline(y=0, color='grey')
plt.axhline(y=1, color='grey') 
plt.axhline(y=-1, color='grey') 
plt.axvline(x=0, color='grey')
plt.title(f'dim={ode.dim}, R={ode.R}, A={ode.A}')
#plt.xlim((0,tf))
plt.ylim((-2, 2))    
#plt.show()            