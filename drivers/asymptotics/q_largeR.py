# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 23:00:48 2020

@author: water
"""

# This script investigate asymptotic solution when the source size is large
# for a few example cases

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')

import SE
import matplotlib.pyplot as plt
import numpy as np

# asymptotic functions
import Asympt_1D
import Asympt_nD

from decimal import Decimal, getcontext

###############
# parameters users are expected to change
# dimension of the problem    
dim = 3 # dim = 1, 2, 3

# weighting factor source term estimate
if dim==2:
    #eta = 0
    eta=0.8 # 0<eta<=1
elif dim==3:
    #eta = 0
    eta=0.95 # 0<eta<=1
    
# final time in unit of source size
tfR = 2.5
    
# for plotting asymptotic solution
na = 1000 # time grid points = na
# y axis limit
ymax = 1.5
# linewidth
lw = 3
# colors for plottig numerical solutions of qs, qn, ql
#colors = ['b', 'k', 'r']
colors = ['lawngreen', 'lawngreen', 'lawngreen']
#colors = ['olive','olive','olive']
#colors = ['g','g','g']

    
###############
# parameters users are not expected to change
# for SolveUntil
Ntmax = 2**24
# precision of Decimal
getcontext().prec = 125 # default is 28


# Data table, cases[dim-1] example parameters for that dimension
# q0n is the initial condition for q->-1 adiabatic solution
# q0l is the initial condition for the nonlinear solution
# q0p is the initial condition for q->1 adiabatic solution
cases = [#{'dim': 1, 'A': 1.0, 'R':10, 'epsilon':1e-3,
#           'q0n': Decimal('-1.0514475018703021148'),   
#           'q0l': Decimal('-0.4795490837798192'),     
#           'q0p': Decimal('.9394609861053205')},
           # {'dim': 1, 'A': 10, 'R':100, 'epsilon':1e-2, # prec=125, 1110 phase
           # 'q0n': Decimal('-1.0522437742821048931304905327153921703124138604488721934041567503637280298241855934478024856735729654186183177546894841788348'),   
           # 'q0l': Decimal('-0.44775725264550740916013298057818732625146333820382203489501340743750672169288247058884272997808279510671746114512161726424912'),     
           # 'q0p': Decimal('0.9379365889161246251280390778963852055450730057940268749239996267391807974093233483171587645300719403076558425659093596802619')},
            {'dim': 1, 'A': 30, 'R':100, 'epsilon':1e-2, # prec=125, 2010 phase
            'q0n': Decimal('-1.1389370739683681307935358390612402156478210245432948627630129880431232804678169459502905084233942661770417600342826668048996'),   
            'q0l': Decimal('0.22351048958877660413776883776372008896149977976752474872686736056824660654989034608390850258174012515988962454335271787976769'),     
            'q0p': Decimal('0.7344164549800330878263162996556651794478507749207947325522132773493591770889538538015981254934796915084580070654403847339073')},
         ##########
          # {'dim': 2, 'A': 78, 'R':10, 'nt':1, 'epsilon':1e-3,
          #  'q0n': None,    
          #  'q0l': Decimal('-1.1867525991907684'),
          #  'q0p': None},
         ###
           # {'dim': 2, 'A': 798.0115618182466, 'R':100, 'nt':1, 'epsilon':1e-2,
           #  'q0n': None,    
           #  'q0l': Decimal('-1.0244854029074789955638436788311736903341876002239685278650242358139154822390830433076773345267490042689766071969021646504190'),
           #  'q0p': None},
         ###
          {'dim': 2, 'A': 798.0115618182466, 'R':100, 'nt':0, 'epsilon':1e-2,
           'q0n': None,    
           'q0l': Decimal('-1.0244669216240160843659507104122193850986009042007807436747126401892856126160112887457646767287848228852358102453533382147240'),
           'q0p': None},
         ###########
         # {'dim': 3, 'A': 3000, 'R':10, 'nt':1, 'epsilon':1e-3,
         #  'q0n': None,      
         #  'q0l': Decimal('-1.3359921711357467'),  
         #  'q0p': None},
         {'dim': 3, 'A': 872152.3564648489, 'R':100.0, 'nt':1, 'epsilon':1e-2,
          'q0n': None,      
          'q0l': Decimal('-1.13007946344758606815186569534559565778834796877065539078482915143599826347857293815698058168961506066667521082209587371055479840343903350288974937232'),  
          'q0p': None},]



# auxilliary tables
initials = ['q0n', 'q0l', 'q0p']
messages = ['Calculating q -> -1 adiabatic solution...',
            'Calculating q -> 1 nonperturbative solution...',
            'Calculating q -> 1 adiabatic solution...']

######################################################
######################################################
# unpack parameters
cd = cases[dim-1]
A = cd['A']
R = cd['R']
# specify the problem
ode = SE.ODE(R, A, dim)
# title of the plot
title = f'{dim}D, R={R}, A={A}'

# time step
epsilon = cd['epsilon']
dt = abs(epsilon*min(1,R))  # dt = abs(epsilon)*min(1,R)
# final time
tf = tfR*R
# number of time steps
Nt = int(tf/dt)

# initialize new figure
#plt.figure(num=None, figsize=(6, 3))

###########################
# compute and plot the numerical solutions for the three cases
for ic in range(3):    
    # initial condition
    q0D = cd[initials[ic]]
    if q0D!=None:    
        # report progress
        print(messages[ic])
    
        # specify initial conditions 
        cds = SE.Initial(Nt=Nt, dt=dt, q0D=q0D) # for critical solution
        # instantiate the solution
        sol = SE.Solution(ode, cds, Ntmax=Ntmax, epsilon=epsilon)
        # print the problem
        print(sol)   
     
        # numerical solution
        #t, yn = sol.Solve(ifplot=False) 
        t, yn, _ = sol.SolveUntil(save='trace') 
        # plot numerical solution 
        plt.plot(t, yn, colors[ic], linestyle='-', linewidth=lw)
    
    
# compute and plot asymptotic solutions   
t = np.linspace(0,tf,na) 
# source term
S = ode.source(t)

# adiabatic solution q->-1
qs = SE.Fixed(S, n=0) # smallest root
plt.plot(t, qs, color='k', linestyle='--')

# adiabatic solution q->1
if ode.S0<1/3/np.sqrt(3):
    ql = SE.Fixed(S, n=2) # largest root
    plt.plot(t, ql, color='k', linestyle='--')

# nonlinear solutions
if dim==1:
    S0 = ode.S0
    qn = Asympt_1D.qConst(t, S0, ifplot=True)
else:
    # select which nonlinear solution branch
    nt = cd['nt']
    # asymptotic solution
    qn = Asympt_nD.q_RlAl(t, R, A, eta=eta, dim=dim, ifplot=True, nt=nt)    
    title = title + f', eta={eta}'
    

# label the figure
plt.xlim((0,tf))
plt.ylim((-ymax,ymax))
plt.title(title)
plt.xlabel('t')
plt.ylabel('q')

# mark special lines
plt.axvline(x=0, color='grey')
plt.axhline(y=1, color='grey') 
plt.axhline(y=0, color='grey',linestyle='--') 
plt.axhline(y=-1, color='grey') 
plt.show()
