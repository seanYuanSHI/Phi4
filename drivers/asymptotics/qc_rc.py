# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 23:27:23 2020

@author: water
"""

# This script compares numerically obtained initial qc = q(t=0) 
# values and light horizon radius rc with asymptotic expressions

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')

import SE
import RA
import matplotlib.pyplot as plt
import numpy as np

# asymptotic functions
import Asympt_1D
import Asympt_nD

############################
# dimension of the problem    
dim = 3 # dim = 1, 2, 3

# plot small R or large R cases
flag = 'small'
#flag = 'large'

# Data table, cases[dim-1] example parameters for that dimension
# iRsmall: representative iR for small R
# iRlarge: representative iR for large R
cases = [{'dim': 1, 'iRsmall': 25, 'iRlarge': 24},
         {'dim': 2, 'iRsmall': 0, 'iRlarge': 53, 'eta': 0.8},
         {'dim': 3, 'iRsmall': 0, 'iRlarge': 61, 'eta': 0.95},]

# for SolveUntil
Ntmax = 2**24
epsilon = 1e-2
# number of points to plot asymptotics
Na = 1000

# file path for batch files
bpath = f'../../../data/{dim}D/batch_e{epsilon}/'

###################################################
# check dim
assert(dim in [1,2,3])
# Initialize new figure
plt.figure(num=None, figsize=(8, 3)) 

# Extract file name
iR = cases[dim-1]['iR'+flag]
# data file name dictionary
fdict = {'dim':dim, 'Ntmax':Ntmax, 'epsilon':epsilon, 'path':bpath}
# plot style
dstyle={'color':'g', 'linestyle':'', 'markersize':8}
# plot data, return R 
R = RA.plotPoints(iR, fdict,dstyle=dstyle)

# prepare A array
# get axis limits
axes = plt.gca()
Amin, Amax = axes.get_xlim()
logAmin=np.log10(Amin)
logAmax=np.log10(Amax)    


# plot asymptotics for small R                
if flag == 'small':       
    # plot qc asymptotics
    plt.subplot(1,2,1)
    if dim==1:
        # A array 
        A = 10**np.linspace(logAmin, logAmax, Na)
    
        # island and separatrix
        Asympt_1D.qcDelta(A,s0=1,f=1,ifplot=True)
        Asympt_1D.qcDelta(A,s0=-1,f=1,ifplot=True)
        Asympt_1D.qcDelta(A,s0=-1,f=-1,ifplot=True)
        
    else:
        # Compute A critical
        Ac = Asympt_nD.Ac_Rs(R, dim=dim)
        logAc = np.log10(Ac)
        
        # A~Ac
        A = 10**np.linspace(logAmin, logAc+1.5, Na)
        # q->1 branch
        Asympt_nD.qc_RsAc(R, A, dim=dim, sign=1, ifplot=True)
        # q->-1 branch
        Asympt_nD.qc_RsAc(R, A, dim=dim, sign=-1, ifplot=True)
        
        # A>>Ac
        A = 10**np.linspace(logAc+2.5, logAmax, Na)
        Asympt_nD.qc_RsAl(R, A, dim=dim, ifplot=True)

        
    # plot rc asymptotics
    plt.subplot(1,2,2)
    if dim==1:
        Asympt_1D.rcDelta(A,ifplot=True)
    else:
        # A~Ac
        A = 10**np.linspace(logAc-1, logAc+1, Na)
        Asympt_nD.rc_RsAc(R, A, dim=dim, ifplot=True)
      

else: # plot asymptotics for large R               
    # A array
    A = 10**np.linspace(logAmin, logAmax, Na)
    # S0 array
    S0 = A/(R*np.sqrt(np.pi))**dim
    
    # plot qc asymptotics
    plt.subplot(1,2,1)
    if dim==1:
        for n in range(3):
            qc = Asympt_1D.qcConst(S0, n=n)
            plt.plot(A, qc, linestyle='--', color='k')
    else:
        for n in [0,2]:
            qc = SE.Fixed(S0, n=n) 
            plt.plot(A, qc, linestyle='-', color='r')

    
    # plot rc asymptotics
    plt.subplot(1,2,2)
    if dim==1:
        qinf = SE.Fixed(S0, n=2) # largest root
        rc = Asympt_1D.rcConst(qinf)
        plt.plot(A, rc, linestyle='--', color='k')
    else:
        # extract eta parameter
        eta = cases[dim-1]['eta']
        
        # Larger root
        _, _, _ = Asympt_nD.Tau1A(R, A, eta=eta, dim=dim, nt=1, ifplot=True)        
        
        # A array upto upper phase boundary
        Amax = Asympt_nD.Ac_Rl(R, dim=dim, phase='2110')
        A = 10**np.linspace(logAmin, np.log10(Amax), Na)
        # Smaller root
        _, _, _ = Asympt_nD.Tau1A(R, A, eta=eta, dim=dim, nt=0, ifplot=True)   
    
