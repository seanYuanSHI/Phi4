# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:20:11 2021

@author: water
"""

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

import RA
import Asympt_nD as AS
import Asympt_1D as AS1
import numpy as np
import matplotlib.pyplot as plt
from decimal import getcontext

# This is the driver function for RA.Energy
# which post process data file generated by RA.Point
# and estimate energy of the critical solutions.

######################################
# run mode 
# mode =  1 : process data and plot figure
#         2 : reprocess data at higher precision and plot figure
#        -1 : load data and plot figure with asymptotics
#        -2 : load and plot data only 

mode = -2

######################################
# dimension of the problem    
dim = 3

# for SolveUntil
Ntmax = 2**24
epsilon = 1e-2    

# R slice to process assuming file already exist
# # targeted final integration time is tf = 5*max(1,R)
iR = 11
tfRmin = 2.2 # only plot data with tfinal>tfRmin*R

# decimal precision
# RA: sdict = {'10':28, '100':125}
#prec = 28 # default 28
prec = 28

# for restart in mode 1
if mode==1:
    # A slice to start processing
    #iAmin = 0 # start from begining 
    #iAmin > 0 # restart from iAmin
    iAmin = 0

# for reprocessing mode
if mode==2:
    # tfinal minimum 
    tfmin = 225
    # increase of precision
    dprec = 5
    
# print debug message
debug = True
#debug = False

# flush output file after each write
ifflush = True # useful for long calculation
#ifflush = False # accelerate IO for short calculation

# file path
path = f'../../data/{dim}D/batch_e{epsilon}/'
#path = '../../data/3D/manual/e0.01/'


######################################
# for plotting asymptotic solution for R>>1
ifasympt = True
#ifasympt = False

NA = 1000 # number of A sampling points, uniform in logA
Amin, Amax = None, None # if none, read range from Point file

# weighting parameter in (0,1) for transition time estimation
#             eta = 0 is the 0-th order scheme
#             eta = 1 is the 1-st order scheme
eta = 0.8

######################################
# set decimal precision
getcontext().prec = prec # default is 28
# file name dictionary
fdict = {'dim':dim, 'Ntmax':Ntmax, 'epsilon':epsilon, 'path':path}

# compute data
if mode == 1:
    print('Computing energy by post processing Point files...')
    RA.Energy(iR, fdict, iAmin=iAmin, debug=debug, ifflush=ifflush)
    
# reprocess data
if mode == 2:
    print('Re-processing energy file at higher precision...')
    print(f'The targeted tfinal > {tfmin}')
    RA.EnergyRefine(iR,prec,fdict,tfmin,dprec=dprec,debug=debug,ifflush=ifflush)
    
###########################
# plot data
prec1 = getcontext().prec
fname = path + f'Energy{dim}D_iR{iR}_d{prec1}_N{Ntmax}_e{epsilon}.txt'
# plot numerical results
print(f'Plotting energy from {fname}...')
Alist, qlist, data = RA.plotEnergy(fname, tfRmin=tfRmin, ifdiag=True) # plot tfinal to check results
# data = [tlist, Elist]

# plot asymptotic results 
if ifasympt and mode==-1:
    # read R and A
    lines = RA.readPoint(iR, fdict) # A is sorted from small to large    
    # R value
    R = lines[0,0]    
    # A array
    if Amin==None: Amin = lines[1, 0]
    if Amax==None: Amax = lines[1, -1]
    A = 10**np.linspace(np.log10(Amin), np.log10(Amax), NA)
        
    if R>=10:
        print('Plotting asymptotic energy for R>>1...')     
        if dim>1:       
            # compute E for all branches
            branches = ['an', 'ap', 'hs', 'hl']
            for b in branches:       
                E = AS.E_RlAs(R, A, branch=b, dim=dim, eta=eta, ifplot=True)
            
        else: # dim=1
            # compute E for all branches
            branches = ['an', 'ap', 'h']
            for b in branches:       
                E = AS1.EConst(R, A, branch=b, ifplot=True)
 
            
    if R<=0.1 and dim==1:
        print('Plotting asymptotic energy for R<<1...')     
        # (-,-) branch
        plt.plot(A, (1-(1+A)**(3/2))/3,'k--')
        # (-,+) branch
        plt.plot(A, (1+(1-A)**(3/2))/3,'k--')
        # (-,-) branch
        plt.plot(A, (1-(1-A)**(3/2))/3,'k--')
        
    plt.legend(loc='best')    
        