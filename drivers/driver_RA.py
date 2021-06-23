# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:52:49 2020

@author: shi9
"""
import os
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

import SE
import QP
import RA
import matplotlib.pyplot as plt
#import sys
import numpy as np
#from IPython import get_ipython
from datetime import datetime

from decimal import Decimal, getcontext

# This script is used to drive member functions in RA
# to classify phase space points by computing order parameters
# qc and rc, and save data.plot figures
######################################
# run mode 
# mode =  1 : run 1D scan in A direction and plot figure
#        -1 : load Point data and plot figure
#
#         2 : run Phase.Diagram and plot figure 
#        -2 : read data and plot Point and raw RA phase diagram

mode = -2

######################################
# parameters needed by all modes 

# dimension of the problem    
dim = 3

# R slice to process if file already exist
iR = 4
# R value to use if iR file does not exist
R = 10
# number of grids to search along A
NA = 16

# for SolveUntil
Ntmax = 2**24
epsilon = 1e-2    

# decimal precision
prec = 28
# precision scheme when running Phase.Diagram
# sdict={'R0':prec0,...}. Use precision prec when R<=R0
#sdict = None # use default precision 
sdict = {'10':28, '100':125}

# for A convergence during refinement
epsilon_bisection_A = 1e-3 # 1e-3 usually sufficient 

# print debug message
debug = True
#debug = False

# parameters for plotPoints
dstyle={'fontsize':12, 'color':'b', 'linestyle':'', 'markersize':4}

# file path
path = f'../../data/{dim}D/batch_e{epsilon}/'
#path = '../../data/3D/manual/e0.01/'

######################################
# parameteres needed by modes 1
# fix R scan A
if abs(mode)==1:

    # range of A array
    Amin = 850
    Amax = 2350
    
    # expected number of islands
    #NIm = None # determined automatically
    NIm = 1
    
    # customize qp phase sapce search
    #customize = True 
    customize = False
    if customize:
        # q box center, specify high precision using string   
        q0D = Decimal('-0.81')
        # q box length
        Lx = 0.1
        # number of sampling points
        Nx = 101   
        
        # initialize sample
        # default theta=0, Ly=2, p0=0, x0=0, y0=0
        sample = QP.Sample(q0D=q0D, Lx=Lx, Nx=Nx, Ny=1)   
        
    else: sample = None
 


######################################
# parameteres needed by modes 2
if abs(mode)==2:      
    # Amin
    logAmin = -2
    
    # range of R array
    Rmin = 0.5
    Rmax = 1
    # number of grids 
    NR = 62
    
    # R slices to process
    iRmin = 0 # default 0
    iRmax = NR # default NR  
    
    # file name iR offset
    offset = 0
    
######################################
# derived parameters
getcontext().prec = prec # default is 28
# for Q convergence during refinement in QP phase space
epsilon_bisection_Q = 10**(-prec+5) # default none, which means auto

# file name dictionary
fdict = {'dim':dim, 'Ntmax':Ntmax, 'epsilon':epsilon, 'path':path}

######################################
# update R from file if exists
fnamePT = path + f'raw/Point{dim}D_iR{iR}_N{Ntmax}_e{epsilon}.txt' 
if os.path.exists(fnamePT): readRfromfile = True
else: readRfromfile = False

if readRfromfile:       
    lines = np.loadtxt(fnamePT, comments="#", delimiter=",", unpack=True) 
    try: R = lines[0,0] # file may be empty
    except IndexError: readRfromfile = False
if readRfromfile: print(f'Read from file: R={R}')
else: print(f'User specified R = {R}')

    
##############################################################################
##############################################################################
# Scan in A direction
if mode == 1:
    # prepare A search array
    Alist = 10**np.linspace(np.log10(Amin), np.log10(Amax), NA)
    
    # print start time 
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print('Scan in A direction started on ', dt_string)
    print(f'dim={dim}, R={R}')
    print('sample=\n', sample)
    print(f'RA refine Q bisection={epsilon_bisection_Q}')
    print(f'Decimal precision is {getcontext().prec}')
    print('Data saved in file: \n', fnamePT)
    # write file header
    fid = open(fnamePT, "a+") 
    if sample==None: s = 10*'#'+'\n'
    else: s = f'## sample: q0={q0D}, Lx={Lx}, Nx={Nx} '+ 10*'#'+'\n'
    fid.write(s)          
    fid.close() 
      
    for iA in range(NA):
        A = Alist[iA]
        print(10*'#'+f' iA/NA={iA}/{NA}, A={A} '+10*'#')        
        # initialize objects
        ode = SE.ODE(R, A, dim)
        point = RA.Point(ode,Ntmax=Ntmax,epsilon=epsilon,sample=sample,NIm=NIm)
        # load point info
        pflag = point.Process(fnamePT, epsilon_bisection_Q, debug)
        print(f'pflag={pflag}')
        
    # print end time 
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print('Scan in A direction finished on ', dt_string)
        
# plot figure
if abs(mode)==1:
    RA.plotPoints(iR, fdict, varname='both', dstyle=dstyle)
    
    
##############################################################################
# initialize phase class
if abs(mode)==2: 
    # prepare R search array
    Rarray = 10**np.linspace(np.log10(Rmin), np.log10(Rmax), NR)
    # instantiate new phase object, unspecifed variables take default values
    phase = RA.Phase(dim,Rarray,NA=NA,logAmin=logAmin,Ntmax=Ntmax,\
                     epsilon=epsilon,offset=offset,rpath=path+'raw/',sdict=sdict)  
    # load other parameters
    phase.debug = debug
    phase.epsilon_bisection_A = epsilon_bisection_A 


# find R-A phase space diagram 
if mode==2:
    # compute phase diagram and save data
    phase.Diagram(iRmin=iRmin, iRmax=iRmax)
 
# plot figure
if mode==-2:
    # plot points
    plt.figure()
    RA.plotPoints(iR, fdict, varname='full', dstyle=dstyle)
    #plt.show()    
    
    # compute phase raw boundaries
    Rpdict = RA.AcPoint(iRmin, iRmax, fdict)
    # plot phase diagram
    plt.figure()
    RA.plotAcPoint(Rpdict) 
    # mark slice
    plt.axvline(x=R, color=dstyle['color'])
    #plt.show()
    
 