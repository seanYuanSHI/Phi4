# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:52:49 2020

@author: shi9
"""

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

import SE
import QP
#import Asympt_nD
import matplotlib.pyplot as plt
#import sys
import numpy as np
#from IPython import get_ipython

from decimal import Decimal, getcontext

# This script is used to drive member functions in QP
# to find phase space islands/separatrices and plot results
######################################
# run mode 
# mode =  1 : run all tests of constituent functions
#             1D Scan, xCritical, critical solutions, 2D scan          
#        -1 : run quick tests of constituent functions
#             1D Scan, xCritical
#             no critical solutions
#             no low-resolution contour
#
#         2 : run Boundary and plot figure. 
#             find phase space island/separatrix for given A and R
#        -2 : read boundary data and plot figure
#
#        otherwise: only load instances of class

mode = -1

######################################
# dimension of the problem    
dim = 2

# parameters of ODE 
R = 10.0 # source size
A = 247.5735556419664 # source amplitude

# final time for plot of critical cases
tf = 2*max(1,R)

# customization of search grid
#customize = True
customize = False

# customized search parameters
if customize:    
    # initial conditions & center of search box, decimal precision
    # specify high precision using string   
    q0D=Decimal('1.4') 
    p0D=Decimal('-1.4')
    
    # box angle
    theta = 10
    # box size
    Lx = 6
    Ly = 3
    # number of grid points
    Nx = 33
    Ny = 33
    
    # expected number of islands in the search interval
    NI = 1
    # expected number of separatrix in the search interval
    NS = 1
    # if automatically scan in addition to customized scan
    ifauto = True

# for SolveUntil
Ntmax = 2**24
epsilon = 1e-2 

# decimal precision
prec = 28
getcontext().prec = prec # default is 28
# for QP convergence during refinement
epsilon_bisection = 10**(-prec+5)
#epsilon_bisection = 1e-6


if abs(mode)==2:
    # file path
    #fhead = f'../../data/{dim}D/manual/e{epsilon}/'
    fhead = '../../data/1D/manual/'
    # file tail
    ftail = f'{dim}D_R{R}_A{A}_N{Ntmax}_e{epsilon}.txt'
        
    # file namefor island boundary
    fnameQPb = fhead + 'Boundary' + ftail
    # file name for separatrix
    fnameQPs = fhead + 'Separatrix' + ftail  


# rescaling q = f + a*w, t = u*z
# defaults are f=0, a=1, and u=1
f, a, u = 0, 1, 1
#f = float(q0D) # requires float
#a = 1e-12
#u = 1

# print debug messages
#ifdebug = False
ifdebug = True

# plot equal scale in q and p directions
equal = True
#equal = False

# parameters for making plots
fontsize = 12 # fontsize of labels
color = 'r' # color of scatter plot
markersize = 8

##############################################################################
##############################################################################
# specify the problem
ode = SE.ODE(R, A, dim, f=f, a=a, u=u)

# instantiate island
island = QP.Island(ode, Ntmax=Ntmax, epsilon=epsilon)
# update default parameters
island.epsilon_bisection = epsilon_bisection
if customize:
    island.NC = 2*NI
    island.NS = NS
#island.Nsample_refine = 101
repr(island)
# updating plotting parameters
island.fontsize = fontsize
island.color = color
island.markersize = markersize

# specify search box
if customize: 
    sample = QP.Sample(theta=theta,Nx=Nx,Ny=Ny,Lx=Lx,Ly=Ly,q0D=q0D,p0D=p0D) 
else:
    sample = QP.Sample() # default low resolution Nx=Ny=33       
    sample.auto(ode) # load auto parameters suggested by ODE in 1D case


##############################################################################
if abs(mode) == 1:# run test of constituent functions
    # test scan in x direction, starting from t=0
    if customize: 
        xsample = QP.Sample(Lx=Lx,Ly=Ly,q0D=q0D,p0D=p0D,ifauto=ifauto,\
                            Nx=Nx,Ny=1,theta=theta)        
    else: 
        xsample = None
    
    print(40*'#'+'\nDetermine critical points,customize =',customize)
    plt.figure()    
    xcD,xfc,xsD = island.xCritical(sample=xsample, ifplot=True, debug=ifdebug)
    print(f'xRefine depth = {island.depth}')
      

if mode==1:  
    ###########################
    print(40*'#'+'\nPlotting critical solutions...')
    # plot near criticl cases (boundary + separatrix), starting from t=0
    ncs = len(xcD) + len(xsD)
    if ncs>0: # if there are critical cases
        # concatenate numpy array
        xcsD = np.concatenate((xcD, xsD))
        # time step
        dt = island.dt  # dt = abs(epsilon)*min(1,R)
        # number of time steps
        Nt = int(tf/dt)
        plt.figure()
        for i in range(ncs):
            # default t0=0
            cds = SE.Initial(Nt=Nt, dt=abs(dt), q0D=xcsD[i], p0D=0) 
            sol = SE.Solution(ode, cds, Ntmax=Ntmax, epsilon=epsilon) 
            sol.fontsize=fontsize
            ttmp, yntmp = sol.Solve(ifplot=True)   
            
    
        plt.ylim((-1.5, 1.5))  
        #plt.legend(bbox_to_anchor=(2, 0.4), loc='right', fontsize=fontsize)
        #plt.legend(loc='best', fontsize=fontsize)
        plt.axhline(y=0, color='grey')
        plt.axhline(y=-1, color='grey', linestyle=':')
        plt.axhline(y=1, color='grey', linestyle=':')
        #plt.show()
    
    
    ###########################
    print(40*'#'+'\nPloting phase space island using contour...')
    # plot phase space island using contour
    if sample.Nx>1 and sample.Ny>1:
        plt.figure()
        Q, P, Flags = island.Scan(sample, ifplot=True)        
        # mark search lines along x and y directions
        sample.drawbox()
        plt.show()


##############################################################################
# find phase space island and save data, starting from t=0   
if mode==2 or (mode==-2 and dim>1 and (not customize)): 
    print('Computing boundary of phase space island, '+repr(ode))
    print('customize = ', customize)
    # compute boundary in q-p space
    island.Boundary(sample) 
    # write data
    if dim == 1 or customize: 
        island.save(fnameQPb, sample,'boundary')
        island.save(fnameQPs, sample,'separatrix')
    # report boundary info
    if dim > 1 and (not customize):
        print('qb =', island.qbD)
        print('asymptotics =', island.fb)
        print('qs =', island.qsD)

if abs(mode) == 2: # read data and plot phase space island
    print('Reading boundary and separatrix files and plot figure...')
    # 1D read file
    if dim==1 or (dim>1 and customize): 
        island.load(fnameQPb)    
        island.load(fnameQPs)    
    # plot island boundary
    #plt.figure()
    island.plot(data='boundary', equal=equal)
    island.plot(data='separatrix', equal=equal)
    # plot search box 
    sample.Ny = max(sample.Ny,2) # also draw box for n-D case
    sample.drawbox()
    #plt.show()
    
    
