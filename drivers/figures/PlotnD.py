# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:14:49 2020

@author: Yuan Shi
"""

# This script plot figures for the 2D and 3D cases

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')

import SE
#import QP
import RA
import Asympt_nD

import matplotlib.pyplot as plt
#import sys
import numpy as np

from decimal import getcontext

#########################
# dimension, dim = 2 or 3
dim = 2

# plot flag
flag = 1

# The following do not require Plot*D_markups
#    flag = 1 # plot phase diagram only

# The following requires Plot*D_markups
#    flag = 2 # plot all critical qc and rc
#    flag = 21 # plot qc for for case 1
#    flag = 22 # plot qc for for case 2, and so on

#    flag = 3 # plot representative solutions

#    flag = 0 # plot all figures

#########################
# parameters for solve until
epsilon = 1e-2
Ntmax = 2**24
# decimal precision
#getcontext().prec = 28 # default is 28

# number of raw files
NR = 62

# if save figure to file
#ifsave = True
ifsave = False

# if show figure to screen
ifshow = not ifsave

# if markup, show R lines and A sample points
#ifmarkup = False
ifmarkup = True


# file path for batch files
bpath = f'../../../data/{dim}D/batch_e{epsilon}/'

# phase correction
pcorrect = {'10':110, '1110':1210, '2000':2110, '1100':1210, '2010':2110, '0':110}

# plotting parameters
# plot fontsize 
fs = 12
# plot marker size
ms = 6
# plot linestyle 
ls='-'
# linewidth
lw = 1.5


###################################################
assert(dim==2 or dim==3)
if dim==2:
    import Plot2D_markups as mks
else:
    import Plot3D_markups as mks

# read R values from file
for rk in mks.Rkeys:
    # iR value
    iR = mks.Rs[rk]['iR']
    # file name
    fname = bpath + f'raw/Point{dim}D_iR{iR}_N{Ntmax}_e{epsilon}.txt' 
    # read file
    lines = np.loadtxt(fname, comments="#", delimiter=",", unpack=True) 
    R = lines[0,0]
    # load R value to dictionary
    mks.Rs[rk]['R'] = R
    # report R values
    print(f'rk={rk}, iR={iR}, R={R}')
 
   # file name dictionary
fdict = {'dim':dim, 'Ntmax':Ntmax, 'epsilon':epsilon, 'path':bpath} 
# plot style         
dstyle={'fontsize':fs, 'linestyle':ls, 'markersize':ms, 'linewidth':lw}

###################################################
# compute, save and plot plot phase diagram
if flag in [0,1]:    
    # initialize figure
    #plt.figure(num=None, figsize=(6, 18), dpi=80, facecolor='w', edgecolor='k')
    plt.figure(num=None, figsize=(16, 6))
    
    # compute phase boundaries
    Rpdict = RA.AcPoint(0, NR, fdict, pcorrect=pcorrect, ifplot=False, ifsave=True)

    # correct phase label
    apc = mks.apc
    for Rkey in apc:
        try:
            Rpdict[Rkey][apc[Rkey]['new']]=Rpdict[Rkey][apc[Rkey]['old']]
        except KeyError:
            pass
        else:
            del Rpdict[Rkey][apc[Rkey]['old']]
    
    # plot phase diagram
    RA.plotAcPoint(Rpdict, dstyle=dstyle)    
    
    # mark characteristic lines
    if ifmarkup:
        for rk in mks.Rkeys:
            # R value
            R = mks.Rs[rk]['R']
            # A values
            As = mks.RAs[rk]
            
            # mark characteristic lines
            plt.axvline(x=R, color=mks.Rs[rk]['color'])
            
            # mark characteristic points
            for iA in range(len(As)):
                c = mks.RAcolors[rk][iA]
                plt.scatter(R, As[iA], s=2*ms, edgecolors=c, marker='s', facecolors=c) 
 
    
    # plot asymptotic phase boudary for small R: 1010->0110
    R = 10**np.linspace(-2.5,-0.2,100)
    Ac = Asympt_nD.Ac_Rs(R, dim=dim)
    plt.plot(R,Ac,color='k',linestyle='--', linewidth=lw)  

    # plot asymptotic phase boudary for large R: 1010->1210
    R = 10**np.linspace(0.8,3,100)
    Ac = Asympt_nD.Ac_Rl(R, dim=dim, phase='1010')
    plt.plot(R,Ac,color='k',linestyle='--', linewidth=lw)  
    
    # plot asymptotic phase boudary for large R: 2110 -> 0110
    R = 10**np.linspace(1.1,3,100)
    Ac = Asympt_nD.Ac_Rl(R, dim=dim, phase='2110')
    plt.plot(R,Ac,color='k',linestyle='--', linewidth=lw)     
    
    plt.xticks([0.1,10])
    plt.xlim((1e-2,1e2))
    
    if dim==2: plt.ylim((0.6,1e4))
    else: plt.ylim((1e-2,1e10))
    
    # save figure
    if ifsave: plt.savefig(bpath + f'Figures/RA{dim}D.png')
    # show figure
    if ifshow: plt.show()


###################################################
# plot qc and rc
for i in range(len(mks.Rkeys)):
    if flag in [0,2,21+i]:
        # key
        rk = mks.Rkeys[i]
        # iR value
        iR = mks.Rs[rk]['iR']
        # A values
        As = mks.RAs[rk]  
        # update color
        dstyle['color'] = mks.Rs[rk]['color']

        ##########################3
        # plot qc
        # initialize new figure
        #plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')  
        plt.figure(num=None, figsize=(3, 4))          
        # mark vertical lines
        if ifmarkup: 
            for iA in range(len(As)): 
                plt.axvline(x=As[iA] , color=mks.RAcolors[rk][iA])       
            
        # plot figures
        RA.plotPoints(iR, fdict, varname='qc', dstyle=dstyle)
        # axis limit
        if dim==2: plt.xlim((1e-2,1e3))
        else: plt.xlim((1e-2,1e4))
        # save figure
        if ifsave: plt.savefig(bpath + f'Figures/qc{dim}D_iR{iR}.png')
        # show figure
        if ifshow: plt.show()
        
        ##########################3
        # plot rc
        # initialize new figure
        #plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')  
        plt.figure(num=None, figsize=(3, 4))   
        # mark vertical lines
        if ifmarkup: 
            for iA in range(len(As)): 
                plt.axvline(x=As[iA] , color=mks.RAcolors[rk][iA])  
            
        # plot figures
        RA.plotPoints(iR, fdict, varname='rc', dstyle=dstyle)
        # axis limit
        if dim==2: plt.xlim((1e-2,1e3))
        else: plt.xlim((1e-2,1e4))
        # save figure
        if ifsave: plt.savefig(bpath + f'Figures/rc{dim}D_iR{iR}.png')
        # show figure
        if ifshow: plt.show()        

    
###################################################
# plot plot representative solutions
if flag in [0,3]:  
    for Rkey in mks.RArep:
        # final time in unit of R
        tfR = mks.tfRrep[Rkey]
        # decimal precision
        getcontext().prec = mks.Rprec[Rkey]      
        # list of A values
        Aindex = mks.RArep[Rkey]
        # list of critical q valus
        qlist = mks.qrep[Rkey]
        
        # initialize figure
        plt.figure(num=None, figsize=(3, 4))   
        #plt.figure(num=None, figsize=(10, 4))          
        # plot specified cases
        for i in range(len(Aindex)):
            # index of A
            ind = Aindex[i]
            R = mks.Rrep[Rkey]
            A = mks.RAs[Rkey][ind]
            # color
            c = mks.RAcolors[Rkey][ind]
            # line width
            w = mks.wrep[i]
            # linestyle
            l = mks.lrep[i]
    
            # representative q values
            qc = qlist[i]
            # plot p and n branches
            for j in range(len(qc)):
                qD = qc[j]
                #l = lrep[j]
    
                # specify the problem
                ode = SE.ODE(R, A, dim)                    
                # time step
                dt = abs(epsilon*min(1,R))  # dt = abs(epsilon)*min(1,R)
                # number of time steps
                Nt = int(tfR*R/dt)
                
                # specify initial conditions 
                cds = SE.Initial(Nt=Nt, dt=dt, q0D=qD)    
                # instantiate the solution
                sol = SE.Solution(ode, cds, Ntmax=16777216, epsilon=1e-3)
                sol.fontsize=fs 
                
                #print(sol)
                #print(f'decimal precision is {getcontext().prec}')
                 
                # numerical solution                  
                t, yn = sol.Solve(ifplot=False) 
                # plot numerical solution
                plt.plot(t, yn,color=c, linestyle=l, linewidth=w)
                        
        #plt.legend(loc='best', fontsize=fontsize )      
        plt.xlabel('t=m*x', fontsize=fs)
        plt.ylabel('phi/v', fontsize=fs)
         
        plt.ylim((-2,1.5)) 
        plt.xlim((0,tfR*R))
        plt.axhline(y=1, color='grey',linestyle=':')     
        plt.axhline(y=0, color='grey',linestyle='-')     
        plt.axhline(y=-1, color='grey',linestyle=':')    
        # save figure
        if ifsave: plt.savefig(bpath + f'Figures/sol{dim}D_R{Rkey}.png')              
        # show figure
        if ifshow: plt.show()
        
