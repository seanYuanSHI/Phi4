# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:14:49 2020

@author: Yuan Shi
"""

# This script plot figures for the 1D case

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')

import SE
import QP
import RA
#import Auxiliary
import matplotlib.pyplot as plt
#import sys
import numpy as np

from decimal import Decimal, getcontext

# This script plot 1D figures
############################
flag = 0

#flag = 0 # plot all figures

#flag = 1 # plot phase diagram only

#flag = 2 # plot all islands
#flag = 21 # plot island for case 1
#flag = 22 # plot island for case 2
#flag = 23 # plot island for case 3

#flag = 3 # plot all critical qc and rc

#flag = 4 # plot rc countor

#flag = 5 # plot representative solutions

############################
# if save figure to file
ifsave = True
#ifsave = False

# if show figures to screen
ifshow = not ifsave

# for SolveUntil
epsilon = 1e-2
Ntmax = 2**24
# decimal precision
getcontext().prec = 28 # default is 28

# total number of R slices
NR = 29

# file path for batch files
bpath = f'../../../data/1D/batch_e{epsilon}/'
# file path for Island
ipath = f'../../../data/1D/manual/e{epsilon}/Production/'
# file path for customized Points
#ppath = f'../../data/1D/manual/e0.01/Production/Point/Sparse/'


###################################################
# parameters 
Rkeys = ['0.1','2','10']

# R values, index, and color
Rs = {'0.1': {'R':0.1, 'iR':0, 'color': 'r'},
      '2': {'R':2.371373705661655, 'iR':11, 'color': 'b'},
      '10': {'R':10, 'iR':16, 'color': 'g'}}

# RA values to draw sampling dots
RAs = {'0.1': [0.2, 0.6, 1.2], 
      '2': [0.6, 1.35, 2.0], 
      '10': [1.0, 3.2, 4.6]}

# colors for drawing RA points
RAcolors = {'0.1': ['brown', 'r', 'orange'],
            '2': ['darkslategrey', 'b', 'c'],
            '10': ['olive','g','lawngreen']}


rinvalid = -333

# plot ontsize 
fontsize = 18
# plot marker size
markersize = 8
# number of contour levels
nlevels = 30
# contour colormap
cmap = 'PuBuGn'
#cmap = "OrRd"
#cmap = "GnBu" 

# tf/R
tfR = 2.5
# representative solutions
Rrep = [10,10]
qrep = [[Decimal('-0.4796223634274262620642231758'),
         Decimal('0.939467288586696049679724953'),
         Decimal('-1.051443578599554218031718966')],
        ####
        [Decimal('0.1555718008342852720016987044'),
         Decimal('0.724988900061230680353488474'),
         Decimal('-1.145049923731027860361429732')]]
Arep = [1.0, 3.2] # same length as Rrep
crep = ['olive','g'] # same length as Rrep

lrep = ['--','-','dashdot'] # same length as qrep
wrep = [3,2] # linewidth

# phase correction
pcorrect = {'1100':1110}

# file name dictionary
fdict = {'dim':1, 'Ntmax':Ntmax, 'epsilon':epsilon, 'path':bpath}

###################################################
# compute phase boundary
Rpdict = RA.AcPoint(0, NR, fdict, pcorrect=pcorrect, ifplot=False, ifsave=True)
    
# plot phase diagram
if flag in [0,1]:     
    # initialize figure
    #plt.figure(num=None, figsize=(6, 18), dpi=80, facecolor='w', edgecolor='k')
    plt.figure(num=None, figsize=(6, 30))
    # plot phase boundaries
    #RA.plotAc(fnameAc, fontsize=fontsize, linestyle='')
    # plot phase diagram
    RA.plotAcPoint(Rpdict, {'clist':['red','orange']})
    
            
    # plot asymptotic phase boudary for small R: 1110->2010
    R_small = 10**np.linspace(-2,0.2,100)
    plt.plot(R_small,np.ones_like(R_small),color='black',linestyle='--')  

    # plot asymptotic phase boudary for large R
    R_large = 10**np.linspace(1,2,100)
    # 2010->0010
    Ac = R_large*np.sqrt(np.pi)/3/np.sqrt(3)
    plt.plot(R_large,Ac,color='black',linestyle='--')      
    # 1110->2010
    plt.plot(R_large,Ac/np.sqrt(2),color='black',linestyle='--')     
    
    plt.xticks([1e-2,0.1,1,10,1e2])
    plt.xlim((1e-2,1e2))
    plt.ylim((0.1,1e2))
    # save figure
    if ifsave: plt.savefig(bpath + 'Figures/RA1D.png')
    # show figure
    if ifshow: plt.show()

###################################################
# plot phase space islands
# file tail
ftail = f'_N{Ntmax}_e{epsilon}.txt' 

for i in range(len(Rkeys)):
    if flag in [0,2,21+i]:
        # key
        rk = Rkeys[i]
        # R value
        R = Rs[rk]['R']
        # A values
        As = RAs[rk]    

        # initialize new figure
        #plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
        f = plt.figure(num=None, figsize=(4, 5))
        # plot figures
        for iA in range(len(As)): 
            # unpack A
            A = As[iA]    
            # specify the problem
            ode = SE.ODE(R, A, 1) # dim = 1    
            # instantiate island
            island = QP.Island(ode)
            # update plot parameters
            island.color = RAcolors[rk][iA]
            island.fontsize = fontsize
            island.markersize = markersize
            
            # file name
            fbody = f'_R{R}_A{A}' # e=0.001
            #fbody = f'_A{A}_R{R}' # e=0.01
            fnameQPb = ipath + 'Boundary1D' + fbody + ftail    
            fnameQPs = ipath + 'Separatrix1D' + fbody + ftail 
            # read data
            island.load(fnameQPb)       
            island.load(fnameQPs)       
            
            # plot island boundary  
            data = ['boundary','separatrix']
            for d in data:
                island.plot(data=d, label=f'A={A}', equal=False)
    
        #plt.legend(loc='lower left')
        #plt.legend(bbox_to_anchor=(1.4, 0.4), loc='right', fontsize=fontsize)

        
        # save figure
        if ifsave: plt.savefig(bpath + f'Figures/QP1D_R{rk}.png')
        # show figure
        if ifshow: plt.show()
        
###################################################
# plot qc and rc
# parameters for plotPoints
dstyle={'fontsize':fontsize, 'linestyle':'', 'markersize':markersize}

if flag in [0,3]:
    for i in range(len(Rkeys)):    
        # key
        rk = Rkeys[i]
        # iR value
        iR = Rs[rk]['iR']
        # A values
        As = RAs[rk]    

        # initialize new figure
        #plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')  
        plt.figure(num=None, figsize=(8, 3))   
        
        # # mark vertical lines
        # for iA in range(len(As)): 
        #     plt.axvline(x=As[iA] , color=RAcolors[rk][iA])       
            
        # plot figures
        dstyle['color']=Rs[rk]['color']
        RA.plotPoints(iR, fdict, varname='both', dstyle=dstyle)
        # axis limit
        plt.xlim((0.01,1.1*max(As)))
        
        # save figure
        if ifsave: plt.savefig(bpath + f'Figures/qc1D_R{rk}.png')
        # show figure
        if ifshow: plt.show()
        
###################################################
# plot rc
if flag in [0,4]:
    # initialize data
    Alist = []
    Rlist = []
    rclist =[]
    
    # read data
    for iR in range(NR):
        # file name 
        fnameRc = bpath + f'raw/Point1D_iR{iR}' +ftail   
        #fnameRc = f'../../data/1D/batch_e0.01/raw/Point1D_iR{iR}_N16777216_e0.01.txt'  
        # read file
        lines = np.loadtxt(fnameRc, comments="#", delimiter=",", unpack=True)                        
        # dimensions
        _, nrow = lines.shape       
        # expect rc from the 8th column
        rc = lines[8]
        
        # loop through the lines
        for i in range(nrow):
            # select valid r values, expect in 5th column
            r = rc[i]
            if r>rinvalid:
                rclist.append(r)
                Rlist.append(lines[0][i])
                Alist.append(lines[1][i])
    
    # initialize new figure
    #fig, ax = plt.subplots(1,1,figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')
    fig, ax = plt.subplots(1,1,figsize=(4, 3))
    # plot phase boundaries
    #RA.plotAc(fnameAc, fontsize=fontsize, marker='')
    RA.plotAcPoint(Rpdict, {'clist':['red','orange']})
    # plot contour data    
    cs=ax.tricontourf(Rlist,Alist,rclist,levels=nlevels,cmap=plt.cm.get_cmap(cmap))
    # add color bar
    fig.colorbar(cs, ax=ax, shrink=0.9, ticks=[0,2,4,6])
    # axis features
    ax.set_xlabel(r'$R$', fontsize=fontsize)
    ax.set_ylabel(r'$A$', fontsize=fontsize)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim((0.1,1e2))
    plt.xticks([0.1,1,10,1e2])

    plt.rcParams.update({'font.size': fontsize})
    # save figure
    if ifsave: plt.savefig(bpath + 'Figures/rc1D_RA.png')
    # show figure
    if ifshow: plt.show()     
    
    
    # initialize new figure
    #plt.figure(num=None, figsize=(4, 4), dpi=80, facecolor='w', edgecolor='k') 
    plt.figure(num=None, figsize=(2, 2)) 
    # plot rc at selected slices
    for i in range(len(Rkeys)):
        # key
        rk = Rkeys[i]
        # iR value
        iR = Rs[rk]['iR']
               
        # file name
        fnameQc = bpath + f'raw/Point1D_iR{iR}' +ftail        
        #fnameQc = f'../../data/1D/batch_e0.01/raw/Point1D_iR{iR}_N16777216_e0.01.txt'   
        
        # plot figures
        dstyle['color']=Rs[rk]['color']
        RA.plotPoints(iR, fdict, varname='rc', dstyle=dstyle)
      
    # axis limit
    plt.xlim((0.1,5))
    plt.ylim((0,5))    
    # save figure
    if ifsave: plt.savefig(bpath + 'Figures/rc1D_A.png')
    # show figure
    if ifshow: plt.show()
    
###################################################
# plot plot representative solutions
if flag in [0,5]:   
    # initialize figure
    plt.figure(num=None, figsize=(2.5, 3.5))     
    # specified cases
    for i in range(2):
        R = Rrep[i]
        A = Arep[i]
        c = crep[i]      
        w = wrep[i]
        l = lrep[i]

        # plot p and n branches
        for j in range(3):
            qD = qrep[i][j]
            #l = lrep[j]

            # specify the problem
            ode = SE.ODE(R, A, 1)                    
            # time step
            dt = abs(epsilon*min(1,R))  # dt = abs(epsilon)*min(1,R)
            # number of time steps
            Nt = int(tfR*R/dt)
            
             # specify initial conditions 
            cds = SE.Initial(Nt=Nt, dt=dt, q0D=qD)    
            # instantiate the solution
            sol = SE.Solution(ode, cds, Ntmax=16777216, epsilon=1e-3)
            sol.fontsize=fontsize 
             
            # numerical solution                  
            sol.Solve(ifplot=True, color=c, linestyle=l, linewidth=w) 
                    
    #plt.legend(loc='best', fontsize=fontsize )               
    plt.ylim((-1.5,1.5)) 
    plt.xlim((0,tfR*R))
    plt.axhline(y=1, color='grey',linestyle=':') 
    plt.axhline(y=-1, color='grey',linestyle=':')             
    # save figure
    if ifsave: plt.savefig(bpath + 'Figures/Solution1D.png')
    # show figure
    if ifshow: plt.show()  

