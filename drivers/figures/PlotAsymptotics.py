# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:14:34 2021

@author: water
"""

# This script plot R<<1 and R>>1 cases
import SE
import RA
import Asympt_nD as AS

import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal, getcontext


# R=0.01 or 100
R = 0.01

mode = 'eg'
# 'eg'  : plot example solution
# 'qc'  : plot qc, critical initial value, 
# 'rc'  : plot rc, light horizon radius
# 'H'   : plot H, normalized energy

# for SolveUntil
Ntmax = 2**24
epsilon = 1e-2

# figure size
fsize = (7,3.5)
# fontsize
fs = 12
# marker size
ms = 8
# linwwidth
lw = 2

# number of asymptotic plots to plot
Na = 1000

##########################################
# parameter dictionary
para = {'0.01':{'prec': 28, # decimal precision
                'dim': 3, # 3D problem
                'clist': ['r','c','r'], # list of colors
                ######## for example solution ###########
                'A': 0.1449559327355391, # for example solution
                'tf': 2, # final time for example solution
                # list of q0, 0110 phase
                'q0list': ['-0.2906961257216834034490719316', # for q->+1 branch
                           '-2.289360831780923586912323796', # for q->-1 branch
                           ],
                # list of corresponding qinf
                'qflist': [1, -1],
                ######## for qc and rc ###################
                'iR': 0, # point file index
                'Amin': 1e-2, # min of A axis
                'Amax': 1e4, # max of A axis                
                },
        '100': {'prec': 155, # decimal precision
                'dim': 3, # 3D problem
                'clist': ['r','orange','y','c','b'], # list of colors
                ######## for example solution ###########
                'A': 552139.12850555, # for example solution
                'tf': 250, # final time for example solution
                # list of q0, 1210 phase, [qa+, q+, q-, qa-]
                'q0list': ['0.8802968724845933157509779571828757089846978934443987421663928450625677251578861667257406129390505312291336534541830957029578', 
                           '-1.0607819954835160590977426921137782512121634856114518172007969464152236040441657234585857342722255669914859394541338824768592', 
                           '-1.08733595297688040225429259846598117773153420785561944922665226063142000316977872635193193712698290237854590837991492957404414199621510038819525597440',
                           '-1.0873359529768804022542925984659811777315342078556194492266522606314200031697787272310481837505631048738331557200171862329532',                           
                           ],
                ######## for qc and rc ###################
                'iR': 61, # point file index
                'Amin': 1e-2, # min of A axis
                'Amax': 1e7, # max of A axis                
                'eta': 0.95, # crossing radius estimation
                }}


# time step
dt = abs(epsilon*min(1,R))  # dt = abs(epsilon)*min(1,R)

##########################################
# unload parameter dictionary
pd = para[str(R)]
# set precision of Decimal
getcontext().prec = pd['prec']    

# unload para
dim = pd['dim']
# file path for batch files
bpath = f'../../../data/{dim}D/batch_e{epsilon}/'    
# data file name dictionary
fdict = {'dim':dim, 'Ntmax':Ntmax, 'epsilon':epsilon, 'path':bpath}
# plot style for point file
dstyle={'linestyle':'', 'markersize':ms}

# prepare A array
logAmin=np.log10(pd['Amin'])
logAmax=np.log10(pd['Amax'])    

# plot example solutions
if mode == 'eg':   
    plt.figure(figsize=fsize)
    # extract A
    A = pd['A']
    # specify the problem
    ode = SE.ODE(R, A, dim)
    # number of time steps
    Nt = int(pd['tf']/dt)    

    # list of initial conditions  
    q0list = pd['q0list']
    Nq = len(q0list)    
    
    # plot each case
    for i in range(Nq):
        # specify initial conditions, decima 
        cds = SE.Initial(Nt=Nt, dt=dt, q0D=Decimal(q0list[i]))    
        # instantiate the solution
        sol = SE.Solution(ode, cds, Ntmax=Ntmax, epsilon=epsilon)
        sol.fontsize=fs
        # print the problem
        print(sol)  
        # solve ODE    
        t, q = sol.Solve(ifplot=True,color=pd['clist'][i], linewidth=lw) 
        
        # asymptotic solution
        if R<=0.01:
            # inner solution
            tmax,qin=AS.qin_RsAc(t,R,A,q0=float(q0list[i]),dim=dim,ifplot=True)            
            # outer solution
            tmin, qout = AS.qout_RsAc(t,R,A,dim=dim,qinf=pd['qflist'][i],ifplot=True)
            
        elif R>=10:
            # source term
            S = ode.source(t)
            
            # adiabatic solution q->-1
            qs = SE.Fixed(S, n=0) # smallest root
            plt.plot(t, qs, color='k', linestyle='--')
            
            # adiabatic solution q->+1
            ql = SE.Fixed(S, n=2) # largest root
            plt.plot(t, ql, color='k', linestyle='--')
            
            # hopping solutions
            for nt in [0,1]:
                _,qt,qn,_=AS.q_RlAl(t,R,A,eta=pd['eta'],dim=dim,ifplot=False,nt=nt)  
                plt.plot(t, qt, 'k', linestyle='-.',linewidth=lw)  
                plt.plot(t, qn, 'k', linestyle=':',linewidth=lw)  
      

if mode in ['qc', 'rc']:  
    # initialize new figure
    plt.figure(figsize=fsize)
    
    # load color
    dstyle['color'] = pd['clist'][-1]
    # plot data
    RA.plotPoints(pd['iR'],fdict,dstyle=dstyle,varname=mode)

    # Compute A critical
    if R<=0.1:
        Ac = AS.Ac_Rs(R, dim=dim)
        logAc = np.log10(Ac)
    
# plot qc asymptotics 
if mode == 'qc':
    if R<=0.1:        
        # A~Ac
        A = 10**np.linspace(logAmin, logAc+2, Na)
        # q->1 branch
        qc = AS.qc_RsAc(R, A, dim=dim, sign=1)
        plt.plot(A, qc, 'k', linestyle='--', linewidth=lw)
        # q->-1 branch
        qc = AS.qc_RsAc(R, A, dim=dim, sign=-1)
        plt.plot(A, qc, 'k', linestyle='--', linewidth=lw)
        
        # A>>Ac
        A = 10**np.linspace(logAc+1.5, logAmax, Na)
        qc = AS.qc_RsAl(R, A, dim=dim)
        plt.plot(A, qc, 'k', linestyle=':', linewidth=lw)
        
    elif R>-10:
        # A array
        A = 10**np.linspace(logAmin, logAmax, Na)
        # S0 array
        S0 = A/(R*np.sqrt(np.pi))**dim
    
        for n in [0,2]:
            qc = SE.Fixed(S0, n=n) 
            plt.plot(A, qc, linestyle='--', color='k', linewidth=lw)
        
    plt.xlim((pd['Amin'], pd['Amax']))
    
    
# plot rc asymptotics
if mode =='rc':
    if R<=0.1:
        # A~Ac
        A = 10**np.linspace(logAc-1, logAc+1, Na)
        rc = AS.rc_RsAc(R, A, dim=dim)
        plt.plot(A, rc, 'k', linestyle='--', linewidth=lw)
        
    elif R>=10:
        # A array from 1010-1210 boundary
        Al = AS.Ac_Rl(R, dim=dim, phase='1010')
        logAl = np.log10(Al)
        A = 10**np.linspace(logAl, logAmax, Na)
        # Larger root
        _, _, _ = AS.Tau1A(R, A, eta=pd['eta'], dim=dim, nt=1, ifplot=True)        
        
        # A array upto upper phase boundary
        Au = AS.Ac_Rl(R, dim=dim, phase='2110')
        A = 10**np.linspace(logAl, np.log10(Au), Na)
        # Smaller root
        _, _, _ = AS.Tau1A(R, A, eta=pd['eta'], dim=dim, nt=0, ifplot=True)   
        
        
# plot energy
if mode == 'H':
    # plot data
    fname=bpath+f'Energy{dim}D_iR{pd["iR"]}_d{pd["prec"]}_N{Ntmax}_e{epsilon}.txt'
    RA.plotEnergy(fname, tfRmin=pd['tf']/2/R, ifdiag=True) # plot tfinal to check results

    # set figure size for E
    fig = plt.gcf()
    fig.set_size_inches(*fsize) # unpack tuple
    # plot asymptotic results 
    if R>=10:
        # array
        A = 10**np.linspace(logAmin, logAmax, Na)
        # compute E for all branches
        branches = ['an', 'ap', 'hs', 'hl']
        for b in branches:       
            E = AS.E_RlAs(R, A, branch=b, dim=dim, eta=pd['eta'], ifplot=True)
                
