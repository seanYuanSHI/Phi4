# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 21:58:24 2020

@author: water
"""

# This script contains functions related to 1D asymptotics
#############################################################  
# Content of this script in order of appearence
# For delta function source S=term A*delta(t)
#     qDelta   : Solution to the ODE
#     rcDelta  : Light horizon radius 
#     qcDelta  : Initial condition for critical solutions
#
# Nonperturbative solution with constant source term S=S0
#    qcConst   : Initial value q0 as a function of S0
#    Q0Const   : Initial value q0 as a function of qinf 
#    Phi0      : Auxilliary function phi0 as a function of qinf
#    t2tau     : Time scaling factor r=t/tau as a function of qinf 
#   tlConst    : For given S0, when t>>tl, solution q approaches qinf
#    qConst    : Solution to the ODE for given S0
#   rcConst    : Light horizon radius as a function of qinf 
#  drcConst    : Derivative of rc w.r.t qinf
#    EConst    : Normalized energy of field configurations

#############################################################  
#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '../')
import SE
 
import numpy as np
#import scipy.special as ss
import matplotlib.pyplot as plt

##########################################
# Solution of q with delta function source term A*delta(t)
# Initial condition q0=q(t=0) is related to A by A=1-q0^2

# Solution to ODE with delta function source term A*delta(t)
# Inputs:
#    t  : normalized time tau
#    q0 : initial condition, float
#    f  : f=1 for branches with q->1
#         f=-1 for branches with q->-1
def qDelta(t, q0, f=1, ifplot=False):
    """Solution with delta function source."""
    # make sure f=1 or -1
    f = np.sign(f)
    assert(f!=0)
    
    y = np.tanh(t/2)
    q = (q0 + f*y)/(1 + f*q0*y)
    
    # plot figure
    if ifplot:
        plt.plot(t, q, linestyle='-', color='r', label='qDelta')
    
    return q
 
    
# Light horizon radius with delta function source term A*delta(t)
def rcDelta(A, ifplot=False):
    """Light horizon radius with delta function source."""
    rc = 2*np.arctanh(np.sqrt(1-A)) 
    
    # plot figure
    if ifplot:
        plt.plot(A, rc, linestyle='--', color='k', label='rcDelta')
        plt.xscale('log')
    
    return rc


# Initial condition for critical solutions
# with delta function source term A*delta(t)
# Inputs:
#    A    : source strength
#    s0   : sign of initial q0, s0=1 or -1
#    f    :  sign of asymptotic values, f=1 or -1
def qcDelta(A, s0=1, f=1, ifplot=False):
    """Critical initial condition for delta function source."""
    # ensure s0 and sinf are signs
    s0 = np.sign(s0)
    f = np.sign(f)
    
    # critical value
    qc = s0*np.sqrt(1-f*A).real 
    
    # plot figure
    if ifplot:
        plt.plot(A, qc, linestyle='--', color='k', label=f'qcDelta_{s0}{f}')
        plt.xscale('log')
    
    return qc


##########################################
# Nonperturbative solution with constant source term S=S0
# For S0>0, the nonperturbative solution increase from the 
# cutoff to the largest stable fixed point

# Initial value q0 as a function of S0
# Inputs:
#    S0  : constant source strength
#    n   : index of cubic root
#          n=0, initial value equals to the smallest fixed point
#          n=1, initial value equals to the cutoff
#          n=2, initial value equals to the largest fixed point
def qcConst(S0, n=1, ifplot=False):    
    """Critical initial condition for constant source."""
    if n!=1:
        # initial value equals to fixed points
        qc = SE.Fixed(S0, n=n) #n=(0, 2): (smallest, largest) fixed point
    else:
        # compute qinf from S0
        f = SE.Fixed(S0, n=2) # largest root
        qc = -f + np.sqrt(2*(1-f**2))
    
    # plot figure
    if ifplot:
        plt.plot(S0, qc, linestyle='--', color='k', label='qcConst')
        plt.xscale('log')
    
    return qc


# Initial value q0 as a function of qinf
def Q0Const(qinf):   
    """Critical initial value q0 as a function of qinf."""
    return -qinf + np.sqrt(2*(1-qinf**2))


# Auxilliary function phi0 as a function of qinf
def Phi0(qinf):   
    """Auxilliary function phi0 as a function of qinf."""
    return 2*qinf/np.sqrt(2*(1-qinf**2)) - 1


# Time scaling factor
# Notice that for t2tau to be real, we need qinf>1/3
# return the real part
def t2tau(qinf):
    """Time scaling factor as a function of qinf."""
    r = np.sqrt((3*qinf**2-1)/2)/2
    return np.real(r)


# t that is sufficiently large that q approaches qinf
# need cosh(r*t)^2 >> phi0. Here, take cosh(tl)=sqrt(1+phi0)
def tlConst(S0):
    """t that is sufficiently large that q approaches qinf."""
    # compute qinf from S0
    qinf = SE.Fixed(S0, n=2) # largest root
    
    # compupte auxsilliary factors
    p0 = Phi0(qinf)
    r = t2tau(qinf)
    
    # a simple estimate of tl
    tl = np.arccosh(np.sqrt(1+p0))/r 
    
    return tl
    

# Solution to ODE with constant source term
def qConst(t, S0, ifplot=False):
    """Solution to ODE with constant source term."""
    # compute qinf from S0
    qinf = SE.Fixed(S0, n=2) # largest root
    
    # compupte auxsilliary factors
    p0 = Phi0(qinf)
    q0 = Q0Const(qinf)
    r = t2tau(qinf)
    
    # function factors
    rt = r*t
    s2 = np.sinh(rt)**2
    c2 = np.cosh(rt)**2
    
    # solution to ODE
    q = ((2+p0)*q0 + 2*qinf*s2)/(2*c2 + p0)
    
    
    # plot figure
    if ifplot:
        # plot up to 5*tl
        tmax = 5*tlConst(S0)
        #tm = np.ma.masked_where(t>tmax, t)
        qm = np.ma.masked_where(t>tmax, q)
        plt.plot(t, qm, linestyle=':', color='k', label='qConst', linewidth=3)       
    
    return q


# Light horizon radius with constant source term
def rcConst(qinf, ifplot=False):
    """Light horizon radius with constant source term."""
    # compupte auxsilliary factors
    p0 = Phi0(qinf)
    q0 = Q0Const(qinf)
    r = t2tau(qinf)
    
    # light horizon radius
    rc = np.arcsinh(np.sqrt(-q0*(2+p0)/2/qinf))/r 
    
    # plot figure
    if ifplot:
        S0 = (qinf**2-1)*qinf/2
        plt.plot(S0, rc, linestyle='-', color='r', label='rcConst')
        plt.xscale('log')
    
    return rc


# derivative of rc w.r.t qinf
def drcConst(qinf):
    """Derivative of rc w.r.t qinf."""
    # compupte auxsilliary factors
    p0 = Phi0(qinf)
    r = t2tau(qinf)
    
    # sqrt factor in arcsinh for rc
    s2 = (p0-1)*(p0+2)/2/(p0+1)
    s = np.sqrt(s2)
    
    # light horizon radius
    rc = np.arcsinh(s)/r 
    
    # factor for derivative of s
    ds = (1-p0)*(1+(1+p0)**2/2) - (2+p0)**2 + 2*s2
    
    # derivative of rc w.r.t. qinf
    drc = -3*qinf*rc/8/r**2 - ds/4/r/qinf/s/np.sqrt(1+s2)
    
    return drc    


#  normalized energy of field configurations
#
# Inputs:
#    R      : source size, scalar
#    A      : source strength, numpy array
#  branch   : 'an' negative adiabatic solution q->-1
#             'ap' positive adiabatic solution q->1
#             'h' hopping solution 
#   ifplot  : when True, plot energy as function of A 
#
# Output:
#    E    : normalized energy as function of A for selected branch, numpy array
#           masked in regions where asymptotics is not valid
def EConst(R, A, branch='an', ifplot=False):
    """Normalized total energy of q for constant source."""  
    # check input
    assert(branch in ['an', 'ap', 'h'])
    # linestyle dictionary
    ls = {'an':'--', 'ap':'--', 'h':':'}
    
    # Prefactor 
    E0 = A/2
    # source strength
    S = A/R/np.sqrt(np.pi)
    
    # compute energy
    if branch == 'an': # adiabatic solution q->-1
        E = -E0
    elif branch == 'ap': # adiabatic solution q->1
        # valid when S<1/3/sqrt(3)
        S0=1/3/np.sqrt(3)
        E = np.ma.masked_where(S>S0, E0)
        
    else: # hopping solution
      
        # largeness factor
        M = 1/2/np.sqrt(S)
        # approximate g factor, next to leading order
        g = 1 - 1/M + 3*np.arcsinh(np.sqrt(M))/2/M**2
 
        # energy due to hopping
        Eh = E0 + 2*g/3
        # valid when M>10
        E = np.ma.masked_where(M<2, Eh)
        
    # plot figure
    if ifplot:
        plt.plot(A, E, 'k', label=branch, linestyle=ls[branch])
    
    
