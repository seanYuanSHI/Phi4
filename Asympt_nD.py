# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 21:58:24 2020

@author: water
"""

# This script contains functions related to 2D and 3D asymptotics
#############################################################  
# Content of this script in order of appearence
# For small source R<<1 and near critical strength A~Ac
#    qc_RsAc    : Initial condition qc=q(t=0) for q->+-1 solutions
#    Ac_Rs      : Critical source strength 
#    qin_RsAc   : Solution with asymptotics q->1 in the inner region 
#    qout_RsAc  : Solution with asymptotics q->1 in the outer region 
#    q_RsAc     : Wrapper function returns both inner and outer solutions
#    rc_RsAc    : Light horizon radius when A~Ac
#
# For small source R<<1 and large source A>>Ac
#    qin_RsAl   : Solution in the inner region t<R
#    pin0_RsAl  : Phase portrait p=p(q) in the inner region t<R for qin0
#    qc_RsAl    : Initial condition qc=q(t=0) 
#
# For large source R>>1
#    x1Large    : 1st order solution to the larger root of x*exp(-x^2)=epsilon
#    x1Small    : 1st order solution to the smaller root of x*exp(-x^2)=epsilon
#    Tau0       : 0th order transition time for the hopping solution
#    Tau1       : 1st order transition time for the hopping solution
#    Tau1A      : Wrapper function for Tau1 with A as an array
#    q_RlAl     : Nonperturbative solution beyond 1010/1210 phase boundary
#    Ac_Rl      : Critical source strength for phase boundaries
#    E_RlAs     : Normalized energy of field configurations
#
#############################################################  
#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '../')
#import SE
 
import numpy as np
#import scipy as sp
from scipy import optimize
import scipy.special as ss
import matplotlib.pyplot as plt

import SE
import Asympt_1D

##########################################
# For small source R<<1 and near critical strength A~Ac

# Initial condition qc=q(t=0) for q->+-1 solutions
# Derived by matching inner and outer solutions
# Inputs:
#    R     : source size
#    A     : source strength
#    dim   : dimension of the problem = 2 or 3
#    sign  : sign=1 for q->1 branch
#            sign=-1 for q->-1 branch
#   ifplot : when True, plot qc as function of A 
def qc_RsAc(R, A, dim=2, sign=1, ifplot=False):
    """q0 for R<<1 and near critical strength A~Ac."""
    # check dimension
    assert(dim==2 or dim==3)
    
    # ensure sign is +1 or -1
    if sign>0:
        sign = 1
        color='r'
    else:
        sign=-1
        color='b'
    
    if dim==2:
        qc = sign - (A/2/np.pi)*(np.log(2/R)-np.euler_gamma/2)
    else:
        qc = sign + (A/2/np.pi)*(1/2-1/R/np.sqrt(np.pi))


    # plot figure
    if ifplot:
        plt.plot(A, qc, linestyle='--', color=color, label='qc_RsAc')
        plt.xscale('log')
    
    return qc

# Critical source strength as a function of R and initial condition
# Derived by matching inner and outer solutions
# Inputs:
#    R     : source size
#    q0    : initial condition, assume |q0| is small
#    dim   : dimension of the problem = 2 or 3
#   ifplot : when True, plot Ac as function of R 
def Ac_Rs(R, q0=0, dim=2, ifplot=False):
    """Critical source strength as a function of R<<1 for given q0."""
    assert(dim==2 or dim==3)
    
    if dim==2:
        Ac = 2*np.pi*(1-q0)/(np.log(2/R)-np.euler_gamma/2)    
    else:
        Ac = 2*np.pi*(1-q0)*R/(1/np.sqrt(np.pi)-R/2)

    # plot figure
    if ifplot:
        plt.plot(R, Ac, linestyle='-', color='r', label='Ac_Rs')
        plt.xscale('log')
        plt.yscale('log')
    
    return Ac

# Solution to ODE with asymptotics q->1
# in the inner region where t<<(A/2/pi)^(1/dim)
# Inputs:
#    t     : normalized time tau
#    R     : source size
#    A     : source strength
#    q0    : if None, determined by asymptotic matching
#            otherwise specified by user
#    dim   : dimension of the problem = 2 or 3
#   ifplot : when True, plot q as function of t 
# Outputs:
#    tmax  : maximum t where the asymptotics is valid
#    qin   : asymptotic solution in the inner region
def qin_RsAc(t, R, A, q0=None, dim=2, ifplot=False):
    """Inner solution for R<<1 when t<<(A/2/pi)^(1/dim)."""
    assert(dim==2 or dim==3)
    # maximum t where the asymptotics is valid
    tmax = (A/2/np.pi)**(1/dim)
    # rescaling time
    tR = t/R
    tR2 = tR**2
    
    # initial condition from asymptotic matching
    if q0==None: q0 = qc_RsAc(R, A, dim=dim)       
    # inner solution
    if dim==2:
        qin = q0 + (A/4/np.pi)*(np.euler_gamma+np.log(tR2)+ss.exp1(tR2))
    else:
        qin = q0 + (A/2/R/np.pi**(3/2))*(1-np.sqrt(np.pi)/2/tR*ss.erf(tR))
    
    # plot figure
    if ifplot:
        # plot in region t<tmax
        ym = np.ma.masked_where(t>tmax, qin)        
        plt.plot(t, ym, linestyle='--', color='k', label='qin_RsAc')
    
    return tmax, qin
 
    
# Solution to ODE with asymptotics q->1
# in the outer region where t>>R
# Inputs:
#    t     : normalized time tau
#    R     : source size
#    A     : source strength
#    dim   : dimension of the problem = 2 or 3
#   ifplot : when True, plot q as function of t 
# Outputs:
#    tmin   : minimum t where the asymptotics is valid
#    qout   : asymptotic solution in the outer region
def qout_RsAc(t, R, A, dim=2, qinf=1, ifplot=False):  
    """Outer solution for R<<1 when t>>R."""
    assert(dim==2 or dim==3)
    # minimum t where the asymptotics is valid
    tmin = R
    # outer solution
    if dim==2:
        qout = qinf - (A/2/np.pi)*ss.kn(0,t) 
    else:
        qout = qinf - (A/2/np.pi**2)*ss.spherical_kn(0,t)    
    
    # plot figure
    if ifplot:
        # plot in region t>tmin
        ym = np.ma.masked_where(t<tmin, qout)        
        plt.plot(t, ym, linestyle=':', color='k', label='qout_RsAc', linewidth=3)
    
    return tmin, qout 
    

# Solution to ODE with asymptotics q->1
# Return both inner and outer solutions
# which are masked in regions where they are not valid
# Inputs:
#    t     : normalized time tau
#    R     : source size
#    A     : source strength
#    dim   : dimension of the problem = 2 or 3
#   ifplot : when True, plot q as function of t 
# Outputs:
#    qin    : asymptotic solution in the inner region
#    qout   : asymptotic solution in the outer region
def q_RsAc(t, R, A, dim=2, ifplot=True):
    """Return both inner and outer solutions for R<<1."""
    assert(dim==2 or dim==3)
    # inner solution
    tmax, qin = qin_RsAc(t, R, A, dim=dim)
    # qin is valid when t<tmax
    qin = np.ma.masked_where(t>tmax, qin)    
    
    # outer solution 
    tmin, qout = qout_RsAc(t, R, A, dim=dim)
    # qout is valid when t>tmin
    qout = np.ma.masked_where(t<tmin, qout)
    
    # plot solution
    if ifplot:
        # asymptotic for small t
        plt.plot(t, qin, 'b',linestyle='-', linewidth=3)
        # asymptotic for large t
        plt.plot(t, qout, 'r',linestyle='-', linewidth=1.5)
        
    return qin, qout
 
    
# Light horizon radius determined from the inner solution
# when the source strength is near critical A~Ac
#
# In 2D, denoting x=tc^2/R^2, then tc is determined by root finding from
#    log(x)+exp1(x) = 2(log(2/R)-2pi/A-gamma)
# In 3D, denoting x=tc/R, then tc is determined by root finding from
#    erf(x)/x = R(1+4*pi/A)
#
# Inputs:
#    R     : source size, scalar
#    A     : source strength
#    dim   : dimension of the problem = 2 or 3
#   ifplot : when True, plot rc as function of A 
# Output:
#    rc    : light horizon radius determined from the inner solution
#            rc = nan if solution not found
def rc_RsAc(R, A, dim=2, ifplot=False):
    """ Light horizon radius when R<<1 near critical A~Ac."""
    assert(dim==2 or dim==3)
    # lenth of A
    try:
        nA = len(A)
    except TypeError:
        # A is a scalar
        nA = 1
        # convert A to numpy array
        A = np.array([A,])    
        
    # initialize output
    rc = np.full_like(A, np.nan)  
    
    if dim==2:
        # LHS as objective function f
        f = lambda x : np.log(x) + ss.exp1(x)
        # derivative of f
        #df = lambda x: (1 - np.exp(-x))/x
        # second derivative of f
        #ddf = lambda x: ((1+1/x)*np.exp(-x)-1/x)/x
        
        # RHS constant as function of A 
        rhs = lambda A : 2*(np.log(2/R)-2*np.pi/A-np.euler_gamma)
        # search root between [xmin, xmax]
        xmin = lambda A : rhs(A) + np.euler_gamma              
        xmax = lambda A : np.exp(rhs(A))
        # convert x to rc
        x2r = lambda x : R*np.sqrt(x)
        
    else:
        # LHS objective function
        f = lambda x : ss.erf(x)/x        
        # RHS constant as function of A 
        rhs = lambda A : R*(1+4*np.pi/A)
        # search root between [xmin, xmax]
        xmin = lambda A : np.sqrt(np.log(2/np.sqrt(np.pi)/rhs(A)))          
        xmax = lambda A : 1/rhs(A)
        # convert x to rc
        x2r = lambda x : R*x
        
    
    # find root for each A
    for iA in range(nA):   
        # unpack A value
        A0 = A[iA]
        
        # RHS 
        rhs0 = rhs(A0)
        # initial guess
        xmin0 = xmin(A0)
        # xmax to search for root
        xmax0 = xmax(A0)
  
        # objective function
        fun = lambda x : f(x) - rhs0 
        # find root
        #print(f'fun(min0)={fun(min0)}, fun(xmax)={fun(xmax)}')
        sol = optimize.root_scalar(fun, method='bisect', bracket=[xmin0, xmax0])

        # load root if found correctly
        if sol.converged:
            rc[iA] = x2r(sol.root) 
    
    # plot figure
    if ifplot:
        # plot where rc is valid, nan are automatically masked
        plt.plot(A, rc, linestyle='-', color='r', label='rc_RsAc')
        
        plt.xscale('log')
    
    return rc


##########################################
# For small source R<<1 and large source A>>Ac

# Solution to ODE 
# in the inner region when A>>1/R and t<R
#
# The dominant balance is q^3/2 + S~0
# The function calculate the next order solution
#
# Inputs:
#    t        : normalized time tau
#    R        : source size
#    A        : source strength
#    dim      : dimension of the problem = 2 or 3
#   ifplot    : when True, plot q as function of t 
# Outputs:
#    qin   : asymptotic solution in the inner region
def qin_RsAl(t, R, A, dim=2, ifplot=False):
    """Inner solution for R<<1 and A>>Ac."""
    assert(dim==2 or dim==3)
    
    # approximate nonlinear term as q^3/2
    # rescaling time
    tR2 = 1/3*(t/R)**2
    # common factor xi
    xi = -(2*A)**(1/3)*R**(1-dim/3)/np.pi**(dim/6)*np.exp(-tR2)
    # inner solution
    qin = (xi + 4/9/xi*(2*tR2-dim))/R 
    
    # plot figure
    if ifplot:
        # plot in region t<tmax
        ym = np.ma.masked_where(t>2*R, qin)        
        plt.plot(t, ym, linestyle='-', color='k', label='qin_RsAl')
    
    return qin



# Phase portrait p=p(q) in the inner region t<R
# for the lowest order solution, whose dominant balance 
# is given by q0^3/2 + S~0. When S>0, we have q<0.
#
# Inputs:
#    q        : normalized field value
#    R        : source size
#    A        : source strength
#    dim      : dimension of the problem = 2 or 3
#   ifplot    : when True, plot q as function of t 
# Outputs:
#    p        : p(t)=q'(t), for 0th order qin
def pin0_RsAl (q, R, A, dim=2, ifplot=False):
    """Phase portrait p=p(q) in the inner region t<R<<1."""
    assert(dim==2 or dim==3)
    
    # source strength at origin
    S0 = A/(R*np.sqrt(np.pi))**dim
    s023 = (2*S0)**(1/3)
    
    # xi=t/R as a function of q
    xi = np.sqrt(3*np.log(abs(s023/q)))
    # p=p(q)
    p = -2*q/3/R*xi
    
    # plot figure
    if ifplot:
        plt.plot(q, p, linestyle='-', color='k', label='pin0_RsAl')
    
    return p
    


# Initial condition qc=q(t=0) 
# Derived from inner solution
# Inputs:
#    R     : source size
#    A     : source strength
#    dim   : dimension of the problem = 2 or 3
#   ifplot : when True, plot qc as function of A 
def qc_RsAl(R, A, dim=2, ifplot=False):
    """Initial condition qc=q(t=0) from inner solution for R<<1 and A>>Ac."""
    assert(dim==2 or dim==3)
    qc = qin_RsAl(0, R, A, dim=dim)

    # plot figure
    if ifplot:
        plt.plot(A, qc, linestyle='--', color='m', label='qc_RsAl')
        plt.xscale('log')
    
    return qc


##########################################
# For large source R>>1 

# first order solution to the larger root of
# x*exp(-x^2)=epsilon<<1
# Input:
#   epsilon : small scalar parameter > 0
#             when epsilon < 1/sqrt(2*e), solution exist
#             otherwise, take x = 1/sqrt(2)
# output:
#    x1     : first order asymptotic solution of larger root
def x1Large(epsilon):
    """1st order solution to the larger root of x*exp(-x^2)=epsilon<<1."""
    # ensure epsilon is positive
    epsilon = abs(epsilon)
    # critical value
    ec = 1/np.sqrt(2*(np.exp(1)))
    
    if epsilon > ec:
        x1 = 1/np.sqrt(2)
    else:
        # lowest order initial guess
        x0 = np.sqrt(np.log(1/epsilon))
        # first order initial guess
        x02=1/2/x0**2
        x1 = (x0 - x02)/(1 - x02)
        
    return x1        

# first order solution to the smaller root of
# x*exp(-x^2)=epsilon<<1
# Input:
#   epsilon : small scalar parameter > 0
#             when epsilon < 1/sqrt(2*e), solution exist
#             otherwise, take x = 1/sqrt(2)
# output:
#    x1     : first order asymptotic solution of smaller root
def x1Small(epsilon):
    """1st order solution to the smaller root of x*exp(-x^2)=epsilon<<1."""
    # ensure epsilon is positive
    epsilon = abs(epsilon)
    # critical value
    ec = 1/np.sqrt(2*(np.exp(1)))
    
    if epsilon > ec:
        x1 = 1/np.sqrt(2)
    else:
        # first order initial guess
        e2 = epsilon**2
        x1 = epsilon*(np.exp(e2)-2*e2)/(1-2*e2)
        
    return x1        
        

# Hopping solution transition from smallest fixed point
# to largest fixed point. Compute lowest-order transition time tau0
# by root finding:
#    lambda = (sqrt(1+xi^2)-1)/xi = sqrt(2*(3*qinf^2-1))/2 = 2r
# where xi = 2*tau/(D-1) and qinf is the largest fixed point
#    qinf*(qinf^2-1)/2 = S(tau)
# where S is the Gaussian source term.
# There may be two solutions to the above equation.
#
# Input:
#    R     : source size, scalar
#    A     : source strength, scalar
#    dim   : dimension of the problem, dim=2 or 3
#    nt    : nt=0, the smaller root, which may not exist
#            nt=1, the larger root, which always exist
# Output:
#    tau   : the selected root
# 
# If the root is found, tau0 is the root
# If the root is not found, tau0 is nan
# smaller root does not exist in 0110 phase
def Tau0(R, A, dim=2, nt=1):
    """0th order transition time for hopping solution."""
    assert(dim>=2)
    # dim-1
    d = dim-1

    # R^2
    R2 = R**2
    # source at t=0 
    S0 = A/(R*np.sqrt(np.pi))**dim    
    #print(f'R={R}, A={A}, S0={S0}')
    
    # smaller root does not exist in 0110 phase
    if R>=10 and S0>1/3/np.sqrt(3) and nt==0:
        tau = np.nan
    else:    
        # define LHS as lambda function
        LHS = lambda t : (np.sqrt(1+(2*t/d)**2)-1)*d/2/t
        # t derivative of LHS
        dLHS = lambda t : -LHS(t)/t + 2/d/np.sqrt(1+(2*t/d)**2)
        
        # source function
        source = lambda t : S0*np.exp(-t**2/R2)
        # qinf as function of t
        qinf = lambda t : SE.Fixed(source(t), n=2) # largest root
        # define RHS as lambda function
        RHS = lambda t : 2*Asympt_1D.t2tau(qinf(t))
        # t derivative og RHS
        dRHS = lambda t : 3*t*qinf(t)*source(t)/8/R2/Asympt_1D.t2tau(qinf(t))**3
        
        # objective function
        f = lambda t : LHS(t) - RHS(t)
        # derivative of objective function for Newton's method
        df = lambda t : dLHS(t) - dRHS(t)
        
        # initial guess of the roots 
        # by solving the asymptotic equation x*exp(-x^2)=epsilon
        # where x=tau/R, and epsilon=(D-1)/(3*R*S0)
        epsilon = d/3/R/S0
        if nt==0:
            # initial guess for smaller root
            t0 = R*x1Small(epsilon)
        else:
            # initial guess for larger root
            t0 = R*x1Large(epsilon)
        
        # root finding, smaller root
        sol = optimize.root_scalar(f, method='newton',fprime=df, x0=t0)    
        # return root if converged
        if sol.converged: tau = sol.root
        else: tau = np.nan
        
    return tau


# Better estimation of transition time use better source strength estimates
# For example, the 1st order scheme estimate the source at light horizon
# and find tau by root finding:
#     rcD(qinf) = tau(qinf) + rc1(qinf)
# where rcD is the light gorizon radius in D-dimensional case
# and rc1 is the light horizon radius in 1D case with a constant source.
#
# More generally, the source strength can be estimated at intermediate time
#    tm = eta*rcD + (1-eta)*tau
#       = tau + eta*rc1
# where 0 < eta < 1 is a weighting parameter. 
#
# The root qinf is converted to tau using:
#    lambda = (sqrt(1+xi^2)-1)/xi = sqrt(2*(3*qinf^2-1))/2 = 2r
# where xi = 2*tau/(D-1) for D-dimensional case
#
# Input:
#    R     : source size, scalar
#    A     : source strength, scalar
#    eta   : weighting parameter in (0,1) for source strength estimation
#            eta = 0 is the 0-th order scheme
#            eta = 1 is the 1-st order scheme
#    dim   : dimension of the problem, dim=2 or 3
#    nt    : nt=0, the smaller root, which may not exist
#            nt=1, the larger root, which always exist
# Output:
#    tau   : the larger root of transition time
#           If the root is found, tau1 is the root
#           If the root is not found, tau1 is nan
#    rcD  : the corresponding light horizon radius
#    qinf : the corresponding qinf
#
# The outputs are np.nan if root not found
def Tau1(R, A, eta=0.65, dim=2, nt=1):
    """1st order transition time for hopping solution."""
    # check eta
    assert(eta>=0 and eta<=1)
    # check dimension
    assert(dim>=2)
    # dim-1
    d = dim -1
    
    # 0-th order solution provides initial guess
    t0 = Tau0(R, A, dim=dim, nt=nt)
    
    # defaul outputs
    tau, rcD, qinf = np.nan, np.nan, np.nan
    if not np.isnan(t0):
        # if t0 is nan, then outouts are nan
        # otherswise, proceed to find next order approximation    
        # source at t=0 
        S0 = A/(R*np.sqrt(np.pi))**dim
    
        # convert t0 to qinf
        xi = 2*t0/d
        L = (np.sqrt(1+xi**2)-1)/xi
        qinf0 = np.sqrt((1+2*L**2)/3)
        
        # tau as a function of qinf
        TAU = lambda q : d*np.sqrt(2*(3*q**2-1))/3/(1-q**2)
        # derivative of tau w.r.t qinf
        dTAU = lambda q : d*2*q*(3*q**2+1)/3/np.sqrt(2*(3*q**2-1))/(1-q**2)**2
        
        # tm as a function of qinf, x=t/R
        XM = lambda q : np.sqrt(np.log(2*S0/q/(1-q**2)))
        TM = lambda q : R*XM(q)
        # derivative of rca w.r.t qinf
        dTM = lambda q : R*(3*q**2-1)/q/(1-q**2)/2/XM(q)
        
        # rc1 as a function of qinf
        RC1 = lambda q : Asympt_1D.rcConst(q)
        # derivative of rc1 w.r.t qinf
        dRC1 = lambda q : Asympt_1D.drcConst(q)
        
        # objective function
        f = lambda q : TM(q) - TAU(q) - eta*RC1(q)
        # derivative of objective function for Newton's method
        df = lambda q : dTM(q) - dTAU(q) - eta*dRC1(q)
        
        # root finding for qinf
        sol = optimize.root_scalar(f, method='newton',fprime=df, x0=qinf0)
    
        # return root if converged
        if sol.converged:
            # numerical root
            qinf = sol.root
            # convert qinf to tau
            tau = TAU(qinf)
            # conpute light horizon radius
            rcD = TAU(qinf) + RC1(qinf)
 
        
    return tau, rcD, qinf
            
# wrapper function for Tau1 with A as an array
def Tau1A(R, A, eta=0.65, dim=2, nt=1, ifplot=False):
    """Wrapper function for Tau1 with A as an array."""
    # ensure A is iterable
    try: NA = len(A)
    except TypeError: # A is a scalar
        A = [A]
        NA = 1
    # ensure A is a numpy array
    A = np.array(A)    
    # initialize output
    tau = np.zeros_like(A)
    rcD = np.zeros_like(A)
    qinf = np.zeros_like(A)
    
    # load values
    for iA in range(NA):
        tau[iA], rcD[iA], qinf[iA] = Tau1(R, A[iA], eta=eta, dim=dim, nt=nt)
        
    # plot figure for rc
    if ifplot:
        plt.plot(A, rcD, linestyle='--', color='k')   
        
    return tau, rcD, qinf
        

# Asymptotic nonperturbative solution for large R and 
# large A>pi**R*sqrt(2*e)/3, namely, beyond the 1010/1210 phase boundary
# The asymptotic solution is patched by four sections:
# adiabatic->transition->nonlinear->adiabatic
#
# Inputs:
#    t     : normalized time 
#    R     : source size
#    A     : source strength
#    eta   : weighting parameter in (0,1) for source strength estimation
#            eta = 0 is the 0-th order scheme
#            eta = 1 is the 1-st order scheme
#    dim   : dimension of the problem, dim=2 or 3
#   ifplot : when True, plot q as function of t 
#    nt    : nt=0, the smaller root, which may not exist
#            nt=1, the larger root, which always exist
# Outputs:
#    qs    : adiabatic solution near the smallest fixed point
#    qt    : transition solution connecting qs and qn
#    qn    : nonlinear transition from qs to ql, 1D like
#    ql    : adiabatic solution near the largest fixed point
# Outputs are masked in regions where the approximations are not applicable
def q_RlAl(t, R, A, eta=0.65, dim=2, ifplot=True, nt=1):
    """Asymptotic solution for R>>1 and A>pi**R*sqrt(2*e)/3."""
    # check eta
    assert(eta>=0 and eta<=1)    
    # transition time and light horizon radius
    tau, rc, qinf = Tau1(R, A, eta=eta, dim=dim, nt=nt) # nt=1 larger root
    # if tau is nan, the nonlinear solution does not exist
    assert(not np.isnan(tau))
    #print(tau, rc, qinf)

    # source function from ODE
    ode = SE.ODE(R, A, dim)
    S = ode.source(t)
    
    #############################
    # adiabatic solutions near smallest fixed point
    qs = SE.Fixed(S, n=0)    
    # qs is valid when t<tau
    qs = np.ma.masked_where(t>tau, qs)
    
    #############################
    # source at transition time
    Stau = ode.source(tau)
    # source derivative at transition time
    dStau = -2*tau*Stau/R**2
    # transition solution connecting qs
    qt = -1 - Stau - (t-tau + (dim-1)/tau)*dStau
    
    # compupte auxsilliary factors
    q0 = Asympt_1D.Q0Const(qinf)
    p0 = Asympt_1D.Phi0(qinf)
    r = Asympt_1D.t2tau(qinf)
    # transition solution connecting exponential
    qt += np.exp(2*r*(t-tau))*(qinf-(1+2/p0)*q0)/2/p0
    
    # qt is valid when tau-dt < t < tau+dt/2, take dt=2/r
    dt = 2/r
    qt = np.ma.masked_where((t-tau-dt/2)*(t-tau+dt)>0, qt)
    
    #############################
    # source estimate for the nonlinear solution
    tm = eta*rc + (1-eta)*tau
    Sm = ode.source(tm)
    # nonlinear solution is 1D like
    qn = Asympt_1D.qConst(t-tau, Sm)
    # time after tau when qn approaches ql
    tl = Asympt_1D.tlConst(Sm)
    
    # qn is valid when tau < t < tau + 5*tl
    qn = np.ma.masked_where((t-tau)*(t-tau-5*tl)>0, qn)
    
    #############################
    # adiabatic solutions near largest fixed point
    ql = SE.Fixed(S, n=2)    
    # ql is valid when t>tau+tl
    ql = np.ma.masked_where(t<tau+tl, ql)   
    
    
    # plot solutions in appropriate regions
    if ifplot:
        plt.plot(t, qs, linestyle='-', color='cyan')  
        plt.plot(t, qt, linestyle='-', color='g')  
        plt.plot(t, qn, linestyle='-', color='y')  
        plt.plot(t, ql, linestyle='-', color='m')  
        
    return qs, qt, qn, ql  


# Critical source strength as a function of R >>1
# for 1010->1210 and 2110->0110 phase boundaries
#
# Inputs:
#    R      : source size
#    dim    : dimension of the problem = 2 or 3
#    phase  : '1010' for 1010->1210 
#             '2110' for 2110->0110
#   ifplot  : when True, plot Ac as function of R 
def Ac_Rl(R, dim=2, phase='1010', ifplot=False):
    """Critical source strength as a function of R >>1."""
    assert(dim==2 or dim==3)
    
    if phase == '1010': #1010->1210  
        d = dim-1
        Ac = d/3*np.sqrt(2*np.exp(1)*np.pi**dim)*R**d
    else: #2110->0110
        S0 = 1/3/np.sqrt(3)
        Ac = S0*(R*np.sqrt(np.pi))**dim
        
    # plot figure
    if ifplot:
        plt.plot(R, Ac, linestyle='-', color='r', label='Ac_Rl')
        plt.xscale('log')
        plt.yscale('log')
    
    return Ac


# Approximate normalized energy of field configurations when R>>1 and S<<1
#
# Inputs:
#    R      : source size, scalar
#    A      : source strength, numpy array
#   dim     : dimension of the problem = 2 or 3
#  branch   : 'an' negative adiabatic solution q->-1
#             'ap' positive adiabatic solution q->1
#             'hs' hopping solution with smaller transition time
#             'hl' hopping solution with larger transition time
#    eta    : weighting parameter in (0,1) for source strength estimation
#             eta = 0 is the 0-th order scheme
#             eta = 1 is the 1-st order scheme
#   ifplot  : when True, plot energy as function of A 
# Output:
#    E    : normalized energy as function of A for selected branch, numpy array
def E_RlAs(R, A, branch='an', dim=2, eta=0.65, ifplot=False):
    """Approximate energy of field configurations when R>>1."""
    # check input
    assert(branch in ['an', 'ap', 'hs', 'hl'])
    # index dictionary for hopping solution
    dh = {'hs':0, 'hl':1}
    # linestyle dictionary
    ls = {'an':'--', 'ap':'--', 'hs':':', 'hl':'-.'}
    
    # prefactor
    d2 = dim/2
    E0 = A*ss.gamma(d2)/2/np.pi**(d2)    
    
    if branch == 'an': # adiabatic solution q->-1
        E = -E0
    elif branch == 'ap': # adiabatic solution q->1
        E = E0
    else: # hopping solution
        # index of transition time
        nt = dh[branch]
        # light horizon radius
        _, rcD, qinf = Tau1A(R, A, eta=eta, dim=dim, nt=nt)
        ## 1D light horizon radius
        #rc1 = Asympt_1D.rcConst(qinf)
        ## radius where source strength is estimated
        #rm = rcD + (eta-1)*rc1        
        # normalize by R
        rcDR2 = (rcD/R)**2
        #rm2 = (rm/R)**2
        
        # explicitly source dependent contribution
        frac = 2*ss.gammaincc(d2, rcDR2) - 1
        #frac = 2*ss.gammaincc(d2, rm2) - 1
        
        # 1D contribution
        # source at transition time, numpy array
        S = -qinf*(qinf**2-1)/2 # expected to be >0
        # largeness factor
        M = 1/2/np.sqrt(S)
        # approximate g factor, next to leading order
        g = 1 - 1/M + 3*np.arcsinh(np.sqrt(M))/2/M**2
        # 1D contribution
        E1 = 2*g*rcD**(dim-1)/3
        #E1 = 2*g*rm**(dim-1)/3
        
        # total energy
        E = E1 + E0*frac
        
    # plot figure
    if ifplot:
        plt.plot(A, E, 'k', label=branch, linestyle=ls[branch])  
        plt.xscale('log')
        plt.xlabel(r'$A$')
        plt.ylabel('Energy')
        plt.title(f'dim={dim}, R={R}, eta={eta}')
        
    return E