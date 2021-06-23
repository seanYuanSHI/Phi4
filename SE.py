# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 13:52:49 2020

@author: shi9@llnl.gov
"""

#############################################################    
# This script contains functions for solving the static-field equation (SE)
#     d^2q/dt^2 + (D-1)/t*dq/dt = 1/2*q*(q^2-1) + S0*exp(-t^2/R^2)
# where A and R are dimensionless constants, and S0 = A/R^D*\pi^{D/2}
# 
# The above equation is obtained from the spherically symetric PDE
#    \nabla^2 \phi = \partial U/\partial\phi + S(x),
# with change of variables t=m*r, q=\phi/v. The source term is  n-D Gaussian
#     S = \alpha/(a^n\pi^{n/2}) exp{-r^2/a^2},
# and the self potential
#     U = \lambda/4! (\phi^2-v^2)^2,
# where v is the Higgs VEV.
#
# The dimensionless constants are defined by
#    A = \alpha*\sqrt{\lambda/3}m^{n-3}
#    R = m*a
# Recall the Higgs VEV is related to the mass of the scalar field by
#    m = \sqrt{\lambda/3} v
#
# Without the source term y and when t->\infty
#    q''=1/2*q(q^2-1) 
# has a phase-space island in the region -1/2<p<1/2 and -1<q<-1. 
# where p = q' is the conjugate momentum.
# Now with a source term, the island is shifted
#
# To improve the memory efficiency, the program solves a rescaled problem
# such that variables are closer to O(1) numbers: 
#    q = f + a*w,   t = u*z
# where f is near fixed point, a>0 is a small and u>0
# are rescaling constants. The equation for w(z) is
#    d^2w/dz^2 + (D-1)/z*dw/dz = (u^2/2)*(c1*w+c2*w^2+c3*w^3) 
#                              + (u^2/a)(c0+S0*exp(-u^2*z^2/R^2))
#
# The ODE is a stiff problem near the separatrix and island boundaries.
# Use Decimal package to improve precision and reduce truncation error
# when solving the ODE. The input initial conditions also need to be
# high precision, but the outputs need not be. Decimal variables are 
# named with a trailing "D".
#
#############################################################  
# Content of this script in order of appearence
#
# Fixed : fixed point of the ODE
#
# class ODE : a class contain the ODE information
#     member functions: 
#        __init__  : load parameters and compute fixed points
#        __repr__  : print contents of the object
#         wD2q     : convert wD to q
#         qD2wD    : convert qD to wD
#         t2zD     : convert t to zD
#         source   : compute the source term at given z
#        sourceD   : compute the source term at given zD, decimal 
#          rhsD    : evaluate the right hand side of the ODE, decimal
#          NI      : return number of islands to expect
#
# class Initial: a class contains initial conditions of the ODE
#    member functions: 
#        __init__  : load parameters 
#        __repr__  : print contents of the object
#
# class Solution: a class contains the solution of the phi4 problem
#    member functions: 
#        __init__  : load parameters 
#        __repr__  : print contents of the object
#          Next    : compute solution at the next step
#          Solve   : solve the problem specified by initial conditions
#         TestCvg  : test onvergence by scan nearby dt at 1/4, 1/2, 1, 2, 4
#      terminatMsg : report termination flag of SolveUntil
#       SolveUntil : solve from t=0 until termination conditions
#          Terms   : compute and plot each term in the ODE
#
#############################################################  

import Auxiliary as AX

#import sys
import numpy as np
import matplotlib.pyplot as plt
import decimal
from decimal import Decimal, getcontext
#getcontext().prec = 28 # default is 28

#################################################################   
# Fix point as a function of source strength
# The fixed point f satisfies
#    f(f^2-1)/2 + S0 = 0
# Inputs:
#    S0     : constant source strength, 
#    n      : select the n-th root to output
#             When n=None, output all roots
# Output:
#    fp     : three roots of the cubic equation
#             fp is ordered by real part of the roots in ascending order
#             fp[n] is numpy array of length len(S0)
def Fixed(S0, n=None): 
    # check input n
    assert(n in [None, 0, 1, 2])
    
    if hasattr(S0, "__len__"):
        # S0 is an array
        nS = len(S0)
        
        # initialie output as real array
        # discard imaginary roots
        fp = np.zeros((3, nS))
    
        # compute roots
        for i in range(nS):
            # coefficients of the cubic equation
            coefs = [0.5, 0, -0.5, S0[i]]
            # find root of cubic equation
            r = np.roots(coefs)
            # sort root
            r = np.sort(r)
            
            # discard imaginary root
            for j in range(3):
                if abs(r[j].imag)>1e-9:
                    r[j] = np.nan
                    
            # load array
            fp[:, i] = r
            
    else:
        # S0 is a scalar        
        # coefficients of the cubic equation
        coefs = [0.5, 0, -0.5, S0]
        #print(S0)
        # find root of cubic equation
        r = np.roots(coefs)
        # sort in ascending order, keep imaginary roots
        fp = np.sort(r)

    # return value
    if n==None: # return all roots
        return fp 
    else: # return selected root
        return fp[n]  
    
 

#################################################################
# Class contain the ODE info
#   R        : radius normalized by m
#   A        : amplitude of source term
#   dim      : dimension of the problem, 1, 2, or 3
#   f, a, u  : rescaling constants
#              q = f + a*w,   t = u*z
#
class ODE:
    """A class contains the ODE information."""
    
    # minimum allowed value for rescaling constants a and u
    aumin = 1e-15    
    
    # initialize object, by loading parameters
    def __init__(self, R, A, dim, f=0, a=1, u=1):
        # check dimension
        assert(dim==1 or dim==2 or dim==3)
        self.dim = dim
        
        # parameters for ODE
        self.A = A
        #assert(A!=0) # ensure nonzero
        self.R = abs(R) # ensure positive
        #decimal version
        AD = Decimal(A)
        RD = Decimal(R)
        
        # rescaling constants
        assert(a>self.aumin) # a should be positive and not too small
        assert(u>self.aumin) # u should be positive and not too small
        aD = Decimal(a)
        uD = Decimal(u)
        fD = Decimal(f)
        self.aD = aD
        self.uD = uD
        self.fD = fD  
        self.u = u
        
        # coefficients for rescaled nonlinear terms
        f2D = fD**2
        self.c0D = fD*(f2D-1)/2
        self.c1D = 3*f2D-1
        self.c2D = 3*fD*aD
        self.c3D = aD**2
        
        u2D = uD**2
        self.u2aD = u2D/aD
        self.u22D = u2D/2       
        
        # derived parameters
        self.uiR2 = (u/R)**2
        S0 = A/(R*np.sqrt(np.pi))**dim
        self.S0 = S0
        # decimal version
        self.uiR2D = (uD/RD)**2
        self.S0D = AD/(RD*Decimal(np.pi).sqrt())**dim
        
        # fixed points are solutions of ode.rhs(t=0,q) == 0
        fp = Fixed(S0) # all roots
        # minimum root separation on real axis
        fpr = fp.real
        df = min([abs(fpr[1]-fpr[0]), abs(fpr[2]-fpr[1]), abs(fpr[0]-fpr[2])])        
        # load variables
        self.fpoints = fp    
        self.df = df
        
        # threshold is given by the max 10 and abs of the fixed points at zero time
        threshold = 2*max(np.insert(abs(fp),0,10))    
        self.threshold = threshold
        # convert to thresholds for w
        self.upper = (threshold-f)/a
        self.lower = (-threshold-f)/a
        
        # separatrix asymptotics determined by sign of A
        fs = -6*np.sign(A) # multiply by 6 to mark as separatrix
        self.sflag = int(fs) # convert to integer
        # default separatrix marker
        if fs>0: self.smarker = '+'
        elif fs<0: self.smarker = 'o'
        else: self.smarker = ''
        
           
    # print parameters
    def __repr__(self):
        return 'A='+str(self.A)+', R='+str(self.R)+', dim='+str(self.dim)
    
    def wD2q(self, wD):
        """Convert wD to q."""        
        return float(self.fD + self.aD*wD)
    
    def qD2wD(self, qD):
        """Convert qD to wD."""      
        return (qD - self.fD)/self.aD
    
    def t2zD(self, t):
        """convert t to zD."""
        return Decimal(t)/self.uD

        
    ########################################    
    # compute source term
    # rescaling  by f, a, u
    # q = f + a*w, t = u*z
    def source(self, z):
        """Compute the source term at given z."""        
        # source term
        S = (self.S0)*np.exp(-self.uiR2*z**2)        
        return S
    
    # compute source term, decimal version
    def sourceD(self, zD):
        """Compute the source term at given z, decimal input/output."""        
        # source term
        SD = (self.S0D)*(-self.uiR2D*zD**2).exp()        
        return SD
    
    ########################################    
    # compute right-hand-side, decimal version
    def rhsD(self, zD, wD):
        """Evaluate the RHS at given z and w, decimal inputs/output."""
        
        # w-dependent terms, derivative of dU/dphi
        try:
            dUD = self.u22D*(self.c1D*wD + self.c2D*wD**2 + self.c3D*wD**3)
        except decimal.Overflow:
            dUD = Decimal('NaN')
            
        # source term
        SD = self.u2aD*(self.c0D + self.sourceD(zD))        
        # RHS = dU + S
        RD = dUD + SD
        
        return RD
    
    
    ########################################    
    # Expected number of phase space island not to exceed NI
    # Use a posteriori knowledge to set NI, or manually set
    # the values excessive large to ensure capturing all possible islands.
    #
    # This information is intrinsic to this ODE problem 
    # Used by QP and RA but not in this module
    #
    # Input:
    #    NIm  : manually set NI if NI is a number
    #           if NIm=None, use a posteriori knowledge
    def NI(self, NIm=None):
        """Number of islands expected in the qp phase space."""
        if NIm!=None: 
            # manually set large enough to capture all possible islands
            NI = int(abs(NIm)) # ensure nonnegative integer    
            
        else:  # set NI using a posteriori knowledge    
            # default
            NI = 1 
            # unpack
            dim, R, A, S0 = self.dim, self.R, self.A, self.S0
            # increase/decrease NI if extra/fewer island
            ifextra, iffewer = False, False 
            
            if R>=10: # large R limit     
                # upper phase boundary for large R is S0=S0u
                S0u = 1/3/np.sqrt(3)                
                if dim==1:
                    iffewer = S0>2*S0u                    
                else: # higher dimension      
                    # two islands exist in 1210 and 2110 phase
                    # 1010<->1210 phase boundary for large R is S0=S0l
                    S0l = (dim-1)*np.sqrt(2*np.exp(1))/3/R                    
                    # determine if this (R,A) is within boundaries
                    ifextra = S0>S0l/2 and S0<2*S0u        
                    #AX.printf(f'S0l={S0l}, S0={S0}, S0u={S0u}')
            
            elif R>1 and R<10: # intermediate R            
                if dim==1: # island vanish for large A
                    iffewer = A>0.3*(R+3)                   
                else:        
                    # two islands exist for A~1
                    pid2 = np.pi**(dim/2)
                    # lower bound R=1, S0>1/3/sqrt(3)
                    Al = pid2/3/np.sqrt(3)
                    # upper bound R=10, S0<1
                    Au = pid2*10**dim
                    # determine if this (R,A) is within boundaries
                    ifextra = A>Al and A<Au
            else: # small R
                # in higher D, always 1 island
                # in 1D, possible to have only separatrix
                iffewer = (dim==1 and A>1.2)
            
            # adjusti number of islands
            # 2 islands at most
            if ifextra: NI = 2
            # 0 island at least
            if iffewer: NI = 0
                
        return NI        



#################################################################
# Class contain initial conditions and related
class Initial:
    """A class contains initial conditions of the ODE.
    
    Parameters for initial conditions:
        User specified:
            Nt      : total number of time steps to solve, 
                      solution is an array of length Nt+1
            dt      : time step, dt>0 solve forward in time
                                 dt<0 solve backward in time               
            q0D     : initial point at t=t0, decimal
            p0D     : initial derivative at t=t0, decimal
            t0      : initial time, default t0 = 0
        Derived:        
            n0      : n0 = int(t0/dt), integer
            tf      : final time is tf = t0 + dt*Nt
            dt2     : dt^2
            q1      : first step at t = t0 + dt, using linear approximation
            ind     : instruction for how to insert next solution
                      dt>0, append, ind = 1
                      dt<0, prepend, ind = 0
    """
    
    # defaul parameters if unspecified by user
    Nt = 1024
    dt = 0.1
    q0D = Decimal(0)
    p0D = Decimal(0)   
    t0 = 0
    
    # initialize the object by loading parameters
    def __init__(self, **kwargs):
        # build input dictionary
        conds = {key: value for key, value in kwargs.items()}
        #AX.printf(conds)
        
        # load optional parameters
        try: self.Nt = int(abs(conds['Nt'])) # ensure positive integer
        except KeyError: pass
        
        try: self.dt = conds['dt'] # ensure positive
        except KeyError: pass
    
        try: self.q0D = Decimal(conds['q0D']) # ensure decimal
        except KeyError: pass
        #else: AX.printf(f'q0 read from input is {self.q0D}')
        
        try: self.p0D = Decimal(conds['p0D']) # ensure decimal
        except KeyError: pass
        #else: AX.printf(f'p0 read from input is {self.p0D}')
    
        try: self.t0 = conds['t0']
        except KeyError: pass
    
        # compute derived parameters
        self.n0 = int(self.t0/self.dt)
        # shifted t0
        self.t0 = self.n0*self.dt
        # final time
        self.tf = self.t0 + self.dt*self.Nt
        
        # instruction for how to insert next solution
        if self.dt>0: # append solution
            self.ind = 1
        else: # prepend solution
            self.ind = 0              
    
    # print contents of the instance
    def __repr__(self):
        AX.printf(f'Nt={self.Nt}, dt={self.dt}, tf={self.tf}')
        AX.printf(f'q0={self.q0D}, p0={self.p0D}')
 
        return '' # need to return str
        
  
#################################################################
# Class contain solution of the phi4 problem
class Solution:
    """A class contains the solution of the phi4 problem."""
    
    # plot parameters
    fontsize=12
    
    # initialize object
    def __init__(self, ode, cds, Ntmax=2**14, epsilon=1e-2):
        # convert q->w, t->z
        cds.z0D = ode.t2zD(cds.t0)
        cds.dzD = ode.t2zD(cds.dt)
        cds.dz2D = cds.dzD**2 
        
        cds.w0D = ode.qD2wD(cds.q0D)
        cds.dw0D = ode.uD/ode.aD*cds.p0D
        cds.w1D = cds.w0D + cds.dw0D*cds.dzD
 
        # load parameters
        self.ode = ode
        self.cds = cds
        
        # default parameters
        self.Ntmax = Ntmax
        self.epsilon = epsilon
        
    # print contents of the instance
    def __repr__(self): 
        ode = self.ode
        AX.printf(20*'#'+'\nSolve ODE for w(z)')
        AX.printf(f'Rescaling q={float(ode.fD)}+{float(ode.aD)}*w, t={ode.u}*z')
        AX.printf(f'Decimal precision is {getcontext().prec}')
        AX.printf(ode)        
        # print initial conditions
        AX.printf(20*'#'+'\nWith initial conditions:')
        AX.printf(self.cds) 
        return '' # need to return str
    
    ########################################    
    # Compute the next step of the ODE using central difference
    # Decimal inputs/output
    #
    # Inputs: 
    #    n1   : current time step, integer
    #    w0D   : solution at previsou time step, decimal
    #    w1D   : solution at current time step, decimal
    # Output: 
    #    w2D   : solution at the nest step, decimal
    def NextD(self, n1, w0D, w1D):
        """Advance one step of the ODE with central difference."""        
        # unpack 
        ode = self.ode
        cds = self.cds
        
        # dimension of the problem
        dim = ode.dim
        
        # unload initial time index
        n0 = cds.n0  
        # current time
        z1D = cds.z0D + cds.dzD*n1   
        
        # current rhs
        SD = ode.rhsD(z1D, w1D)  
        # multiply time step
        Sdz2D = SD*cds.dz2D
    
        # advance one step
        # central difference of Laplacian as two operators
        nd = 2*(n0+n1) # interger denominator
        c = 0 # default value for 1D
        if dim!=1 and nd!=0:
            c = (dim-1)/Decimal(nd)
            
        # coefficients of derivatives, either interger or decimal
        cr = 1 + c
        cl = 1 - c
               
        # next step
        w2D = (2*w1D - cl*w0D + Sdz2D)/cr  
        
        return w2D

    ########################################    
    # Solve the ODE using central difference for Nt time steps
    # Solve the equation forward (backward) in time if dt>0 (dt<0)  
    # The solution series is of length Nt+1, and is always ordered forward in time
    #
    # Input (optional):
    #    ifplot = True: plot figure
    # Output: 
    #    t    : time basis, numpy arrays of length Nt + 1, numpy array
    #    q    : solution, numpy arrays of length Nt + 1, numpy array 
    def Solve(self, ifplot=False, color='b', linestyle='-', linewidth=3):
        """Solve the ODE with central difference."""
        # unpack 
        ode = self.ode
        cds = self.cds
        
        # unload initial conditions
        w0D = cds.w0D
        w1D = cds.w1D     
        t0 = cds.t0
        
        # unload resolution
        Nt = cds.Nt
        dt = cds.dt   
        # current time
        t1 = t0 + dt     
    
        # instruction for how to insert next solution
        ind = cds.ind
        
        # initialize arrays
        time = [t0]
        phiv = [ode.wD2q(w0D)]
        time.insert(ind, t1)
        phiv.insert(ind, ode.wD2q(w1D))
    
        # load next steps, starting from n=1
        for n in range(1, Nt):
            # next step
            w2D = self.NextD(n, w0D, w1D)
            
            # updates
            w0D, w1D = w1D, w2D # multi assignments        
            # next time
            t1 += dt      
            
            # abort when devide by zero
            if w1D.is_nan(): break
            
            # record values
            time.insert(ind*(n+1), t1) 
            phiv.insert(ind*(n+1), ode.wD2q(w1D))
            
        # convert w(z) to q(t)
        phiv = np.array(phiv)
        # convert z to t
        time = np.array(time)
            
        # plot figure
        if ifplot:           
            #plt.figure()
            fs = self.fontsize
            plt.plot(time,phiv,color=color, linestyle=linestyle, linewidth=linewidth, \
                label="q0={0:.2f}".format(cds.q0D)+", p0={0:.2f}".format(cds.p0D))
                
            #plt.legend(loc='best', fontsize=fs)
            plt.xlabel('t=m*x', fontsize=fs)
            plt.ylabel('q=phi/v', fontsize=fs)
            plt.axhline(y=0, color='grey')
            plt.axhline(y=1, color='grey', linestyle='--') 
            plt.axhline(y=-1, color='grey', linestyle='--') 
            plt.axvline(x=0, color='grey')
            plt.title(f'dim={ode.dim}, R={ode.R}, A={ode.A}', fontsize=fs)
            #plt.xlim((0,tf))
            plt.ylim((-2, 2))    
            #plt.show()            
     
        return time, phiv

    ########################################    
    # test convergence of the solution by solving again with dt/2 and 2*dt
    # for fixed tf. The number of time step is change accordingly to 2*Nt and Nt/2
    def TestCvg(self, ifnorm=False, ifplot=True):
        """Test convergence of solutions."""
        # unpack 
        ode = self.ode
        cds = self.cds
        
        # copy new initial condition dictionary 
        od = dict(vars(cds)) # copy dictionary of member variables in cds
        # Nt for the baseline case
        Nt = od['Nt']
        # dt for the baseline case
        dt = od['dt']
     
        # linestyle
        styles = [':','--','-','--',':']
        # line thickness
        widths = [4,3,2,3,4]
        # labels
        labels = ['2*better','better','this','worse','2*worse']
        # colors 
        colors = ['g','c','b','r','m']
        # initialize figure
        plt.figure()
        
        # record arrays if computing norm
        if ifnorm: record = []
        # solve ODE
        for i in range(5):
            # improvement factor
            imp = 2**(2-i)
            # update Nt
            od['Nt'] = int(Nt*imp)
            # update dt
            od['dt'] = dt/imp
            # initialize new initial condition with improved parameters
            icds = Initial(**od) # kwargs
            #AX.printf(icds)
            #AX.printf(18*'#')
            
            # initiate a new instance of improved solution
            isolution = Solution(ode, icds)            
            # new solution
            (t, q) = isolution.Solve()   
            #AX.printf(len(q))
            
            # record array
            if ifnorm: record.append(q)
            
            # plot figures
            if ifplot:                
                plt.plot(t, q, color=colors[i], label=labels[i], \
                         linewidth=widths[i], linestyle=styles[i])
            
        # mark axis
        if ifplot:
            fs = self.fontsize
            plt.legend(loc='best', fontsize=fs)
            plt.xlabel('t=m*x', fontsize=fs)
            plt.ylabel('q=phi/v', fontsize=fs)
            plt.title('convergence test', fontsize=fs)
            #plt.legend(bbox_to_anchor=(2, 0.4), loc='right', fontsize=fs)
            #plt.ylim((-qmax,qmax))    
            plt.show()
            
        # compute error norms 
        if ifnorm:
            # initialize empty dictionary
            norm={}            
            # prepare array
            for i in range(5):
                # load array
                if i==2: # compute norm of reference solution
                    array = record[i]
                else: # compute error norm
                    # index of worse solution
                    ind = int(i-np.heaviside(i-2,0))
                    
                    # worse solution
                    qw = record[ind+1]
                    Nw = len(qw)
                    
                    # better solution
                    qb = record[ind]
                    # sub array of better solution
                    qbs = qb[::2]
                    Nb = len(qbs)
                    
                    # equate length of arrays
                    Nc = min(Nw,Nb)
                    array = qw[:Nc]-qbs[:Nc]                    
                    
                # mask invalid numbers
                err = np.ma.masked_invalid(array**2)
                # norm
                e2 = np.sqrt(np.sum(err))
                # number of points
                Ne = np.ma.count(err)
                # normalized error norm
                norm[labels[i]] = e2/Ne  # per valid point
                
            return norm            
            
        
    ########################################    
    # Solve the ODE until termination conditions are satisfied 
    # Termination conditions are marked by the following numerical flags
    def terminatMsg(self, flag):
        """Translate flag to message."""        
        # translate flag to message
        # initialize ctionary
        Msg = {}
        # add entries
        Msg['0']  = 'Termination: exceed maximum number of time steps'
        Msg['1']  = 'Termination: increase above threshold'
        Msg['-1'] = 'Termination: decrease below threshold'
        Msg['2']  = 'Termination: within range but oscillation up'
        Msg['-2'] = 'Termination: within range but oscillation down'
        Msg['3']  = 'Termination: cross zero from negative to positive'
        Msg['-3'] = 'Termination: cross zero from positive to negative'        
        Msg['6']  = 'Separatrix asymptotes to +1'
        Msg['-6'] = 'Separatrix asymptotes to -1'
        Msg['999'] = 'Termination: exceed user specified final time'
        
        try:
            s = Msg[str(int(flag))]  
        except KeyError:
            AX.printf(f'Termination: unknown flag = {flag}')
        else:
            AX.printf(s)
          
    ########################################    
    # Solve the ODE until termination conditions are satisfied 
    # always solve forward in time from t0
    #
    # Inputs (Optional):
    #        criteria : termination criteria
    #                   'type'  : terminate after solution type can be determined
    #                   'cross' : terminate after solution cross zero
    #        save     : variables to keep in the memory
    #                   'plot'  : keep the entire time trace and plot figure
    #                   'trace' : keep the entire time trace
    #                   'end'   : only keep the last step  
    #        tfmax    : when specified, terminate when t exceed tfmax               
    #
    # Output: (t, q, flag)
    #        t    : time basis, float
    #        q    : solution, float
    #        flag : termination flag, integer      
    def SolveUntil(self, criteria='type', save='end', tfmax=None):
        """Solve the ODE with central difference until termination conditions."""
        # unpack 
        ode = self.ode
        cds = self.cds
        Ntmax = self.Ntmax
        epsilon = self.epsilon
        
        # unload initial conditions
        w0D = cds.w0D
        w1D = cds.w1D   
        # sign of the initial derivatives
        s01 = np.sign(w1D-w0D) # first order
            
        # unload resolution
        dt = abs(cds.dt) # always solve forward in time    
        # current time
        t1 = cds.t0 + dt   
        
        # minimum final time s.t. A/(R^n*pi^{n/2})*exp(-t^2/R^2)<epsilon
        ARe = abs(ode.S0/epsilon) # ensure positive
        if ARe<1: # source is not a limiting factor
            tf = 1
        else: #source is a limiting factor
            tf = ode.R*np.sqrt(np.log(ARe)) # ensure positive    
        # intrinsic time scale for linearized, no source is 1
        tfmin = max(1,tf)
    
        # record values
        if save in ['plot', 'trace']: # keep the entire time trace
            # instruction for how to insert next solution
            ind = cds.ind
            # initialize arrays
            time = [cds.t0]
            phiv = [ode.wD2q(w0D)]
            time.insert(ind, t1)
            phiv.insert(ind, ode.wD2q(w1D))
            
        # termination parameters
        # thresholds 
        upper = ode.upper
        lower = ode.lower
        
        # initial flag
        flag = 0
        # advance solution
        count = 0 # number of times derivative changes sign
        # initial step count
        nsteps = 1 # count from 1
        while flag == 0 and nsteps < Ntmax+1:
            # next step
            w2D = self.NextD(nsteps, w0D, w1D)
            
            # updates
            w0D, w1D = w1D, w2D # multi assignments        
            # next time
            t1 += dt       
            nsteps += 1            
            
            # record values
            if save in ['plot', 'trace']: # keep the entire time trace
                time.insert(ind*nsteps, t1) 
                phiv.insert(ind*nsteps, ode.wD2q(w1D)) 
            
            # check if termination conditions are satisfied  
            # primary conditions
            if w1D>upper: # increase above threshold 
                flag = 1
            elif w1D<lower: # decrease below threshold
                flag = -1
            else: # within range, check secondary conditions
                # check final time
                if tfmax!=None:
                    if t1>tfmax:
                        flag = 999
                else: 
                    if criteria == 'cross': # q cross zero, need convert w->q
                        # 3: corss - to +, -3: cross + to -, 0: otherwise
                        dq = np.sign(ode.wD2q(w1D))-np.sign(ode.wD2q(w0D))
                        flag = 3*int(0.5*dq) 
    
                    else: # oscillation
                        # sign of current 1st order derivative 
                        s11 = np.sign(w1D-w0D)       
                        # check oscillation only at asymptotic time
                        if abs(t1)>tfmin: 
                            # update count
                            count += np.heaviside(-s01*s11, 0) # +1 if 1st derivative change
                            #AX.printf('t=',t1,', s01=',s01,', s11=',s11,', count=',count)
                        # update reference point
                        s01 = s11                    
                        # update flag
                        if count>1:
                            # convert t->z, decimal
                            z1D = ode.t2zD(t1)
                            # sign of current 2nd order derivative 
                            flag = int(2*np.sign(ode.rhsD(z1D, w1D))) 
     
        # report warning message
        if flag==0: 
            AX.printf(f'SolveUntil indetermined at \nq0={cds.q0}, \np0={cds.p0}') 
        
        # convert w(z) to q(t), float
        q0 = ode.wD2q(w0D)
        q1 = ode.wD2q(w1D)
        # linear estimation of zero crossing time
        if abs(flag)==3:      
            # correct for zero crossing time
            t1 -= cds.dt*q1/(q1-q0)                
        
        # plot figure
        if save == 'plot':
            plt.figure()
            plt.plot(time, phiv)
            # mark threshold
            if abs(flag) == 1:
                ymax = ode.threshold
                plt.axhline(ymax,linestyle='--',color='k')
                plt.axhline(-ymax,linestyle='--',color='k')
            fs = self.fontsize
            plt.xlabel('t=m*x', fontsize=fs)
            plt.ylabel('q=phi/v', fontsize=fs)
            plt.title(f'Solve until termination, precision={getcontext().prec}')
            plt.axhline(y=0, color='grey') 
            plt.show()
            
            if abs(flag)==3:
                plt.axhline(y=0, color='grey')
                plt.axvline(x=t1, color='grey')
            #plt.show()
            
        if save in ['plot', 'trace']:
            # convert w(z) to q(t)
            time = np.array(time) # numpy array
            phiv = np.array(phiv) # numpy array of decimals
            return time, phiv, flag
        else:
            # t1 and q1 are float, flag is integer
            return t1, q1, flag
        
        
    ########################################    
    # Compute each term in ODE for diagnostic purpose
    # This information could help determine dominant
    # balance in asymptotic solutions.
    #            
    # Input:
    #      tfmax  : when specified, terminate no later than tfmax
    #      ifplot : if plot diagnopstic figures
    #
    # Output: 
    #     dictionary that contains the folliwing numpy arrays 
    #        t     : time basis
    #        q     : numerical solution of q(t)
    #        ddq   : q'' of the numerical solution
    #        dq    : q' of the numerical solution
    #        dqt   : (D-1)/t *q' of the numerical solution
    #        qn    : nonlinear term of the numerical solution
    #        s     : source term
    #        dE    : normalized field energy density
    #        E     : integrated field energy density, normalized
    #        it    : index >0 of t where abs(dE) is at minimum 
    def Terms(self, tfmax=None, ifplot=True):
        """Compute each term in ODE for diagnostic purpose."""        
        
        # unpack ode
        ode = self.ode
        # dimension of the problem
        dim = ode.dim
        
        # compute numerical solution
        #t, yn = self.Solve()
        t, yn, _ = self.SolveUntil(save='trace', tfmax=tfmax)
 
        # unload resolution
        Nt = len(yn)
        dt = t[1]-t[0] # assume uniform time grid
        
        # terms in the differential equation
        y0 = yn[0:Nt-2]
        y1 = yn[1:Nt-1]
        y2 = yn[2:Nt]
        
        t1 = t[1:Nt-1]
        
        # q'' of the numerical solution
        ddq = (y2 -2*y1+ y0)/dt**2
        # q'' of the numerical solution
        dq = (y2-y0)/2/dt
        # (D-1)/t *q' of the numerical solution
        dqt = (dim-1)/t1*dq
        # nonlinear term of the numerical solution
        qn = y1*(y1**2-1)/2
        # source term 
        s1 = ode.source(t1/ode.u)
        # normalized field energy density
        dE = t1**(dim-1)*(0.5*dq**2 + 0.125*(y1**2-1)**2 + s1*y1)
        
        # LHS-RHS
        lmr = ddq + dqt - qn - s1
        
        # cummulated energy
        E = np.zeros_like(dE)
        ae = 0
        # loop though elements of dE
        for i in range(Nt-2):
            # unpack value
            dEi = dE[i]
 
            # integratation by trapezoidal rule
            if i==0: ae += dEi*dt  
            else: ae += dt*(dEi+dE[i-1])/2
                    
            # store value    
            E[i] = ae
 
        # Optimal time to estimate energy 
        it = AX.WidestPlateau(dE, debug=ifplot)      
            
        # load output dictionary
        dterms=dict()
        dterms['t'] = t1
        dterms['q'] = y1
        dterms['ddq'] = ddq
        dterms['dqt'] = dqt
        dterms['dq'] = dq
        dterms['qn'] = qn
        dterms['s'] = s1
        dterms['dE'] = dE
        dterms['E'] = E
        dterms['it'] = it
        
        # plot figures
        if ifplot:
            # plot terms of ODE #############            
            # initialize new figure
            plt.figure()        
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))
            
            # plot each term
            # q''
            ax1.plot(t1, ddq, marker='.', label="q''")
            # (D-1)/t *q'
            ax1.plot(t1, dqt, marker='.', label="(D-1)*q'/t")
            # nonlinear
            ax1.plot(t1, qn, marker='.', label="q*(q**2-1)/2")
            # source
            ax1.plot(t1, s1, marker='.', label="source")
            # mark figure
            ax1.set_title(f'{dim}D, R={ode.R}, A={ode.A}')
            ax1.set_xlabel('t')
            ax1.legend(loc='best')            
            ax1.axhline(y=0, color='grey')
            ax1.axvline(x=0, color='grey')
            ##ax1.annotate(r'$R/\sqrt{2}$', xy =(Rm, 0), xytext =(Rm, ax1.get_ylim()[0])) 
            
            # plot LHS-RHS
            ax2.plot(t1,lmr,'k', label="LHS-RHS")
            # mark figure
            ax2.set_xlabel('t')
            #ax2.legend(loc='best')
            ax2.set_title('LHS-RHS')            
            ax2.axhline(y=0, color='grey',linestyle='--') 
            ax2.axvline(x=0, color='grey')
            #ax2.annotate(r'$R/\sqrt{2}$', xy =(R2, 0), xytext =(R2, ax2.get_ylim()[0]))
            
            plt.subplots_adjust(hspace=1.5)
            plt.show()
             
            # plot energy ###################           
            # new figure
            fig = plt.figure()        
            ax3 = plt.subplot2grid((3, 1), (0, 0), rowspan=2,fig=fig)
            ax5 = plt.subplot2grid((3, 1), (2, 0))
            
            # energy density
            color='tab:blue'
            ax3.plot(t1, dE, marker='.', color=color, label="energy density")
            # mark figure
            ax3.set_xlabel('t')
            ax3.set_title('Normalized energy')            
            ax3.set_ylabel('energy density', color=color)
            ymin, ymax = ax3.get_ylim()
            aymax = max(abs(ymin), abs(ymax))
            ax3.set_ylim(-aymax, aymax)
            ax3.axhline(y=0, color='grey') 
            ax3.axvline(x=t1[it], color='blue') 
            
            # twin axis
            color = 'tab:red'
            ax4 = ax3.twinx() # instantiate a second axes that shares the same x-axis         
            # cummulated energy
            ax4.plot(t1, E, marker='.', color=color, label="cummulated energy")            
            # mark secondary axis            
            ax4.set_ylabel('cummulated energy', color=color)
            ax4.tick_params(axis='y', labelcolor=color)    
            ymin, ymax = ax4.get_ylim()
            aymax = max(abs(ymin), abs(ymax))
            ax4.set_ylim(-aymax, aymax)
            
            # plot solution as reference
            ax5.plot(t1, y1, 'k')
            # mark figure
            ax5.set_title(f'{dim}D, R={ode.R}, A={ode.A}')
            ax5.set_xlabel('t=m*x')
            ax5.set_ylabel('q=phi/v')
            ax5.set_ylim(y1[0]-1, 2)
            ax5.axhline(y=-1, color='grey',linestyle='--') 
            ax5.axhline(y=0, color='grey')   
            ax5.axhline(y=1, color='grey',linestyle='--')  
            ax5.axvline(x=t1[it], color='blue') 
            
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.subplots_adjust(hspace=1.5)
            plt.show()
            
        return dterms
    