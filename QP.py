# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:47:40 2020

@author: shi9@llnl.gov
"""

#################################################################
# This script contain functions used to map out 
# the q-p phase space boundary for given A and R
#
#####################
# Contents of this script in order of appearence
# class sample : a class contains sampling info in the q-p plane
#    member functions:
#        __init__    : load parameters
#        __repr__    : print contents
#          xstep     : step size in x direction
#          ystep     : step size in y direction
#          steps     : return both x and y steps
#           xyD      : sampling array in x-y plane
#          RotD      : rotation matrix counter-clockwise by degree theta
#           qpD      : rotate arrays from (x,y) coordinates to (q,p) coordinates
#           QPD      : sampling grid in q-p plane
#         verticesD  : coordinates of vertices of sampling box in q-p frame
#          edgesD    : coordinates of edge centers of sampling box in q-p frame
#          drawbox   : draw sample box in current figure
#          auto      : suggested sampling parameters based on data
#
# x2qp:  for given x return (q0, p0) for the sample
#        when sample not specified, default search along (q,0)
#
# class Island: a class for q-p phase space island and separatrix
#    member functions:
#        __init__    : load parameters
#        __repr__    : print contents
#         Shoot      : Call SolveUntil to solve ODE for given initial conditions 
#         Asympt     : compute asymptotic flag for given initial conditions of ODE
#         Scan       : scan q0-p0 Samples to determine solution type by SolveUntil
#        xExtent     : find xmax or xmin beyond which no island exists
#        xRefine     : zoom in possibly under-resolved transitions along x 
#        xCritical   : find transition points along x direction
#        xBisect     : use bisection to find single transition along x 
#        xInterval   : extend interval to contain a known transition along x
#        Boundary    : compute a list of boundary points within Sample box
#         save       : QP boundary points by appending to txt file
#         load       : load QP boundary points from txt file      
#         plot       : plot phase space island boundary
#
# 
# Use Decimal package to improve precision and reduce truncation error.
# Decimal variables are named with a trailing "D".
#
####################
# Below is a list of artificial parameters. 
# Users are not recommended to adjust these parameters unless proficient
# Local to Island:
#     epsilon_bisection  : control convergence criteria 
#                          if too large, may omit qp phase boundary/separatrix
#                          if too small, ODE convergence poor near boundary, 
#                          in which case need higher precision/resolution
# Local to Island.xExtent:
#      Next          : exdend xmax/xmin at least Next times
#                      if too small, not sure have reached boundary
#                      if too large, unnecessary computation cost
#      xext          : each extension add/subtract xmax/xmin by xext
#                      intrinsic island width is 1
#      Niter         : maximum number of iterations before stopping further extension
# Local to Island.xRefine:
#     xflag = 1      : asymptotics when x->+infty
#                      do not change this!
#   Nsample_refine   : number of q sampling points when refinining separatrix
#                      if too small, may not resolve separatrix
#                      if too large, may take too long
#
#################################################################

import SE
import Auxiliary as AX
#import Asympt_nD
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
rc('text', usetex=True)

from decimal import Decimal, getcontext
#getcontext().prec = 28 # default is 28

# Maximum number of QP phase space island
#   too small: may not capture all islands
#   too large: takes unnecessary long time to search
#NIm = None  # determine automatically, using a posterori knowledge
#NIm = 4 # user specified value

#################################################################
# Class contain the sampling info in q-p plane
# parameters are initialized to default values
# user can change the stored values later.
#
# To allow versatile sampling, the sampling grid (x,y) is centered at 
# (q0, p0) and is rotated counterclockwise by angle theta. We have the 
# following coordinate transform
#    (q)   (q0)   (\cos\theta   -\sin\theta) (x)
#    ( ) = (  ) + (                        ) ( )
#    (p)   (p0)   (\sin\theta    \cos\theta) (y)
#
# Member variables:
#    q0D, p0D : center of the x-y origin in the q-p frame
#    theta    : degree of the x-y frame w.r.t. the q-p frame
#    Nx, Ny   : number of points in x, y direction
#    Lx, Ly   : length of the box in x, y direction
#    x0D, y0D : center of the search box in x-y frame
#    ifauto   : flag to tell if sample has been/should be auto loaded
#               using a posteriori knowledge of the QP phase diagram
#
# The default values are theta=0, and Nx=Ny=Ns
# The default box length is given by the intrinsic scales of the ODE 

        
class Sample:
    """A class contains sampling info in the q-p plane."""
    
    # initialize object 
    def __init__(self,theta=0,Nx=33,Ny=33,Lx=4,Ly=2,\
                 q0D=0,p0D=0,x0D=0,y0D=0,ifauto=False):
        """Load parameters."""   
        # convert angle to within [0,360)
        self.theta = AX.Angle(theta)
        
        self.Nx = Nx
        self.Ny = Ny
       
        self.Lx = Lx
        self.Ly = Ly
        
        # ensure decimal
        self.q0D = Decimal(q0D)    
        self.p0D = Decimal(p0D)
        
        self.x0D = Decimal(x0D)
        self.y0D = Decimal(y0D)
        
        self.ifauto = ifauto
        
        
    # print contents of the instance
    def __repr__(self):
        AX.printf(f'Nx={self.Nx}, Ny={self.Ny}, theta={self.theta} deg')
        AX.printf(f'q0={self.q0D}, p0={self.p0D}')
        AX.printf(f'x0={self.x0D}, y0={self.y0D}')
        AX.printf(f'Lx={self.Lx}, Ly={self.Ly}')
 
        return '' # need to return str
        
    # uniform step size in x direction
    def xstep(self):
        """Step size in x direction."""
        Nx = self.Nx
        if Nx>1: # multiple points in x direction
            dx = self.Lx/(Nx-1)
        else: # single point in x direction
            dx = 0
            
        return dx
            
    # uniform step size in y direction
    def ystep(self):
        """Step size in y direction."""
        Ny = self.Ny
        if Ny>1: # multiple points in y direction
            dy = self.Ly/(Ny-1)
        else: # single point in y direction
            dy = 0
            
        return dy
    
    # return both steps
    def steps(self):
        """Step sizes in x and y directions."""
        return self.xstep(), self.ystep()
    
    # sampling array in x-y plane, decimal
    def xyD(self):
        """Sampling arrays in x,y plane, decimal precision."""
        Nx = self.Nx
        if Nx>1: # multiple points in x direction
            xD = Decimal(-self.Lx/2) + np.arange(Nx)*Decimal(self.xstep())
        else: # single point in x direction
            xD = Decimal(0) # at origin of x axis
        xD = xD + Decimal(self.x0D) # shift along x by x0, ensure decimal
        
        Ny = self.Ny
        if Ny>1: # multiple points in x direction
            yD = Decimal(-self.Ly/2) + np.arange(Ny)*Decimal(self.ystep())
        else: # single point in x direction
            yD = Decimal(0) # at origin of y axis
        yD = yD + Decimal(self.y0D) # shift along y by y0, ensure decimal
        
        return xD, yD
    
    
    # rotation matrix, decimal
    # not putting this into initialization to allow updated theta
    def RotD(self):
        """Rotation matrix counter-clockwise by degree theta"""
        ctD = Decimal(np.cos(self.theta/180*np.pi))
        stD = Decimal(np.sin(self.theta/180*np.pi))
        RmD = np.array([[ctD, -stD], [stD, ctD]])
        
        return RmD
    

    # rotate a list of points with coordinate (x,y) to (q,p), decimal
    # xD and yD are numpy arrays of length N
    def qpD(self, xD, yD):
        """Rotate arrays from (x,y) coordinates to (q,p) coordinates, decimal precision."""
        # angle to within [0,360)
        if self.theta>1e-3: # rotation needed
            # rotation matrix
            RmD = self.RotD()
            # cast into matrix
            XYD = np.array([xD, yD]) # 2-by-N
            # rotate coordinate'
            QPD = RmD @ XYD # matrix multiplication
        else: # rotation ignored
            QPD = np.array([xD, yD]) # 2-by-N
            
        # extract and shift coordinate
        qD = QPD[0] + Decimal(self.q0D) # 0-th row, ensure decimal
        pD = QPD[1] + Decimal(self.p0D) # 1-st row, ensure decimal
        
        return qD, pD
    
    # sampling grid in q-p plane, decimal
    def QPD(self):
        """Rotate x-y grid to q-p grid, decimal precision."""
        # x-y sampling points
        xD, yD = self.xyD()
        # meshgrid
        XD, YD = np.meshgrid(xD, yD) # Ny-by-Nx
        
        # reshape
        N = self.Nx*self.Ny        
        # rotate and shift
        qD, pD = self.qpD(XD.reshape(N), YD.reshape(N)) # 2-by-N        
        # reshape
        QD = qD.reshape(self.Ny, self.Nx)
        PD = pD.reshape(self.Ny, self.Nx)
        
        return QD, PD
    
    # q-p coordinates of vertices of the sampling box, decimal
    def verticesD(self):
        """Coordinates of vertices of the sampling box in q-p frame."""
        Lx2D = Decimal(self.Lx/2)
        Ly2D = Decimal(self.Ly/2)
        
        # first quadrant
        x1D = self.x0D + Lx2D
        y1D = self.y0D + Ly2D
        q1D, p1D = self.qpD(x1D, y1D)
        
        # second quadrant
        x2D = self.x0D - Lx2D
        y2D = self.y0D + Ly2D
        q2D, p2D = self.qpD(x2D, y2D)
        
        # third quadrant
        x3D = self.x0D - Lx2D
        y3D = self.y0D - Ly2D
        q3D, p3D = self.qpD(x3D, y3D)
        
        # fourth quadrant
        x4D = self.x0D + Lx2D
        y4D = self.y0D - Ly2D
        q4D, p4D = self.qpD(x4D, y4D)
        
        # repead the point for easy plots
        return np.array([q1D,q2D,q3D,q4D,q1D]), np.array([p1D,p2D,p3D,p4D,p1D])
    
    # q-p coordinates of edge centers of the sampling box, decimal
    def edgesD(self):
        """Coordinates of edge centers of the sampling box in q-p frame."""
        Lx2D = Decimal(self.Lx/2)
        Ly2D = Decimal(self.Ly/2)
        
        # +x
        x1D = self.x0D + Lx2D
        y1D = self.y0D 
        q1D, p1D = self.qpD(x1D, y1D)
        
        # -x
        x2D = self.x0D - Lx2D
        y2D = self.y0D 
        q2D, p2D = self.qpD(x2D, y2D)
        
        # +y
        x3D = self.x0D
        y3D = self.y0D + Ly2D 
        q3D, p3D = self.qpD(x3D, y3D)       
        
        # -y
        x4D = self.x0D
        y4D = self.y0D - Ly2D 
        q4D, p4D = self.qpD(x4D, y4D)
        
        return np.array([q1D,q2D,q3D,q4D]), np.array([p1D,p2D,p3D,p4D])
    
    # draw sample box in current figure
    def drawbox(self, ifprint=False, ifortho=False, color='b', linestyle=':'):
        """Draw sampling box in current figure."""
        # connect vertices
        if self.Nx>1 and self.Ny>1:
            qvD, pvD = self.verticesD()
            plt.plot(qvD, pvD, color=color, linestyle=linestyle)
        
        # draw center lines
        qeD, peD = self.edgesD()
        if (self.Nx>1 and not ifortho) or (ifortho and self.Ny>1): # draw x axis
            plt.plot(qeD[:2], peD[:2], color=color, linestyle=linestyle)
        if (self.Ny>1 and not ifortho) or (ifortho and self.Nx>1): # draw y axis
            plt.plot(qeD[2:], peD[2:], color=color, linestyle=linestyle)
        
        # mark axis
        plt.axhline(y=0, color='grey')
        plt.axvline(x=0, color='grey')
        
        if ifortho: plt.axes().set_aspect('equal') 
        
        # plot title
        #plt.title('Box q0={0:.2f}'.format(self.q0)+', p0={0:.2f}'.format(self.p0))
        if ifprint:
            AX.printf('Box q0={0:.2f}'.format(self.q0D)+', p0={0:.2f}'.format(self.p0D))
        
        
    # automatically determine parameters from the ODE
    def auto(self, ode):
        """Modify parameters to ODE suggested values."""    
        self.ifauto = True
        # unpack parameters in ode
        A = ode.A
        R = ode.R
        dim = ode.dim
        
        if dim == 1: # 1D, phase space island, tested R=0.1 to 1000
            fpr = ode.fpoints.real # real value of fixed points
            
            # weighting factor of the source term
            ws = 1/(1+(R/2)**4)**0.25
            #AX.printf(ws)
                    
            # search box size ######
            # non-linearity dominated width of the island
            Ln = 2*ode.threshold        
            # fixed points dominated island width
            Lf = 2*ode.df
            
            # intrinsic scale in q direction
            Lx = ws*(Ln + 1) + (1-ws)*Lf
            # intrinsic scale in p direction
            Ly = ws*Ln + (1-ws)*Lf
            
            
            # search box center #####
            # non-linearity dominated stable fixed point
            qn = min(fpr, key=abs)          
            # source dominated center of the island
            # simple integration of the Gaussian source
            qs = 0.5*A*R**(2-dim)/np.pi**(dim/2)
            ps = 0.5*A*(R*np.sqrt(np.pi))**(1-dim)
            #AX.printf('qs=',qs,', ps=',ps)
            
            # source dominant R-dependent factors
            fp = 1/np.sqrt(1+R**2/4) # rp->1 when R->0, rp~1/R when R>>1
            fq = fp**2
            #AX.printf('fq=',fq,', fp=',fp)        
    
            # simple integration of the Gaussian source
            q0 = ws*fq*qs + (1-ws)*qn       
            # simple integration of the Gaussian source
            p0 = -ws*fp*ps
     
            # load parameteres
            self.Lx = Lx
            self.Ly = Ly
            self.q0D = Decimal(q0)
            self.p0D = Decimal(p0)
            #AX.printf('q0=',self.q0,', p0=',self.p0)

######################################################################
# utility function, for given x return (q0, p0) for the sample
# xD is scalar or numpy array
def x2qp(xD, sample):
    """Convert x to (q0, p0) for sample or None."""
    if sample!=None:
        # use sample specified convertion
        return sample.qpD(xD, sample.y0D*np.ones_like(xD)) 
    else: 
        # default search along q axis
        # p=0, integer compatible with decimal
        return xD, Decimal(0)*np.ones_like(xD)
    
    

######################################################################
# A class containing q-p phase space island info.
# Key parameter:
#    epsilon : access as input 
#              (1) time step size dt=epsilon*min(1,ode.R)
#                  the scale of nonlinearity is 1
#                  the scale of the source is R
#              (2) passed on to Solution
#                  SolveUntil abort only after source vanish S < epsilon 
#
#    epsilon_bisection : acceess by island.epsilon_bisection
#                        Control convergence criteria for bisection
#                        Used to resolve separatrix/islands in xRefine
#
#    Nsample_refine    : acceess by island.Nsample_refine
#                        default number of q sampling points 
#                        when refinining separatrix
#
# Key member variables:
#    qbD, pbD  : list of boundary points, decimal
#    fb        : asymptotic flag of boundary points, integer
#    qsD, psD  : list of presumed separatrix points, decimal
#
# Useful metadata: (updated after each call of xRefine)
#    xlistD    : refined search array along x direction, decimal
#    flist     : asymptotic flags corresponding to xlistD
#    bcount    : number of boundary points identified
#    scount    : number of separatrix points identified
#    depth     : number of times refinement is carried out
class Island:
    """A class for identifying q-p phase space island."""  

    # plot parameters
    fontsize = 12
    color = 'r'
    markersize = 10
    
    # initialize object 
    def __init__(self, ode, Ntmax=2**14, epsilon=1e-2, NIm=None):
        """Load parameters.""" 
        # A->-A and p->-p is a symmetry of the ODE
        # This program solves is designed for the case A>=0
        assert(ode.A>=0)
        # store parameters
        self.ode = ode
        
        # time step for solving ODE
        self.dt = abs(epsilon)*min(1,ode.R) # ensure positive  
        
        # expected number of island boundary points
        self.NC = 2*ode.NI(NIm)
        # expected number of separatrix points
        if ode.dim==1: self.NS = 1
        else: self.NS = 0
        
        # asymptotics when x->+infty
        #self.xflag = int(np.sign(ode.A))
        self.xflag = 1
        
        # for SolveUntil
        self.Ntmax=Ntmax
        self.epsilon=epsilon
        
        # epsilon_bisection used to resolve separatrix and narrow islands
        # initialize with default values. Users can change later
        if ode.dim==1: # in 1D, the feature is a separatrix
            self.epsilon_bisection = 1e-6
        else: # in higher dim, the feature is a narrow island
            self.epsilon_bisection = 1e-9
            
        # default number of q sampling points when refinining separatrix
        self.Nsample_refine = 33 # recommended value
        #Nsample_refine = 101
        #Nsample_refine = 1001
        
    # print contents of the instance
    def __repr__(self):
        AX.printf(self.ode)
        AX.printf(f'Ntmax={self.Ntmax}, epsilon={self.epsilon}, dt={self.dt}')
        AX.printf(f'Expected counts (boundary, separatrix)=({self.NC}, {self.NS})')
        AX.printf(f'Refinement convergence criteria is {self.epsilon_bisection}')
        AX.printf(f'Decimal precision is {getcontext().prec}')
        return '' # need to return str   


    #################################################################
    # Solve the ODE for given initial conditions until 
    # the solution type is determined
    # Inputs:
    #    q0D, p0D  : initial conditions, decimal
    # Output: from SolveUntil
    #    t       : termination time 
    #    f       : termination flag 
    def Shoot(self, q0D, p0D):
        """Solve ODE for given initial conditions until termination."""
        
        # initial condition
        cds = SE.Initial(dt=self.dt, q0D=q0D, p0D=p0D) # default t0=0
        #AX.printf(f'q0={q0D}, p0={p0D}, cds= ', cds)
        
        # instantiate solution class
        sol = SE.Solution(self.ode, cds, Ntmax=self.Ntmax, epsilon=self.epsilon)
        #AX.printf(sol)
        
        # termination reason, criteria = 'type' and save='end' by default
        t, _, f = sol.SolveUntil()
        
        return t, f

    #################################################################
    # Compute asymptotics flag for given initial conditions
    # Inputs:
    #    q0D, p0D  : initial conditions, decimal
    # Output:
    #    f       : termination flag from SolveUntil
    def Asympt(self, q0D, p0D):
        """Compute asymptotic flag for given initial conditions of the ODE."""
        
        # This is a wrapper function calling Shoot
        _, f = self.Shoot(q0D, p0D)
        
        return f

    #################################################################
    # Scan over initial conditions to determine reason of termination,
    # which determines the type of solution: oscillatory or diverging.
    # Always solve from t=0.
    #
    # The scan is carried out in the x-y coordinate, which is rotated 
    # counter-clockwise from the q-p coordinate by degree theta. The center 
    # of the x-y frame is located at (q0, p0)
    #
    # When Nx or Ny is 1, carry out 1D scan, otherwise 2D scan.
    #
    # Inputs:
    #    sample : an instance of class Sample, specifying the search box
    #    ifplot = True: plot figure
    # Output: 
    #    QD, PD   : sampled values, numpy 2D array of length Ny-by-Nx, decimal
    #    flags    : termination flags, numpy 2D array of length Ny-by-Nx           
    def Scan(self, sample, ifplot=False):
        """Scan q0-p0 plane to determine solution type."""         
        # unpack
        fs = self.fontsize
        
        # prepare sampling arrays
        QD, PD = sample.QPD()
        Nx = sample.Nx
        Ny = sample.Ny
        
        # initialize output
        flags = np.zeros((Ny, Nx), dtype=int)
        # loop through x and y, column major by default
        for i in range(Nx):
            #AX.printf(f'i/Nx={i}/{Nx}')
            for j in range(Ny):            
                #AX.printf(f'j/Ny={j}/{Ny}')
                flags[j, i] = self.Asympt(QD[j,i], PD[j,i])
                
        # reshape arrays if 1D
        if Nx==1:
            QD = QD.reshape(Ny)
            PD = PD.reshape(Ny)
            flags = flags.reshape(Ny)            
            # plot figure
            if ifplot:
                # rescaling x axis
                rPD = (PD-Decimal(sample.p0D))/Decimal(sample.Ly)
                #sample.drawbox()  
                plt.plot(rPD, flags, marker='.',linestyle='')
                plt.xlabel('(p-p0)/Ly', fontsize=fs)
                plt.ylabel('flag', fontsize=fs)
                plt.title('scan along y, fixed q0={0:.2f}'.format(sample.q0D) \
                          + f', theta={sample.theta}')
                plt.grid(True)
                #plt.show()   
            
        if Ny==1:
            QD = QD.reshape(Nx)
            PD = PD.reshape(Nx)
            flags = flags.reshape(Nx)
            # plot figure
            if ifplot:
                # rescaling x axis
                rQD = (QD-Decimal(sample.q0D))/Decimal(sample.Lx)
                #sample.drawbox()  
                plt.plot(rQD, flags, marker='.',linestyle='')
                plt.xlabel('(q-q0)/Lx', fontsize=fs)
                plt.ylabel('flag', fontsize=fs)
                plt.title('scan along x, fixed p0={0:.2f}'.format(sample.p0D) \
                          + f', theta={sample.theta}')
                plt.grid(True)
                #plt.show()   
                
        # plot 2D figure
        if ifplot and Nx>1 and Ny>1:
            #sample.drawbox()            
            # boundary separated by abs(flag) = 1 and 2
            plt.contour(QD, PD, abs(flags), levels=[1.5])
            plt.xlabel('q0', fontsize=fs)
            plt.ylabel('p0', fontsize=fs)
            plt.title('Low resolution contour')
            # mark axis
            plt.axhline(y=0, color='grey')
            plt.axvline(x=0, color='grey')
            plt.axes().set_aspect('equal')
            #plt.show()
        
        return QD, PD, flags
    
    
    #################################################################
    # Determine xmax and xmin where island can exist
    # Search along x direction, with y=y0.
    # Always solve from t=0. 
    #
    # Input:
    #    flag    : flag=1  search xmax, at which termination flag is 1
    #              flag=-1 search xmin, at which termination flag is -1
    #    sample  : if None, default search along q axis
    #              if specified, search along x axis
    #              only q0, p0, theta, Lx are significant member variables
    #    debug   : if True, print debug messages and plot figure
    # Output: 
    #    xm  : xmax or xmin beyond which no island exists, integer 
    def xExtent(self, flag, sample=None, debug=False):
        """Find xmax or xmin beyond which no island exists.""" 
        ############################
        # fixed internal parameters    
        # exdend xmax/xmin at least Next times
        Next = 5 
        # each extension add/subtract xmax/xmin by xext
        xext = 1 # intrinsic island width is 1, integer compatible with decimal
        # maximum number of iterations before abort
        Niter = 1000
        ############################
        
        # make sure flag is +-1
        flag=np.sign(flag)
        assert(flag!=0)
        # extend direction
        xflag = flag # integer
        if debug: AX.printf(f'xflag={xflag}')
        #AX.printf(self.__repr__())
        
        # preparation for searching along x direction
        if sample!=None: # search box specified
            # angle of x axis, within [0,360)
            theta = sample.theta
            # correct for flag when q along positive x direction asymptotes to -1
            if theta>90 and theta<=270: # in 2 or 3 uadrant, or along -y
                xflag = -xflag                
        
        # intrinsic scale of the ode for large q
        qm = -int((2*self.ode.S0)**(1/3))
        # ensure qm beyond bare island, xm is integer
        if xflag==1: xm = max(1,qm)
        else: xm = min(-1,qm)  
        # qm in numpy int, for later compatibility, convert xm to decimal
        xm = Decimal(xm)

        # determine bound
        # number of times already extended
        count = 0
        # number of iterations already examined
        ii = 0
        while count<Next and ii<Niter:
            # update count
            ii += 1            
            # conver x to (q,p)
            q0D, p0D = x2qp(xm, sample)  
            #AX.printf(f'type of xm={type(xm)})), q0={type(q0D)}, p0={type(p0D)}')
            # asymptotic flag
            f = self.Asympt(q0D, p0D)
            if debug: AX.printf(f'xm={xm}, q0={q0D}, p0={p0D}, flag={f}')
            
            # update count
            if f==flag: # beyond threshold
                count +=1
            else: # reset 
                count = 0
            # extend qmax 
            xm += xflag*xext
        
        # having made sure boundary is reached,
        # pull xm back towards island
        xm -= xflag*Next*xext
        
        # print warining message
        if ii>=Niter: 
            AX.printf('warning: Island.xExtent reached maximum number of iterations!!')
            AX.printf(f'         xExtent in flag={flag} direction')
            
        return xm
    

    #################################################################
    # Refine sample so that 1D search array capture possible island boundaries.
    # Search along x axis, with y=y0.
    # Always solve from t=0.     
    #
    # Increasing from the lower bound, flag transition from -1 to +1 through 
    # intermediate +2 or -2 if the search passes through an island. 
    # On the other hand, when passes through a separatrix, flag transition 
    # directly from -1 to +1, with no intermediate oscillations.
    #    
    # The program refine +1/-1 transitions to look for possible islands,
    # and refine -1->(-2,2) jumps to look for a separatrix or a narrow 
    # secondary island. 
    #
    # The refinement process stops either when the expected number of 
    # islands and separatrix are found, or when the search box size 
    # Lx<epsion_bisection.
    #
    # Input (optional):
    #    sample  : if None, default search along q axis
    #              if specified, search along x axis
    #              only q0, p0, theta, Lx, Nx are significant member variables
    #              Lx>0: refine within sample specified bounds
    #              Lx=0: exhaustive refine, use xExtent bounds
    #    ifplot  : if True, plot figure and print results
    #    debug   : if True, print debug messages and plot figure
    #            
    # Key outputs (stored internally as metadata): 
    #    xlistD  : sampled array, numpy array of length N, decimal
    #    flags   : termination flags, numpy array of length N     
    #     
    def xRefine(self, sample=None, ifplot=False, debug=False):
        """Refine 1D search array along x axis."""  
        
        # asymptotics when x->+infty
        xflag = self.xflag
        # number of q sampling points when refinining separatrix
        Nsample_refine = self.Nsample_refine
        
        # prepare search sample 
        if sample!=None: # search box specified
            Nx = sample.Nx
            # search in x direction should have Nx>1
            if Nx<2: Nx = Nsample_refine
            # copy parameters of sample into dictionary
            ds = dict(vars(sample)) # copy dictionary of member variables  
            # copy sample for later modification
            sx = Sample(**ds) # kwargs 
            # update default parameters
            sx.Nx = Nx
            sx.Ny = 1
            # correct xflag if needed 
            theta = sample.theta
            if theta>90 and theta<=270: xflag = -xflag
        else: # default search along q axis
            Nx = Nsample_refine
            # initialize sample, change default values
            sx = Sample(Nx=Nx,Ny=1,Lx=0)
            
            
        # xmin and xmax of the search    
        if sx.Lx>0 and sample!=None:
            if debug: AX.printf('Search boundary from sample')
            # xmin and xmax specified by sample
            x0D = sx.x0D
            Lx2D = Decimal(sx.Lx/2)
            xmaxD = x0D + Lx2D
            xminD = x0D - Lx2D
        else: 
            if debug: AX.printf('Search boundary from xExtent')
            # determin xminx and xmax from xExtent, integers
            xp = self.xExtent(1, sample=sample, debug=debug)  # asymptotic to +1
            xn = self.xExtent(-1, sample=sample, debug=debug) # asymptotic to -1
            # ensure correct ordering, integer compatible with decimal
            xmaxD = max(xp,xn)
            xminD = min(xp,xn)        
        if debug: AX.printf(10*'#'+f'\n{self.ode}\nxmin={xminD},xmax={xmaxD},Nx={Nx}')   
        

        ##############
        # refine search array within xmax and xmin
        # initial box size
        L0D = xmaxD - xminD   
        # initial lists, ensure at least one search
        slist = [1] #1: -1->1 jump, 2: -1->(2,-2) jump, 0: no refinement needed
        xllistD = [xminD]
        xrlistD = [xmaxD]
        fllist = [-xflag]
        frlist = [xflag]
        
        # boundary, separatrix, and narrow island boundary counts
        # boundary is (-1,1)<->(-2,2) jump
        # separatrix is unresolved -1 <-> 1 jump
        bcount, scount, nncount, npcount = 0, 0, 0, 0 
        # expected boundary/separatrix counts
        NC, NS = self.NC, self.NS
        # number of times the refinement has been carried out
        depth = 0
        # ensure at least one run
        iffirst = True
        # if -1->1, -1->(-2,2), jump have been discovered
        ifseparatrix, ifboundary, ifjump = False, False, False
        # convergence criteria
        dLe = self.epsilon_bisection
        
        # iterative refinement. Do at least once        
        while (L0D>dLe and (bcount<NC or scount<NS)) or iffirst: 
            # update box length
            sx.Lx = L0D
            # update termination flags
            iffirst = False
            # number of jumps 
            njumps = len(slist)
            # number of new jumps 
            new = 0
            
            if debug:
                 AX.printf(50*'#'+f'\nBefore refine\nL0={L0D}')
                 AX.printf(f'boundary (count, expected)=({bcount}, {NC})')
                 AX.printf(f'separatrix (count, expected)=({scount}, {NS})')
            #     AX.printf('slist=',slist,'\nxllist=',xllistD,'\nxrlist=',xrlistD)
            #     AX.printf('fllist=',fllist,'\nfrlist=',frlist)
                    
            # refine each jump
            for i in range(njumps):
                # old index before any insertion
                i0 = i + new
                if debug: AX.printf(10*'#'+f'\ni={i}, i0={i0}, s={slist[i0]}')
                # refine -1/+1 and -1->(-2,2) jumps
                if slist[i0]>0: 
                    # sample center
                    x0D = (xllistD[i0] + xrlistD[i0])/2                      
                    # update values
                    sx.x0D = x0D
                    # scan for given 1D sample
                    _, _, flags = self.Scan(sx)
                    # search x array
                    xD, _ = sx.xyD()
                    
                    # index where jump occurs
                    index, _ = AX.Jump(flags, 0.5)
                    # number of jumps
                    nj = len(index) # >=1 at least one jump
                    
                    if debug:
                        AX.printf(f'Refine jump {i}, sample:\n {sx}')
                        AX.printf(f'index={index}')
                    
                    # list of x values
                    xlD = [xD[j] for j in index] # before jump
                    xrD = [xD[j+1] for j in index] # after jump
                    # list of flag values
                    fl = [flags[j] for j in index] # before jump
                    fr = [flags[j+1] for j in index] # after jump           
                    
                    # insert or replace jumps to the list
                    for j in range(nj):  
                        ifjump = True # at least one jump is found
                        # if island boundary : (-1,+1)<->(-2,+2)
                        b = int(np.heaviside(1-abs(abs(fl[j]*fr[j])-2), 0))
                        # count new boundary
                        bcount += b   
                        
                        # determine if the jump needs to be further refines                        
                        # if -1/+1 jump
                        if fl[j]*fr[j]==-1: 
                            s = 1
                            # count separatrix only the first time
                            if not ifseparatrix:  scount = 1                               
                            # set true after first discover the jump   
                            ifseparatrix = True
                        elif fl[j]==-1 and abs(fr[j])==2:
                            s = 2
                            # remove repeated boundary count
                            if ifboundary: bcount -= 1          
                            # set true after first discover the jump   
                            ifboundary = True                            
                        else: s = 0     
                                                                     
                        if debug: AX.printf(f'j={j}, b={b}, s={s}')
                        
                        # update list of jump points
                        ind = i0+j
                        AX.InsertReplace(slist, ind, s, j)
                        AX.InsertReplace(xllistD, ind, xlD[j], j)
                        AX.InsertReplace(xrlistD, ind, xrD[j], j)
                        AX.InsertReplace(fllist, ind, fl[j], j)
                        AX.InsertReplace(frlist, ind, fr[j], j)     
                        
                        # register narrow island boundaries
                        # "-"->"-" boundary
                        if fl[j]==-1 and abs(fr[j])==2: nncount += 1
                        # "-"->"+" boundary
                        if abs(fl[j])==2 and fr[j]==1:
                            qD, _ = sx.qpD(xlD[j],sx.y0D)
                            if qD<0: npcount += 1
                      
                    # number of new jumps
                    new += max(nj-1, 0)
                    
                    # debug message and plots
                    if debug:
                        AX.printf(f'counts (boundary,separatrix)=({bcount},{scount})')
                        AX.printf(f'nj={nj}, new = {new}')
                        AX.printf(f'slist={slist}\nxllist={xllistD}\nxrlist={xrlistD}')
                        AX.printf(f'fllist={fllist}\nfrlist={frlist}')
                    if debug or ifplot: 
                        # search q array
                        qD =[x2qp(x0D, sample)[0] for x0D in xD]
                        # rescaling x axis
                        qD = (np.array(qD)-sx.q0D)/(xmaxD-xminD)
                        # plot as function of rescaled q
                        plt.plot(qD,flags,'o')  
                
                # if no jump found, the presumed jump does not exist
                if nj==0: slist[i0] = 0
                    
            # new patch size
            L0D = L0D/(Nx-1)  
            # refinement depth
            depth += 1
            
            # when above the upper phase boundary, NC=0 in 1D
            # and NC=2 in 2D and 3D. Can early terminate the iteration
            # if the upper boundary is already reached
            if nncount>0 and npcount>0:
                AX.printf('  Upper phase boundary is presumably reached.')
                if debug: AX.printf(f'  nncount={nncount}, npcount={npcount}')
                break                        
        
        ##############
        # join lists
        njumps = len(slist)
        xD = [[xllistD[i], xrlistD[i]] for i in range(njumps)]
        f = [[fllist[i], frlist[i]] for i in range(njumps)]
        # flatten lists
        xlistD = [item for sublist in xD for item in sublist]
        flags = [item for sublist in f for item in sublist]
        # insert original initial and final points for record
        if ifjump:
            xlistD.insert(2*njumps, xmaxD)
            xlistD.insert(0, xminD)
            flags.insert(2*njumps, xflag)
            flags.insert(0, -xflag)             
               
        # store meta data
        self.xlistD, self.flist = xlistD, flags
        self.bcount, self.scount, self.depth = bcount, scount, depth
        #return np.array(xlistD), np.array(flags)
    

    #################################################################
    # Find critical points where solution transition from oscillatory to diverging
    # Search along x direction, with y=y0. 
    # Always solve from t=0.
    #
    # Notice that in addition to phase space island, beyond whose
    # boundary the solution changes from oscillatory to diverging,
    # there also exists separatix in the phase space. On one side
    # of the separatrix, the solution diverges to +infty, and on 
    # the other side of the separatix, the solution goes to -infty,
    # while there is no oscillatory solution along the separatix.
    #
    # Inputs (optional):
    #    sample : an instance of class Sample, specifying the search box
    #             if None, default search along q direction by xRefine
    #    ifplot : if True, plot figure as function of rescaled q and print results
    #    debug  : if True, print debug messages
    # Output: 
    #    xcD       : critical x values, numpy array, decimal
    #    fc        : a list of asymptotic values
    #                fc =  1: the critical solution asymptotes to 1
    #                fc = -1: the critical solution asymptotes to -1  
    #   xsD        : eparatrix q, p values, numpy array, decimal
    def xCritical(self, sample=None, ifplot=False, debug=False):
        """Find transition points between oscillatory/diverging along x axis."""
        ###########################
        # Since xRefine has already narrowed down the range
        # use relative convergence criteria
        epsilon = Decimal(1e-3)
        ###########################           
        
        # preliminary search using refine, ifplot: plot as function of rescaled q
        self.xRefine(sample=sample,ifplot=ifplot,debug=debug)
        # extract meta data
        xlistD, flist = self.xlistD, self.flist
        # remove leading and trailing points, which were for record purpose
        # and may contain spurious jumps. Convert to array to allow manupulations        
        x0D, flags = self.xlistD[1:-1], np.array(self.flist[1:-1])
        if debug: AX.printf(10*'#'+f'\nxCritical raw:\nxlist={xlistD},\nflags={flist}')
        
        # prepare function handle for converting x to (q,p) coordinate
        if sample!=None: # search box specified         
            q0D = sample.q0D 
            y0D = sample.y0D
            # objective function, expand tuple with "*"
            fun = lambda xD: self.Asympt(*sample.qpD(xD,y0D)) 
        else: # default search along q axis
            q0D = 0
            # objective function. p=0, integer compatible with decimal 
            fun = lambda xD: self.Asympt(xD,0)            
        if debug: AX.printf(f'xCritical Sample:\n{sample}')

        # compute island boundary
        # search transition from oscillatory (flags=+-2) to diverging (flags=+-1)
        index, _ = AX.Jump(abs(flags),0.5)    
        # objective function
        f = lambda xD: abs(fun(xD)) - 1.5
        # search for critical values with bisection
        xcD = []
        fc = []       
        # search near each transition
        if debug: AX.printf('Boundary search near each jump...')
        for ind in index: # empty loop if index=[]           
            # root finding using bisection
            xlD, xrD = x0D[ind], x0D[ind+1]
            # use relative convergence criteria
            xD, _ = AX.SignChangeBisec(f, xlD, xrD, epsilon*(xrD-xlD))                
            # register mid point
            xcD.append(xD)
            # asymptotic flag
            fc.append(min([flags[ind], flags[ind+1]],key=abs)) # +1 or -1
            
            
        # compute separatrix q, p coordinates
        # search transition from -1 to +1
        index, _ = AX.Jump(flags, 1.5, cap=2.5)    
        # objective function
        f = lambda xD: fun(xD)
        # search for critical values with bisection
        xsD = []     
        # search near each transition
        if debug: AX.printf('Separatrix search near each jump...')
        for ind in index: # empty loop if index=[]           
            # root finding using bisection
            xlD, xrD = x0D[ind], x0D[ind+1]
            # use relative convergence criteria
            xD, _ = AX.SignChangeBisec(f, xlD, xrD, epsilon*(xrD-xlD))                
            # register mid point
            xsD.append(xD)    
            

        # print results and plot figure
        if ifplot:
            # print results
            AX.printf(f'boundary xc = {xcD}')
            AX.printf(f'asymptotics fc = {fc}')    
            AX.printf(f'separatrix xs = {xsD}')
            # mark characteristic lines in figure
            for xD in xcD: 
                # rescaled q
                qD, _ = x2qp(xD, sample)
                x = (qD-q0D)/(xlistD[-1]-xlistD[0])                
                plt.axvline(x=x, color='r', linestyle='--')
            for xD in xsD: 
                # rescaled q
                qD, _ = x2qp(xD, sample)
                x = (qD-q0D)/(xlistD[-1]-xlistD[0])        
                plt.axvline(x=x, color='grey')
            # mark up figure
            plt.grid(True)    
            plt.title(f'xCritical: {self.ode}')
            plt.xlabel('(q-q0)/Lx', fontsize=self.fontsize)
                
        return np.array(xcD), fc, np.array(xsD)
    
    
    
    #################################################################
    # Use bisection to find a single transition point along x.
    # The goal is to resolve island boundaries.
    # The island boundary is a (-1,1)<->(-2,2) transition.
    # A separatrix -1 <-> 1 may be an underresolved island boundary
    #
    # Stop when lower bound of termination time is reached.
    # Inputs:
    #    xlDi  : initial left boundary, decimal
    #    xrDi  : initial right boundary, decimal
    #    tf    : targeted termination time
    #    fl,fr : termination flags on left and right boundary if known
    #            if None, the program will determine internally 
    #    side  : for treating separatrix when island emerge
    #            side=0: search for island/separatrix boundary on left
    #            otherwise: search for island/separatrix boundary on right
    #    sample : an instance of class Sample, specifying the search box
    #             if None, default search along q direction 
    #    debug  : if True, print debug messages
    #
    # Output:
    #    x0D  : transition point. None if no transition found   
    #    t    : actual termination time correspond to x0D
    #    xlD, xrD : the latest search interval
    def xBisect(self,xlDi,xrDi,tf,fl=None,fr=None,side=0,sample=None,debug=False):
        """Use bisection to find single transition along x if exist."""
        ###########################
        # number of repeated t termination value before abort
        nr = 5
        ###########################                   
        # check input, ensure decimal and l<r
        if xlDi < xrDi: xlD, xrD = Decimal(xlDi), Decimal(xrDi)
        else: xlD, xrD = Decimal(xrDi), Decimal(xlDi)
        #assert(xrD != xlD) # right boundary should be larger than left

        # prepare function handle for bisection, (t,f)=Shoot(q,p)
        if sample!=None: # search box specified         
            y0D = sample.y0D
            # objective function, expand tuple with "*"
            fun = lambda xD: self.Shoot(*sample.qpD(xD,y0D))
        else: # default search along q axis
            # objective function. p=0, integer compatible with decimal 
            fun = lambda xD: self.Shoot(xD,0)    

        # determine termination flags at boundary
        if fl==None: _, fl = fun(xlD)
        if fr==None: _, fr = fun(xrD)
        
        # possible flags that can appear on left and right
        NoTransition = False
        Separatrix = False
        left, right = [], []      
        if abs(fl)==2 and abs(fr)==1: # island is on left
            left = [-2,2] 
            right = [fr] 
        elif abs(fl)==1 and abs(fr)==2: # island is on right
            left = [fl] 
            right = [-2,2] 
        elif fl*fr==-1: # separatrix
            left = [fl]
            right = [fr]
            Separatrix = True
        else: # no island in the interval            
            NoTransition = True
            
        # bisection
        if NoTransition: # not a transition
            xl, xr = float(xlD), float(xrD)
            AX.printf(f'No transition expected. fl={fl}, fr={fr}, xl={xl}, xr={xr}')
            AX.printf(f'ODE: {self.ode}')
            if xlD == xrD: # no bisection expected 
                x0D = xlD
                t, f0 = fun(x0D)
                AX.printf(f't/tf={t}/{tf},x0={x0D},flag={f0}')
            else: # bisection expected but fails to identify
                x0D, t = None, None
        else: # expect transition
            # determine the maximum number of iterations 
            # allowed by decimal precision
            prec = getcontext().prec
            # dx/2^Niter = 10^(-prec)
            Niter = int((prec + float((xrD-xlD).log10()))/np.log10(2))  
            if debug: AX.printf(f'prec={prec}, Niter={Niter}, fl={fl}, fr={fr}')
        
            # initial time, previous time, iteration counter, repeat counter
            t, tp, ic, rc = 0, -1, 0, 0
            # search transition using bisection
            while t < tf and ic < Niter and rc < nr:
                # mid point
                x0D = (xlD + xrD)/2
                # ode SolveUntil
                t, f = fun(x0D)
                
                # print every 10 iterations in debug mode
                if debug and (ic % 10 == 0):
                    ti, tfi, xl, xr = int(t), int(tf), float(xlD), float(xrD)
                    AX.printf(f'{ic}/{Niter}:t/tf={ti}/{tfi},xl={xl},xr={xr},flag={f}')
            
                # update separatrix case
                if Separatrix and abs(f)!=1:
                    Separatrix = False # no longer a separatrix
                    if side==0: # island on right
                        AX.printf('Separatrix resolved. Further search on left')
                        left = [fl]
                        right = [-2,2]
                    else: # island on left
                        AX.printf('Separatrix resolved. Further search on right')
                        right = [fr]
                        left = [-2,2]
            
                # update bisection points
                if f in right: # move right boundary left
                    xrD = x0D
                elif f in left: # move left boundary right
                    xlD = x0D
                else:
                    AX.printf(f'Unexpected! fl={left},fr={right},flag={f},ode={self.ode}')   
                    break
                
                # check repeat t value
                if t == tp: rc += 1
                else: rc = 0
                
                # count iterations
                ic += 1                
                # record previous time value
                tp = t
            
            # report abort reasons if tf not reached
            if ic>=Niter: 
                AX.printf(f'Maximum iterations reached before t>=tf. ode={self.ode}')                
            if rc>=nr: 
                AX.printf(f'Aborted: same t repeated for {rc} times. ode={self.ode}')
                
        return x0D, t, xlD, xrD  


    #################################################################
    # Give the a priori knowledge that a transition exists near 
    # the interval (xl,xr), extend the bounds such that the interval 
    # indeed contains the transition.
    #
    # Inputs:
    #    xlDi    : initial left boundary, decimal
    #    xrDi    : initial right boundary, decimal
    #    dxD     : extend the interval by dxD, decimal
    #              when dxD=None, use dxD = xrDi-xlDi   
    #    fli,fri : termination flags on left and right boundary
    #              if None, the program will determine internally 
    #    sample  : an instance of class Sample, specifying the search box
    #              if None, default search along q direction 
    #    debug   : if True, print debug messages
    #
    # Output:
    #    xlD, xrD : the minimal search interval that contains the transition
    #    fl, fr   : termination flags on left and right boundaries
    def xInterval(self,xlDi,xrDi,dxD=None,fli=None,fri=None,sample=None,debug=False):
        """Extend interval to contain a known transition along x."""
        #############################
        # maximum number of iteration
        Niter = 16 # initial interval is presumably already close to transition
        # The extension is expected to terminate in finite number of iterations
        # because f->1 when q->infty and f->-1 when q->-infty.
        # However, set maximum number of iterations in case of unexpected error
        #############################
        
        # check input, ensure decimal and l<r
        if xlDi < xrDi: xlD, xrD = Decimal(xlDi), Decimal(xrDi)
        else: xlD, xrD = Decimal(xrDi), Decimal(xlDi)
        #assert(xrD != xlD) # right boundary should be larger than left
        
        # step size for extension
        if dxD==None: dxD = xrD - xlD
        # ensure decimal
        dxD = Decimal(dxD)

        # prepare function handle for bisection, (t,f)=Shoot(q,p)
        if sample!=None: # search box specified         
            y0D = sample.y0D
            # objective function, expand tuple with "*"
            fun = lambda xD: self.Shoot(*sample.qpD(xD,y0D))
        else: # default search along q axis
            # objective function. p=0, integer compatible with decimal 
            fun = lambda xD: self.Shoot(xD,0)   
        
        # initialize left and right flag
        if fli==None: _, fli = fun(xlD)
        if fri==None: _, fri = fun(xrD)
        fl, fr = fli, fri
        if debug: print(f'Initial flags are fl={fl},fr={fr}')

        # extend boundary if not an island boundary
        ii = 0 # counter
        while (fl==fr or fl*fr==-4) and ii<Niter and dxD>0: # expand interval 
        #while abs(fl) == abs(fr): # extend interval
            # left flag
            xlD -= dxD
            _, fl = fun(xlD)         
            # right flag
            xrD += dxD
            _, fr = fun(xrD)
            # update counter
            ii += 1        
            if debug: print(f'ii/N={ii}/{Niter},fl={fl},fr={fr}')
            #print(f'ql={qlD},qr={qrD}')
            
        # shrink interval to minimal 
        if ii>1: # boundary not within initial interval 
            if fl==fli:
                # move left boundary to right
                xlD = xrD - dxD
            elif fr==fri:
                # move right boundary to left
                xrD = xlD + dxD    
                
        # proper transition not identified, return original boundaries
        if ii>=Niter:            
            xlD, xrD = xlDi, xrDi
            print('Transition not identified after maximum number of iteraction!')
                
        return xlD, xrD, fl, fr
         
       
    #################################################################
    # Solving from t=0 and generate a list of boundary points in q0-p0 space 
    # where the solution transition from one asymptotic behavior to another. 
    # The list of boundary points is unordered to allow arbitrary shape.
    # Scan in both x and y directions to capture extremum points.
    #
    # Inputs: 
    #    sample : an instance of class Sample, specifying the search box
    def Boundary(self, sample):
        """Compute a list of boundary points of phase space island."""
        
        # copy parameters of sample into dictionary
        ds = dict(vars(sample)) # copy dictionary of member variables  
        # copy sample for later modification
        sx = Sample(**ds) # kwargs        
        # sampling arrays
        xD, yD = sx.xyD() # x0 lengt Nx, y0 length Ny
        # default march along y, and for given y search along x
        sx.Ny = 1
    
        # dictionary for march in x and y directions
        # march along x and y are equivalent up to coordinate rotation 
        dxy = {'y': {'Nx':ds['Nx'], 'Ny':ds['Ny'], 'theta':ds['theta'],
                     'x0D':ds['x0D']*np.ones_like(yD), 'y0D':yD},
               'x': {'Nx':ds['Ny'], 'Ny':ds['Nx'], 'theta':ds['theta']+90, 
                     'x0D':ds['y0D']*np.ones_like(xD), 'y0D':-xD}}
        if self.ode.dim != 1: # no need to march along x
            dxy['x']['Ny'] = 0
            
        # initialize search list
        qblistD = [] # boundary q coordinate 
        pblistD = [] # boundary p coordinate
        fblist = [] # asymptotics
        qslistD = [] # separatrixq coordinate
        pslistD = [] # separatrix p coordinate
        
        # march in y directions
        for d in dxy:            
            AX.printf(f'Marching along {d} direction...')
            # unpack dictionary
            dz = dxy[d]
            #AX.printf(dz)
            Ny = dz['Ny']
            
            # prepare march
            r0 = max(0.1, 1/(Ny+1)) # report every 10%   
            prog  = r0
            count = 0 # progress counter
            bc = 0 # boundary points counter
            sc = 0 # separatrix point counter              
            # search in x direction
            for i in range(Ny):
                # report progress
                count +=1
                if count/Ny>=prog:
                    AX.printf(f'{int(prog*100)} % finished')
                    prog += r0
                    
                # update sample 
                sx.Nx = dz['Nx']
                sx.theta = dz['theta']
                
                # update x0 and y0
                try: sx.x0D = dz['x0D'][i]
                except (IndexError,TypeError): sx.x0D = dz['x0D'] # int not a list
                
                # update x0 and y0
                try: sx.y0D = dz['y0D'][i]
                except (IndexError,TypeError): sx.y0D = dz['y0D'] # int not a list
                
                # search along x direction 
                xcD, fc, xsD =self.xCritical(sample=sx)
                # convert to (q, p)
                qcD, pcD = x2qp(xcD, sx)
                qsD, psD = x2qp(xsD, sx)
                
                # load boundary values
                nc = len(xcD)
                bc += nc
                for j in range(nc):
                    qblistD.append(qcD[j])
                    pblistD.append(pcD[j])
                    fblist.append(fc[j])
                    
                # load separatrix values
                ns = len(xsD)
                sc += ns
                for j in range(ns):
                    qslistD.append(qsD[j])
                    pslistD.append(psD[j])
            
            AX.printf(f'Found {bc} boundary and {sc} separatrix points for {d} scan')
        
        # load data, convert to numpy array
        self.qbD = np.array(qblistD)
        self.pbD = np.array(pblistD)
        self.fb = np.array(fblist)
        self.qsD = np.array(qslistD)
        self.psD = np.array(pslistD)
        
    
    #################################################################
    # save boundary or separatrix data to txt file
    def save(self, fname=None, sample=None, data='boundary'):
        """Save boundary points by appending to txt file"""    
        
        # parameters for file name
        dim = self.ode.dim
        A = self.ode.A
        R = self.ode.R   
        # precision of decimal
        prec = getcontext().prec

        # default file name
        if fname == None:
            if data=='boundary': # island boundary
                fname = f'../data/Boundary{dim}D_A{A}_R{R}.txt'   
            else: # separatrix
                fname = f'../data/Separatrix{dim}D_A{A}_R{R}.txt' 
        
        # open file, appending
        fid = open(fname, "a+")
        
        # write header
        if sample!=None:
            fid.write(50*'#'+'\n')
            fid.write(f'# Ntmax={self.Ntmax}, epsilon={self.epsilon}, decimal prec={prec} \n')
            fid.write(f'# q0={sample.q0D}, p0={sample.p0D} \n')
            fid.write(f'# Lx={sample.Lx}, Ly={sample.Ly}, theta={sample.theta} \n')
            fid.write(f'# Nx={sample.Nx}, Ny={sample.Ny} \n')
            
        if data=='boundary': # island boundary           
            # write file header
            fid.write(8*'#'+' qb '+prec*'#'+' pb '+prec*'#'+' fb '+5*'#'+'\n')                      
            # write data
            for i in range(len(self.qbD)):
                fid.write(f'{self.qbD[i]}, {self.pbD[i]}, {self.fb[i]}\n') 
                
        else: # separatrix
            # write file header
            fid.write(8*'#'+' qs '+prec*'#'+' ps '+8*'#'+'\n')                      
            # write data
            for i in range(len(self.qsD)):
                fid.write(f'{self.qsD[i]}, {self.psD[i]}\n') 
            
        fid.close() 
        
        
    #################################################################
    # read data from txt file, float precision sufficient for plotting
    def load(self, fname):
        """Load QP boundary points from txt file"""         
        # read file
        lines = np.loadtxt(fname, comments="#", delimiter=",", unpack=True)
        # dimensions
        ncol= len(lines) 

        
        if ncol==3: # boundary file
            # load value as float
            self.qbD = lines[0] # q coordinate of boundary points
            self.pbD = lines[1] # p coordinate of boundary points
            self.fb = lines[2] # asymptotics of boundary points
        elif ncol==2: # separatrix file
            # load value
            self.qsD = lines[0] # q coordinate of boundary points
            self.psD = lines[1] # p coordinate of boundary points
        else:
            AX.printf('Unexpected number of columns in file!')   
        
        
        
    #################################################################
    # plot island boundary or separatrix
    def plot(self, data='boundary', label='', equal=True):
        """Plot q-p phase space island boundary."""
        # unpack
        c = self.color
        fs = self.fontsize       
        ms = self.markersize
        
        if data=='boundary': # plot island boundary
            # load data
            qb = self.qbD
            pb = self.pbD
            fb = self.fb    
            
            # data with f=1, mask data with f<fmax
            qr, pr = AX.fyMask(qb, pb, fb, fmax=0.5)
            # data with f=-1, mask data with f<fmax
            ql, pl = AX.fyMask(qb, pb, -fb, fmax=0.5)
            
            # sort by phi
            #qra, pra = AX.ASort(qr,pr)
            #qla, pla = AX.ASort(ql,pl)
                
            # plot points with f=1
            plt.plot(qr,pr,linestyle='',marker='+',color=c,markersize=ms,label=label)
            #plt.plot(qra, pra, linestyle='-',marker='+',color=c, markersize=ms)
            # plot points with f=-1
            plt.plot(ql,pl,linestyle='',marker='o',color=c,markersize=ms,fillstyle='none')
            #plt.plot(qla,pla,linestyle='-',marker='o',color=c,markersize=ms,fillstyle='none')
            
        else: # plot separatrix
            # load data
            qs = self.qsD
            ps = self.psD    

            # sort by phi
            #qsa, psa = AX.ASort(qs,ps)            
            
            # plot with separatrix marker
            plt.plot(qs, ps, color=c,marker='.',linestyle='',markersize=ms)  
            #plt.plot(qsa, psa, color=c,marker='.',linestyle='-',markersize=ms)  
                
        
        plt.xlabel(r'$q_0$', fontsize=fs)
        plt.ylabel(r'$p_0$', fontsize=fs)
        plt.rcParams.update({'font.size': fs})
        
        plt.title('phase space: '+repr(self.ode))
        plt.axhline(y=0, color='grey')
        plt.axvline(x=0, color='grey')
        if equal: plt.axes().set_aspect('equal') 
        

