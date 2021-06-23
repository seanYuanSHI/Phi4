# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 17:34:05 2020

@author: Yuan Shi
"""

import numpy as np
import matplotlib.pyplot as plt

#################################################################
# This script contain auxiliary functions not specific to any class 
#
# Contents of this script in order of appearence
#    Angle        : convert angle (in degree) to within [0, 360)
#    ASort        : sort 2D data (x,y) by polar angle
#    Jump         : detect jumps in a search array
# InsertReplace   : insert or replace an element at the index location of a list
#   fyMask        : mask 1D data by asymptotic flags and invalid values
# SignChangeBisec : use bisection to find x where f(x) changes sign
#   printf        : print with flush
#  WidestPlateau  : identify the widest plateau of an array

#################################################################
# Convert angle (in degree) to within [0, 360)
def Angle(theta_in):
    """Convert angle (in degree) to within [0, 360)"""
    # initialize output angle
    theta_out = theta_in
    
    # convert negative angle
    while theta_out<0:
        theta_out += 360
        
    # convert positive angle
    while theta_out>=360:
        theta_out -=360
        
    return theta_out


#################################################################
# Sort 2D data (x,y) by polar angle in ascending order
# The origin is picked to be the center of mass of the data
# 
# Inputs:
#    x, y   : numpy arrays
# Outputs:
#    xa, ya : data sorted by polar angle, numpy array
def ASort(x, y):
    # center of mass
    x0 = np.average(x)
    y0 = np.average(y)
    
    # compute polar angle
    phi = np.arctan2(y-y0, x-x0)    
    # sort according to phi
    ind = phi.argsort()
    
    return x[ind], y[ind]
    

#################################################################
# Detect jump in an array
#
# Inputs:
#    array     : input data array
#    threshold : successive data larger than threshold is considered a jump
#    cap       : omit jumps larger than cap
# Output: 
#    index : a list of index after which jumps occur
#    midpt : a list of mid point values at the jump      
def Jump(array, threshold, cap=None):
    """Detect jumps in a search array.""" 
    
    # length of array
    N = len(array) 
    #assert(N>1) # need to be an array
    
    # initialize output
    index = []
    midpt = []
        
    # find index after which jump occurs
    if N>1:
        for i in range(N-1):
            # unpack array values
            a0 = array[i]
            a1 = array[i+1]
            
            # check cap
            c = True
            if cap!=None: c = abs(a0-a1)<cap
            
            # check threshold
            if abs(a0-a1)>threshold and c: 
                # record index
                index.append(i)
                # record mid point
                midpt.append((a0+a1)/2)
    
    return index, midpt


#################################################################
# insert or replace an element of mylist at the index location
# method = 0         : replace
#          otherwise : insert       
def InsertReplace(mylist, index, element, method):
    """Insert or replace an element at the index location of a list."""    
    if method == 0: # replace
        mylist[index] = element
    else: # insert
        mylist.insert(index, element)    
        
        
#################################################################
# Mask 1D data by asymptotic flags and invalid values
# Inputs:
#     x        : x array of the plot, length N
#     y        : y array of the plot, length N
#     f        : asymptotic flag, length N
#     fmax     : mask data for which f<fmax
#     yinvalid : y<yinvalid are masked, scalar     
def fyMask(x, y, f, fmax=2, yinvalid=None):
    """Mask 1D data by asymptotic flags and invalid values."""
    
    # mask invalid values
    if yinvalid != None:
        xy = np.ma.masked_where(y<=yinvalid, x)
        yy = np.ma.masked_where(y<=yinvalid, y)
    else:
        xy = x
        yy = y
    
    # mask data with f<fmax
    xm = np.ma.masked_where(f<fmax, xy)
    ym = np.ma.masked_where(f<fmax, yy)
    
    return xm, ym


#################################################################
# Use bisection to find x0 where fun(x) changes sign.
# x0 is assumed to be within (xl, xr)
# fun(x) is not assumed to be continous, but has only one sign change
# Bisect at least once.
#
# Inputs:
#     fun      : function hangle to evaluate fun(x)
#     xl       : left bound
#     xr       : right bound
#     epsilon  : convergence criteria is (xr-xl)<epsilon
#     Nmax     : maximum number of iterations before abortion
# Outputs:
#     x0       : estimated value of the location of sign change
#     error    : error = 0, no error
#                        1, potentially under resolved sign change
#                       -1, no sign difference at the two boundaries
#                        2, not convergent before abortion
#
# x related variables are either float for decimal 
def SignChangeBisec(fun, xl, xr, epsilon=1e-3, Nmax=100):
    """Use bisection to find x where f(x) changes sign."""
    
    # initial left and right values
    fl = np.sign(fun(xl))
    fr = np.sign(fun(xr))
    # initial mid point
    x0 = (xr + xl)/2 # bisect at least once 
    
    if fl*fr<0: # exist sign change
        # initial separation
        dx = xr - xl
        # initial number of iterations
        ix = 0
        # bisection
        while dx>epsilon and ix<Nmax:
            # update counter
            ix += 1
            # value at mid point
            f0 = np.sign(fun(x0))
            #print('ix=',ix,' fl=',fl,' f0=',f0,' fr=',fr)
            
            # update boundaries
            if f0==fr: # move zr left
                xr = x0
                #print('move xr left')
            elif f0==fl: # move zl right
                xl = x0
                #print('move xl right')
            else: 
                print('Warning: Potentially under resolved sign change!')
                error = 1
                break
            
            # update mid point
            x0 = (xr + xl)/2
            # update dx
            dx = xr - xl
            
        # check convergence
        if ix==Nmax: # not convergent before abortion
            error = 2 
        else: # convergent
            error = 0
        
    else:
        error = -1
        print('Warning: No sign difference at the two boundaries!')
    
    return x0, error   


# print with flush
def printf(message):
    """ Print message and flush output."""
    print(message, flush=True)
    
    
    
# identify the widest plateau of y=f(x), assuming f is continuous
#
# A plateau is where y is near zero.
# The widest plateau is where the curvature is the smallest.
# The curvature is estimated nonlocally using "width at half maximum".
#
# To find plateau at nonzero f, can use derivative as y
#
# Inputs:
#    y     : 1D array of length n
#            assuming x is evenly spaced
#    debug : if true, print debug message and plot diagnostic figure
# Outputs:
#    ix : index at the center of the plateau
def WidestPlateau(y, debug=False):
    """Identify the widest plateau of y=f(x) and return x index at center."""
    # length of data
    n = len(y)
    # absolute value of data
    ya = abs(y)
    if debug: print(f'size of array is {n}')
    
    # index of min and max of ya
    imin, imax = [], []
    for i in range(1,n-1):
        yl, y0, yr = ya[i-1], ya[i], ya[i+1]
        if y0<=yl and y0<=yr: # local minimum
            imin.append(i)
        if y0>yl and y0>yr: # local maximum
            imax.append(i)
    if debug: print(f'min index={imin}, max index={imax}')
                    
    # half min of maximum
    try: h = min(ya[imax])/2 # imax may be empty, no local maximum
    except ValueError: h = max([ya[0],ya[-1]])/2 # maximum at boundary
    if debug: print(f'half height ={h}')
                        
    # plateau width at half height 
    wid = []
    for i in imin: # imin shoulr be none empty
        # search left
        il = i-1
        while il>0 and ya[il]<=h:
            il -=1
            
        # search right
        ir = i+1
        while ir<n and ya[ir]<=h:
            ir +=1
            
        # characteristic width of this pleteau
        wid.append(ir-il)
    if debug: print(f'widths are {wid}')
            
    # widest plateau
    mw = max(wid)     
    iw = wid.index(mw)
    ix = imin[iw]
    
    # plot debug figure
    if debug:
        plt.figure()
        plt.plot(ya, color='k', marker='.')
        plt.plot(imin,ya[imin], linestyle='', marker='o', color='b')
        plt.plot(imax,ya[imax], linestyle='', marker='s', color='r')
        plt.axvline(x=ix, color='grey')
        plt.axhline(y=h, color='grey')
        plt.title('diagnostic figure of WidestPlateau')
    
    return ix
    
            
    
    
