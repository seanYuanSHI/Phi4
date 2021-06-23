# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:47:59 2020

@author: Yuan Shi
"""

# This script compute upper phase boundary from AcPoint dictionary files
# plot upper boundary and compare with complete phase plot

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')

import RA
import numpy as np
import matplotlib.pyplot as plt


############################
flag = 0

#flag = 0 # compute data and plot figure, use when AcPoint is updated
#flag = 1 # plot figures, compute data only if upper file dosen't exist

############################

# dimentions of the problem
dim = 1

epsilon = 1e-2
Ntmax = 2**24

# file path
fpath = f'../../../data/{dim}D/batch_e{epsilon}/'
# file tail
ftail = f'_N{Ntmax}_e{epsilon}.txt'

# plot style
dstyle = {'linestyle':'', 'marker':'.'}


###################################                  
# read dictionary file
print('Reading phase dictionary in ', fpath)
fname = fpath + f'AcPoint{dim}D' + ftail
try:
    Rpdict = RA.readAcPoint(fname)
except FileNotFoundError:
    print(f'AcPoint{dim}D nonexists! Run ../plots/Plot{dim}D.py first!')
else:    
    # plot complete phase diagram
    RA.plotAcPoint(Rpdict,dstyle=dstyle)

    # name for upper boundary file
    fname = fpath + f'Ac{dim}D_upper' + ftail
     
    # check if upper file exist
    try:
        data = np.loadtxt(fname, comments="#",delimiter=",",unpack=True)
    except OSError:
        ifnotexist = True
        print('Ac upper file does not exist!')
    else:
        ifnotexist = False
        print('Ac upper file found')        
    
    # compute upper boundary of flag=0 or ifnotexist=True
    if flag==0 or ifnotexist:  
        if flag==0: print('Updating upper boundary from phase diactionary...')
        if ifnotexist: print('Computing upper boundary from phase diactionary...')
        # initialize list
        R, A = [], []
        
        # read data from dictionary
        for Rkey in Rpdict:
            # store R
            R.append(float(Rkey))
            
            # phase dictionary
            pd = Rpdict[Rkey]
            # initialize empty list for phase boundary
            Ac = []
            # load A boundaries
            for pkey in pd:
                Ac.append(pd[pkey])
            # store maximum
            A.append(max(Ac))
            
        
        # prepare data
        data0 = np.array([R, A])
        # sort data according to R
        data = data0[:, data0[0].argsort()]
    
        # write file
        # open file, overwrite
        fid = open(fname, "w+")           
        # write content
        for iR in range(len(Rpdict)):
            # string for this line
            s = f'{data[0,iR]}, {data[1,iR]} \n'
            # write to file
            fid.write(s)
        # close file           
        fid.close()        
    
    # plot upper boundary
    plt.plot(data[0],data[1],color='k')