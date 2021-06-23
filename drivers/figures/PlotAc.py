# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 20:57:29 2020

@author: Yuan Shi
"""

# This script plot upper boundaries of 1D/2D/3D RA phase diagram 
# data are generated using ./tools/Acupper.py

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')

#import RA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)


epsilon = 1e-2
Ntmax = 2**24
fontsize = 12

# file path
fpath = '../../../data/'
# file tail
ftail = f'_N{Ntmax}_e{epsilon}.txt'

# plot colors
colors = ['red','green','blue']
# dimensions
dims = [1,2,3]
# max Rs
Rsmax = [1.5, 0.6, 0.8]

plt.figure(num=None, figsize=(10, 15))
for i in range(len(dims)):    
    # dimension
    dim = dims[i]
    
    #########
    # boundary from numerical solution data 
    # file name
    fname = fpath + f'{dim}D/batch_e{epsilon}/Ac{dim}D_upper' + ftail
    # read upper boundary file
    try:
        lines = np.loadtxt(fname, comments="#",delimiter=",",unpack=True)
    except OSError:
        print(f'Ac{dim}D_upper file nonexists. Run ../tools/AcUpper.py first!')
    else:                       
        # source size
        R = lines[0]
        # normalization factor
        RG = R**dim*np.pi**(dim/2)
        
        # source amplitude
        A = lines[1]
        # normalized amplitude
        AG = A/RG
         
        # plot upper boundary
        plt.loglog(R, AG, color=colors[i], label=f'{dim}D', marker='.')
        
        #########
        # asymptotic boundary at R<<1
        Rs = np.linspace(min(R), Rsmax[i], len(R))
        # normalization factor
        RsG = Rs**dim*np.pi**(dim/2)
        
        if dim == 1:
            As = np.ones_like(Rs)
        elif dim ==2:
            As = 2*np.pi/(np.log(2/Rs) - np.euler_gamma/2)
        else: # dim==3
            As = 2*np.pi*Rs/(1/np.sqrt(np.pi) - Rs/2)
            
        # normalized As
        AsG = As/RsG
        
        # plot asymptotic upper boundary
        plt.loglog(Rs, AsG, color=colors[i], linestyle=':')
    
    
# mark critical lines
plt.axhline(y=1/3/np.sqrt(3), color='k',linestyle=':')

plt.xlabel(r'$R$', fontsize=fontsize)
plt.ylabel(r'$A/(R^D\pi^{D/2})$', fontsize=fontsize)
plt.title('Phase diagram')

plt.legend(loc='best', fontsize=fontsize) 
plt.ylim((0.1,1e5))

# save figure
plt.savefig(fpath + 'RA_all.png')
