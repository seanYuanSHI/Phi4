# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:10:58 2020

@author: Yuan Shi
"""

# This script plot all raw figures and save to file
# compare two resolutions

# This script serves as a scratch paper
#import os
#os.chdir('../')
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')

import matplotlib.pyplot as plt
import RA


##############################################
# dimension of the problem    
dim = 2

# for SolveUntil
Ntmax = 2**24  

# figure path
#path = f'../../../data/{dim}D/batch_e{epsilon}/'
path = f'../../../data/{dim}D/Figures/Comparison/'
#path = f'../../../data/{dim}D/'
# file path
pathL = f'../../../data/{dim}D/batch_e0.01/'
pathH = f'../../../data/{dim}D/batch_e0.001/'
# file taile
ftailL = f'_N{Ntmax}_e0.01'
ftailH = f'_N{Ntmax}_e0.001'
# file name dictionary
fdictL = {'dim':dim, 'Ntmax':Ntmax, 'epsilon':1e-2, 'path':pathL}   
fdictH = {'dim':dim, 'Ntmax':Ntmax, 'epsilon':1e-3, 'path':pathH}     
    
# parameters for making plots
dstyleL={'fontsize':24, 'color':'b', 'linestyle':'', 'markersize':8, 'label':'e0.01'}
dstyleH={'fontsize':24, 'color':'r', 'linestyle':'', 'markersize':8, 'label':'e0.001'}

iRmin = 0
iRmax = 59

##############################################
# plot and save figures
for iR in range(iRmin,iRmax):
    print(f'iR={iR}')
    
    # file name
    fnameL = pathL + f'/raw/Point{dim}D_iR{iR}' + ftailL + '.txt'
    fnameH = pathH + f'/raw/Point{dim}D_iR{iR}' + ftailH + '.txt'
    
    fnameFig = path + f'Point{dim}D_iR{iR}.png'
    
    # initialize new figure
    fig = plt.figure(figsize=(20,10))
    # plot figure
    RA.plotPoints(iR, fdictL, varname='full', dstyle=dstyleL)
    RA.plotPoints(iR, fdictH, varname='full', dstyle=dstyleH)
 
    # show legend    
    plt.legend(loc='best')
    
    # save figure        
    plt.savefig(fnameFig)
    plt.close(fig)