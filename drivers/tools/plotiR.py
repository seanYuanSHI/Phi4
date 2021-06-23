# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:10:58 2020

@author: Yuan Shi
"""

# This script plot all raw figures and save to file


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
epsilon = 1e-2

# file path
#path = './'
path = f'../../../data/{dim}D/batch_e{epsilon}/'
#path = f'../data/{dim}D/batch/raw/'
# file taile
ftail = f'_N{Ntmax}_e{epsilon}'

# file name dictionary
fdict = {'dim':dim, 'Ntmax':Ntmax, 'epsilon':epsilon, 'path':path}    
# parameters for plotPoints
dstyle={'fontsize':24, 'color':'b', 'linestyle':'', 'markersize':8}

iRmin = 0
iRmax = 62

##############################################
# plot and save figures
for iR in range(iRmin,iRmax):
    print(f'iR={iR}')
    
    # file name
    fnamePT = path + f'/raw/Point{dim}D_iR{iR}' + ftail + '.txt'
    fnameFig = path + f'/Figures/raw/Point{dim}D_iR{iR}' + ftail + '.png'
    
    # initialize new figure
    fig = plt.figure(figsize=(20,10))
    # plot figure
    RA.plotPoints(iR, fdict, dstyle=dstyle)
    
    # save figure        
    plt.savefig(fnameFig)
    plt.close(fig)