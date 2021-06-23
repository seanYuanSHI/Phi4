# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:44:38 2020

@author: shi9
"""

# This script is used to carryout further 1D scan in A direction 
# by mannual specifying sample for a number of selected cases in batch

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')


import SE
import QP
import RA
#import Auxiliary

#import matplotlib.pyplot as plt
import numpy as np
#from IPython import get_ipython
from datetime import datetime

from decimal import Decimal, getcontext

######################################
# parameters needed by all modes 
# for SolveUntil
Ntmax = 2**24
epsilon = 1e-2  

# decimal precision
prec = 28
getcontext().prec = prec # default is 28
# for Q convergence during refinement in QP phase space
epsilon_bisection_Q = 10**(-prec+5) # default none, which means auto

# number of gridsfor each scan
NA = 4

######################################
# initialize empty diactionary
paras = {}
# list of parameters in the following order
# iR, Amin, Amax, sample dictionary
# sd = None: when sample is not specified
# sd = {...}: specify sample parameters, always have Ny=1
###############
paras['2']={# example for small R
            '0': [{'min':0.2, 'max':1.2, 
                   'sd':{'q0D':Decimal('0.005'), 
                          'Lx':4, 'Nx':16}}],
            '8': [{'min':5.7, 'max':10.7, 'sd':None}],
            # 0310 phase
            '47': [{'min':8.66, 'max':8.677, 'sd':None}],
            '48': [{'min':8.75, 'max':8.8, 'sd':None},
                   {'min':8.8, 'max':8.88, 'sd':None},
                   {'min':8.25, 'max':10.12, 'sd':None},
                   {'min':8.65, 'max':8.83, 'sd':None}],
            '49': [{'min':8.88, 'max':8.93, 'sd':None}],
            '52': [{'min':9.17, 'max':9.48, 'sd':None}],
            '59': [{'min':11.17, 'max':11.034, 'sd':None}],
            '60': [{'min':11.29, 'max':11.3, 'sd':None},
                   {'min':9.7, 'max':11, 'sd':None}],
            '61': [{'min':11.575, 'max':11.578, 'sd':None}],
            '62': [{'min':11.8712, 'max':11.8715, 'sd':None}],
            '63': [{'min':12.1792, 'max':12.17925, 
                    'sd':{'q0D':Decimal('0.005'), 
                          'Lx':0.02, 'Nx':256}}],
            # 2110 phase
            '64': [{'min':12.5, 'max':12.502, 'sd':None}],
            # example for large R
            # '18':[{'min':72, 'max':221, 'sd':None},
            #       {'min':40, 'max':81, 'sd':None},
            #       {'min':43, 'max':57, 'sd':None},                  
            #       {'min':44, 'max':49, 
            #         'sd':{'q0D':Decimal('-1.0435'), 
            #               'Lx':0.003, 'Nx':256}},
            #       {'min':44, 'max':45.8, 
            #        'sd':{'q0D':Decimal('-1.042'), 
            #              'Lx':0.002, 'Nx':256}},
            #       {'min':44.2, 'max':44.4, 
            #        'sd':{'q0D':Decimal('-1.04147'), 
            #              'Lx':0.0002, 'Nx':256}},
            #       {'min':44.1246, 'max':44.2203, 
            #        'sd':{'q0D':Decimal('-1.04137'), 
            #              'Lx':0.0002, 'Nx':256}}],           
            '17':[{'min':28, 'max':126, 'sd':None},
                  {'min':26, 'max':52, 'sd':None},
                  {'min':30, 'max':41, 'sd':None},   
                  {'min':105, 'max':119, 'sd':None},  
                  {'min':110, 'max':113.5, 'sd':None},  
                  {'min':31, 'max':40, 
                   'sd':{'q0D':Decimal('-1.0575'), 
                         'Lx':0.02, 'Nx':256}}],                  
            }

###############
paras['3']={# 0310 phase, left of critical point
            '46':[{'min':176, 'max':188, 'sd':None},
                  {'min':180, 'max':183, 'sd':None}],
            '45': [{'min':172.2, 'max':172.6, 'sd':None}],
            # 0310 phase, right of critical point
            '54': [{'min':296, 'max':296.6, 'sd':None}],
            '55': [{'min':316.3, 'max':316.6, 'sd':None}],
            '56': [{'min':337.8, 'max':338.2, 'sd':None}],
            '57': [{'min':361.45, 'max':361.57, 'sd':None}],
            '58': [{'min':386.88, 'max':386.92, 'sd':None}],
            # example of 1210 phase
            '49': [{'min':208.5, 'max':223, 'sd':None}],
            # 2110 phase
            '59': [{'min':414.33, 'max':414.34, 
                    'sd':{'q0D':Decimal('0.008'), 
                          'Lx':0.02, 'Nx':256}}],
            '60': [{'min':443.95, 'max':444.05, 'sd':None}],
            '61': [{'min':475.8, 'max':476.3, 'sd':None}],
            '62': [{'min':510.5, 'max':511.2, 'sd':None}],
            '63': [{'min':548.2, 'max':549.2, 'sd':None}],
            '64': [{'min':588.6, 'max':590, 'sd':None}],
            '65': [{'min':632.8, 'max':634.4, 'sd':None}],
            '66': [{'min':681.2, 'max':682.8, 'sd':None}],
            # example for large R
            '18':[{'min':2000, 'max':7500, 'sd':None},
                  {'min':2687, 'max':4000, 'sd':None},
                  {'min':2645, 'max':2488, 
                   'sd':{'q0D':Decimal('-1.087'), 
                         'Lx':0.02, 'Nx':256}},
                  {'min':2767, 'max':2876, 
                   'sd':{'q0D':Decimal('-1.079'), 
                         'Lx':0.002, 'Nx':256}},
                  {'min':2792, 'max':2806, 
                   'sd':{'q0D':Decimal('-1.0786'), 
                         'Lx':0.0005, 'Nx':256}}],                  
            }
            

# dimension of the problem    
#dims = [2, 3] 
dims = [2]
    
 ########################
for dim in dims:
    # file head
    fpath = f'../../../data/{dim}D/batch_e{epsilon}/raw/'
    # fname tail
    ftail = f'_N{Ntmax}_e{epsilon}.txt'    
    
    # print start time 
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f'Refine {dim}D points started on ', dt_string)
    print('Data is loaded and save in', fpath)
    
    # dictionary of parameters
    para = paras[str(dim)]
    
    # compute all cases
    for iR in para:
        # current list
        lp = para[iR] 
        
        # file name
        fname = fpath + f'Point{dim}D_iR{iR}' + ftail   
        
        # read R from file
        lines = np.loadtxt(fname, comments="#", delimiter=",", unpack=True) 
        R = lines[0,0]
        print(f'iR={iR}, read R from file: R={R}')    
        
        # compute all cases in the list
        Nc = len(lp)
        for ic in range(Nc):
            # parameter dictionary
            pd = lp[ic]
        
            # Amin
            Amin = pd['min']
            # Amax
            Amax = pd['max']
            # sample dictionary
            sd = pd['sd']            
            
            # report current status
            print(f'iR={iR}, scan Amin={Amin}, Amax={Amax}, sample=', sd)
            
            # prepare sample
            if sd==None:
                # no sample
                sample = None
            else:
                # add Ny=1
                sd['Ny'] = 1
                # initialize sample
                sample = QP.Sample(**sd) # kwargs           
            
            # prepare A search array
            Alist = 10**np.linspace(np.log10(Amin), np.log10(Amax), NA)
        
            # write file header
            fid = open(fname, "a+") 
            if sample==None: s = 10*'#'+'\n'
            else: s = f"## sample: q0={sd['q0D']}, Lx={sd['Lx']}, Nx={sd['Nx']} "+ 10*'#'+'\n'
            fid.write(s)          
            fid.close() 
              
            for iA in range(NA):
                A = Alist[iA]
                print(f'iA/NA={iA}/{NA}, A={A}')        
                # initialize objects
                ode = SE.ODE(R, A, dim)
                point = RA.Point(ode,Ntmax=Ntmax,epsilon=epsilon,sample=sample)
                # load point info
                pflag = point.Process(fname, epsilon_bisection_Q)
                print(f'pflag={pflag}')
            
    # print end time 
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print('Refine finished on ', dt_string)
        
