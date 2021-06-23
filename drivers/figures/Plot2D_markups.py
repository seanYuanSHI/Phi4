# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:14:49 2020

@author: Yuan Shi
"""
from decimal import Decimal

# This script contains parameters for marking up
# Plot2D, which plot figures for the 2D case

###################################################
# parameters for phase diagram
Rkeys = ['1.0','1.9','2.1','13.3']
# decimal precision
Rprec = {'1.0':28, '4.0':28, '4.3':28, '17.8':155}

# R values, file index, and color
Rs = {'1.0': {'iR':12, 'color': 'r'},
      '1.9': {'iR':25, 'color': 'm'},
      '2.1': {'iR':30, 'color': 'g'},
      '13.3': {'iR':54, 'color': 'b'}}

# RA values to draw sampling dots
RAs = {'1.0':  [2.3, 8], 
      '1.9': [2.302969187006411, 8.4, 8.902598196495218, 13.692460376260554], 
      '2.1': [4, 9.25, 9.4, 15],
      '13.3':  [10, 34.39008246926193, 115.00851673937414, 140]}

# colors for drawing RA points
RAcolors = {'1.0'  : ['orange', 'r'],
            '1.9'  : ['pink', 'magenta','m','purple'],
            '2.1'  : ['lightblue', 'skyblue', 'b', 'black'],
            '13.3' : ['olive','g','lawngreen','mediumspringgreen']}


###################################################
# parameters for representative solutions
#epsilon=1e-2
# tf/R
tfRrep = {'1.9': 10, '13.3': 3}
# decimal precision
Rprec = {'1.0':28, '1.9':28, '2.1':28, '13.3':28}
# representative cases, 
Rrep = {'1.9':1.9007373500989018,
        '13.3':13.33521432163324}
# Rkey: Aindex
RArep = {'1.9':[0,2,3],
        '13.3':[1,2]} 
# initial q values
qrep = {'1.9':
        [# R=1.9..., A=2.3, pflag = 1010
         [Decimal('-1.107524470106669969345447992'), 
          Decimal('0.870276964140998643414443359')], 
         # R=1.9..., A=8.9, pflag = 0310
         [Decimal('-1.346142622459217708444026071'), 
          Decimal('-0.791712731642429857994014198'), 
          Decimal('-0.502320467243958829026841585'), 
          Decimal('-0.1656463513169331814766948380')],
         # R=1.9..., A=13.7, pflag = 0110
         [Decimal('-1.481377198077869693268260694'), 
          Decimal('-1.396420388168101481189728077')]],
        '13.3':
        [# R=13.3, A=34.4, pflag=1210
         [Decimal('-1.055677397751206282514644810'), 
          Decimal('-1.055555200768673403086333356'), 
          Decimal('-1.053263168514936945052667648'), 
          Decimal('0.933580880765085513436752462')],                
         # R=13.3, A=115, pflag=2110
         [Decimal('-1.161603185636360119517607384'), 
          Decimal('-1.161603185123379053469645836'), 
          Decimal('0.04306319628718854995194789512'), 
          Decimal('0.639380243489063699259197650')]]}

# line styles
lrep = ['-', '--','-.',':']
# linewidth
wrep = [3,2.5,2]

###################################################
# additional phase correction
# dictionary of the form
# "Rkey:{'old':'pflag', 'new': 'pflag'}
apc={'2.740522449800859':{'old':'110', 'new': '310'}}
