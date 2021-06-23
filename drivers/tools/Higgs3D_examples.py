# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:38:29 2021

@author: water
"""

# This script run Higgs3D for a number of examples listed below
import Higgs3D as H3
import numpy as np

################################
# fermions, mass in GeV
flist = [{'name':'electron', 'mass': 0.511e-3},
         {'name':'top quark', 'mass': 173},
         {'name':'hypothetical', 'mass': 1e3}
        ]
# add type
for d in flist: d['type']='fermion'

################################
# nucleus, speficy atomic mass number
nlist = [{'name':'nucleon', 'number': 1},
         {'name':'carbon', 'number': 12},
         {'name':'gold', 'number': 197}
        ]
# add type
for d in nlist: d['type']='nucleus'

################################
# baryonic matter, size in meter, mass in kg
# solar mass in kg
Msun = 1.98847e30
# solar radius in meter
asun = 6.96e8
# parsec in meter
parsec = 3.0857e16
# uranium density in kg/m^3
rhoU = 1.9e4

blist = [{'name':'fusion fule at stagnation', 'size':5e-5, 'mass':1e-6},
         {'name':'uranium ball', 'size':1, 'mass':4*np.pi/3*rhoU},
         {'name':'neutron star', 'size':1e4, 'mass':1.5*Msun},
         {'name':'Earth', 'size':6.37e6, 'mass':5.97e24},
         {'name':'Jupiter', 'size':7e7, 'mass':1.9e27},
         {'name':'white dwarf', 'size':0.013*asun, 'mass':0.6*Msun},
         {'name':'M star', 'size':0.7*asun, 'mass':0.4*Msun},
         {'name':'G star (sun)', 'size':asun, 'mass':Msun},
         {'name':'O star', 'size':6.6*asun, 'mass':16*Msun},
         {'name':'red giant', 'size':100*asun, 'mass':6*Msun},
         {'name':'super giant', 'size':500*asun, 'mass':15*Msun},
         {'name':'dwarf galaxy halo', 'size':25e3*parsec, 'mass':1e9*Msun,
          'luminous size': 1e3*parsec, 'luminous mass':1e6*Msun},
         {'name':'milky way halo', 'size':280e3*parsec, 'mass':1e12*Msun,
          'luminous size':1.5e4*parsec, 'luminous mass':1e11*Msun},
        ]
# add type
for d in blist: d['type']='baryon'

################################
# process data
dlist = [flist, nlist, blist]
for l in dlist:
    for d in l:
        name = d['name']
        print('\n\n'+20*'#'+f'\nSource: {name}')
        H3.ProcessData(d)
    
