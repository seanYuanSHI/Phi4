# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 22:47:21 2021

@author: water
"""

# This script compute parameters in 3D
# for the Higgs field 

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../')

import numpy as np
import scipy.special as ss
import Asympt_nD as AS

################################
# basic parameters
vevGeV =246 # GeV, Higgs VEV
mhGeV = 125.2 # GeV, Higgs mass

muMeV = 2.2 # up quark mass in MeV
mdMeV = 4.7 # down quark mass in MeV
mpGeV = 0.938 # proton mass in GeV

# physical constants in SI unit
hbar34 = 1.0546 # hbar in units of 10^(-34) J*s
c8 = 2.9979 # speed of light in units of 10^8 m/s
GeV_J10 = 1.6022 # 1 GeV = 1.6022*10^(-10) J 
GeV_kg27 = 1.782662 # 1GeV/c^2 = 1.782*10^(-27) kg

Lsun26 = 3.828 # solar luminosity in 10^26 W
year7 = 3.15576 # Julian year in 10^7 seconds
#rnfm = 1.25 # nuclean size in 10^(-15)m
rnfm = 1.4 # nuclean size in 10^(-15)m

################################
# derived parameters
mudMeV = (muMeV+mdMeV)/2 # average mass of up and down quarks
lc18 = 1e2*hbar34*c8/mhGeV/GeV_J10 # Compton wavelength in 10^(-18) meter
H_EGeV = 4*np.pi*vevGeV**2/mhGeV # convert dimensionless energy to GeV
Lyr34 = 1e-1*Lsun26*year7 # solar energy release per year in 10^34 J
Rn = 1e3*rnfm/lc18 # normalized nucleon size


################################
# Convert fermion mass to dimensionless A
# mfGeV is the fermion mass in GeV
def Afermion(mfGeV, ifprint=True):
    """Convert mass to dimensionless source strength A for a single single fermion."""
    # compute A
    A = mfGeV*mhGeV/vevGeV**2
    # print in scientific format
    if ifprint: print('A='+"{:e}".format(A))
    
    return A


################################
# Convert baryon mass to dimensionless A
# M is the total mass
# unit of M is 'GeV', 'number', or 'kg'
# Estimate mf=mud for baryonic matter
def Abaryon(M, unit='GeV', ifprint=True):
    """Convert mass to dimensionless source strength A for baryoninc matter."""
    # check unit
    if unit=='GeV':
        MGeV = M
    elif unit=='number':
        # atomic number
        MGeV = M*mpGeV
    else: # assume M is in kg
        if unit!='kg': print('Unknown unit! Assume M is in kg')
        unit = 'kg'
        MGeV = 1e27*M/GeV_kg27    
        
    # number of quarks
    N = 3*MGeV/mpGeV
    # average quark mass
    mfGeV = 1e-3*mudMeV
    
    if ifprint: print('M='+"{:e} ".format(M)+unit)
    A = Afermion(N*mfGeV, ifprint=ifprint)
    
    return A


################################
# Convert source size to dimensionless R
# a is source size
# unit='GeV': compute Compton wavelength
# unit='meter': normalize to lc18
# unit='number': compute for nucleons
def Rnormal(a, unit='meter', ifprint=True):
    """Compute dimensionless source size."""
    # check unit
    if unit=='GeV':
        R = mhGeV/a
    elif unit=='number':
        R = a**(1/3)*Rn
    else: # assume a in meter
        if unit!='meter': print('Unknown unit! Assume a is in meter')
        unit = 'meter'
        R = 1e18*a/lc18
        
    if ifprint:
        # formatted strings
        As = "{:e}".format(a)
        Rs = "{:e}".format(R)
        print(f'a={As} {unit}, R={Rs}')
    
    return R
 
################################
# When in 1210 phase, compute light horizon radius in units of R.
# If there is no light horizon, return 0
def rhoR_A1210(R, A, ifprint=True):
    """Compute light horizon radius rho/R in 1210 phase."""
    # 1010-1210 phase boundary
    Ac = AS.Ac_Rl(R, dim=3)
    # ratio A/Ac
    AAc = A/Ac
    # report A/Ac
    if ifprint: print('A/Ac='+"{:e}".format(AAc))
    
    if AAc<1: # no light horizon
        if ifprint: print('1010 phase, no light horizon')
        rhoRs, rhoRl = 0, 0
    else:
        # source at t=0 
        S0R = A/R**2/np.pi**(3/2)  
        # smallness parameter
        epsilon = 2/3/S0R
        
        # 1st order asymptotic solutions to x*exp(-x^2)=epsilon<<1
        rhoRs = AS.x1Small(epsilon) # smaller root
        rhoRl = AS.x1Large(epsilon) # larger root
        
        if ifprint: 
            print('1210 phase')
            # convert to meter and format string
            rs = "{:e}".format(1e-18*R*rhoRs*lc18)
            print(f'Small rho/R={rhoRs}, rs={rs} meter')
            rl = "{:e}".format(1e-18*R*rhoRl*lc18)
            print(f'Large rho/R={rhoRl}, rl={rl} meter')
 
    return rhoRs, rhoRl
    
      
################################
# Convert dimensionless H to dimensionful energy
def H2E(H):    
    """Convert dimensionless H to dimensionful E."""
    # convert to GeV
    EGeV = H*H_EGeV    
    # convert to J
    EJ = 1e-10*EGeV*GeV_J10
    # convert to Lyr
    ELyr = 1e-34*EJ/Lyr34
    
    # formatted strings
    Hs = "{:e}".format(H)
    EGeVs = "{:e}".format(EGeV)
    EJs = "{:e}".format(EJ)
    ELyrs = "{:e}".format(ELyr)
    print(f'  H={Hs}\n  E={EGeVs} GeV \n   ={EJs} J\n   ={ELyrs} L*yr')   
    

################################
# Compute energy of solutions for R>>1
def EnergyRl(R, A):
    """Compute energy of field configurations."""
    # adiabatic energy
    H0 = A/4/np.pi
    print('Energy of adiabatic potential')
    H2E(H0)
    
    # compute hopping energy when in 1210 phase
    rhoRsl = rhoR_A1210(R, A, ifprint=False)
    labels=['inner', 'outer']
    nl = 0
    for rhoR in rhoRsl:
        if rhoR>0:
            rhoR2 = rhoR**2
            # adiabatic contribution
            Ha = H0*(2*ss.gammaincc(3/2, rhoR2)-1)
            # 1D hopping contribution
            H1 = 2/3*rhoR2*R**2
            # report total energy
            H = Ha + H1
            print(f'\nEnergy of {labels[nl]} hopping potential is')
            H2E(H)
            print('Energy gap E-E0')
            H2E(H+H0)
        # next label
        nl +=1
    
    

################################
# Process a dicionary of the type
# For macroscopic objects, specify size in meter and mass in kg
#   data={'type': 'baryon', 
#         'size': a, 'mass': M}
# For atomic nuclei, specify atomic mass number
#   data={'type': 'nucleus', 
#         'number': N}
# For single fermion, specify mass in GeV
#   data={'type':'fermion',
#         'mass': M}
def ProcessData(data):
    """Process and report data for fermion, nucleus, and baryon."""
    # check source type and compute R and A
    source = data['type']
    if source=='fermion': # single fermion
        print('Source type: fermion')
        # extract parameters
        mfGeV = data['mass']   
        # normalized R, A
        R = Rnormal(mfGeV, unit='GeV')
        A = Afermion(mfGeV)
        # report Compton wavelength in meter
        print(f'a={1e-18*R*lc18} meter')
    elif source=='nucleus': # single nucleus
        print('Source type: nucleus')
        # extract parameters
        N = data['number']
        # normalized R, A
        R = Rnormal(N, unit='number')
        A = Abaryon(N, unit='number')
        # report size and mass
        print(f'a={1e-18*R*lc18} meter, M={N*mpGeV} GeV')
    else:  # baryonic matter
        print('Source type: baryonic matter')
        # extract parameters
        a = data['size']
        M = data['mass']
        # normalized R, A
        R = Rnormal(a, unit='meter')
        A = Abaryon(M, unit='kg')

 
    # process info 
    if R<1: # small source
        print('Small source. See Energy files for rc and E')
        if R<0.5: # Check 1010-0110 phase boundary
            Ac = AS.Ac_Rs(R, dim=3)
            print('1010-0110 A/Ac='+"{:e}".format(A/Ac))
        else: 
            print('See numerical phase diagram for A_1010^0110')
            
    else: # large source
        print('Large source')
        # Check 1010-1210 phase boundary
        rhoR_A1210(R, A)
        # Energy of adiabatic and hopping solutions
        EnergyRl(R, A)


 

