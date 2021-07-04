##############################################################
This collection of programs provide tools for solving the static phi4 problem. The functions are organized according to their purposes. Detailed descriptions of the functions can be found as comments in the scripts. The programs were written in Python 3.7

This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344. This code is released under LLNL-MI-823772.


##########
SE.py : Solve the ODE with initial conditions (q,p), for a given source term of parameter (R,A)

The key function is "SE.Solution.SolveUntil", which is used to determine whether a solution is oscillatory or diverging. Its key input parameters are "epsilon" (determines step size) and "Ntmax" (determines when to abort). The solution type is labled by "flag", whose options are explained by "SE.Solution.terminatMsg".


##########
QP.py : Identify phase space island boundary/separatrix of the ODE for a given source term of parameter (R,A)

The key function is "QP.Island.xCritical", which zooms in x values near which the "flag" switches. Its key static parameters are "epsilon_bisection" (convergence criteria for bisection critical values) and "Nsample_refine" (determines number sampling points when distinguishing separatrix from narrow island). Here, (x,y) refers to a rotated search box in the qp phase space. The search box is specified by the "Sample" class. 


##########
RA.py  : Scan (R,A) parameters of the source to determine the phase diagram of phi4 potentials.

Regions of the phase diagram are labeled by 4-digit numbers as explained in the "Point" class. Due to the stiffness of the problem, unresolved narrow islands may be misidentified as separatrix, which results in misclassifications of phase regions. One should mannnually inspect the "Point" raw data files, and may correct phase identification mannully in "ABoundary", which post processes raw data files. 

##########
Anxiliary.py : Contains generic utility functions 

Asympt_1D : Contains asymptotic solutions for the 1D case in various R and A limits

Asympt_nD : Contains asymptotic solutions for the 2D and 3D cases in various R and A limits



##############################################################
Member classes and functions may be used alone. Their typical usages are demonstrated in driver programs.

##########
drivers/driver_SE.py : Solve the initial-value problem, compare terms in the ODE, test for convergence, test termination reasons, and plot figures

drivers/driver_QP.py : 1D/2D scan in qp phase space, bisection critical points, find phase space island/separatrix, and plot figures

drivers/driver_RA.py : Run 1D scan in A direction, compute 2D RA phase diagram, and plot figures 

drivers/Energy.py : Refine finite energy potentials, estimate their energy, and plot figures



##############################################################
Scripts in ./drivers/plots/ are used to generate figures for the paper by plotting 1D/2D/3D figures and asymptotic figures. These scripts assume data has already been generated and is stored in folders like ../data/1D/batch_e0.001/raw/. 

Example data can be found at https://doi.org/10.5281/zenodo.5021378, which is used for figures in the paper "Nonperturbative phi4 potentials: Phase transitions and light horizons".

For 0.01<R<1000, reasonable convergence can already be obtained for epsilon=0.01. However, since the ODE is stiff, the convergence is always poor near boundaries/separatrics.

Parameters used for plotting the figures are saved within scripts.

##########
Plot1D.py : Plot 1D figures including RA phase diagram, QP phase space islands, initial qc = q(t=0) values, light horizon radius rc, representative solutions

PlotnD.py : Plot 2D and 3D figures including RA phase diagram, initial qc = q(t=0) values, light horizon radius rc, representative solutions

Plot2D_markups.py : Contains parameters for marking up Plot2D

Plot3D_markups.py : Contains parameters for marking up Plot3D

PlotAc.py : Plot upper boundaries of 1D/2D/3D RA phase diagram and compare them with asymptotic expressions

PlotAsymptotics.py : For R>>1 and R<<1, plot example solutions, initial qc = q(t=0) values, light horizon radius rc, normalized energy H, and compare them with asymptotic results.
 

##############################################################
Scripts in ./drivers/asymptotics/ are used to plot 1D/2D/3D figures and compare numerical and asymptotic results in various limits.

Parameters used for plotting the figures are saved within scripts

##########
q_smallR.py : Compare numerical and asymptotic solutions when the R is small and A is large 

q_critical.py : Compare numerical and asymptotic solutions when the R is small and A is near critical. The critical solution satisfies q(t=0)=0 and q(t=infty)=1

q_largeR.py : Compare numerical and asymptotic solutions when R is large and A is large

qc_rc.py : Compare numerically obtained initial qc = q(t=0) values and light horizon radius rc with asymptotic expressions


##############################################################
Scripts in ./tools/ are drivers that serve specific purposes. 
The most useful tools for end users are indicated by *

### for computating #######
ACritical.py : Search for critical A values for given q0 and R 

BoundaryBatch: Run QP island boundary/separatrix search with mannually specifyied samples in batch.

Higgs3D.py : Contain functions for computing source parameters in 3D for the Higgs field.

*Higgs3D_examples.py : Driver for Higgs3D.py. Contain source examples and generate most entries of Table I in the paper.

PhaseBatch.py : Run RA.Phase for a number of user-specified cases in batch.

PointBatch.py : Run RA.Point to scan A using user-specified samples in batch.

*qCritical.py : Search for critical q0 values for given A and R

ScanQ.py : Zoom in QP island boundary/separatrix in a user-specified search interval along the q direction


### for plotting ##########
*AcUpper.py :  Extract and plot upper RA phase boundaries from raw data files

plotCases.py : plot solutions of the ODE for a list of initial conditions.

*plotiR.py : Plot all raw "Point" data and save figures to file

plotiRcomp.py : Plot all raw "Point" data at two resolutions and save comparison figures to file
