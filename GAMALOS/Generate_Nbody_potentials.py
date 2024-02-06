#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np 
import pandas as pd
from scipy.interpolate import UnivariateSpline as us
import matplotlib.pyplot as plt


#should we use this also in this case?

def cdf_disk(R,Rd, Rmin=0.1, Rmax=30):
    """
    Cumulative distribution of the  exponential 2D density profile pdf
    Sigma(R) =  e^{-R/Rd}  / (2*pi*Rd^2 (e^(-Rmin/Rd)(1+Rmin/Rd) - e^(-Rmax/Rd)(1+Rmax/Rd)) )
    cdf = int^R_Rmin 2 pi R Sigma(R) dR  = e^(-Rmin/Rd)(1+Rmin/Rd)-e^(-R/Rd)(1+R/Rd) / (e^(-Rmin/Rd)(1+Rmin/Rd) - e^(-Rmax/Rd)(1+Rmax/Rd))
    
    @param R:  cylindrical radius
    @param Rd: exponential scale length
    @param Rmin: minimum radius of the domain
    @param Rmax: maximum radius of the domain
    @return: the cdf at R
    
    @ NOTICE, R,Rd, Rmin, Rmax must have the same units
    
    """
    
    xmin = Rmin/Rd
    xmax = Rmax/Rd
    
    minexp = np.exp(-xmin)*(1+xmin)
    
    norm = minexp - np.exp(-xmax)*(1+xmax)
    
    x=R/Rd
    
    return (minexp-np.exp(-x)*(1+x))/norm

#should we use this also in this case?

def R_from_u(u,Rd,Rmin=0.1,Rmax=30):
    """
    Draw the radius sampling from the cdf of and exponential 2D density profile pdf
    using the inverse sampling an random uniformly generated number u
    
    @param u:  a number between 0 and 1 generted from a uniform distribution
    @param Rd: exponential scale length
    @param Rmin: minimum radius of disc to consider for the  exctraction
    @param Rmax: maximum radius of disc to consider for the  exctraction
    @return: the cylindrical radius R
    
    @ NOTICE, Rd, Rmin, Rmax must have the same units, R will be in these same units
    
    
    """
    
    # Notice the cdf of fhte 2D density profile is not invertible analytically, 
    # so we interpolate it.
    
    # Generate the interpolaton grid from Rmin to Rmaz
    xx=np.linspace(Rmin,Rmax,100000)
    
    # Estimate the cdf at xx
    cdf=cdf_disk(xx,Rd,Rmin=Rmin,Rmax=Rmax)
    
    # Interpolate the inverse function 
    fus=us(cdf,xx,k=2,s=0)
    
    # Finally get R from u
    return fus(u)


def plummer_and_hernquist_vel_Nbody(r,Mtot,a):
    """
    Return the expected circular velocity for a body orbiting in a plummer sphere
    assumed at the centre of the frame of reference.
    
    @param r: spherical radius (r=0, means centre of the plummer sphere) 
    @param Mtot: total mass of the plummer sphere 
    @param a: plummer scale length  in kpc
    @return the circular velocity at radius r
    
    @NOTICE: all the units are in N-body units and G=1
    
    
    """
        
    Mr = Mtot*(r*r*r)/(r*r + a*a)**1.5  # Plummer mass profile
    Mh = Mtot*(r*r)/(r+a)**2            # Hernquist mass profile
    Mt = mr + Mh                        # Total mass profile
    Vc = np.sqrt(Mt/r)                  # circular velocity at radius r 
    
    return Vc