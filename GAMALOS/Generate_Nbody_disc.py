#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np 
import pandas as pd
from scipy.interpolate import UnivariateSpline as us
import matplotlib.pyplot as plt


# In[37]:


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

def plummer_vel_Nbody(r,Mtot,a):
    """
    Return the expected circular velocity for a body orbiting in a plummer sphere
    assumed at the centre of the frame of reference.
    
    @param r: spherical radius (r=0, means centre of the plummer sphere) 
    @param Mtot: total mass of the plummer sphere 
    @param a: plummer scale length  in kpc
    @return the circular velocity at radius r
    
    @NOTICE: all the units are in N-body units and G=1
    
    
    """
        
    Mr = Mtot*(r*r*r)/(r*r + a*a)**1.5 # Plummer mass profile
    Vc = np.sqrt(Mr/r) # circular velocity at radius r 
    
    return Vc


def generate_thin_disc_nbody(N, Mgalaxy=1, Rd=3, Mtracers=1e-11, a_plummer=5, Rmin=0.1, Rmax=30):
    """
    Generate a simplified realisation of a disc galaxy. 
    The total mass distribuition of the galaxy is representes by a central body (always
    as posiiton 0 of the arrays) with mass equal to the mass of the galaxy. 
    The disc tracers are sampled considering an exponential disc distribution. 
    The velocity of all the particles is set so that they are in a circular orbit consider 
    a total distribution of matter following a Plummer sphere. 
    This simplified realisation can bse considered as composed by disc tracers and a central
    body generating a plummer potential.
    
    @param N: total number of particles to draw. NOTICE the actual number of particles
    will be always N+1, because the first particle is the central body containing the total
    mass of the galaxy
    @param Mgalaxy: total mass of the galaxy (in Nbody units) this will be the mass of the central body
    @param Rd: scale length of the explonentail disc (in Nbody units)
    @param Mtracers: mass of the disc tracers (in Nbody units), this should be many order of magnitude smaller than the Mgalaxy
    @param a_plummer: scale length (in Nbody units) of the plummer sphere representing the total mass of the galaxy 
    @param Rmin: minimum radius of disc to consider for the  exctraction of the disc tracer
    @param Rmax: maximum radius of disc to consider for the  exctraction of the disc tracer
    @return:
        - pos array: (N+1)x3 numpy array containing the Cartesian position (in Nbody units) of the particles 
        - vel array: (N+1)x3 numpy array containing the Cartesian velocity (in Nbody units) of the particles 
        - mass array: N+1 1D numpy array containing the mass of the particles (in Nbody units) 
    
    @ NOTICE: in order to integrate this Nbody realisaiton, the force estiamator needs to use
    a plummer softening kernel with the SOFTENING PARAMETER EQUAL TO THE PARAMETER a_plummer
    @ NOTICE-2: all the i/o values are in Nbody units and assume G=1, they can be used directly in fireworks
    
    """
    
    pos  = np.zeros(shape=(N+1,3)) # To store positions 
    vel  = np.zeros(shape=(N+1,3)) # To store velocities 
    mass = np.ones(N+1)*Mtracers   # Containg the mass of the tracers
    
    #Step0, generate, position, velocity and mass of the central body
    mass[0] = Mgalaxy # position and velocity are 0, not need to modify
    
    #Step1, generate the  position on the disc
    # A random sample the radius 
    u    = np.random.uniform(0,1,int(N)) # generate N random deviate from 0 to 1
    Rcyl = R_from_u(u,Rd=Rd,Rmin=Rmin,Rmax=Rmax)
    # Random sample the azimuthal  angle
    phi  = np.random.uniform(0,2*np.pi,N)
    # Transform from Cylindrical to Cartesian
    pos[1:,0] = Rcyl*np.cos(phi) # x
    pos[1:,1] = Rcyl*np.sin(phi) # y
    pos[1:,2] = 0 #  by default, partciles just in the disc plane
    
    #Step 2, set velocity
    Vphi     = plummer_vel_Nbody(Rcyl, Mtot=Mgalaxy, a=a_plummer) # circular velocity
    # Transform from cylindrical to cartesian 
    vel[1:,0] = -Vphi*np.sin(phi) #vx
    vel[1:,1] = Vphi*np.cos(phi) #vy
    vel[1:,2] = 0 # by default, no vertical motions
    
    return pos,vel,mass
    


# In[15]:


if __name__=="__main__":

    import pandas as pd 

    np.random.seed(42)
    a_plummer=5
    # Generate the Nbody realisation
    pos,vel,mass = generate_thin_disc_nbody(100000,a_plummer=a_plummer)

    # Put it in a DataFrame
    df=pd.DataFrame({
                "mass":mass,
                "x":pos[:,0],
                "y":pos[:,1],
                "z":pos[:,2],
                "vx":vel[:,0],
                "vy":vel[:,1],
                "vz":vel[:,2],
                })

    # Save it to a csv file
    df.to_csv("Nbody_disc.csv",index=False)


    # In[35]:


    # Example, how to read
    dfr = pd.read_csv("Nbody_disc.csv")
    # Take the first 100000 particles 
    dfr = dfr.iloc[:100000]

    # Cylindrical radius 
    Rcyl = np.sqrt(dfr.x*dfr.x + dfr.y*dfr.y)

    # Estimate the azimuthal velocity
    phi = np.arctan2(dfr.y,dfr.x)
    Vphi = dfr.vy*np.cos(phi) - dfr.vx*np.sin(phi)

    # Make some plot
    fig,axl = plt.subplots(1,2,figsize=(10,5))

    # Estimate the surface density 
    plt.sca(axl[0])
    H,edge=np.histogram(Rcyl, bins=20)
    Area = 2*np.pi*(edge[1:]**2 - edge[0:-1]**2) # Area of a circular anuli 
    Rave = 0.5*(edge[1:] + edge[0:-1]) # Average radius of the bin
    plt.plot(Rave,H/Area)
    plt.xlabel("Rcyl [Nbody]")
    plt.ylabel("Surface number density [Nbody]")
    plt.yscale("log")

    plt.sca(axl[1])
    plt.scatter(Rcyl,Vphi)
    plt.xlabel("Rcyl [Nbody]")
    plt.ylabel("Vphi [Nbody]")

    fig.savefig("Nbody_disc_summary.png")

