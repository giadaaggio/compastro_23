"""
====================================================================================================================
Collection of functions to estimate the Gravitational forces and accelerations (:mod:`fireworks.nbodylib.dynamics`)
====================================================================================================================

This module contains a collection of functions to estimate acceleration due to
gravitational  forces.

Each method implemented in this module should follow the input-output structure show for the
template function  :func:`~acceleration_estimate_template`:

Every function needs to have two input parameters:

    - particles, that is an instance of the class :class:`~fireworks.particles.Particles`
    - softening, it is the gravitational softening. The parameters need to be included even
        if the function is not using it. Use a default value of 0.

The function needs to return a tuple containing three elements:

    - acc, the acceleration estimated for each particle, it needs to be a Nx3 numpy array,
        this element is mandatory it cannot be 0.
    - jerk, time derivative of the acceleration, it is an optional value, if the method
        does not estimate it, just set this element to None. If it is not None, it has
        to be a Nx3 numpy array.
    - pot, gravitational potential at the position of each particle. it is an optional value, if the method
        does not estimate it, just set this element to None. If it is not None, it has
        to be a Nx1 numpy array.


"""
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
from ..particles import Particles

try:
    import pyfalcon
    pyfalcon_load=True
except:
    pyfalcon_load=False

def acceleration_estimate_template(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    """
    This an empty functions that can be used as a basic template for
    implementing the other functions to estimate the gravitational acceleration.
    Every function of this kind needs to have two input parameters:

        - particles, that is an instance of the class :class:`~fireworks.particles.Particles`
        - softening, it is the gravitational softening. The parameters need to be included even
          if the function is not using it. Use a default value of 0.

    The function needs to return a tuple containing three elements:

        - acc, the acceleration estimated for each particle, it needs to be a Nx3 numpy array,
            this element is mandatory it cannot be 0.
        - jerk, time derivative of the acceleration, it is an optional value, if the method
            does not estimate it, just set this element to None. If it is not None, it has
            to be a Nx3 numpy array.
        - pot, gravitational potential at the position of each particle. it is an optional value, if the method
            does not estimate it, just set this element to None. If it is not None, it has
            to be a Nx1 numpy array.

    :param particles: An instance of the class Particles
    :param softening: Softening parameter
    :return: A tuple with 3 elements:

        - acc, Nx3 numpy array storing the acceleration for each particle
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None
    """

    acc  = np.zeros(len(particles))
    jerk = None
    pot = None

    return (acc,jerk,pot)


def acceleration_pyfalcon(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    """
    Estimate the acceleration following the fast-multipole gravity Dehnen2002 solver (https://arxiv.org/pdf/astro-ph/0202512.pdf)
    as implementd in pyfalcon (https://github.com/GalacticDynamics-Oxford/pyfalcon)

    :param particles: An instance of the class Particles
    :param softening: Softening parameter
    :return: A tuple with 3 elements:

        - Acceleration: a NX3 numpy array containing the acceleration for each particle
        - Jerk: None, the jerk is not estimated
        - Pot: a Nx1 numpy array containing the gravitational potential at each particle position
    """

    if not pyfalcon_load: return ImportError("Pyfalcon is not available")

    acc, pot = pyfalcon.gravity(particles.pos,particles.mass,softening)
    jerk = None

    return acc, jerk, pot








# ACCELERATION ESTIMATE

def acceleration_direct(particles: Particles, softening: float =0.) \
    -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:

#    N           = 100
#    pos_min     = 5.0
#    pos_max     = 10.0
#    vel_min     = 1.0
#    vel_max     = 10.0
#    mass_min    = 1.0
#    mass_max    = 100.0

    pos     = particles.pos # particles'positions
    v       = particles.vel # particles'velocities
    mass    = particles.mass # particles'masses
    N       = len(particles)

    acc     = np.zeros((N,3),float)
    force   = np.zeros((N, 3),float)


    def Force(mass, acc):
        Force = mass*acc
        
        return Force

   
    for i in range(N):
        for j in range(N):
            if i != j:
                denom      = np.linalg.norm(pos[i] - pos[j])**3
                temp       = - mass[j]*(pos[i] - pos[j]) / denom
                acc[i,:]   = acc[i,:] + temp


        force[i,:]    = Force(mass[i], acc[i,:])

    
    jerk = None
    pot = None

    return (acc,jerk,pot)


def acceleration_direct_vectorized(particles: Particles, softening: float =0.) \
    -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:

    pos     = particles.pos # particles'positions
    v       = particles.vel # particles'velocities
    mass    = particles.mass # particles'masses
    N       = len(particles)

    #x
    ax = pos[:,0]              # I concentrate on x
    bx = ax.reshape((N,1))     # I need to reshape to transform it in a vector (vertical)
    cx = bx - ax               # this is xi-xj, the delta. I am creating a matrix xi-xj (NxN)
    #y
    ay = pos[:,1]              #same for y
    by = ay.reshape((N,1))
    cy = by - ay
    #z
    az = pos[:,2]              #same for z
    bz = az.reshape((N,1))
    cz = bz - az

    r = np.array((cx, cy, cz)) #I put everything into a sole tenson (3,N,N)
    
    deltax2 = r[0,:,:]**2
    deltay2 = r[1,:,:]**2
    deltaz2 = r[2,:,:]**2
    normr  = np.sqrt(deltax2 + deltay2 + deltaz2)    #I calculate |r| and |r|**3
    normr3 = normr**3

    factor = r / normr3        # I calculate the factor in the expression for a; I still have a tensor (3,N,N)
    addend = mass*factor       # I construct my addend multiplying by the mass
    addend[np.isnan(addend)] = 0    #I substitute the nans in the diagonal with zeros

    addendx = addend[0,:,:]    #I devide in the three components
    addendy = addend[1,:,:]
    addendz = addend[2,:,:]
    ax = - addendx.sum(axis=1)   #and I sum axis by axis
    ay = - addendy.sum(axis=1)
    az = - addendz.sum(axis=1)

    acc = np.array((ax, ay, az)) #this is the acceleration suffered by each particle by each other in the three components matrix(N,3)
    acc = acc.T                    # To get a matrix (3,N)
    
    jerk = None
    pot = None

    return (acc,jerk,pot)


# JERK ESTIMATE

def acceleration_jerk_direct(particles: Particles, softening: float =0.) \
    -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:

    pos = particles.pos       # particles'positions
    v = particles.vel         # particles'velocities
    mass = particles.mass     # particles'masses
    N = len(particles)

    acc, _, _ = acceleration_pyfalcon(particles)    #(3,N) matrix
    jerk=np.zeros([N,3],float)

    for i in range(N):
        temp_jerk = np.zeros(3, float)
        for j in range(N):
            if (j!=i):
                x_ij = (pos[i,:] - pos[j,:])
                v_ij = (v[i,:] - v[j,:])
                vet = np.dot(x_ij, v_ij)
                x_norm = np.linalg.norm(x_ij)
                temp_jerk = temp_jerk + (mass[j]*((v_ij/x_norm**3. - 3.*vet*x_ij)/x_norm**5.))
            
        jerk[i,:] = -temp_jerk

    pot = None
        
    return (acc, jerk, pot)



