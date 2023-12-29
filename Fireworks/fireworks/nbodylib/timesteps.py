"""
====================================================================================================================
Collection of functions to estimate the timestep of the Nbody integrations (:mod:`fireworks.nbodylib.timesteps`)
====================================================================================================================

This module contains functions and utilities to estimate the timestep for the Nbody integrations.
There are no strict requirements for these functions. Obviously  it is important that they return a timestep.
It could be also useful to have as inputs a minimum and maximum timestep


"""
from typing import Optional, Tuple, Callable, Union
import numpy as np
import numpy.typing as npt
from ..particles import Particles

def adaptive_timestep_simple(particles: Particles, tmin: Optional[float] = None, tmax: Optional[float] = None) -> float:
    """
    Very simple adaptive timestep based on the ratio between the position and the velocity of the particles

    :param particles: that is an instance of the class :class:`~fireworks.particles.Particles`
    :return: estimated timestep
    """

    # Simple idea, use the R/V of the particles to have an estimate of the required timestep
    # Take the minimum among all the particles

    ts = np.nanmin(particles.radius()/particles.vel_mod())

    # Check tmin, tmax
    if tmin is not None: ts=np.max(ts,tmin)
    if tmax is not None: ts=np.min(ts,tmax)

    return ts




def adaptive_timestep_r(particles: Particles, tmin: Optional[float] = None, tmax: Optional[float] = None) -> float:

    #I use the R/V of the particles to have an estimate of the required timestep
    #I don't want the zeros in this procedure
    r  = particles.radius()
    v  = particles.vel_mod()
    ts = r/v
    eta = 0.001             #proportionality constant  
    ts = eta * np.nanmin(ts[np.nonzero(ts)])

    # Check tmin, tmax
    if tmin is not None: ts=np.max(ts,tmin)
    if tmax is not None: ts=np.min(ts,tmax)

    return ts , tmin, tmax



'''
    def adaptive_timestep_a(particles: Particles, acceleration_estimator: Union[Callable], tmin: Optional[float] = None, tmax: Optional[float] = None) -> float:

    #I use the R/V of the particles to have an estimate of the required timestep
    #I don't want the zeros in this procedure
    particles.acc = acceleration_estimator(Particles(particles.pos ,
                                                particles.vel ,
                                                particles.mass ))
    radius  = particles.radius()
    veloc   = particles.vel_mod()
    accel   = np.sqrt(np.sum(particles.acc*particles.acc))

    #we could use both rad/vel and vel/acc. We will use vel/acc

    ts = veloc/accel
    ts = np.min(np.nonzero(~np.isnan(ts)))
    
    #analogously we could write
    #ts1 = np.min(np.nonzero(~np.isnan(ts)))
    #ts2 = np.nanmin(np.divide(a, b, where= a.any or b.any != 0))
    #ts3 = np.min(np.nonzero(~np.isnan(c)))
    #ts4 = np.nanmin(np.nonzero(~np.isnan(c)))

    # Check tmin, tmax
    if tmin is not None: ts=np.max(ts,tmin)
    if tmax is not None: ts=np.min(ts,tmax)

    #it could be useful to have as imputs a minimum and a maximum timestep
    #if tmin is not None: 
    tmin = np.min(ts,tmin)
    #if tmax is not None: 
    tmax = np.max(ts,tmax)

    return ts, tmin, tmax

    '''