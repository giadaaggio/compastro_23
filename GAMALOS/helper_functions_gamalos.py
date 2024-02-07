import numpy as np
import pandas as pd
from fireworks.ic import ic_two_body
import matplotlib.pyplot as plt
from fireworks.nbodylib.integrators import integrator_leapfrog_galaxy, integrator_leapfrog, integrator_leapfrog_galaxy_encounter, integrator_tsunami
from fireworks.nbodylib.dynamics import acceleration_pyfalcon
from fireworks.nbodylib.timesteps import adaptive_timestep_r
from fireworks.particles import Particles
from typing import Optional, Tuple, Callable, Union
import time

from mpl_toolkits.mplot3d import Axes3D


def rotation_curve_rescaled(galaxy: Particles, GC: float):
    
    # rescale the position and velocity of the particles
    # GC is the particle assumed as the center of the galaxy
    dx = galaxy.pos[:,0] - galaxy.pos[GC,0]
    dy = galaxy.pos[:,1] - galaxy.pos[GC,1]
    dz = galaxy.pos[:,2] - galaxy.pos[GC,2]

    dvx = galaxy.vel[:,0] - galaxy.vel[GC,0]
    dvy = galaxy.vel[:,1] - galaxy.vel[GC,1]
    dvz = galaxy.vel[:,2] - galaxy.vel[GC,2]

    # compute the cylindrical radius and azimuthal angle
    Rcyl = np.sqrt(dx**2 + dy**2)                   # cylindrical radius
    phi = np.arctan2(dy, dx)                        # azimuthal angle
    z = dz                                          # z-coordinate

    # compute velocities    
    vel_phi = dvy * np.cos(phi) - dvx * np.sin(phi)       # azimuthal velocity
    vel_r = np.cos(phi) * dvx + np.sin(phi) * dvy  # radial velocity
    vel_z = dvz                                           # vertical velocity

    return Rcyl, dz, vel_phi, vel_r, vel_z

def surface_density(Rcyl):
    H, edge = np.histogram(Rcyl, bins=20)                                          # histogram of the cylindrical radius
    area =  np.pi * (edge[1:]**2 - edge[0:-1]**2)                                  # area of every bin (every concentric ring) 
    mid = 0.5 * (edge[1:] + edge[0:-1])                                            # midpoint of every bin  

    return mid, H, area


def integration_leapfrog(galaxy: Particles, h: float, tsimulation: float, t: float, soft: float, GC: float):
    N = int((tsimulation - t) / h)  # number of time steps
    num_particles = len(galaxy.mass)  # number of particles in the galaxy
    path = np.empty((N, num_particles, 3))  # array to store the position of the galaxy
    velocity = np.empty((N, num_particles, 3))  # array to store the velocity of the galaxy
    time = np.empty(N)
    timestep = np.empty(N)
    
    R_cyl, _, V_phi, V_r, V_z = rotation_curve_rescaled(galaxy, GC)
    R_cyl = [R_cyl]
    V_phi = [V_phi]
    V_r = [V_r]
    V_z = [V_z]

    i = 0
    while t < tsimulation - h:
        updated_galaxy, _,  updated_acc, _, _ = integrator_leapfrog(particles=galaxy, tstep=h, acceleration_estimator=acceleration_pyfalcon, softening=soft)
        path[i] = updated_galaxy.pos
        velocity[i] = updated_galaxy.vel
            
        R, z, Vphi, Vr, Vz = rotation_curve_rescaled(updated_galaxy, GC)

        timestep[i] = h
        t += h
        time[i] = t

        R_cyl.append(R)
        V_phi.append(Vphi) 
        V_r.append(Vr)
        V_z.append(Vz)

        i += 1

    R_cyl = np.array(R_cyl)
    V_phi = np.array(V_phi)
    V_r = np.array(V_r)
    V_z = np.array(V_z)

    return path, velocity, time, R_cyl, V_phi, V_r, V_z


def plot_orbit_single_galaxy(Galaxy_orbit: np.ndarray, title: str):
    plt.figure(figsize=(5,5))
    plt.scatter(Galaxy_orbit[:,1:,0], Galaxy_orbit[:,1:,1], s=0.005, color='b', alpha=0.1)
    plt.plot(Galaxy_orbit[0,1:,0], Galaxy_orbit[0,1:,1], 'o', markersize=0.5, c='b', label='initial position', alpha=0.5)
    plt.plot(Galaxy_orbit[-1,1:,0], Galaxy_orbit[-1,1:,1], 'o', markersize=1, c='g', label='final position', alpha=0.5, zorder=10)
    plt.scatter(Galaxy_orbit[:,0,0], Galaxy_orbit[:,0,1], s=10, color='r', label='GC', zorder=10)
    plt.xlabel('X [Nbody]')
    plt.ylabel('Y [Nbody]')
    plt.title(title)
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.5)


def rotcurve_plot(Galaxy_Rcyl, Galaxy_Vphi, title_fig: str):
    plt.figure(figsize=(5, 5))
    n = Galaxy_Rcyl.shape[0]
    snapshots = np.arange(0, n, n//2)
    title = ['initial', 'mid', 'final']
    for i in range (len(snapshots)):
        plt.scatter(Galaxy_Rcyl[snapshots[i],:], Galaxy_Vphi[snapshots[i],:], s=0.5, label=title[i], alpha=0.5)
    
    plt.xlabel('$R_{cycl}$')
    plt.ylabel('$V_{\phi}$')
    plt.title(title_fig)
    plt.legend(loc='lower right')
    plt.show()  


def plot_surface_density(mid, H, area, mid_end, H_end, area_end, title: str):
    plt.figure(figsize=(5,5))
    plt.plot(mid, H / area, label='initial')
    plt.plot(mid_end, H_end / area_end, label='final')
    plt.yscale('log')
    plt.xlabel("Rcyl [Nbody]")  
    plt.ylabel("Surface Number Density [Nbody]")
    plt.title("Surface Density Profile Galaxy 1")
    plt.legend()
    plt.show()


def integration_leapfrog_encounter(galaxy: Particles, h: float, tsimulation: float, t: float, soft: float):
    N = int((tsimulation - t) / h)  # number of time steps
    num_particles = len(galaxy.mass)  # number of particles in the galaxy
    path = np.empty((N, num_particles, 3))  # array to store the position of the galaxy
    velocity = np.empty((N, num_particles, 3))  # array to store the velocity of the galaxy
    time = np.empty(N)
    timestep = np.empty(N)
    
    R_cyl, _, V_phi, V_r, V_z = rotation_curve_rescaled(galaxy, GC=np.where(galaxy.ID < (len(galaxy.ID)//2), 0, len(galaxy.ID)//2))
    R_cyl = [R_cyl]
    V_phi = [V_phi]
    V_r = [V_r]
    V_z = [V_z]

    i = 0
    while t < tsimulation:
        updated_galaxy, _,  updated_acc, _, _ = integrator_leapfrog(particles=galaxy, tstep=h, acceleration_estimator=acceleration_pyfalcon, softening=soft)
        path[i] = updated_galaxy.pos
        velocity[i] = updated_galaxy.vel
            
        R, z, Vphi, Vr, Vz = rotation_curve_rescaled(updated_galaxy, GC=np.where(updated_galaxy.ID < (len(updated_galaxy.ID)//2), 0, len(updated_galaxy.ID)//2))

        timestep[i] = h
        t += h
        time[i] = t

        R_cyl.append(R)
        V_phi.append(Vphi) 
        V_r.append(Vr)
        V_z.append(Vz)

        i += 1

    R_cyl = np.array(R_cyl)
    V_phi = np.array(V_phi)
    V_r = np.array(V_r)
    V_z = np.array(V_z)

    return path, velocity, time, R_cyl, V_phi, V_r, V_z


def plot_encounter(Combined_Galaxies_orbit, title: str, div: int):
    plt.figure(figsize=(8,8))

    # stars trajectory
    plt.scatter(Combined_Galaxies_orbit[:,1:div,0], Combined_Galaxies_orbit[:,1:div,1], s=0.005, color='lightblue',  alpha=0.05)
    plt.scatter(Combined_Galaxies_orbit[:,(div+1):,0],  Combined_Galaxies_orbit[:,(div+1):,1],  s=0.005, color='lightgreen', alpha=0.05)

    # initial and final position of galaxy 1
    plt.plot(Combined_Galaxies_orbit[0,1:div,0],  Combined_Galaxies_orbit[0,1:div,1],  'o', markersize=2, c='cornflowerblue', label='initial position of Gal1', alpha=0.3)
    plt.plot(Combined_Galaxies_orbit[-1,1:div,0], Combined_Galaxies_orbit[-1,1:div,1], '*', markersize=3, c='b', label='final position of Gal1',   alpha=0.3)

    # initial and final position of galaxy 2
    plt.plot(Combined_Galaxies_orbit[0,(div+1):,0],  Combined_Galaxies_orbit[0,(div+1):,1],  'o', markersize=2, c='limegreen', label='initial position of Gal2', alpha=0.3)
    plt.plot(Combined_Galaxies_orbit[-1,(div+1):,0], Combined_Galaxies_orbit[-1,(div+1):,1], '*', markersize=3, c='g', label='final position of Gal2',   alpha=0.3)

    # initial and final position of GC1
    plt.plot(Combined_Galaxies_orbit[0,0,0],  Combined_Galaxies_orbit[0,0,1],  'o', markersize=4, color='r', label='GC 1', zorder=10)
    plt.plot(Combined_Galaxies_orbit[-1,0,0], Combined_Galaxies_orbit[-1,0,1], 'o', markersize=4, color='r',               zorder=10)

    # initial and final position of GC2
    plt.plot(Combined_Galaxies_orbit[0,div,0],  Combined_Galaxies_orbit[0,div,1],  'o', markersize=4, color='orange', label='GC 2', zorder=10)
    plt.plot(Combined_Galaxies_orbit[-1,div,0], Combined_Galaxies_orbit[-1,div,1], 'o', markersize=4, color='orange',               zorder=10)

    # trajectory of GC1
    plt.scatter(Combined_Galaxies_orbit[:,0,0],    Combined_Galaxies_orbit[:,0,1],    s=0.05, color='r',      zorder=10, alpha=0.5)
    # trajectory of GC2
    plt.scatter(Combined_Galaxies_orbit[:,div,0], Combined_Galaxies_orbit[:,div,1], s=0.05, color='orange', zorder=10, alpha=0.5)

    plt.xlabel('X [Nbody]')
    plt.ylabel('Y [Nbody]')
    plt.title(title)
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, alpha=0.5)
    plt.show()


def subplot_positions_rotcurve_density(Combined_Galaxies_orbit, Combined_Galaxies_Rcyl, Combined_Galaxies_Vphi, 
                                       mid_in_1, H_in_1, area_in_1, mid_end_1, H_end_1, area_end_1, 
                                       mid_in_2, H_in_2, area_in_2, mid_end_2, H_end_2, area_end_2,
                                       div: int):
    
    plt.subplots(3,3,figsize=(12,12))

    plt.subplot(3, 3, 1)
    # final position of galaxy 1
    plt.plot(Combined_Galaxies_orbit[-1,1:div,0], Combined_Galaxies_orbit[-1,1:div,1], 'o', markersize=1, c='b', label='final position of Gal1',   alpha=0.5)
    # position of GC1
    plt.plot(Combined_Galaxies_orbit[-1,0,0], Combined_Galaxies_orbit[-1,0,1], 'o', markersize=4, color='r', label='GC 1', zorder=10)
    # final position of galaxy 2
    plt.plot(Combined_Galaxies_orbit[-1,(div+1):,0], Combined_Galaxies_orbit[-1,(div+1):,1], 'o', markersize=1, c='g', label='final position of Gal2',   alpha=0.5)
    # position of GC2
    plt.plot(Combined_Galaxies_orbit[-1,div,0], Combined_Galaxies_orbit[-1,div,1], 'o', markersize=4, color='orange', label='GC 2', zorder=10)

    plt.xlabel('X [Nbody]')
    plt.ylabel('Y [Nbody]')
    plt.title('Final position of the Galaxies')
    plt.legend(loc='lower right', fontsize=8)

    plt.subplot(3, 3, 2)
    # final position of galaxy 1
    plt.plot(Combined_Galaxies_orbit[-1,1:div,0], Combined_Galaxies_orbit[-1,1:div,1], 'o', markersize=1, c='b', label='final position of Gal1',   alpha=0.5)
    # position of GC1
    plt.plot(Combined_Galaxies_orbit[-1,0,0], Combined_Galaxies_orbit[-1,0,1], 'o', markersize=4, color='r', label='GC 1', zorder=10)

    plt.xlabel('X [Nbody]')
    plt.ylabel('Y [Nbody]')
    plt.title('Galaxy 1')
    plt.legend(loc='lower right', fontsize=8)

    plt.subplot(3, 3, 3)
    # final position of galaxy 2   
    plt.plot(Combined_Galaxies_orbit[-1,(div+1):,0], Combined_Galaxies_orbit[-1,(div+1):,1], 'o', markersize=1, c='g', label='final position of Gal2',   alpha=0.5)
    # position of GC2
    plt.plot(Combined_Galaxies_orbit[-1,div,0], Combined_Galaxies_orbit[-1,div,1], 'o', markersize=4, color='orange', label='GC 2', zorder=10)

    plt.xlabel('X [Nbody]')
    plt.ylabel('Y [Nbody]')
    plt.title('Galaxy 2')
    plt.legend(loc='lower right', fontsize=8)

    plt.subplot(3, 3, 5)
    plt.scatter(Combined_Galaxies_Rcyl[-1,1:div], Combined_Galaxies_Vphi[-1,1:div], s=0.5, color='r', label='final', alpha=0.5)
    plt.scatter(Combined_Galaxies_Rcyl[0,1:div], Combined_Galaxies_Vphi[0,1:div], s=0.5, color='b', label='initial', alpha=0.5)
    plt.legend(loc='best', fontsize=8)
    plt.xlabel('$R_{cycl}$')
    plt.ylabel('$V_{\phi}$')
    plt.title('Rotation curve Galaxy 1')

    plt.subplot(3, 3, 6)
    plt.scatter(Combined_Galaxies_Rcyl[-1,(div+1):], Combined_Galaxies_Vphi[-1,(div+1):], s=0.5, color='g', label='final', alpha=0.5)
    plt.scatter(Combined_Galaxies_Rcyl[0,(div+1):], Combined_Galaxies_Vphi[0,(div+1):], s=0.5, color='darkorange', label='initial', alpha=0.5)
    plt.legend(loc='best', fontsize=8)
    plt.xlabel('$R_{cycl}$')
    plt.ylabel('$V_{\phi}$')
    plt.title('Rotation curve Galaxy 2')

    plt.subplot(3, 3, 4)
    plt.scatter(Combined_Galaxies_Rcyl[-1,1:div], Combined_Galaxies_Vphi[-1,1:div], s=0.5, color='r', label='final G1', alpha=0.5)
    plt.scatter(Combined_Galaxies_Rcyl[-1,(div+1):], Combined_Galaxies_Vphi[-1,(div+1):], s=0.5, color='g', label='final G2', alpha=0.5)
    plt.scatter(Combined_Galaxies_Rcyl[0,1:div], Combined_Galaxies_Vphi[0,1:div], s=0.5, color='b', label='initial G1', alpha=0.5)
    plt.scatter(Combined_Galaxies_Rcyl[0,(div+1):], Combined_Galaxies_Vphi[0,(div+1):], s=0.5, color='darkorange', label='initial G2', alpha=0.5)
    plt.legend(loc='best', fontsize=8)
    plt.xlabel('$R_{cycl}$')
    plt.ylabel('$V_{\phi}$')
    plt.title('Rotation curve (G1 and G2)')

    plt.subplot(3, 3, 7)
    plt.plot(mid_in_1, H_in_1 / area_in_1, label='initial G1 and G2', c='b')
    plt.plot(mid_end_1, H_end_1 / area_end_1, label='final G1', c='r')
    plt.plot(mid_end_2, H_end_2 / area_end_2, label='final G2', c='g')
    plt.yscale('log')
    plt.xlabel("Rcyl [Nbody]")  
    plt.ylabel("Surface Number Density [Nbody]")
    plt.legend(loc='best', fontsize=8)
    plt.title("Surface Density Profile (G1 and G2)")

    plt.subplot(3, 3, 8)
    plt.plot(mid_in_1, H_in_1 / area_in_1, label='initial', c='b')
    plt.plot(mid_end_1, H_end_1 / area_end_1, label='final', c='r')
    plt.yscale('log')
    plt.xlabel("Rcyl [Nbody]")  
    plt.ylabel("Surface Number Density [Nbody]")
    plt.legend(loc='best', fontsize=8)
    plt.title("Surface Density Profile Galaxy 1")

    plt.subplot(3, 3, 9)
    plt.plot(mid_in_2, H_in_2 / area_in_2, label='initial', c='b')
    plt.plot(mid_end_2, H_end_2 / area_end_2, label='final', c='g')
    plt.yscale('log')
    plt.xlabel("Rcyl [Nbody]")  
    plt.ylabel("Surface Number Density [Nbody]")
    plt.legend(loc='best', fontsize=8)
    plt.title("Surface Density Profile Galaxy 2")


    plt.tight_layout()
    plt.show()


def subplots_radial_vertical_vel(Combined_Galaxies_Rcyl, Combined_Galaxies_Vr, Combined_Galaxies_Vz, div):
    plt.subplots(2,3,figsize=(12,8), sharey=True)

    plt.subplot(2, 3, 2)
    plt.scatter(Combined_Galaxies_Rcyl[-1,1:div], Combined_Galaxies_Vr[-1,1:div], s=0.5, color='r', label='final', alpha=0.5)
    plt.scatter(Combined_Galaxies_Rcyl[0,1:div], Combined_Galaxies_Vr[0,1:div], s=0.5, color='b', label='initial', alpha=0.5)
    plt.legend(loc='best')
    plt.xlabel('$R_{cycl}$')
    plt.ylabel('$V_{r}$')
    plt.title('Radial velocity Galaxy 1')

    plt.subplot(2, 3, 3)
    plt.scatter(Combined_Galaxies_Rcyl[-1,(div+1):], Combined_Galaxies_Vr[-1,(div+1):], s=0.5, color='g', label='final', alpha=0.5)
    plt.scatter(Combined_Galaxies_Rcyl[0,(div+1):], Combined_Galaxies_Vr[0,(div+1):], s=0.5, color='darkorange', label='initial', alpha=0.5)
    plt.legend(loc='best')
    plt.xlabel('$R_{cycl}$')
    plt.title('Radial velocity Galaxy 2')

    plt.subplot(2, 3, 1)
    plt.scatter(Combined_Galaxies_Rcyl[-1,1:div], Combined_Galaxies_Vr[-1,1:div], s=0.5, color='r', label='final G1', alpha=0.5)
    plt.scatter(Combined_Galaxies_Rcyl[-1,(div+1):], Combined_Galaxies_Vr[-1,(div+1):], s=0.5, color='g', label='final G2', alpha=0.5)
    plt.scatter(Combined_Galaxies_Rcyl[0,1:div], Combined_Galaxies_Vr[0,1:div], s=0.5, color='b', label='initial G1', alpha=0.5)
    plt.scatter(Combined_Galaxies_Rcyl[0,(div+1):], Combined_Galaxies_Vr[0,(div+1):], s=0.5, color='darkorange', label='initial G1', alpha=0.5)
    plt.legend(loc='best')
    plt.xlabel('$R_{cycl}$')
    plt.title('Radial velocity of both Galaxies')

    plt.subplot(2, 3, 4)
    plt.scatter(Combined_Galaxies_Rcyl[-1,1:div], Combined_Galaxies_Vz[-1,1:div], s=0.5, color='r', label='final G1', alpha=0.5)
    plt.scatter(Combined_Galaxies_Rcyl[-1,(div+1):], Combined_Galaxies_Vz[-1,(div+1):], s=0.5, color='g', label='final G2', alpha=0.5)
    plt.scatter(Combined_Galaxies_Rcyl[0,1:div], Combined_Galaxies_Vz[0,1:div], s=0.5, color='b', label='initial G1', alpha=0.5)
    plt.scatter(Combined_Galaxies_Rcyl[0,(div+1):], Combined_Galaxies_Vz[0,(div+1):], s=0.5, color='darkorange', label='initial G2', alpha=0.5)
    plt.legend(loc='best')
    plt.xlabel('$R_{cycl}$')
    plt.title('Vertical velocity of both Galaxies')

    plt.subplot(2, 3, 5)
    plt.scatter(Combined_Galaxies_Rcyl[-1,1:div], Combined_Galaxies_Vz[-1,1:div], s=0.5, color='r', label='final', alpha=0.5)
    plt.scatter(Combined_Galaxies_Rcyl[0,1:div], Combined_Galaxies_Vz[0,1:div], s=0.5, color='b', label='initial', alpha=0.5)
    plt.legend(loc='best')
    plt.xlabel('$R_{cycl}$')
    plt.ylabel('$V_{z}$')
    plt.title('Vertical velocity Galaxy 1')

    plt.subplot(2, 3, 6)
    plt.scatter(Combined_Galaxies_Rcyl[-1,(div+1):], Combined_Galaxies_Vz[-1,(div+1):], s=0.5, color='g', label='final', alpha=0.5)
    plt.scatter(Combined_Galaxies_Rcyl[0,(div+1):], Combined_Galaxies_Vz[0,(div+1):], s=0.5, color='darkorange', label='initial', alpha=0.5)
    plt.legend(loc='best')
    plt.xlabel('$R_{cycl}$')
    plt.title('Vertical velocity Galaxy 2')


    plt.tight_layout()
    plt.show()



def plot_3d_initial_final(Combined_Galaxies_inclined_orbit, div, num, title:str):

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(Combined_Galaxies_inclined_orbit[0,0,0], Combined_Galaxies_inclined_orbit[0,0,1], Combined_Galaxies_inclined_orbit[0,0,2], s=10, c='r', label='initial GC 1', zorder=10)
    ax1.scatter(Combined_Galaxies_inclined_orbit[0,div,0], Combined_Galaxies_inclined_orbit[0,div,1], Combined_Galaxies_inclined_orbit[0,num,2], s=10, c='orange', label='initial GC 2', zorder=10)
    ax1.scatter(Combined_Galaxies_inclined_orbit[0,1:div,0], Combined_Galaxies_inclined_orbit[0,1:div,1], Combined_Galaxies_inclined_orbit[0,1:num,2], s=0.5, c='b', label='initial Galaxy 1', alpha=0.5)
    ax1.scatter(Combined_Galaxies_inclined_orbit[0,div+1:,0], Combined_Galaxies_inclined_orbit[0,div+1:,1], Combined_Galaxies_inclined_orbit[0,num+1:,2], s=0.5, c='g', label='initial Galaxy 2', alpha=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Initial position of the Galaxies')
    ax1.legend(loc='upper right')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(Combined_Galaxies_inclined_orbit[-1,1:div,0], Combined_Galaxies_inclined_orbit[-1,1:div,1], Combined_Galaxies_inclined_orbit[-1,1:num,2], s=0.5, c='b', label='final Galaxy 1', alpha=0.5)
    ax2.scatter(Combined_Galaxies_inclined_orbit[-1,div+1:,0], Combined_Galaxies_inclined_orbit[-1,div+1:,1], Combined_Galaxies_inclined_orbit[-1,num+1:,2], s=0.5, c='g', label='final Galaxy 2', alpha=0.5)
    ax2.scatter(Combined_Galaxies_inclined_orbit[-1,0,0], Combined_Galaxies_inclined_orbit[-1,0,1], Combined_Galaxies_inclined_orbit[-1,0,2], s=10, c='r', label='final GC 1', zorder=-10)
    ax2.scatter(Combined_Galaxies_inclined_orbit[-1,div,0], Combined_Galaxies_inclined_orbit[-1,div,1], Combined_Galaxies_inclined_orbit[-1,num,2], s=10, c='orange', label='final GC 2', zorder=-10)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Final position of the Galaxies')
    ax2.legend(loc='upper right')

    plt.suptitle(title)

    plt.show()


def plot_orbit_3d(Combined_Galaxies_inclined_orbit, div):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    # trajectories
    ax.scatter(Combined_Galaxies_inclined_orbit[:,1:div,0], Combined_Galaxies_inclined_orbit[:,1:div,1], Combined_Galaxies_inclined_orbit[:,1:div,2], s=0.005, c='lightblue', alpha=0.05)
    ax.scatter(Combined_Galaxies_inclined_orbit[:,div+1:,0], Combined_Galaxies_inclined_orbit[:,div+1:,1], Combined_Galaxies_inclined_orbit[:,div+1:,2], s=0.005, c='lightgreen', alpha=0.05)
    ax.scatter(Combined_Galaxies_inclined_orbit[:,0,0], Combined_Galaxies_inclined_orbit[:,0,1], Combined_Galaxies_inclined_orbit[:,0,2], s=0.5, c='r', zorder=10)
    ax.scatter(Combined_Galaxies_inclined_orbit[:,div,0], Combined_Galaxies_inclined_orbit[:,div,1], Combined_Galaxies_inclined_orbit[:,div,2], s=0.5, c='orange', zorder=10)

    # initial position
    ax.scatter(Combined_Galaxies_inclined_orbit[0,1:div,0], Combined_Galaxies_inclined_orbit[0,1:div,1], Combined_Galaxies_inclined_orbit[0,1:div,2], s=0.5, c='cornflowerblue', label='initial Galaxy 1')
    ax.scatter(Combined_Galaxies_inclined_orbit[0,div+1:,0], Combined_Galaxies_inclined_orbit[0,div+1:,1], Combined_Galaxies_inclined_orbit[0,div+1:,2], s=0.5, c='limegreen', label='initial Galaxy 2')
    ax.scatter(Combined_Galaxies_inclined_orbit[0,0,0], Combined_Galaxies_inclined_orbit[0,0,1], Combined_Galaxies_inclined_orbit[0,0,2], s=10, c='r', label='GC 1', zorder=10)
    ax.scatter(Combined_Galaxies_inclined_orbit[0,div,0], Combined_Galaxies_inclined_orbit[0,div,1], Combined_Galaxies_inclined_orbit[0,div,2], s=10, c='orange', label='GC 2', zorder=10)

    # final position
    ax.scatter(Combined_Galaxies_inclined_orbit[-1,1:div,0], Combined_Galaxies_inclined_orbit[-1,1:div,1], Combined_Galaxies_inclined_orbit[-1,1:div,2], s=1, c='b', label='final Galaxy 1')
    ax.scatter(Combined_Galaxies_inclined_orbit[-1,div+1:,0], Combined_Galaxies_inclined_orbit[-1,div+1:,1], Combined_Galaxies_inclined_orbit[-1,div+1:,2], s=1, c='g', label='final Galaxy 2')
    ax.scatter(Combined_Galaxies_inclined_orbit[-1,0,0], Combined_Galaxies_inclined_orbit[-1,0,1], Combined_Galaxies_inclined_orbit[-1,0,2], s=10, c='r', zorder=10)
    ax.scatter(Combined_Galaxies_inclined_orbit[-1,div,0], Combined_Galaxies_inclined_orbit[-1,div,1], Combined_Galaxies_inclined_orbit[-1,div,2], s=10, c='orange', zorder=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Orbit of the Combined Galaxies')
    ax.legend(loc='upper right')

    plt.show()