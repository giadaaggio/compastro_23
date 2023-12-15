import numpy as np
import matplotlib.pyplot as plt
from fireworks.particles import Particles
from fireworks.nbodylib.integrators import integrator_tsunami
import fireworks.nbodylib.dynamics as fdyn

# INITIAL CONDITION

mass = np.array([3., 4., 5.])
pos = np.array([[0., 0., 0.], [0.5, 0.866, 0.], [0., 0., 0.]])
vel = np.zeros_like(pos)
tevolve = 65.

particles_0 = Particles(position=pos, velocity=vel, mass=mass)

print(particles_0.mass)

particles, time, _ , _ , _ = integrator_tsunami(particles=particles_0, tstep=tevolve)

print(time.shape)



