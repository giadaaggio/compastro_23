import pytest
import numpy as np
import fireworks.nbodylib.dynamics as fdyn
from fireworks.particles import Particles
import math

@pytest.fixture
def sample_particles():
    # sample data for mass and positions
    mass = [1.0, 2.0, 3.0]  

    pos = [
        [0, 0, 0], 
        [1, 0, 0],
        [0, 1, 0]
    ]

    vel = [
        [1, 1, 1],  
        [0, 2, 0],
        [1, 1, 0]
    ]
    return pos, vel, mass


def expected_pot_energy(mass, pos, softening):
    # calculate expected energy 
    Epot = 0.0

    for i in range(len(mass)):
        for j in range(len(mass)):
            if i != j:
                rij_sq = sum((a-b)**2. for a, b in zip(pos[i],pos[j]))
                rij = np.sqrt(rij_sq + softening**2.)
                Epot_particle = (mass[i] * mass[j]) / (rij)
                Epot += Epot_particle
        
    Epot = - 0.5 * Epot

    return Epot


def test_Epot(sample_particles):
    pos, vel, mass = sample_particles
    part = Particles(pos, vel, mass)
    softening = 0.1  # softening parameter for the test

    # calculate expected energy
    expected_energy_pot = expected_pot_energy(mass, pos, softening)

    # calculate potential energy using the Epot 
    calculated_pot_energy = part.Epot(softening)

    # Check if the calculated energy matches the expected energy within a small tolerance
    assert abs(calculated_pot_energy - expected_energy_pot) < 1e-6  



def expected_kin_energy(mass, vel):
    # calculate expected energy
    Ekin = 0.0

    for i in range(len(mass)):
        vel_sq = sum(v**2. for v in vel[i])
        Ekin_particle = mass[i] * vel_sq
        Ekin += Ekin_particle

    Ekin = 0.5 * Ekin

    return Ekin


def test_Ekin(sample_particles):
    pos, vel, mass = sample_particles
    part = Particles(pos, vel, mass)

    # calculate expected energy
    expected_energy_kin = expected_kin_energy(mass, vel)

    # calculate potential energy using the Epot 
    calculated_kin_energy = part.Ekin()

    # Check if the calculated energy matches the expected energy within a small tolerance
    assert abs(calculated_kin_energy - expected_energy_kin) < 1e-6 


def test_Etot(sample_particles):
    pos, vel, mass = sample_particles
    part = Particles(pos, vel, mass)
    softening = 0.1

    expected_energy_kin = expected_kin_energy(mass, vel)
    expected_energy_pot = expected_pot_energy(mass, pos, softening)
    expected_energy_tot = expected_energy_pot + expected_energy_kin

    calculated_tot_energy, Ekin, Epot = part.Etot(softening)

    assert abs(calculated_tot_energy - expected_energy_tot) < 1e-6 