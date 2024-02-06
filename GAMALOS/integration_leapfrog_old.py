def integration_leapfrog_old(galaxy: Particles, h: float, tsimulation: float, t: float, soft: float, GC: float):
    N = len(galaxy.mass)         # number of particles in the galaxy
    path = [galaxy.pos]          # list to store the position of the galaxy
    velocity = [galaxy.vel]      # list to store the velocity of the galaxy
    time = []
    timestep = []
    
    Etot_leapfrog = []
    Ekin_leapfrog = []
    Epot_leapfrog = []

    R_cyl, _, vel_phi = rotation_curve_rescaled(galaxy, GC)
    R_cyl = [R_cyl]
    V_phi = [vel_phi]

    while t < tsimulation:
        result = integrator_leapfrog(particles=galaxy, tstep=h, acceleration_estimator=acceleration_pyfalcon, softening=soft)
        updated_galaxy, _,  updated_acc, _, _ = result
        #updated_galaxy.pos -= updated_galaxy.pos[0]
        #updated_galaxy.vel -= updated_galaxy.vel[0]
        path.append(updated_galaxy.pos)
        velocity.append(updated_galaxy.vel)
            
        Etot_n, Ekin_n, Epot_n = updated_galaxy.Etot_vett()
        Etot_leapfrog.append(Etot_n)
        Ekin_leapfrog.append(Ekin_n)
        Epot_leapfrog.append(Epot_n)
    
        R, z, V = rotation_curve_rescaled(updated_galaxy, GC)

        timestep.append(h)
        t += h
        time.append(t)

        R_cyl.append(R)
        V_phi.append(V) 


    path = np.array(path)
    velocity = np.array(velocity)
    time   = np.array(time)
    timestep  = np.array(timestep)

    R_cyl = np.array(R_cyl)
    V_phi = np.array(V_phi)

    Etot_leapfrog = np.array(Etot_leapfrog)
    Ekin_leapfrog = np.array(Ekin_leapfrog)
    Epot_leapfrog = np.array(Epot_leapfrog)

    return path, velocity, Etot_leapfrog, time, R_cyl, V_phi
                                                                                          