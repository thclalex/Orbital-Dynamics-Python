##########################
# 
# Two Body Problem 
#
##########################

# Basic imports --------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

# Code start -----------------------------------------------

R_earth = 6378 # [km]
mu_earth = 398600 # [km^3/s^2]


def diffy_q(t,y,mu): # Integration of the 2BP Equation
    
    # Unpack state
    rx,ry,rz,vx,vy,vz = y
    r = np.array([rx,ry,rz])
    
    # Norm of the position vector
    norm_r = np.linalg.norm(r)
    
    # Acceleration of the second body
    ax,ay,az = -(r*mu/norm_r**3)
    
    return [vx,vy,vz,ax,ay,az]

# Define conditions of orbit
r_mag = R_earth + 500 # [km] Radius of the example orbit
v_mag = np.sqrt(mu_earth/r_mag) # [km/s] Velocity of the example orbit which is circular

# Define the intial position and velocity vectors
r0 = [r_mag,0,0]
v0 = [0,v_mag,0] 

# Define grid for simulation
tspan = 100*60 # [s] Time span for the complete simulation 100 minutes
dt = 100 # [s] Time step intervals
n_steps = int(np.ceil(tspan/dt)) # Number of steps for the simulation (Ceil rounds to integer number of steps)
    
# Initialize arrays
ys = np.zeros((n_steps,6))
ts = np.zeros((n_steps,1))

# Set intial conditions for the ODE
ys[0] = r0 + v0 # Initial state
i = 1 # Dummy variable to not overwrite intial state

# Initiate solver
solver = ode(diffy_q)
solver.set_integrator('lsoda')
solver.set_initial_value(ys[0],0)
solver.set_f_params(mu_earth)

# Propagate orbit
while solver.successful() and i<n_steps:
    solver.integrate(solver.t+dt)
    ts[i] = solver.t
    ys[i] = solver.y
    i = i+1
    
rs = ys[:,:3]


def plot3D(r):
    ax = plt.axes(projection="3d")
    x = r[:,0]
    y = r[:,1]
    z = r[:,2]
    ax.plot3D(x,y,z) 
    plt.show()

plot3D(rs)


    









