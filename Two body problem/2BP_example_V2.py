##########################
# 
# Two Body Problem V2
#
##########################

# Basic imports ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# Code start ------------------------------------------------------------------

# Body parameters
R_earth = 6378 # [km]
G = 6.6743e-11/1e9 # [km^3/kg*s^2]
m_earth = 5.972e24 # [kg]
mu_earth = m_earth*G # [km^3/s^2]
h = 5000 # [km]
r_mag = R_earth+h # [km]
v_mag = np.sqrt(mu_earth/r_mag) # [km/s]

# Definition of the ODE function to set before integrating, (The function needs the state vector and time vector)
def model_2BP(st,t):
    
    x = st[0] # Position of the body in x
    y = st[1] # Position of the body in y
    z = st[2] # Position of the body in z
    dx_dt = st[3] # Velocity of the body in x
    dy_dt = st[4] # Velocity of the body in y
    dz_dt = st[5] # Velocity of the body in z
    ddx_dt = -mu_earth*x/(x**2+y**2+z**2)**(3/2) # Acc of the body in x
    ddy_dt = -mu_earth*y/(x**2+y**2+z**2)**(3/2) # Acc of the body in y
    ddz_dt = -mu_earth*z/(x**2+y**2+z**2)**(3/2) # Acc of the body in z
    dst_dt = np.array([dx_dt, dy_dt, dz_dt, ddx_dt, ddy_dt, ddz_dt])
    
    return dst_dt

# Definition of the initial conditions to set before integrating
r0 = [r_mag, 0, 0] # Python list indicating the initial position of our body
v0 = [0, v_mag, 0] # Python list indicating the initial velocity of our body
st0 = np.array(r0+v0) # Array storing our initial state vector

# Difinition of the time grid for the simulation
t = np.linspace(0, 4*3600, 100) # Grid of 3 hours in intervals of 100 seconds

# Solving the ODE
solution_vec = odeint(model_2BP, st0, t) # Easier approach than before to solve the ODE
r_body = solution_vec[:,:3] # [km] Position vector of our body (satellite) over the time interval
v_body = solution_vec[:,3:6] # [km/s] Velocity vector of our body (satellite) over the time interval

# Plot start ------------------------------------------------------------------

# Setting up Spherical Earth to Plot   
N = 50
phi = np.linspace(0, 2 * np.pi, N)
theta = np.linspace(0, np.pi, N)
theta, phi = np.meshgrid(theta, phi)
numDataPoints = len(t)

r_Earth = 6378.14  # Average radius of Earth [km]
X_Earth = r_Earth * np.cos(phi) * np.sin(theta)
Y_Earth = r_Earth * np.sin(phi) * np.sin(theta)
Z_Earth = r_Earth * np.cos(theta)


def animate_func(num):
    ax.clear()  # Clears the figure to update the line, point,   
                # title, and axes
    # Updating Trajectory Line (num+1 due to Python indexing)
    ax.plot3D(r_body[:num+1, 0], r_body[:num+1, 1], 
              r_body[:num+1, 2], c='black')
    # Updating Point Location 
    ax.scatter( r_body[num, 0],  r_body[num, 1],  r_body[num, 2], 
               c='red', marker='o')
    # Adding Constant Origin
    ax.plot3D( r_body[0, 0], r_body[0, 1],  r_body[0, 2],     
               c='red', marker='*')
    # Setting Axes Limits
    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(),      
                       ax.get_zlim3d()]).T
    XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim * 3/4)
    
    ax.plot_surface(X_Earth, Y_Earth, Z_Earth, color='blue', alpha=0.7)

    # Adding Figure Labels
    ax.set_title('Orbital trajectory \nTime = ' + str(np.round(t[num],    
                 decimals=2)) + ' [s]')
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')
    plt.legend(['Trajectory','Satellite'])

# Plotting the Animation
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(30, 145)  # Changing viewing angle (adjust as needed)
line_ani = animation.FuncAnimation(fig, animate_func, interval=100, frames=numDataPoints)
plt.show()
