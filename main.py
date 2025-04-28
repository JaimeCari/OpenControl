#### ICM2813: Drone height control ####
# 
# Professor: David E. Acuña-Ureta, PhD
# E-mail: david.acuna@uc.cl
#
#######################################

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import animation


#### Parameters and variables ####

# Model parameters
g = 9.8                     # gravity acceleration
m = 0.5                    # drone mass
l = 0.2                     # radius of the drone (spherical approximation)
viscosity = 0.0000174       # viscosity: air = 0.0000174; hidrogen = 0.00084; xenon = 0.000212
cf = 6*np.pi*viscosity*l    # friction coefficient/ Stokes Law
# F_{d} = -c_{f}v and F_{d} or drag force = 6*pi*viscocity*radius_sphere

#Laminar fluid aproximation
air_density = 1.225 #kg/m^3
wind_velocity_x = 125000 #


# Simulation parameters
Th = 40     # time horizon
Ts = 0.03   # sample time
t0 = 0.0    # initial time
ref_y = 3
ref_x = 0     # reference: desired position
y0 = 0.0    # initial condition for position
dy0 = 0.0   # initial condition for velocity
c0 = 0.0    # initial control

x0 = 0.0    # initial condition for position
dx0 = 0.0   # initial condition for velocity

Kp_y = 1.5    # controller: proportional
Ki_y = 0.2    # controller: integrative
Kd_y = 0.3    # controller: derivative

Kp_x = 7    # controller: proportional
Ki_x = 2    # controller: integrative
Kd_x = 0.5  # controller: derivative


# Variables
t = [t0]        # time signal: list
y = [y0]        # position signal: list
dy = [dy0]      # velocity signal: list

x = [x0]        # position signal: list
dx = [dx0]      # velocity signal: list

c_y = [c0]        # control signal: list
c_x = [c0]        # control signal: list

e_y = [0.0, 0.0]  # error signal: list
e_x = [0.0, 0.0]  # error signal: list

# Animation
bool_animate = True # bool to either generate a MP4 file (True) or not (False)


#### Simulation ####

def model_y(z, t, c_y):
    return np.array([z[1], c_y/m - g - (cf/m)*z[1]])  # y'' = - g - cf*y'/m + c/m

def model_x(X, t, c_x):
    return np.array([X[1], c_x/m + (cf/m)*(wind_velocity_x - X[1])])  

#array returned [y', y'']
while(t[-1] < Th):#condition for stop
    t.append(t[-1] + Ts) #time added for each iteration
    e_y.append(ref_y - y[-1]) #error of the position in y
    c_y.append(c_y[-1] + (Kp_y + Ts*Ki_y + Kd_y/Ts)*e_y[-1] + (-Kp_y - 2*Kd_y/Ts)*e_y[-2] + (Kd_y/Ts)*e_y[-3])#PID
    e_x.append(ref_x - x[-1]) #error of the position in x
    c_x.append(c_x[-1] + (Kp_x + Ts*Ki_x + Kd_x/Ts)*e_x[-1] + (-Kp_x - 2*Kd_x/Ts)*e_x[-2] + (Kd_x/Ts)*e_x[-3])#PID

    sol_y = odeint(model_y, [y[-1], dy[-1]], t[-2:], (c_y[-1],))
    y.append(sol_y[1, 0])
    dy.append(sol_y[1, 1])

    sol_x = odeint(model_x, [x[-1], dx[-1]], t[-2:], (c_x[-1],))
    x.append(sol_x[1, 0])
    dx.append(sol_x[1, 1])

plt.plot(t, y)
plt.ylabel('Altura (m)')
plt.xlabel('Tiempo (s)')
plt.show()

plt.plot(t, x)
plt.ylabel('Distancia en x (m)')
plt.xlabel('Tiempo (s)')
plt.show()



#### Animation ####

if bool_animate:
    fig = plt.figure()
    ax = plt.axes(xlim=(-(max(x) + 1.5), max(x) + 1.5), ylim=(-1, max(y) + 1.5))
    drone = mpimg.imread('drone.png')
    background = mpimg.imread('windy_background.webp')
    imagebox = OffsetImage(drone, zoom=0.5)
    ab = AnnotationBbox(imagebox, (0.0, 0.0), xycoords='data', frameon=False)

    ax.imshow(background, extent=[-(max(x) + 1.5), max(x) + 1.5, -1, max(y) + 1.5], aspect='auto', zorder=0)

    def init():
        ax.add_artist(ab)
        return ab,

    def animate(i):
        ab.xybox = (x[i], y[i])
        return ab,

    subsampling = 1
    anim = animation.FuncAnimation(fig, animate,
                                init_func=init,
                                frames=range(0,len(y)-1, subsampling),
                                interval=int(Ts*subsampling*1000),
                                blit=True)

    anim.save('height_control.mp4', writer = 'ffmpeg', fps = len(y)//Th)
    #anim.save('height_control.mp4', writer = 'ffmpeg', fps = 30)
    #anim.save('height_control.gif', writer=animation.PillowWriter(fps=30))