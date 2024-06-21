"""
Imports
"""
import math

import pandas as pd

from rotorpy.controllers.quadrotor_control import SE3Control
# The simulator is instantiated using the Environment class
from rotorpy.environments import Environment
from rotorpy.utils.postprocessing import unpack_sim_data
from rotorpy.vehicles.multirotor import Multirotor

# Vehicles. Currently there is only one. 
# There must also be a corresponding parameter file. 
from rotorpy.vehicles.tiltquad import TiltQuad
# from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.vehicles.hummingbird_params import quad_params  # There's also the Hummingbird

from rotorpy.controllers.tiltquad_control import SE3TiltControl
from rotorpy.controllers.tilt_lbf_control import SE3TiltLBFControl

# And a trajectory generator
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import CircularTraj, ThreeDCircularTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.trajectories.speed_traj import ConstantSpeed
from rotorpy.trajectories.minsnap import MinSnap

# You can optionally specify a wind generator, although if no wind is specified it will default to NoWind().
from rotorpy.wind.default_winds import NoWind, ConstantWind, SinusoidWind, LadderWind
from rotorpy.wind.dryden_winds import DrydenGust, DrydenGustLP
from rotorpy.wind.spatial_winds import WindTunnel

# You can also optionally customize the IMU and motion capture sensor models. If not specified, the default parameters will be used. 
from rotorpy.sensors.imu import Imu
from rotorpy.sensors.external_mocap import MotionCapture

# You can also specify a state estimator. This is optional. If no state estimator is supplied it will default to null. 
from rotorpy.estimators.wind_ukf import WindUKF

# Also, worlds are how we construct obstacles. The following class contains methods related to constructing these maps. 
from rotorpy.world import World

import matplotlib.pyplot as plt

# Plotter for TiltQuad
from rotorpy.utils.tiltquad_plotter import TiltPlotter
from rotorpy.utils.tiltquad_postprocessing import unpack_tiltquad_sim_data

# Reference the files above for more documentation. 

# Other useful imports
import numpy as np                  # For array creation/manipulation
import matplotlib.pyplot as plt     # For plotting, although the simulator has a built in plotter
from scipy.spatial.transform import Rotation  # For doing conversions between different rotation descriptions, applying rotations, etc. 
import os                           # For path generation

# phi = [1e-3, 10, 30, 60]
# phi = np.array([-1, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]) * (math.pi / 180)
# print(phi)
phi = np.array([-1, 0, 5, 10, 15, 20]) * (math.pi / 180)
# phi = np.array([-1, 0])
dist = []


for i in range(len(phi)):
    print(i)
    """
    Instantiation
    """

    # Obstacle maps can be loaded in from a JSON file using the World.from_file(path) method. Here we are loading in from
    # an existing file under the rotorpy/worlds/ directory. However, you can create your own world by following the template
    # provided (see rotorpy/worlds/README.md), and load that file anywhere using the appropriate path.

    # "world" is an optional argument. If you don't load a world it'll just provide an empty playground!

    # An instance of the simulator can be generated as follows:
    if phi[i] < 0:
        sim_instance = Environment(vehicle=Multirotor(quad_params),  # vehicle object, must be specified.
                                   controller=SE3Control(quad_params, phi[1]),
                                   # controller object, must be specified.
                                   trajectory=HoverTraj(),
                                   # trajectory=CircularTraj(radius=1),         # trajectory object, must be specified.
                                   wind_profile=NoWind(),
                                   # OPTIONAL: wind profile object, if none is supplied it will choose no wind.
                                   sim_rate=100,
                                   # OPTIONAL: The update frequency of the simulator in Hz. Default is 100 Hz.
                                   imu=None,
                                   # OPTIONAL: imu sensor object, if none is supplied it will choose a default IMU sensor.
                                   mocap=None,
                                   # OPTIONAL: mocap sensor object, if none is supplied it will choose a default mocap.
                                   estimator=None,  # OPTIONAL: estimator object
                                   world=None,
                                   # OPTIONAL: the world, same name as the file in rotorpy/worlds/, default (None) is empty world
                                   safety_margin=0.25
                                   # OPTIONAL: defines the radius (in meters) of the sphere used for collision checking
                                   )
    else:
        sim_instance = Environment(vehicle=TiltQuad(quad_params),  # vehicle object, must be specified.
                                   controller=SE3TiltLBFControl(quad_params, phi[i]),
                                   # controller object, must be specified.
                                   trajectory=HoverTraj(),
                                   # trajectory=CircularTraj(radius=1),         # trajectory object, must be specified.
                                   wind_profile=NoWind(),
                                   # OPTIONAL: wind profile object, if none is supplied it will choose no wind.
                                   sim_rate=100,
                                   # OPTIONAL: The update frequency of the simulator in Hz. Default is 100 Hz.
                                   imu=None,
                                   # OPTIONAL: imu sensor object, if none is supplied it will choose a default IMU sensor.
                                   mocap=None,
                                   # OPTIONAL: mocap sensor object, if none is supplied it will choose a default mocap.
                                   estimator=None,  # OPTIONAL: estimator object
                                   world=None,
                                   # OPTIONAL: the world, same name as the file in rotorpy/worlds/, default (None) is empty world
                                   safety_margin=0.25
                                   # OPTIONAL: defines the radius (in meters) of the sphere used for collision checking
                                   )

    # This generates an Environment object that has a unique vehicle, controller, and trajectory.
    # You can also optionally specify a wind profile, IMU object, motion capture sensor, estimator,
    # and the simulation rate for the simulator.

    """
    Execution
    """

    # Setting an initial state. This is optional, and the state representation depends on the vehicle used.
    # Generally, vehicle objects should have an "initial_state" attribute.
    if phi[i] < 0:
        x0 = {'x': np.array([1, 0, 0]),
              'v': np.zeros(3, ),
              'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
              'w': np.zeros(3, ),
              'wind': np.array([0, 0, 0]),  # Since wind is handled elsewhere, this value is overwritten
              # 'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
              'rotor_speeds': np.array([0, 0, 0, 0])}
    else:
        x0 = {'x': np.array([1, 0, 0]),
              'v': np.zeros(3, ),
              'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
              'w': np.zeros(3, ),
              'wind': np.array([0, 0, 0]),  # Since wind is handled elsewhere, this value is overwritten
              'rotor_thrusts': np.zeros((4, 3))}
    sim_instance.vehicle.initial_state = x0

    # Executing the simulator as specified above is easy using the "run" method:
    # All the arguments are listed below with their descriptions.
    # You can save the animation (if animating) using the fname argument. Default is None which won't save it.

    if phi[i] < 0:
        results = sim_instance.run(t_final=5,  # The maximum duration of the environment in seconds
                                   use_mocap=False,
                                   # Boolean: determines if the controller should use the motion capture estimates.
                                   terminate=False,
                                   # Boolean: if this is true, the simulator will terminate when it reaches the last waypoint.
                                   plot=False,  # Boolean: plots the vehicle states and commands
                                   plot_mocap=False,  # Boolean: plots the motion capture pose and twist measurements
                                   plot_estimator=False,
                                   # Boolean: plots the estimator filter states and covariance diagonal elements
                                   plot_imu=False,  # Boolean: plots the IMU measurements
                                   animate_bool=False,
                                   # Boolean: determines if the animation of vehicle state will play.
                                   animate_wind=False,
                                   # Boolean: determines if the animation will include a scaled wind vector to indicate the local wind acting on the UAV.
                                   verbose=True,  # Boolean: will print statistics regarding the simulation.
                                   fname=None
                                   # Filename is specified if you want to save the animation. The save location is rotorpy/data_out/.
                                   )
    else:
        results = sim_instance.run(t_final=5,  # The maximum duration of the environment in seconds
                                   use_mocap=False,
                                   # Boolean: determines if the controller should use the motion capture estimates.
                                   terminate=False,
                                   # Boolean: if this is true, the simulator will terminate when it reaches the last waypoint.
                                   plot=False,  # Boolean: plots the vehicle states and commands
                                   plot_mocap=False,  # Boolean: plots the motion capture pose and twist measurements
                                   plot_estimator=False,
                                   # Boolean: plots the estimator filter states and covariance diagonal elements
                                   plot_imu=False,  # Boolean: plots the IMU measurements
                                   animate_bool=False,  # Boolean: determines if the animation of vehicle state will play.
                                   animate_wind=False,
                                   # Boolean: determines if the animation will include a scaled wind vector to indicate the local wind acting on the UAV.
                                   verbose=True,  # Boolean: will print statistics regarding the simulation.
                                   fname="tiltquad_test",
                                   # Filename is specified if you want to save the animation. The save location is rotorpy/data_out/.
                                   custom_plotter=TiltPlotter,
                                   custom_logger=unpack_tiltquad_sim_data)

    # There are booleans for if you want to plot all/some of the results, animate the multirotor, and
    # if you want the simulator to output the EXIT status (end time reached, out of control, etc.)
    # The results are a dictionary containing the relevant state, input, and measurements vs time.

    # To save this data as a .csv file, you can use the environment's built in save method. You must provide a filename.
    # The save location is rotorpy/data_out/
    sim_instance.save_to_csv("basic_tiltquad_usage" + str(i) + ".csv")

    # Instead of producing a CSV, you can manually unpack the dictionary into a Pandas DataFrame using the following:
    if phi[i] < 0:
        dataframe = unpack_sim_data(results)
    else:
        dataframe = unpack_tiltquad_sim_data(results)

    dist.append(np.sqrt(dataframe['x'] ** 2 + dataframe['y'] ** 2 + dataframe['z'] ** 2))

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
plt.figure()
for y, color in zip(dist, colors):
    plt.plot(np.arange(len(y)), y, color=color)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Distance from Hover for Various Phi Angles')
# plt.legend(['Quad', 'Tilt'])
plt.legend(np.round(phi * (180 / math.pi), 3))

# Show the plot
plt.show()


