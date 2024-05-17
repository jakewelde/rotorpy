import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path

# from rotorpy.utils.axes3ds import Axes3Ds
from rotorpy.utils.animate import _decimate_index, ClosingFuncAnimation
from rotorpy.utils.shapes import Quadrotor

import os

"""
Functions for showing the results from the simulator.

"""

def animate_tiltquad(time, position, rotation, wind, rotor_thrust_vecs, animate_wind, animate_thrust_vecs, world, filename=None, blit=False, show_axes=True, close_on_finish=False):
    """
    Animate a completed simulation result based on the time, position, and
    rotation history. The animation may be viewed live or saved to a .mp4 video
    (slower, requires additional libraries).

    For a live view, it is absolutely critical to retain a reference to the
    returned object in order to prevent garbage collection before the animation
    has completed displaying.

    Below, M corresponds to the number of drones you're animating. If M is None, i.e. the arrays are (N,3) and (N,3,3), then it is assumed that there is only one drone.
    Otherwise, we iterate over the M drones and animate them on the same axes.

    N is the number of time steps in the simulation.

    Parameters
        time, (N,) with uniform intervals
        position, (N,M,3)
        rotation, (N,M,3,3)
        wind, (N,M,3) world wind velocity
        rotor_thrust_vecs, (N, M, 4, 3)
        animate_wind, if True animate wind vector
        animate_thrust_vecs, if True animate the thrust vectors on each rotor.
        world, a World object
        filename, for saved video, or live view if None
        blit, if True use blit for faster animation, default is False
        show_axes, if True plot axes, default is True
        close_on_finish, if True close figure at end of live animation or save, default is False
    """

    # Check if there is only one drone.
    if len(position.shape) == 2:
        position = np.expand_dims(position, axis=1)
        rotation = np.expand_dims(rotation, axis=1)
        wind = np.expand_dims(wind, axis=1)
        rotor_thrust_vecs = np.expand_dims(rotor_thrust_vecs, axis=1)
    M = position.shape[1]

    # Temporal style.
    rtf = 1.0 # real time factor > 1.0 is faster than real time playback
    render_fps = 30

    # Normalize the wind by the max of the wind magnitude on each axis, so that the maximum length of the arrow is decided by the scale factor
    wind_mag = np.max(np.linalg.norm(wind, axis=-1), axis=1)             # Get the wind magnitude time series
    max_wind = np.max(wind_mag)                         # Find the maximum wind magnitude in the time series

    if max_wind != 0:
        wind_arrow_scale_factor = 1                         # Scale factor for the wind arrow
        wind = wind_arrow_scale_factor*wind / max_wind

    thrust_mag = np.max(np.linalg.norm(rotor_thrust_vecs, axis=-1), axis=1)
    max_thrust = np.max(thrust_mag)

    rotor_thrust_vecs = rotor_thrust_vecs / max_thrust

    # Decimate data to render interval; always include t=0.
    if time[-1] != 0:
        sample_time = np.arange(0, time[-1], 1/render_fps * rtf)
    else:
        sample_time = np.zeros((1,))
    index = _decimate_index(time, sample_time)
    time = time[index]
    position = position[index,:]
    rotation = rotation[index,:]
    wind = wind[index,:]
    rotor_thrust_vecs = rotor_thrust_vecs[index,:]

    # Set up axes.
    if filename is not None:
        if isinstance(filename, Path):
            fig = plt.figure(filename.name)
        else:
            fig = plt.figure(filename)
    else:
        fig = plt.figure('Animation')
    fig.clear()
    ax = fig.add_subplot(projection='3d')
    if not show_axes:
        ax.set_axis_off()

    quads = [Quadrotor(ax, wind=animate_wind, wind_scale_factor=1, thrust_vector=True, thrust_vec_scale_factor=1) for _ in range(M)]

    world_artists = world.draw(ax)

    title_artist = ax.set_title('t = {}'.format(time[0]))

    def init():
        ax.draw(fig.canvas.get_renderer())
        # return world_artists + list(cquad.artists) + [title_artist]
        return world_artists + [title_artist] + [q.artists for q in quads]

    def update(frame):
        title_artist.set_text('t = {:.2f}'.format(time[frame]))
        for i, quad in enumerate(quads):
            quad.transform(position=position[frame,i,:], rotation=rotation[frame,i,:,:], wind=wind[frame,i,:], thrust_vectors=rotor_thrust_vecs[frame,i,:,:])
        # [a.do_3d_projection(fig.canvas.get_renderer()) for a in quad.artists]   # No longer necessary in newer matplotlib?
        # return world_artists + list(quad.artists) + [title_artist]
        return world_artists + [title_artist] + [q.artists for q in quads]

    ani = ClosingFuncAnimation(fig=fig,
                        func=update,
                        frames=time.size,
                        init_func=init,
                        interval=1000.0/render_fps,
                        repeat=False,
                        blit=blit,
                        close_on_finish=close_on_finish)

    if filename is not None:
        print('Saving Animation')
        if not ".mp4" in filename:
            filename = filename + ".mp4"
        path = os.path.join(os.path.dirname(__file__),'..','data_out',filename)
        ani.save(path,
                writer='ffmpeg',
                fps=render_fps,
                dpi=100)
        if close_on_finish:
            plt.close(fig)
            ani = None

    return ani

class TiltPlotter():

    def __init__(self, results, world):

        (self.time, self.x, self.x_des, self.v, 
        self.v_des, self.q, self.q_des, self.w, 
        self.rotor_thrust_vecs, self.rotor_thrust_vecs_des, self.M, self.T, self.wind,
        self.accel, self.gyro, self.accel_gt,
        self.x_mc, self.v_mc, self.q_mc, self.w_mc, 
        self.filter_state, self.covariance, self.sd) = self.unpack_results(results)

        self.R = Rotation.from_quat(self.q).as_matrix()
        self.R_mc = Rotation.from_quat(self.q_mc).as_matrix() # Rotation as measured by motion capture.

        self.world = world

        return

    def plot_results(self, plot_mocap, plot_estimator, plot_imu):
        """
        Plot the results

        """

        # 3D Paths
        fig = plt.figure('3D Path')
        # ax = Axes3Ds(fig)
        ax = fig.add_subplot(projection='3d')
        self.world.draw(ax)
        ax.plot3D(self.x[:,0], self.x[:,1], self.x[:,2], 'b.')
        ax.plot3D(self.x_des[:,0], self.x_des[:,1], self.x_des[:,2], 'k')

        # Position and Velocity vs. Time
        (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Pos/Vel vs Time')
        ax = axes[0]
        ax.plot(self.time, self.x_des[:,0], 'r', self.time, self.x_des[:,1], 'g', self.time, self.x_des[:,2], 'b')
        ax.plot(self.time, self.x[:,0], 'r.',    self.time, self.x[:,1], 'g.',    self.time, self.x[:,2], 'b.')
        ax.legend(('x', 'y', 'z'))
        ax.set_ylabel('position, m')
        ax.grid('major')
        ax.set_title('Position')
        ax = axes[1]
        ax.plot(self.time, self.v_des[:,0], 'r', self.time, self.v_des[:,1], 'g', self.time, self.v_des[:,2], 'b')
        ax.plot(self.time, self.v[:,0], 'r.',    self.time, self.v[:,1], 'g.',    self.time, self.v[:,2], 'b.')
        ax.legend(('x', 'y', 'z'))
        ax.set_ylabel('velocity, m/s')
        ax.set_xlabel('time, s')
        ax.grid('major')

        # Orientation and Angular Velocity vs. Time
        (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Attitude/Rate vs Time')
        ax = axes[0]
        ax.plot(self.time, self.q_des[:,0], 'r', self.time, self.q_des[:,1], 'g', self.time, self.q_des[:,2], 'b', self.time, self.q_des[:,3], 'm')
        ax.plot(self.time, self.q[:,0], 'r.',    self.time, self.q[:,1], 'g.',    self.time, self.q[:,2], 'b.',    self.time, self.q[:,3],     'm.')
        ax.legend(('i', 'j', 'k', 'w'))
        ax.set_ylabel('quaternion')
        ax.set_xlabel('time, s')
        ax.grid('major')
        ax = axes[1]
        ax.plot(self.time, self.w[:,0], 'r.', self.time, self.w[:,1], 'g.', self.time, self.w[:,2], 'b.')
        ax.legend(('x', 'y', 'z'))
        ax.set_ylabel('angular velocity, rad/s')
        ax.set_xlabel('time, s')
        ax.grid('major')

        if plot_mocap:  # if mocap should be plotted. 
            # Motion capture position and velocity vs time
            (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Motion Capture Pos/Vel vs Time')
            ax = axes[0]
            ax.plot(self.time, self.x_mc[:,0], 'r.', self.time, self.x_mc[:,1], 'g.',    self.time, self.x_mc[:,2], 'b.')
            ax.legend(('x', 'y', 'z'))
            ax.set_ylabel('position, m')
            ax.grid('major')
            ax.set_title('MOTION CAPTURE Position/Velocity')
            ax = axes[1]
            ax.plot(self.time, self.v_mc[:,0], 'r.',    self.time, self.v_mc[:,1], 'g.',    self.time, self.v_mc[:,2], 'b.')
            ax.legend(('x', 'y', 'z'))
            ax.set_ylabel('velocity, m/s')
            ax.set_xlabel('time, s')
            ax.grid('major')
            # Motion Capture Orientation and Angular Velocity vs. Time
            (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Motion Capture Attitude/Rate vs Time')
            ax = axes[0]
            ax.plot(self.time, self.q_mc[:,0], 'r.',    self.time, self.q_mc[:,1], 'g.',    self.time, self.q_mc[:,2], 'b.',    self.time, self.q_mc[:,3],     'm.')
            ax.legend(('i', 'j', 'k', 'w'))
            ax.set_ylabel('quaternion')
            ax.set_xlabel('time, s')
            ax.grid('major')
            ax.set_title("MOTION CAPTURE Attitude/Rate")
            ax = axes[1]
            ax.plot(self.time, self.w_mc[:,0], 'r.', self.time, self.w_mc[:,1], 'g.', self.time, self.w_mc[:,2], 'b.')
            ax.legend(('x', 'y', 'z'))
            ax.set_ylabel('angular velocity, rad/s')
            ax.set_xlabel('time, s')
            ax.grid('major')

        num_rotors = self.rotor_thrust_vecs.shape[1]
        num_subplots = int(np.sqrt(num_rotors))
        fig, axes = plt.subplots(nrows=num_subplots, ncols=num_subplots, num="Rotor Forces")
        for i in range(num_rotors):
            ax = axes.ravel()[i]
            ax.plot(self.time, self.rotor_thrust_vecs[:, i, 0], 'r.', label='X')
            ax.plot(self.time, self.rotor_thrust_vecs[:, i, 1], 'g.', label='Y')
            ax.plot(self.time, self.rotor_thrust_vecs[:, i, 2], 'b.', label='Z')
            ax.plot(self.time, self.rotor_thrust_vecs_des[:, i, 0], 'r', label='X')
            ax.plot(self.time, self.rotor_thrust_vecs_des[:, i, 1], 'g', label='Y')
            ax.plot(self.time, self.rotor_thrust_vecs_des[:, i, 2], 'b', label='Z')
            ax.set_xlabel("time, s")
            ax.set_ylabel("thrust, N")
            ax.set_title("Rotor "+str(i))
            ax.legend()
        fig.tight_layout()

        # Winds
        (fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Winds vs Time')
        ax = axes[0]
        ax.plot(self.time, self.wind[:,0], 'r')
        ax.set_ylabel("wind X, m/s")
        ax.grid('major')
        ax.set_title('Winds')
        ax = axes[1]
        ax.plot(self.time, self.wind[:,1], 'g')
        ax.set_ylabel("wind Y, m/s")
        ax.grid('major')
        ax = axes[2]
        ax.plot(self.time, self.wind[:,2], 'b')
        ax.set_ylabel("wind Z, m/s")
        ax.set_xlabel("time, s")
        ax.grid('major')

        # IMU sensor
        if plot_imu:
            (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num="IMU Measurements vs Time")
            ax = axes[0]
            ax.plot(self.time, self.accel[:,0], 'r.', self.time, self.accel[:,1], 'g.', self.time, self.accel[:,2], 'b.')
            ax.plot(self.time, self.accel_gt[:,0], 'k', self.time, self.accel_gt[:,1], 'c', self.time, self.accel_gt[:,2], 'm')
            ax.set_ylabel("linear acceleration, m/s/s")
            ax.grid()
            ax = axes[1]
            ax.plot(self.time, self.gyro[:,0], 'r.', self.time, self.gyro[:,1], 'g.', self.time, self.gyro[:,2], 'b.')
            ax.set_ylabel("angular velocity, rad/s")
            ax.grid()
            ax.legend(('x','y','z'))
            ax.set_xlabel("time, s")

        if plot_estimator:
            if self.estimator_exists:
                N_filter = self.filter_state.shape[1]
                (fig, axes) = plt.subplots(nrows=N_filter, ncols=1, sharex=True, num="Filter States vs Time")
                fig.set_size_inches(11, 8.5)
                for i in range(N_filter):
                    ax = axes[i]
                    ax.plot(self.time, self.filter_state[:,i], 'k', )
                    ax.fill_between(self.time, self.filter_state[:,i]-self.sd[:,i], self.filter_state[:,i]+self.sd[:,i], alpha=0.3, color='k')
                    ax.set_ylabel("x"+str(i))
                ax.set_xlabel("Time, s")

                (fig, axes) = plt.subplots(nrows=N_filter, ncols=1, sharex=True, num="Filter Covariance vs Time")
                fig.set_size_inches(11, 8.5)
                for i in range(N_filter):
                    ax = axes[i]
                    ax.plot(self.time, self.sd[:,i]**2, 'k', )
                    ax.set_ylabel("cov(x"+str(i)+")")
                ax.set_xlabel("Time, s")

        plt.show()

        return

    def animate_results(self, animate_wind, animate_thrust_vecs=True, fname=None):
        """
        Animate the results
        
        """

        # Animation (Slow)
        # Instead of viewing the animation live, you may provide a .mp4 filename to save.
        # ani = animate(self.time, self.x, self.R, self.wind, animate_wind, world=self.world, filename=fname)
        ani = animate_tiltquad(self.time, self.x, self.R, self.wind, self.rotor_thrust_vecs, animate_wind, animate_thrust_vecs, world=self.world, filename=fname)
        plt.show()

        return

    def unpack_results(self, result):

        # Unpack the dictionary of results
        time                = result['time']
        state               = result['state']
        control             = result['control']
        flat                = result['flat']
        imu_measurements    = result['imu_measurements']
        imu_gt              = result['imu_gt']
        mocap               = result['mocap_measurements']
        state_estimate      = result['state_estimate']

        # Unpack each result into NumPy arrays
        x = state['x']
        x_des = flat['x']
        v = state['v']
        v_des = flat['x_dot']

        q = state['q']
        q_des = control['cmd_q']
        w = state['w']

        # s_des = control['cmd_motor_speeds']
        # s = state['rotor_speeds']
        rotor_thrust_vecs = state['rotor_thrusts']
        rotor_thrust_vecs_des = control['cmd_motor_thrusts']
        M = control['cmd_moment']
        T = control['cmd_thrust']

        wind = state['wind']

        accel   = imu_measurements['accel']
        gyro    = imu_measurements['gyro']

        accel_gt = imu_gt['accel']

        x_mc = mocap['x']
        v_mc = mocap['v']
        q_mc = mocap['q']
        w_mc = mocap['w']

        filter_state = state_estimate['filter_state']
        covariance = state_estimate['covariance']
        if filter_state.shape[1] > 0:
            sd = 3*np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
            self.estimator_exists = True
        else:
            sd = []
            self.estimator_exists = False

        return (time, x, x_des, v, v_des, q, q_des, w, rotor_thrust_vecs, rotor_thrust_vecs_des, M, T, wind, accel, gyro, accel_gt, x_mc, v_mc, q_mc, w_mc, filter_state, covariance, sd)

def plot_map(ax, world_data, equal_aspect=True, color=None, edgecolor=None, alpha=1, world_bounds=True, axes=True):
    """
    Plots the map in the world data in a top-down 2D view. 
    Inputs:
        ax: The axis to plot on
        world_data: The world data to plot
        equal_aspect: Determines if the aspect ratio of the plot should be equal.
        color: The color of the buildings. If None (default), it will use the color of the buildings. 
        edgecolor: The edge color of the buildings. If None (default), it will use the color of the buildings.
        alpha: The alpha value of the buildings. If None (default), it will use the color of the buildings.
        world_bounds: Whether or not to plot the world bounds as a dashed line around the 2D plot. 
        axes: Whether or not to plot the axis labels
    Outputs:
        Plots the map in the axis of interest. 
    """
    from matplotlib.patches import Rectangle

    # Add a dashed rectangle for the world bounds
    if world_bounds:
        world_patch = Rectangle((world_data['bounds']['extents'][0], world_data['bounds']['extents'][2]), 
                                world_data['bounds']['extents'][1]-world_data['bounds']['extents'][0], world_data['bounds']['extents'][3]-world_data['bounds']['extents'][2], 
                                linewidth=1, edgecolor='k', facecolor='none', linestyle='dashed')
        ax.add_patch(world_patch)

    plot_xmin = world_data['bounds']['extents'][0]
    plot_xmax = world_data['bounds']['extents'][1]
    plot_ymin = world_data['bounds']['extents'][2]
    plot_ymax = world_data['bounds']['extents'][3]

    for block in world_data['blocks']:
        xmin = block['extents'][0]
        xmax = block['extents'][1]
        ymin = block['extents'][2]
        ymax = block['extents'][3]
        if color is None:
            building_color = tuple(block['color'])
        else:
            building_color = color
        if edgecolor is None:
            building_edge_color = tuple(block['color'])
        else:
            building_edge_color = edgecolor
        block_patch = Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), linewidth=1, edgecolor=building_edge_color, facecolor=building_color, alpha=alpha, fill=True)
        ax.add_patch(block_patch)

        if xmin < plot_xmin:
            plot_xmin = xmin
        if xmax > plot_xmax:
            plot_xmax = xmax
        if ymin < plot_ymin:
            plot_ymin = ymin
        if ymax > plot_ymax:
            plot_ymax = ymax

    ax.set_xlim([plot_xmin, plot_xmax])
    ax.set_ylim([plot_ymin, plot_ymax])

    if axes:
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    # Set the aspect ratio equal
    if equal_aspect:
        ax.set_aspect('equal')

    return

if __name__ == "__main__":

    from rotorpy.world import World

    # Get a list of the maps available under worlds. 
    available_worlds = [fname for fname in os.listdir(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','worlds'))) if 'json' in fname]

    # Load a random world
    world_fname = np.random.choice(available_worlds)
    world = World.from_file(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','worlds', world_fname)))
    
    # Plot the world. 
    (fig, ax) = plt.subplots(nrows=1, ncols=1, num="Top Down World View")
    plot_map(ax, world.world)
    ax.set_title(world_fname)
    plt.show()