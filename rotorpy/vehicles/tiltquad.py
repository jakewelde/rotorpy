import numpy as np
from numpy.linalg import inv, norm
import scipy.integrate
from scipy.spatial.transform import Rotation
from rotorpy.vehicles.hummingbird_params import quad_params

import time

"""
Multirotor models
"""

def quat_dot(quat, omega):
    """
    Parameters:
        quat, [i,j,k,w]
        omega, angular velocity of body in body axes

    Returns
        duat_dot, [i,j,k,w]

    """
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
    G = np.array([[ q3,  q2, -q1, -q0],
                  [-q2,  q3,  q0, -q1],
                  [ q1, -q0,  q3, -q2]])
    quat_dot = 0.5 * G.T @ omega
    # Augment to maintain unit quaternion.
    quat_err = np.sum(quat**2) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot - quat_err * quat_err_grad
    return quat_dot

class TiltQuad(object):
    """
    Multirotor forward dynamics model augmented with propellers that can tilt to create lateral forces. 

    states: [position, velocity, attitude, body rates, wind, rotor thrusts]

    rotor_thrusts is a numpy array of shape (4, 3), where the i'th row corresponds to the i'th rotor, and the columns are the x/y/z axes. 

    We assume that the rotor thrusts are nominally aligned with the body z axis of the quadrotor. 

    Parameters:
        quad_params: a dictionary containing relevant physical parameters for the TiltQuad. 
        initial_state: the initial state of the vehicle. 
        aero: boolean, determines whether or not aerodynamic drag forces are computed. 
    """
    def __init__(self, quad_params, aero=True, initial_state = {'x': np.array([0,0,0]),
                                            'v': np.zeros(3,),
                                            'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                                            'w': np.zeros(3,),
                                            'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                                            'rotor_thrusts': np.zeros((4, 3))},
                ):
        """
        Initialize quadrotor physical parameters.
        """

        # Inertial parameters
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.Ixy             = quad_params['Ixy']  # kg*m^2
        self.Ixz             = quad_params['Ixz']  # kg*m^2
        self.Iyz             = quad_params['Iyz']  # kg*m^2

        # Frame parameters
        self.c_Dx            = quad_params['c_Dx']  # drag coeff, N/(m/s)**2
        self.c_Dy            = quad_params['c_Dy']  # drag coeff, N/(m/s)**2
        self.c_Dz            = quad_params['c_Dz']  # drag coeff, N/(m/s)**2

        self.num_rotors      = quad_params['num_rotors']
        self.rotor_pos       = quad_params['rotor_pos']

        self.rotor_dir       = quad_params['rotor_directions']

        self.extract_geometry()

        # Rotor parameters    
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s

        self.k_eta           = quad_params['k_eta']     # thrust coeff, N/(rad/s)**2
        self.k_m             = quad_params['k_m']       # yaw moment coeff, Nm/(rad/s)**2
        self.k_d             = quad_params['k_d']       # rotor drag coeff, N/(m/s)
        self.k_z             = quad_params['k_z']       # induced inflow coeff N/(m/s)
        self.k_flap          = quad_params['k_flap']    # Flapping moment coefficient Nm/(m/s)

        # Motor parameters
        self.tau_m           = quad_params['tau_m']     # motor reponse time, seconds
        self.motor_noise     = quad_params['motor_noise_std'] # noise added to the actual motor speed, rad/s / sqrt(Hz)

        # Additional constants.
        self.inertia = np.array([[self.Ixx, self.Ixy, self.Ixz],
                                 [self.Ixy, self.Iyy, self.Iyz],
                                 [self.Ixz, self.Iyz, self.Izz]])
        self.drag_matrix = np.array([[self.c_Dx,    0,          0],
                                     [0,            self.c_Dy,  0],
                                     [0,            0,          self.c_Dz]])
        self.g = 9.81 # m/s^2

        self.inv_inertia = inv(self.inertia)
        self.weight = np.array([0, 0, -self.mass*self.g])

        # Set the initial state
        self.initial_state = initial_state

        self.aero = aero

    def extract_geometry(self):
        """
        Extracts the geometry in self.rotors for efficient use later on in the computation of 
        wrenches acting on the rigid body.
        The rotor_geometry is an array of length (n,3), where n is the number of rotors. 
        Each row corresponds to the position vector of the rotor relative to the CoM. 
        """

        self.rotor_geometry = np.array([]).reshape(0,3)
        for rotor in self.rotor_pos:
            r = self.rotor_pos[rotor]
            self.rotor_geometry = np.vstack([self.rotor_geometry, r])

        return

    def statedot(self, state, control, t_step):
        """
        Integrate dynamics forward from state given constant cmd_rotor_speeds for time t_step.
        """

        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control)

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = np.clip(cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max) 

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds)
        s = TiltQuad._pack_state(state)
        
        s_dot = s_dot_fn(0, s)
        v_dot = s_dot[3:6]
        w_dot = s_dot[10:13]

        state_dot = {'vdot': v_dot,'wdot': w_dot}
        return state_dot 


    def step(self, state, control, t_step):
        """
        Integrate dynamics forward from state given constant control for time t_step.
        """

        cmd_rotor_thrusts = control['cmd_motor_thrusts']

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_thrusts)
        s = TiltQuad._pack_state(state)

        # Option 1 - RK45 integration
        sol = scipy.integrate.solve_ivp(s_dot_fn, (0, t_step), s, first_step=t_step)
        s = sol['y'][:,-1]
        # Option 2 - Euler integration
        # s = s + s_dot_fn(0, s) * t_step  # first argument doesn't matter. It's time invariant model

        state = TiltQuad._unpack_state(s)

        # Re-normalize unit quaternion.
        state['q'] = state['q'] / norm(state['q'])

        return state

    def _s_dot_fn(self, t, s, cmd_rotor_thrusts):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """

        state = TiltQuad._unpack_state(s)

        rotor_thrusts = state['rotor_thrusts']
        inertial_velocity = state['v']
        wind_velocity = state['wind']
        rotor_thrusts = state['rotor_thrusts']

        R = Rotation.from_quat(state['q']).as_matrix()

        # Position derivative.
        x_dot = state['v']

        # Orientation derivative.
        q_dot = quat_dot(state['q'], state['w'])

        # Rotor acceleration
        rotor_acc = (1/self.tau_m)*(cmd_rotor_thrusts - rotor_thrusts)

        # Compute airspeed vector in the body frame
        body_airspeed_vector = R.T@(inertial_velocity - wind_velocity)

        # Compute total wrench in the body frame based on the current rotor speeds and their location w.r.t. CoM
        (FtotB, MtotB) = self.compute_body_wrench(state['w'], rotor_thrusts, body_airspeed_vector)

        # Rotate the force from the body frame to the inertial frame
        Ftot = R@FtotB

        # Velocity derivative.
        v_dot = (self.weight + Ftot) / self.mass

        # Angular velocity derivative.
        w = state['w']
        w_hat = TiltQuad.hat_map(w)
        w_dot = self.inv_inertia @ (MtotB - w_hat @ (self.inertia @ w))

        # NOTE: the wind dynamics are currently handled in the wind_profile object. 
        # The line below doesn't do anything, as the wind state is assigned elsewhere. 
        wind_dot = np.zeros(3,)

        # Pack into vector of derivatives.
        s_dot = np.zeros((16+self.num_rotors*3,))
        s_dot[0:3]   = x_dot
        s_dot[3:6]   = v_dot
        s_dot[6:10]  = q_dot
        s_dot[10:13] = w_dot
        s_dot[13:16] = wind_dot
        s_dot[16:] = rotor_acc.ravel()

        return s_dot

    def compute_body_wrench(self, body_rates, rotor_thrusts, body_airspeed_vector):
        """
        Computes the wrench acting on the rigid body based on the rotor speeds for thrust and airspeed 
        for aerodynamic forces. 
        The airspeed is represented in the body frame.
        The net force Ftot is represented in the body frame. 
        The net moment Mtot is represented in the body frame. 
        """

        # Add in aero wrenches (if applicable)
        if self.aero:
            # Parasitic drag force acting at the CoM
            D = -TiltQuad._norm(body_airspeed_vector)*self.drag_matrix@body_airspeed_vector
        else:
            D = np.zeros(3,)

        # Compute the moments due to the rotor thrusts, rotor drag (if applicable), and rotor drag torques
        # M_force = -np.einsum('ijk, ik->j', TiltQuad.hat_map(self.rotor_geometry), rotor_thrusts.T)
        M_force = np.zeros(3,)
        for i in range(self.num_rotors):
            M_force += TiltQuad.hat_map(self.rotor_geometry[i, :])@rotor_thrusts[i, :]
        M_yaw = self.rotor_dir*(np.array([0, 0, self.k_m/self.k_eta])[:, np.newaxis]*np.linalg.norm(rotor_thrusts, axis=1))

        # Sum all elements to compute the total body wrench
        FtotB = np.sum(rotor_thrusts, axis=0) + D
        MtotB = M_force + np.sum(M_yaw, axis=1)

        return (FtotB, MtotB)

    @classmethod
    def rotate_k(cls, q):
        """
        Rotate the unit vector k by quaternion q. This is the third column of
        the rotation matrix associated with a rotation by q.
        """
        return np.array([  2*(q[0]*q[2]+q[1]*q[3]),
                           2*(q[1]*q[2]-q[0]*q[3]),
                         1-2*(q[0]**2  +q[1]**2)    ])

    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        In the vectorized implementation, we assume that s is in the shape (N arrays, 3)
        """
        if len(s.shape) > 1:  # Vectorized implementation
            return np.array([[ np.zeros(s.shape[0]), -s[:,2],  s[:,1]],
                             [ s[:,2],     np.zeros(s.shape[0]), -s[:,0]],
                             [-s[:,1],  s[:,0],     np.zeros(s.shape[0])]])
        else:
            return np.array([[    0, -s[2],  s[1]],
                             [ s[2],     0, -s[0]],
                             [-s[1],  s[0],     0]])

    @classmethod
    def _pack_state(cls, state):
        """
        Convert a state dict to Quadrotor's private internal vector representation.
        """
        s = np.zeros((16+12,))   # FIXME: this shouldn't be hardcoded. Should vary with the number of rotors. 
        s[0:3]   = state['x']       # inertial position
        s[3:6]   = state['v']       # inertial velocity
        s[6:10]  = state['q']       # orientation
        s[10:13] = state['w']       # body rates
        s[13:16] = state['wind']    # wind vector
        s[16:]   = state['rotor_thrusts'].ravel()     # rotor speeds

        return s

    @classmethod
    def _norm(cls, v):
        """
        Given a vector v in R^3, return the 2 norm (length) of the vector
        """
        norm = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
        return norm

    @classmethod
    def _unpack_state(cls, s):
        """
        Convert Quadrotor's private internal vector representation to a state dict.
        x = inertial position
        v = inertial velocity
        q = orientation
        w = body rates
        wind = wind vector
        rotor_speeds = rotor speeds
        """
        state = {'x':s[0:3], 'v':s[3:6], 'q':s[6:10], 'w':s[10:13], 'wind':s[13:16], 'rotor_thrusts':s[16:].reshape(4,3)}
        return state

if __name__ == "__main__":

    from rotorpy.vehicles.hummingbird_params import quad_params
    import matplotlib.pyplot as plt 

    quad_params['tau_m'] = 0.5

    quad = TiltQuad(quad_params, aero=False)

    control = {'cmd_motor_thrusts': np.array([[0, 0, quad.mass*quad.g/4],
                                              [0, 0, quad.mass*quad.g/4],
                                              [0, 0, quad.mass*quad.g/4],
                                              [0, 0, quad.mass*quad.g/4]])}
    
    initial_state={'x': np.zeros(3,), 'v':np.zeros(3,), 'q': np.array([0, 0, 0, 1]), 'wind': np.zeros(3,), 'w': np.zeros(3,), 'rotor_thrusts': control['cmd_motor_thrusts']}
    quad.initial_state = initial_state

    tf = 10
    dt = 0.01
    time = np.arange(0, tf+dt, dt)

    state = [quad.initial_state]
    rotor_thrusts = np.zeros((len(time), 4, 3))
    position = np.zeros((len(time), 3))
    w = np.zeros((len(time), 3))

    for i, t in enumerate(time):
        rotor_thrusts[i] = state[-1]['rotor_thrusts'].reshape(4, 3)
        position[i] = state[-1]['x']
        w[i] = state[-1]['w']
        state.append(quad.step(state[-1], control, t_step=dt))

    fig, axes = plt.subplots(nrows=2, ncols=2, num="Rotor Forces")
    for i in range(4):
        ax = axes.ravel()[i]
        ax.plot(time, rotor_thrusts[:, i, 0], label='X')
        ax.plot(time, rotor_thrusts[:, i, 1], label='Y')
        ax.plot(time, rotor_thrusts[:, i, 2], label='Z')
        ax.set_xlabel("Time, s")
        ax.set_ylabel("Thrust, N")
        ax.legend()
    fig.tight_layout()

    fig, axes = plt.subplots(nrows=1, ncols=1, num="Position")
    axes.plot(time, position[:, 0], 'r-', label='X')
    axes.plot(time, position[:, 1], 'g-', label='Y')
    axes.plot(time, position[:, 2], 'b-', label='Z')
    axes.set_xlabel("Time, s")
    axes.set_ylabel("Position, m")

    fig, axes = plt.subplots(nrows=1, ncols=1, num="Body Rates")
    axes.plot(time, w[:, 0], 'r-', label='X')
    axes.plot(time, w[:, 1], 'g-', label='Y')
    axes.plot(time, w[:, 2], 'b-', label='Z')
    axes.set_xlabel("Time, s")
    axes.set_ylabel("Body Rates, rad/s")

    plt.show()