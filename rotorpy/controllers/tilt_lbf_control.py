import math

import numpy as np
from scipy.spatial.transform import Rotation
from rotorpy.controllers.ControlAllocation import ControlAllocation
from sympy import Matrix, symbols, lambdify


def hat_map(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


s1, s2, s3 = symbols('s1 s2 s3')
H_2_var = Matrix([1 - pow(s1, 2) / (1 + s3), - (s1 * s2) / (1 + s3), s1,
                  -(s1 * s2) / (1 + s3), 1 - pow(s2, 2) / (1 + s3), s2,
                  -s1, -s2, s3])
H_2_jac = H_2_var.jacobian([s1, s2, s3])
H_2_jac_func = lambdify((s1, s2, s3), H_2_jac, modules='numpy')


class SE3TiltLBFControl(object):
    """
    """

    def __init__(self, quad_params, phi):
        """
        Parameters:
            quad_params, dict with keys specified in rotorpy/vehicles
        """

        # Quadrotor physical parameters.
        # Inertial parameters
        self.mass = quad_params['mass']  # kg
        self.Ixx = quad_params['Ixx']  # kg*m^2
        self.Iyy = quad_params['Iyy']  # kg*m^2
        self.Izz = quad_params['Izz']  # kg*m^2
        self.Ixy = quad_params['Ixy']  # kg*m^2
        self.Ixz = quad_params['Ixz']  # kg*m^2
        self.Iyz = quad_params['Iyz']  # kg*m^2

        # Frame parameters
        self.c_Dx = quad_params['c_Dx']  # drag coeff, N/(m/s)**2
        self.c_Dy = quad_params['c_Dy']  # drag coeff, N/(m/s)**2
        self.c_Dz = quad_params['c_Dz']  # drag coeff, N/(m/s)**2

        self.num_rotors = quad_params['num_rotors']
        self.rotor_pos = quad_params['rotor_pos']
        self.rotor_dir = quad_params['rotor_directions']

        # Rotor parameters
        self.rotor_speed_min = quad_params['rotor_speed_min']  # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max']  # rad/s

        self.k_eta = quad_params['k_eta']  # thrust coeff, N/(rad/s)**2
        self.k_m = quad_params['k_m']  # yaw moment coeff, Nm/(rad/s)**2
        self.k_d = quad_params['k_d']  # rotor drag coeff, N/(m/s)
        self.k_z = quad_params['k_z']  # induced inflow coeff N/(m/s)
        self.k_flap = quad_params['k_flap']  # Flapping moment coefficient Nm/(m/s)

        # Motor parameters
        self.tau_m = quad_params['tau_m']  # motor reponse time, seconds

        # You may define any additional constants you like including control gains.
        self.inertia = np.array([[self.Ixx, self.Ixy, self.Ixz],
                                 [self.Ixy, self.Iyy, self.Iyz],
                                 [self.Ixz, self.Iyz, self.Izz]])  # kg*m^2
        self.g = 9.81  # m/s^2

        # Gains
        self.kp_pos = np.array([6.5, 6.5, 15]) * self.mass
        self.kd_pos = np.array([4.0, 4.0, 9]) * self.mass
        self.kp_att = self.inertia @ (np.eye(3) * 544)
        self.kd_att = self.inertia @ (np.eye(3) * 46.64)
        self.kp_vel = 0.1 * self.kp_pos / self.mass  # P gain for velocity controller (only used when the control abstraction is cmd_vel)
        # w_n = 1.5  # Freq
        # xi = 2.0  # Damping
        # self.kp_pos = pow(2 * math.pi * w_n, 2) * self.mass
        # self.kd_pos = 2 * xi * 2 * math.pi * w_n * self.mass
        #
        # a_w_n = 5.0  # Freq
        # a_xi = 2.0  # Damping
        # self.kp_att = self.inertia @ (np.eye(3) * pow(2 * math.pi * a_w_n, 2))
        # self.kd_att = self.inertia @ (np.eye(3) * (2 * a_xi * 2 * math.pi * a_w_n))
        # self.kp_vel = 0.1 * self.kp_pos  # P gain for velocity controller (only used when the control abstraction is cmd_vel)

        # Linear map from individual rotor forces to scalar thrust and vector
        # moment applied to the vehicle.
        k = self.k_m / self.k_eta  # Ratio of torque to thrust coefficient.

        # Below is an automated generation of the control allocator matrix. It assumes that all thrust vectors are aligned
        # with the z axis and that the "sign" of each rotor yaw moment alternates starting with positive for r1. 'TM' = "thrust and moments"
        self.f_to_TM = np.vstack((np.ones((1, self.num_rotors)),
                                  np.hstack(
                                      [np.cross(self.rotor_pos[key], np.array([0, 0, 1])).reshape(-1, 1)[0:2] for key in
                                       self.rotor_pos]),
                                  (k * self.rotor_dir).reshape(1, -1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)

        r = np.vstack([self.rotor_pos['r1'], self.rotor_pos['r2'], self.rotor_pos['r3'], self.rotor_pos['r4']])
        a = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
        # self.phi = math.pi / 2
        # phi = 1e-3
        self.phi = phi
        min_f = self.k_eta * self.rotor_speed_min * self.rotor_speed_min
        max_f = self.k_eta * self.rotor_speed_max * self.rotor_speed_max
        kQ = k * self.rotor_dir
        self.control_allocation = ControlAllocation(r, a, kQ, self.phi, min_f, max_f)

    def find_r_des_lbf(self, t, F):
        R_r = np.eye(3)
        b_3 = R_r[:, 2]
        d = np.cross(b_3, F)
        if np.linalg.norm(d) == 0:
            raise Exception("d is norm 0")
        else:
            d_n = d / np.linalg.norm(d)
        trip_prod = np.dot(np.cross(b_3, F), d_n)
        theta = trip_prod / abs(trip_prod) * np.arccos(np.dot(b_3, F) / (np.linalg.norm(b_3) * np.linalg.norm(F)))

        R_des = R_r
        if abs(theta) > self.phi:
            a = theta - theta / abs(theta) * self.phi
            a = (a + 2 * np.pi) % (2 * np.pi)
            if abs(a) > np.pi:
                a -= a / abs(a) * 2 * np.pi
            a = -a
            R_matrix = np.matrix(np.array([[np.cos(a) + pow(d_n[0], 2) * (1 - np.cos(a)),
                                            d_n[0] * d_n[1] * (1 - np.cos(a)) - d_n[2] * np.sin(a),
                                            d_n[0] * d_n[2] * (1 - np.cos(a) + d_n[1] * np.sin(a))],
                                           [d_n[1] * d_n[0] * (1 - np.cos(a)) + d_n[2] * np.sin(a),
                                            np.cos(a) + pow(d_n[1], 2) * (1 - np.cos(a)),
                                            d_n[1] * d_n[2] * (1 - np.cos(a)) - d_n[0] * np.sin(a)],
                                           [d_n[2] * d_n[0] * (1 - np.cos(a)) - d_n[1] * np.sin(a),
                                            d_n[2] * d_n[1] * (1 - np.cos(a)) + d_n[0] * np.sin(a),
                                            np.cos(a) + pow(d_n[2], 2) * (1 - np.cos(a))]])).T
            b_d_3 = R_matrix @ b_3
            b_d_3 = np.array(b_d_3)[0]
            R_des[:, 0] = np.cross(np.cross(b_d_3, R_r[:, 0]), b_d_3)
            R_des[:, 1] = np.cross(b_d_3, R_r[:, 0])
            R_des[:, 2] = b_d_3

        return R_des

    def find_o_des(self, r_des, F, R, state, flat_output):
        F = R @ F
        F_des_n = F / np.linalg.norm(F)
        F_n = F_des_n.reshape((3, 1))
        F_dot = -self.kp_pos * (state['v'] - flat_output['x_dot']) - self.kd_pos * (-self.g * np.array([0, 0, 1]) +
                                                                                    (np.array(F) / self.mass) -
                                                                                    flat_output['x_ddot'])
        F_dot = F_dot.reshape((3, 1))
        F_n_dot = ((np.eye(3) - (F_n @ F_n.T)) @ F_dot) / np.linalg.norm(F)
        r_des_dot = H_2_jac_func(F_n[0][0], F_n[1][0], F_n[2][0]) @ F_n_dot
        o_des_hat = r_des.T @ r_des_dot.reshape((3, 3))
        o_des = np.array([o_des_hat[2][1], o_des_hat[0][2], o_des_hat[1][0]])
        return o_des

    def control_allocation_slidyquad(self, F, T):
        self.control_allocation.solve_prob(F, T)
        f = self.control_allocation.res.x.reshape(4, 3)
        # self.control_allocation.plot_forces()
        return f

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.
        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_motor_thrusts, N
                cmd_thrust, N
                cmd_moment, N*m
                cmd_q, quaternion [i,j,k,w]
                cmd_w, angular rates in the body frame, rad/s
                cmd_v, velocity in the world frame, m/s
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        def normalize(x):
            """Return normalized vector."""
            return x / np.linalg.norm(x)

        def vee_map(S):
            """Return vector corresponding to given skew symmetric matrix."""
            return np.array([-S[1, 2], S[0, 2], -S[0, 1]])

        # Get the desired force vector.
        pos_err = state['x'] - flat_output['x']
        dpos_err = state['v'] - flat_output['x_dot']
        F_des = - self.kp_pos * pos_err - self.kd_pos * dpos_err + \
                self.mass * (flat_output['x_ddot'] + np.array([0, 0, self.g]))

        R_des = self.find_r_des_lbf(t, F_des)

        # Convert F_des to body frame + Saturation
        R = Rotation.from_quat(state['q']).as_matrix()
        u1 = R.T @ F_des
        u1_xy = np.array([u1[0], u1[1]])
        r = u1[2] * np.tan(self.phi)
        if r < np.linalg.norm(u1_xy):
            u1_xy_n = normalize(u1_xy)
            u1 = np.array([r * u1_xy_n[0], r * u1_xy_n[1], u1[2]])

        # M and o_des
        # o_des = self.find_o_des(R_des, u1, R, state, flat_output)
        o_des = np.zeros(3)
        rt_rd = R.T @ R_des
        S_err = 0.5 * (R_des.T @ R - rt_rd)
        att_err = vee_map(S_err)
        w_err = state['w'] - rt_rd @ o_des
        O_hat = hat_map(state['w'])
        o_des_dot = np.array([0.0, 0.0, 0.0])
        u2 = -self.kp_att @ att_err - self.kd_att @ w_err + np.cross(state['w'], self.inertia @ state[
            'w']) - self.inertia @ (O_hat @ rt_rd @ o_des - rt_rd @ o_des_dot)

        # Compute command body rates by doing PD on the attitude error.
        # cmd_w = np.linalg.inv(self.inertia) @ (-self.kp_att * att_err - self.kd_att * w_err)
        cmd_w = np.array([0.0, 0.0, 0.0])

        """
        control slidy - Returns a dictionary of force values in all three direction for 4 rotors
        """
        cmd_rotor_thrusts = self.control_allocation_slidyquad(u1.tolist(), u2.tolist())

        # Compute motor speeds. Avoid taking square root of negative numbers.
        # TM = np.array([u1, u2[0], u2[1], u2[2]])
        # cmd_rotor_thrusts = self.TM_to_f @ TM

        # cmd_motor_speeds = np.linalg.norm(cmd_rotor_thrusts, axis=1) / self.k_eta
        # cmd_motor_speeds = cmd_rotor_thrusts / self.k_eta
        # cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))
        cmd_motor_speeds = np.zeros(3)

        # Assign controller commands.
        cmd_thrust = u1  # Commanded thrust, in units N.
        cmd_moment = u2  # Commanded moment, in units N-m.
        cmd_q = Rotation.from_matrix(R_des).as_quat()  # Commanded attitude as a quaternion.
        cmd_v = -self.kp_vel * pos_err + flat_output[
            'x_dot']  # Commanded velocity in world frame (if using cmd_vel control abstraction), in units m/s

        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_motor_thrusts': cmd_rotor_thrusts,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment,
                         'cmd_q': cmd_q,
                         'cmd_w': cmd_w,
                         'cmd_v': cmd_v}

        return control_input

# if __name__ == "__main__":
#     from rotorpy.trajectories.hover_traj import HoverTraj
#     from rotorpy.vehicles.hummingbird_params import quad_params
#
#     controller = SE3TiltLBFControl(quad_params)
#
#     traj = HoverTraj()
#     flat_output = traj.update(0)
#
#     state = {'x': np.array([0, 0, -1]), 'v': np.zeros(3, ), 'q': np.array([0, 0, 0, 1]), 'w': np.zeros(3, ),
#              'wind': np.zeros(3, )}
#
#     import time
#
#     start = time.time()
#     cmd = controller.update(0, state, flat_output)
#     print((time.time() - start) * 1000)
#     print(cmd)
