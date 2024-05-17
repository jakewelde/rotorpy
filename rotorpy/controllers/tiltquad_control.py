import numpy as np
from scipy.spatial.transform import Rotation
import cvxpy as cp

class SE3TiltControl(object):
    """

    """
    def __init__(self, quad_params):
        """
        Parameters:
            quad_params, dict with keys specified in rotorpy/vehicles
        """

        # Quadrotor physical parameters.
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

        # You may define any additional constants you like including control gains.
        self.inertia = np.array([[self.Ixx, self.Ixy, self.Ixz],
                                 [self.Ixy, self.Iyy, self.Iyz],
                                 [self.Ixz, self.Iyz, self.Izz]]) # kg*m^2
        self.g = 9.81 # m/s^2

        # Gains  
        self.kp_pos = np.array([6.5,6.5,15])
        self.kd_pos = np.array([4.0, 4.0, 9])
        self.kp_att = 544
        self.kd_att = 46.64
        self.kp_vel = 0.1*self.kp_pos   # P gain for velocity controller (only used when the control abstraction is cmd_vel)

        # Linear map from individual rotor forces to scalar thrust and vector
        # moment applied to the vehicle.
        k = self.k_m/self.k_eta  # Ratio of torque to thrust coefficient. 

        # Below is an automated generation of the control allocator matrix. It assumes that all thrust vectors are aligned
        # with the z axis and that the "sign" of each rotor yaw moment alternates starting with positive for r1. 'TM' = "thrust and moments"
        self.f_to_TM = np.vstack((np.ones((1,self.num_rotors)),
                                  np.hstack([np.cross(self.rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos]), 
                                 (k * self.rotor_dir).reshape(1,-1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)

    def update_ref(self, t, flat_output):
        """
        This function receives the current time, and desired flat
        outputs. It returns the reference command inputs.
        Follows https://repository.upenn.edu/edissertations/547/

        Inputs:
            t, present time in seconds
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2  a
                x_dddot,  jerk, m/s**3          a_dot
                x_ddddot, snap, m/s**4          a_ddot
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
                yaw_ddot, yaw acceleration, rad/s**2  #required! not the same if computing command using controller

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
                cmd_w, angular velocity
                cmd_a, angular acceleration
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_q = np.zeros((4,))

        def normalize(x):
            """Return normalized vector."""
            return x / np.linalg.norm(x)

        # Desired force vector.
        t = flat_output['x_ddot']+ np.array([0, 0, self.g])
        b3 = normalize(t) 
        F_des = self.mass * (t)# this is vectorized

        # Control input 1: collective thrust. 
        u1 = np.dot(F_des, b3)

        # Desired orientation to obtain force vector.
        b3_des = normalize(F_des) #b3_des and b3 are the same
        yaw_des = flat_output['yaw']
        c1_des = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
        b2_des = normalize(np.cross(b3_des, c1_des))
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.stack([b1_des, b2_des, b3_des]).T

        R = R_des # assume we have perfect tracking on rotation
        
        # Following section follows Mellinger paper to compute reference angular velocity
        dot_u1 = np.dot(b3,flat_output['x_dddot'])
        hw = self.mass/u1*(flat_output['x_dddot']-dot_u1*b3)
        p  = np.dot(-hw, b2_des)
        q  = np.dot(hw, b1_des)
        w_des = np.array([0, 0, flat_output['yaw_dot']])
        r  = np.dot(w_des, b3_des)
        Omega = np.array([p, q, r])

        wwu1b3 = np.cross(Omega, np.cross(Omega, u1*b3))
        ddot_u1 = np.dot(b3, self.mass*flat_output['x_ddddot']) - np.dot(b3, wwu1b3)
        ha = 1.0/u1*(self.mass*flat_output['x_ddddot'] - ddot_u1*b3 - 2*np.cross(Omega,dot_u1*b3) - wwu1b3)
        p_dot = np.dot(-ha, b2_des)
        q_dot = np.dot(ha, b1_des)
        np.cross(Omega, Omega)
        r_dot = flat_output['yaw_ddot'] *np.dot(np.array([0,0,1.0]), b3_des) #uniquely need yaw_ddot
        Alpha = np.array([p_dot, q_dot, r_dot]) 

        # Control input 2: moment on each body axis
    
        u2 =  self.inertia @ Alpha + np.cross(Omega, self.inertia @ Omega)

        # Convert to cmd motor speeds. 
        TM = np.array([u1, u2[0], u2[1], u2[2]])
        cmd_motor_forces = self.TM_to_f @ TM
        cmd_motor_speeds = cmd_motor_forces / self.k_eta
        cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        cmd_q = Rotation.from_matrix(R_des).as_quat()


        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                        'cmd_thrust':u1,
                        'cmd_moment':u2,
                        'cmd_q':cmd_q,
                        'cmd_w':Omega,
                        'cmd_a':Alpha}
        return control_input
    

    def C_matrix_force(self,rotor_pos):
    
        Torque_body_matrix = np.zeros((3,3))

        r1_x, r1_y, r1_z = rotor_pos['r1']
        r2_x, r2_y, r2_z = rotor_pos['r2']
        r3_x, r3_y, r3_z = rotor_pos['r3']
        r4_x, r4_y, r4_z = rotor_pos['r4']


        R_hat_1 = np.array([[0, -r1_z, r1_y], [r1_z, 0, -r1_x], [-r1_y, r1_x, 0]])
        # print("R hat 1 shape", R_hat_1.shape)
        # print("Rhat matrix", R_hat_1)

        R_hat_2 = np.array([[0, -r2_z, r2_y], [r2_z, 0, -r2_x], [-r2_y, r2_x, 0]])

        R_hat_3 = np.array([[0, -r3_z, r3_y], [r3_z, 0, -r3_x], [-r3_y, r3_x, 0]])

        R_hat_4 = np.array([[0, -r4_z, r4_y], [r4_z, 0, -r4_x], [-r4_y, r4_x, 0]])


        C_matrix_top_half = np.tile(np.eye(3), (1, 4))
        C_matrix_bottom_half_raw = np.array(
            [[R_hat_1 - Torque_body_matrix], [R_hat_2 + Torque_body_matrix], [R_hat_3 - Torque_body_matrix],
            [R_hat_4 + Torque_body_matrix]])
        C_matrix_collapsed = C_matrix_bottom_half_raw.reshape(4, 3, 3)
        C_matrix_bottom_half = np.hstack(C_matrix_collapsed)

        C = np.vstack((C_matrix_top_half, C_matrix_bottom_half))

        return C





    def control_allocation_slidyquad(self,F_x, F_y, F_z, T_x, T_y, T_z):
        

        # Define variables
        F1_x = cp.Variable()
        F1_y = cp.Variable()
        F1_z = cp.Variable()

        F2_x = cp.Variable()
        F2_y = cp.Variable()
        F2_z = cp.Variable()

        F3_x = cp.Variable()
        F3_y = cp.Variable()
        F3_z = cp.Variable()

        F4_x = cp.Variable()
        F4_y = cp.Variable()
        F4_z = cp.Variable()

        Fs = [[F1_x, F1_y, F1_z], [F2_x, F2_y, F2_z], [F3_x, F3_y, F3_z], [F4_x, F4_y, F4_z]]

        # # print("Fs", Fs[0])

        F_all = cp.vstack([F1_x, F1_y, F1_z, F2_x, F2_y, F2_z, F3_x, F3_y, F3_z, F4_x, F4_y, F4_z])
        # F = cp.Variable((12,1))

        # print("F shape", F_all.shape)


        # desired_T_F = cp.Parameter(6, value = np.array([F_x, F_y, F_z, T_x, T_y, T_z]))

        desired_T_F = np.array([[F_x], [F_y], [F_z], [T_x], [T_y], [T_z]])

        # print("desired T_f ", desired_T_F.shape)

        # C_matrix = np.array([[1, 0,0,1,0,0,1,0,0,1,0,0],[0,1,0,0,1,0,0,1,0,0,1,0],[0,0,1,0,0,1,0,0,1,0,0,1],
        #                         [0.26916547, - 0.6259074,   0.15,        0.26916547, - 0.7159074 ,  0.15, 0.26916547, - 0.7159074 ,- 0.15 ,  0.26916547, - 0.7159074 ,- 0.15],
        #                         [0.28172339, - 0.62176047, - 0.2, 0.37172339, - 0.62176047, 0.2, 0.37172339, - 0.62176047, 0.2 ,0.37172339, - 0.62176047, 0.2],
        # [-0.09090219,  0.06477175, - 0.03553641, - 0.09090219, - 0.33522825, - 0.03553641, 0.20909781, - 0.33522825, - 0.03553641,  0.20909781, - 0.33522825, - 0.03553641]])

        C_matrix = self.C_matrix_force(self.rotor_pos)

        C = cp.Parameter((6, 12), value=C_matrix)

        # print(C.value)
        # print(" c shape", C.shape)

        # objective_new =
        # Define the constraints
        # constraints = [0.2173145989732144*x - 0.2560408941681052*y + 0.9419221972045823*z <= -1.6770471008442362]

        # constraint_new = [C @ F_all == desired_T_F]

        inequalities_sh = [[8.02533240e-01, - 5.91539818e-01, - 7.75953821e-02, - 0.00000000e+00],
        [9.13555129e-01, 3.99244264e-01, - 7.75953821e-02, - 0.00000000e+00],
        [-9.13555129e-01, - 3.99244264e-01, - 7.75953821e-02,  1.38777878e-17],
        [-1.11021889e-01, - 9.90784082e-01, - 7.75953821e-02, - 0.00000000e+00],
        [-8.02533240e-01, 5.91539818e-01, - 7.75953821e-02, - 0.00000000e+00],
        [1.11021889e-01,  9.90784082e-01, - 7.75953821e-02, - 0.00000000e+00],
        [1.40062414e-01, - 1.03238708e-01, 9.84745799e-01, - 8.35441668e-01],
        [-1.59438550e-01, - 6.96782544e-02,  9.84745799e-01, - 8.35441668e-01],
        [1.59438550e-01, 6.96782544e-02, 9.84745799e-01, - 8.35441668e-01],
        [-1.93761367e-02, - 1.72916962e-01,  9.84745799e-01, - 8.35441668e-01],
        [1.93761367e-02, 1.72916962e-01, 9.84745799e-01, - 8.35441668e-01],
        [-1.40062414e-01,  1.03238708e-01,  9.84745799e-01, - 8.35441668e-01]]
        constraint_new = []

        for i, (a_new, b_new, c_new, d_new) in enumerate(inequalities_sh):
            for F in Fs:
                constraint_new.append(a_new * F[0] + b_new * F[1] + c_new * F[2] <= -d_new)

        # for constraint in constraint_new:
            # print(constraint)

        # print(len(constraint_new))
        #
        # print(constraint_new)

        # Define the objective function
        # objective = cp.Minimize(x**2 + y**2 + z**2)
        alpha = 1e5
        effort_cost = alpha * ((F1_x ** 2 + F1_y ** 2) + (F2_x ** 2 + F2_y ** 2) + (F3_x ** 2 + F3_y ** 2) + (F4_x ** 2 + F4_y ** 2)) \
                    + (F1_z ** 2 + F2_z ** 2 + F3_z ** 2 + F4_z ** 2)
        # effort_cost = ((F1_x**2 + F1_y**2 + F1_z**2) + (F2_x**2 + F2_y**2 + F2_z**2) + (F3_x**2 + F3_y**2 + F3_z**2)+ (F4_x**2 + F4_y**2 + F4_z**2))
        force_cost = cp.sum_squares(C @ F_all - desired_T_F)
        # objective_new = cp.Minimize((0.5* cp.quad_form(F_all, np.eye(12))))

        for epsilon in np.logspace(1, -10, num=10, endpoint=True):
            total_cost = force_cost + (epsilon * effort_cost)

            # objective_new = cp.Minimize(force_cost + (epsilon * effort_cost))

            objective_new = cp.Minimize(total_cost)
            problem_1 = cp.Problem(objective_new, constraint_new)
            result = problem_1.solve(verbose=False)
            status = problem_1.status

            # print("Solution for eps=", epsilon, "Status: ", status, "Effort cost:", effort_cost.value, "Force cost:",
                # force_cost.value)

        # Print the results
        # # print(f"x = {x.value}")
        # # print(f"y = {y.value}")
        # # print(f"z = {z.value}")

        # Output the optimized values of F

        # print("Optimized Forces:")
        # print(f"F1_x: {F1_x.value}, F1_y: {F1_y.value}, F1_z: {F1_z.value}")
        # print(f"F2_x: {F2_x.value}, F2_y: {F2_y.value}, F2_z: {F2_z.value}")
        # print(f"F3_x: {F3_x.value}, F3_y: {F3_y.value}, F3_z: {F3_z.value}")
        # print(f"F4_x: {F4_x.value}, F4_y: {F4_y.value}, F4_z: {F4_z.value}")

        force_vector = np.array(
            [F1_x.value, F1_y.value, F1_z.value, F2_x.value, F2_y.value, F2_z.value, F3_x.value, F3_y.value, F3_z.value,
            F4_x.value, F4_y.value, F4_z.value])
        # print("force vector stacked", force_vector)
        # print("shape of forced vector", force_vector.shape)

        resulting_value = C.value @ force_vector

        # print("forces", resulting_value)

        results = (f"F1_x: {F1_x.value}, F1_y: {F1_y.value}, F1_z: {F1_z.value}\n"
                f"F2_x: {F2_x.value}, F2_y: {F2_y.value}, F2_z: {F2_z.value}\n"
                f"F3_x: {F3_x.value}, F3_y: {F3_y.value}, F3_z: {F3_z.value}\n"
                f"F4_x: {F4_x.value}, F4_y: {F4_y.value}, F4_z: {F4_z.value}")

        return results
    
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
            return np.array([-S[1,2], S[0,2], -S[0,1]])

        # Get the desired force vector.
        pos_err  = state['x'] - flat_output['x']
        dpos_err = state['v'] - flat_output['x_dot']
        F_des = self.mass * (- self.kp_pos*pos_err
                             - self.kd_pos*dpos_err
                             + flat_output['x_ddot']
                             + np.array([0, 0, self.g]))

        # Desired thrust is force projects onto b3 axis.
        R = Rotation.from_quat(state['q']).as_matrix()
        b3 = R @ np.array([0, 0, 1])
        u1 = np.dot(F_des, b3)

        # Desired orientation to obtain force vector.
        b3_des = normalize(F_des)
        yaw_des = flat_output['yaw']
        c1_des = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
        b2_des = normalize(np.cross(b3_des, c1_des))
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.stack([b1_des, b2_des, b3_des]).T

        # Orientation error.
        S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
        att_err = vee_map(S_err)

        # Angular velocity error (this is oversimplified).
        w_des = np.array([0, 0, flat_output['yaw_dot']])
        w_err = state['w'] - w_des

        # Desired torque, in units N-m.
        u2 = self.inertia @ (-self.kp_att*att_err - self.kd_att*w_err) + np.cross(state['w'], self.inertia@state['w'])  # Includes compensation for wxJw component

        # Compute command body rates by doing PD on the attitude error. 
        cmd_w = -self.kp_att*att_err - self.kd_att*w_err

        # control allocation
        F_x = F_des[0]
        F_y = F_des[1]
        F_z = F_des[2]
        T_x = u2[0]
        T_y = u2[1]
        T_z = u2[2]

        """
        control slidy - Returns a dictionary of force values in all three direction for 4 rotors
        """
        control_slidy = self.control_allocation_slidyquad(F_x, F_y, F_z, T_x, T_y, T_z)

        # Compute motor speeds. Avoid taking square root of negative numbers.
        TM = np.array([u1, u2[0], u2[1], u2[2]])
        # cmd_rotor_thrusts = self.TM_to_f @ TM

        ############ control allocation force extraction #####################
        # Extracting the values from the results string
        lines = control_slidy.split('\n')
        forces = []

        for line in lines:
            parts = line.split(',')
            force_values = [float(part.split(':')[1].strip()) for part in parts]
            forces.append(force_values)

        # Convert the list of forces to a NumPy array
        cmd_rotor_thrusts = np.array(forces)
        
        cmd_motor_speeds = np.linalg.norm(cmd_rotor_thrusts, axis=1) / self.k_eta
        cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        # Assign controller commands.
        cmd_thrust = u1                                             # Commanded thrust, in units N.
        cmd_moment = u2                                             # Commanded moment, in units N-m.
        cmd_q = Rotation.from_matrix(R_des).as_quat()               # Commanded attitude as a quaternion.
        cmd_v = -self.kp_vel*pos_err + flat_output['x_dot']     # Commanded velocity in world frame (if using cmd_vel control abstraction), in units m/s





        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_motor_thrusts':cmd_rotor_thrusts,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q,
                         'cmd_w':cmd_w,
                         'cmd_v':cmd_v}
        
        return control_input

if __name__ == "__main__":

    from rotorpy.trajectories.hover_traj import HoverTraj
    from rotorpy.vehicles.hummingbird_params import quad_params

    controller = SE3TiltControl(quad_params)


    traj = HoverTraj()
    flat_output = traj.update(0)

    state = {'x': np.array([0, 0, -1]), 'v': np.zeros(3,), 'q': np.array([0, 0, 0, 1]), 'w': np.zeros(3,), 'wind': np.zeros(3,)}
    
    import time
    start = time.time()
    cmd = controller.update(0, state, flat_output)
    print((time.time() - start)*1000)
    print(cmd)