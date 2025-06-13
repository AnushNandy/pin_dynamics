import numpy as np
from dynamics.rnea import RobotDynamics

class RegressorBuilder:
    """
    Constructs the dynamics regressor matrix numerically for a serial-link manipulator.
    This approach is based on the linearity of the manipulator dynamic equations.
    The regressor Y(q, qd, qdd) maps a vector of dynamic parameters P to joint torques tau:
        tau = Y(q, qd, qdd) * P
    """

    def __init__(self, config):
        """
        Initializes the regressor builder.

        :param config: The robot configuration object, containing kinematic and dynamic info.
        """
        self.config = config
        self.num_joints = config.NUM_JOINTS

        # 10 base parameters per link:
        # [Ixx, Ixy, Ixz, Iyy, Iyz, Izz, mcx, mcy, mcz, m]
        # and 2 friction parameters per joint:
        # [Fv (viscous), Fc (Coulomb)]
        self.num_link_params = 10
        self.num_friction_params = 2

        self.total_link_params = self.num_joints * self.num_link_params
        self.total_friction_params = self.num_joints * self.num_friction_params
        self.total_params = self.total_link_params + self.total_friction_params

        self.robot_dyn = RobotDynamics(config)

    def _create_param_vector_from_config(self):
        """Helper to create the full parameter vector P from the config file."""
        P = np.zeros(self.total_params)
        for i in range(self.num_joints):
            link_name = self.config.LINK_NAMES_IN_KDL_ORDER[i + 1]  # Skip base link
            dyn_params = self.config.LINK_DYNAMIC_PARAMETERS[link_name]

            mass = dyn_params['mass']
            com = np.array(dyn_params['com'])
            mc = mass * com

            I_origin = np.array(dyn_params['inertia_tensor'])
            skew_pc = self._get_skew_symmetric_matrix(com)
            I_com = I_origin + mass * (skew_pc @ skew_pc.T)

            p_link = np.array([
                I_com[0, 0], I_com[0, 1], I_com[0, 2],
                I_com[1, 1], I_com[1, 2], I_com[2, 2],
                mc[0], mc[1], mc[2], mass
            ])

            start_idx = i * self.num_link_params
            end_idx = start_idx + self.num_link_params
            P[start_idx:end_idx] = p_link

        # Friction parameters from config are not added here, as they are typically unknown and what we want to identify.
        return P

    def _get_skew_symmetric_matrix(self, v):
        """Converts a 3D vector to a skew-symmetric matrix."""
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    def _update_dynamics_from_param_vector(self, P_vec):
        """
        Updates the internal state of the RobotDynamics object with a given parameter vector P.
        This is the key function for the numerical regressor calculation.
        """
        for i in range(self.num_joints):
            link_name = self.robot_dyn.link_names[i + 1]
            link_idx_in_dyn = self.robot_dyn.link_names.index(link_name)

            start_idx = i * self.num_link_params
            end_idx = start_idx + self.num_link_params
            p_link = P_vec[start_idx:end_idx]

            ixx, ixy, ixz, iyy, iyz, izz, mcx, mcy, mcz, mass = p_link

            if np.abs(mass) < 1e-9:
                com = np.zeros(3)
            else:
                com = np.array([mcx, mcy, mcz]) / mass

            I_com = np.array([[ixx, ixy, ixz],
                              [ixy, iyy, iyz],
                              [ixz, iyz, izz]])

            self.robot_dyn.links[link_idx_in_dyn]['mass'] = mass
            self.robot_dyn.links[link_idx_in_dyn]['p_com'] = com
            self.robot_dyn.links[link_idx_in_dyn]['I_com'] = I_com

    def compute_regressor_matrix(self, q, qd, qdd, gravity=np.array([0, 0, -9.81])):
        """
        Computes the full dynamics regressor matrix Y for a given state.

        :param q: Joint positions
        :param qd: Joint velocities
        :param qdd: Joint accelerations
        :param gravity: Gravity vector
        :return: The (num_joints x total_params) regressor matrix Y.
        """
        Y = np.zeros((self.num_joints, self.total_params))

        # --- Part 1: Rigid Body Parameter Columns ---
        for i in range(self.total_link_params):

            P_test = np.zeros(self.total_params)
            P_test[i] = 1.0

            # Configure RNEA engine
            self._update_dynamics_from_param_vector(P_test)
            tau_col = self.robot_dyn.compute_rnea(q, qd, qdd, gravity, f_ext=np.zeros(6))
            Y[:, i] = tau_col

        # --- Part 2: Friction Parameter Columns ---
        # For each joint, add columns for its friction model.
        # simple linear model: tau_friction = Fv * qd + Fc * sign(qd)
        for i in range(self.num_joints):
            # Viscous friction column: contribution is qd[i]
            viscous_col_idx = self.total_link_params + i * self.num_friction_params
            Y[i, viscous_col_idx] = qd[i]

            # Coulomb friction column: contribution is sign(qd[i])
            coulomb_col_idx = self.total_link_params + i * self.num_friction_params + 1
            Y[i, coulomb_col_idx] = np.sign(qd[i])

        return Y