import pinocchio as pin
import numpy as np

class PinocchioRegressorBuilder:
    """
    Constructs the dynamics regressor matrix using Pinocchio's analytical functions.
    """
    def __init__(self, urdf_path: str, num_joints: int):
        """
        Initializes the regressor builder.
        
        :param str urdf_path: Path to the robot's URDF file.
        :param int num_joints: The number of actuated joints.
        """
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.num_joints = num_joints
        
        # 10 base parameters per link for the rigid body part
        self.num_link_params = 10
        self.total_link_params = self.num_joints * self.num_link_params
        
        # 2 friction parameters per joint
        self.num_friction_params = 2
        self.total_friction_params = self.num_joints * self.num_friction_params
        
        self.total_params = self.total_link_params + self.total_friction_params

    def compute_regressor_matrix(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """
        Computes the full dynamics regressor matrix Y for a given state.

        :param q: Joint positions
        :param qd: Joint velocities
        :param qdd: Joint accelerations
        :return: The (num_joints x total_params) regressor matrix Y.
        """
        # Pinocchio state vectors
        q_full = np.concatenate([pin.neutral(self.model)[:7], q])
        qd_full = np.concatenate([np.zeros(6), qd])
        qdd_full = np.concatenate([np.zeros(6), qdd])

        # --- Part 1: Compute Rigid Body Parameter Regressor ---
        # This is the magic function. It computes the regressor for all link parameters.
        # Y_rnea = pin.computeRNEARegressor(self.model, self.data, q_full, qd_full, qdd_full)
        pin.computeAllTerms(self.model, self.data, q_full, qd_full, qdd_full)
        Y_rnea = pin.computeJointTorqueRegressor(self.model, self.data, q_full, qd_full, qdd_full)

        print("RNEA REGRESSOR : ", Y_rnea)
        
        # Pinocchio's parameter order: [m, mcx, mcy, mcz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
        # Your parameter order:        [Ixx, Ixy, Ixz, Iyy, Iyz, Izz, mcx, mcy, mcz, m]
        
        Y_rigid_body = np.zeros((self.num_joints, self.total_link_params))
        pin_order = [9, 1, 2, 3, 4, 5, 6, 7, 8, 0]
        user_order_map = [4, 5, 7, 6, 8, 9, 1, 2, 3, 0]

        pin.bodyRegressor

        for i in range(self.num_joints):
            # Columns for link i in Pinocchio's regressor
            pin_cols_start = (i + 1) * 10 # +1 to skip fixed base
            pin_cols_end = pin_cols_start + 10
            Y_pin_link_i = Y_rnea[:, pin_cols_start:pin_cols_end]

            # Columns for link i in our user-defined regressor
            user_cols_start = i * 10
            user_cols_end = user_cols_start + 10
            
            # Reorder the columns
            Y_rigid_body[:, user_cols_start:user_cols_end] = Y_pin_link_i[:, user_order_map]

        # --- Part 2: Append Friction Parameter Columns ---
        Y_friction = np.zeros((self.num_joints, self.total_friction_params))
        for i in range(self.num_joints):
            viscous_col_idx = i * self.num_friction_params
            Y_friction[i, viscous_col_idx] = qd[i]

            coulomb_col_idx = i * self.num_friction_params + 1
            Y_friction[i, coulomb_col_idx] = np.sign(qd[i])
            
        Y_full = np.hstack([Y_rigid_body, Y_friction])
        
        return Y_full