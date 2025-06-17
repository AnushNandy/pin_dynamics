import pinocchio as pin
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import robot_config

def smooth_sign(v, threshold=0.01):
    return np.tanh(v / threshold)

class PinocchioAndFrictionRegressorBuilder:
    """
    Constructs the dynamics regressor matrix for BOTH rigid-body parameters
    and friction parameters.

    This definitive version is robust to different URDF structures (fixed base vs. floating)
    by correctly constructing the full state vectors.
    """
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.num_joints = robot_config.NUM_JOINTS

        print("\n--- REGRESSOR BUILDER MODEL INSPECTION ---")
        print(f"Model nq: {self.model.nq}") 
        print(f"Model nv: {self.model.nv}") 
        print(f"Number of bodies in model: {self.model.nbodies}")
        print("------------------------------------------\n")

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.num_link_params = 10
        self.total_link_params = self.num_joints * self.num_link_params
        
        # 2 friction parameters (Viscous, Coulomb) per joint
        self.num_friction_params = 2
        self.total_friction_params = self.num_joints * self.num_friction_params
        
        self.total_params = self.total_link_params + self.total_friction_params
        self.rnea_param_col_indices = []
        first_actuated_body_idx = 2
        for i in range(self.num_joints):
            # The body index corresponding to the i-th actuated joint
            body_id = first_actuated_body_idx + i
            
            # The regressor columns are indexed starting from body 1.
            # So, parameters for body_id=1 start at col 0.
            # Parameters for body_id=2 start at col 10.
            # Parameters for body_id=j start at col (j-1)*10.
            start_col = (body_id - 1) * self.num_link_params
            self.rnea_param_col_indices.extend(range(start_col, start_col + self.num_link_params))

    def compute_regressor_matrix(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, gravity: np.ndarray = np.array([0, 0, -9.81])) -> np.ndarray:
        """
        Computes the full dynamics regressor matrix Y for a given state.
        τ = Y * [P_rnea, P_friction]^T
        """
        # --- 1. Construct Full State Vectors ---
        # Create zero-vectors of the correct, full size for the model
        q_full = pin.neutral(self.model) # Creates a valid q vector of size model.nq
        qd_full = np.zeros(self.nv)      # Creates a v vector of size model.nv
        qdd_full = np.zeros(self.nv)     # Creates an a vector of size model.nv

        # Place the actuated joint values into the correct slice.
        q_full[7 : 7 + self.num_joints] = q
        qd_full[6 : 6 + self.num_joints] = qd
        qdd_full[6 : 6 + self.num_joints] = qdd
        self.model.gravity.linear = gravity

        # --- 2. Compute Rigid Body Parameter Regressor ---
        full_rnea_regressor = pin.computeJointTorqueRegressor(self.model, self.data, q_full, qd_full, qdd_full)
        
        # The output regressor has shape (nv, num_bodies * 10).
        # We only care about the rows corresponding to the actuated joint torques.
        # These are the last `num_joints` rows of the regressor matrix.
        actuated_torque_rows = slice(6, 6 + self.num_joints)

        # We also only care about the columns corresponding to the parameters of the
        # actuated links (bodies). These are bodies 1 through 7 in a standard URDF.
        # Body 0 is the base link attached to the floating joint.
        # actuated_param_cols = slice(self.num_link_params, (self.num_joints + 1) * self.num_link_params)


        # first_actuated_body_idx = 2
        # start_col = (first_actuated_body_idx - 1) * self.num_link_params
        # end_col = start_col + self.total_link_params
        # actuated_param_cols = slice(start_col, end_col)

        # Slice the full regressor to get the one for our actuated system
        # Y_rnea = full_rnea_regressor[actuated_torque_rows, actuated_param_cols]
        Y_rnea = full_rnea_regressor[actuated_torque_rows, :][:, self.rnea_param_col_indices]

        # --- 3. Append Friction Parameter Columns ---
        Y_friction = np.zeros((self.num_joints, self.total_friction_params))

        for i in range(self.num_joints):
            # Viscous friction column: Fv * qd
            viscous_col_idx = i * self.num_friction_params
            Y_friction[i, viscous_col_idx] = qd[i]

            # Coulomb friction column: Fc * sign(qd)
            coulomb_col_idx = i * self.num_friction_params + 1
            # Y_friction[i, coulomb_col_idx] = np.sign(qd[i])
            Y_friction[i, coulomb_col_idx] = smooth_sign(qd[i])
            
        # Combine the two parts horizontally
        Y_full = np.hstack([Y_rnea, Y_friction])
        
        # Final sanity check on the shape
        expected_shape = (self.num_joints, self.total_params)
        if Y_full.shape != expected_shape:
            raise ValueError(f"Final regressor shape is {Y_full.shape}, but expected {expected_shape}.")

        return Y_full