import pinocchio as pin
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import robot_config

def smooth_sign(v, threshold=0.01):
    return np.tanh(v / threshold)

def compute_stribeck_friction(qd, threshold=0.01):
    """
    Stribeck friction model: Fc * (sign(qd) + alpha * exp(-|qd|/vs))
    where alpha is the Stribeck parameter and vs is the Stribeck velocity
    """
    return np.tanh(qd / threshold) * (1 + 0.5 * np.exp(-np.abs(qd) / 0.1))

def compute_asymmetric_friction(qd, threshold=0.01):
    """
    Asymmetric Coulomb friction
    """
    positive_friction = np.where(qd > threshold, 1.0, 0.0)
    negative_friction = np.where(qd < -threshold, -1.0, 0.0)
    return positive_friction + negative_friction

def compute_nonlinear_viscous(qd):
    """
    Quadratic viscous friction: Fv * qd * |qd|
    """
    return qd**3

class PinocchioAndFrictionRegressorBuilder:
    """
    Constructs the dynamics regressor matrix for BOTH rigid-body parameters
    and friction parameters.
    """
    def __init__(self, urdf_path: str):
        # self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.model = pin.buildModelFromUrdf(urdf_path) # add pin.JointModelFreeFlyer() 
        self.data = self.model.createData()
        self.num_joints = robot_config.NUM_JOINTS
        self.num_moving_bodies = self.model.nbodies - 1

        print("\n--- REGRESSOR BUILDER MODEL INSPECTION ---")
        print(f"Model nq: {self.model.nq}") 
        print(f"Model nv: {self.model.nv}") 
        print(f"Number of moving bodies in model: {self.num_moving_bodies}")
        print("------------------------------------------\n")
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.num_link_params = 10
        self.total_link_params = self.num_moving_bodies * self.num_link_params

        if self.model.joints[1].shortname() == "JointModelFreeFlyer":
            self.idx_q_actuated_start = self.model.joints[1].nq
            self.idx_v_actuated_start = self.model.joints[1].nv
        else: #fixed base robot
            self.idx_q_actuated_start = 0
            self.idx_v_actuated_start = 0
        
        print(f"Determined actuated joint start index in q: {self.idx_q_actuated_start}") 
        print(f"Determined actuated joint start index in v: {self.idx_v_actuated_start}")
        
        # 2 friction parameters (nonlinear Viscous, Coulomb) per joint
        self.num_friction_params = 2
        self.total_friction_params = self.num_joints * self.num_friction_params
        
        # self.rnea_param_col_indices = []

        self.total_params = self.total_link_params + self.total_friction_params
        # self.total_params = self.total_link_params


    def compute_regressor_matrix(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, gravity: np.ndarray = np.array([0, 0, -9.81])) -> np.ndarray:
        """
        Computes the full dynamics regressor matrix Y for a given state.
        τ = Y * [P_rnea, P_friction]^T
        """
        # --- 1. Construct Full State Vectors ---
        # Create zero-vectors of the correct, full size for the model
        q_full = pin.neutral(self.model) 
        qd_full = np.zeros(self.nv)      
        qdd_full = np.zeros(self.nv)     

        # Place the actuated joint values into the correct slice.
        q_start = self.idx_q_actuated_start
        v_start = self.idx_v_actuated_start
        a_start = self.idx_v_actuated_start
        q_full[q_start : q_start + self.num_joints] = q
        qd_full[v_start : v_start + self.num_joints] = qd
        qdd_full[a_start : a_start + self.num_joints] = qdd 
        self.model.gravity.linear = gravity

        # --- 2. Compute Rigid Body Parameter Regressor ---
        full_rnea_regressor = pin.computeJointTorqueRegressor(self.model, self.data, q_full, qd_full, qdd_full)
        
        # The output regressor has shape (nv, num_bodies * 10).
        # Only care about the rows corresponding to the actuated joint torques.
        # last `num_joints` rows of of regressor
        actuated_torque_rows = slice(v_start, v_start + self.num_joints)

        num_rnea_cols = self.num_moving_bodies * self.num_link_params
        actuated_param_cols = slice(0, num_rnea_cols)
        Y_rnea = full_rnea_regressor[actuated_torque_rows, actuated_param_cols]

        # --- 3. Append Friction Parameter Columns ---
        Y_friction = np.zeros((self.num_joints, self.total_friction_params))

        for i in range(self.num_joints):
            # Viscous friction column: Fv * qd
            viscous_col_idx = i * self.num_friction_params
            # Y_friction[i, viscous_col_idx] = qd[i]
            Y_friction[i, viscous_col_idx] = compute_nonlinear_viscous(qd[i])

            # Coulomb friction column: Fc * sign(qd)
            coulomb_col_idx = i * self.num_friction_params + 1
            Y_friction[i, coulomb_col_idx] = smooth_sign(qd[i])
            
        Y_full = np.hstack([Y_rnea, Y_friction])
        # Y_full = Y_rnea
        
        expected_shape = (self.num_joints, self.total_params)
        if Y_full.shape != expected_shape:
            raise ValueError(f"Final regressor shape is {Y_full.shape}, but expected {expected_shape}.")

        return Y_full