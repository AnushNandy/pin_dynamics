# In file: pinocchio_dynamics.py

import pinocchio as pin
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import robot_config

class PinocchioRobotDynamics:
    """
    A robot dynamics calculator using Pinocchio.
    """
    def __init__(self, urdf_path: str):
        try:
            self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        except Exception as e:
            print(f"Failed to load URDF from {urdf_path}")
            raise e
            
        print("\n--- PINOCCHIO MODEL INSPECTION ---")
        print(self.model)
        print(f"Total configuration dimension (model.nq): {self.model.nq}")
        print(f"Total velocity dimension (model.nv): {self.model.nv}")
        print(f"Number of joints in model: {self.model.njoints}")
        print("------------------------------------\n")

        self.data = self.model.createData()
        
        self.nq = self.model.nq
        self.nv = self.model.nv

        self.num_actuated_joints = robot_config.NUM_JOINTS

    def set_parameters_from_vector(self, P_vec: np.ndarray):
        """
        Update the link dynamic parameters from a flat parameter vector P.
        """
        num_link_params = 10
        for i in range(self.num_actuated_joints):
            body_idx = i + 2 
            
            start_idx = i * num_link_params
            end_idx = start_idx + num_link_params
            p_link_user_format = P_vec[start_idx:end_idx]

            ixx, ixy, ixz, iyy, iyz, izz, mcx, mcy, mcz, mass = p_link_user_format
            
            m = max(mass, 1e-6)
            c = np.array([mcx, mcy, mcz]) / m

            I_com_identified = np.array([
                [ixx, ixy, ixz],
                [ixy, iyy, iyz], 
                [ixz, iyz, izz]
            ])

            I_com_physically_valid = self._get_nearest_spd_matrix(I_com_identified)
            
            skew_c = pin.skew(c)
            I_origin = I_com_physically_valid - m * (skew_c @ skew_c)
            
            inertia_pin = pin.Inertia(m, c, I_origin)

            self.model.inertias[body_idx] = inertia_pin
            
        print("Successfully updated Pinocchio model with identified parameters.")
        self.data = self.model.createData()

    @staticmethod
    def _get_nearest_spd_matrix(M):
        """
        Finds the nearest symmetric positive-definite matrix to M.
        """
        B = (M + M.T) / 2
        _, s, V = np.linalg.svd(B)
        H = np.diag(np.maximum(s, 1e-6))
        return V @ H @ V.T

    def compute_rnea(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, 
                 gravity: np.ndarray = np.array([0, 0, -9.81])) -> np.ndarray:
        """
        Compute the inverse dynamics using Pinocchio's RNEA.
        """
        # 1. Create full-sized vectors using the model's properties.
        q_full = pin.neutral(self.model) #  nq (21)
        qd_full = np.zeros(self.nv)      # nv (13)
        qdd_full = np.zeros(self.nv)     #  nv (13)

        # 2. Place the actuated joint values into the correct slice.
        #      q_full[7 : 7 + 7]  (indices 7 to 13)
        #      qd_full[6 : 6 + 7] (indices 6 to 12)
        q_full[7 : 7 + self.num_actuated_joints] = q
        qd_full[6 : 6 + self.num_actuated_joints] = qd
        qdd_full[6 : 6 + self.num_actuated_joints] = qdd

        # 3. Set gravity and run the algorithm.
        self.model.gravity.linear = gravity 
        
        tau_full = pin.rnea(self.model, self.data, q_full, qd_full, qdd_full)

        # 4. Return the torques for the actuated joints.
        return tau_full[6 : 6 + self.num_actuated_joints]
    
    def compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Computes the geometric Jacobian for the end-effector in the world frame.
        The Jacobian maps joint velocities to end-effector spatial velocity.
        v_ee = J(q) * q_dot

        :param q: Joint positions (for ACTUATED joints only).
        :return: The 6xN Jacobian matrix.
        """
        q_full = pin.neutral(self.model)
        q_full[7 : 7 + self.num_actuated_joints] = q
        # q_full = q
        
        ee_frame_id = self.model.getFrameId(robot_config.END_EFFECTOR_FRAME_NAME)
        pin.computeJointJacobians(self.model, self.data, q_full)
        full_J = pin.getFrameJacobian(self.model, self.data, ee_frame_id, pin.ReferenceFrame.LOCAL)
        
        actuated_J = full_J[:, 6:]
        
        return actuated_J
    
    def compute_task_space_inertia(self, q: np.ndarray, J: np.ndarray) -> np.ndarray:
        """
        Computes the operational space (task space) inertia matrix, Lambda.
        """
        # Get the actuated joint mass matrix
        M_actuated = self.compute_mass_matrix(q)
        
        # Solve M * x = J^T for x 
        try:
            M_inv_J_T = np.linalg.solve(M_actuated, J.T)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            M_inv_J_T = np.linalg.pinv(M_actuated) @ J.T
        
        # Compute J * M^-1 * J^T
        J_M_inv_J_T = J @ M_inv_J_T
        
        try:
            Lambda = np.linalg.inv(J_M_inv_J_T)
        except np.linalg.LinAlgError:
            Lambda = np.linalg.pinv(J_M_inv_J_T)
        
        return Lambda
    
    def compute_gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """
        Computes the gravity compensation torque vector g(q).
        """
        q_full = pin.neutral(self.model)
        q_full[7 : 7 + self.num_actuated_joints] = q        
        g_full = pin.computeGeneralizedGravity(self.model, self.data, q_full)
        return g_full[6 : 6 + self.num_actuated_joints]

    def compute_coriolis_matrix(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """
        Computes the Coriolis and centrifugal matrix C(q, qd).
        The torque vector is then C @ qd.
        """
        q_full = pin.neutral(self.model)
        qd_full = np.zeros(self.nv)
        q_full[7 : 7 + self.num_actuated_joints] = q
        qd_full[6 : 6 + self.num_actuated_joints] = qd

        pin.computeCoriolisMatrix(self.model, self.data, q_full, qd_full)
        
        C_full = self.data.C
        
        actuated_slice = slice(6, 6 + self.num_actuated_joints)
        return C_full[actuated_slice, actuated_slice]
    
    def compute_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Computes the joint space mass matrix M(q)."""
        q_full = pin.neutral(self.model)
        q_full[7 : 7 + self.num_actuated_joints] = q
        
        # This function computes M and stores it in data.M
        pin.crba(self.model, self.data, q_full)
        M_full = self.data.M
        
        actuated_slice = slice(6, 6 + self.num_actuated_joints)
        return M_full[actuated_slice, actuated_slice]

    def compute_nonlinear_effects(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Computes the nonlinear effects vector n(q, qd) = C(q,qd)qd + g(q)."""
        q_full = pin.neutral(self.model)
        qd_full = np.zeros(self.nv)
        q_full[7 : 7 + self.num_actuated_joints] = q
        qd_full[6 : 6 + self.num_actuated_joints] = qd
        
        n_full = pin.nonLinearEffects(self.model, self.data, q_full, qd_full)
        
        return n_full[6 : 6 + self.num_actuated_joints]