import pinocchio as pin
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import robot_config

class PinocchioRobotDynamics:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())            
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
        num_moving_bodies = self.model.nbodies - 1

        for i in range(num_moving_bodies):
            body_idx = i + 1
            
            start_idx = i * num_link_params
            end_idx = start_idx + num_link_params
            p_link_user_format = P_vec[start_idx:end_idx]

            m = p_link_user_format[0]
            mc = p_link_user_format[1:4]

            m_safe = max(m, 1e-6)
            c = mc / m_safe
            ixx, ixy, ixz, iyy, iyz, izz = p_link_user_format[4], p_link_user_format[5], p_link_user_format[6], p_link_user_format[7], p_link_user_format[8], p_link_user_format[9]
            I_origin_identified = np.array([
                [ixx, ixy, ixz],
                [ixy, iyy, iyz], 
                [ixz, iyz, izz]
            ])

            # I_com = I_origin + m * skew(c) * skew(c)
            c_skew = pin.skew(c)
            I_com_identified = I_origin_identified + m * (c_skew @ c_skew)
            
            # ESymmetric Positive Definite)
            I_com_valid = self._get_nearest_spd_matrix(I_com_identified)
            inertia_pin = pin.Inertia(m, c, I_com_valid)

            self.model.inertias[body_idx] = inertia_pin
            
        print("Successfully updated Pinocchio model with identified parameters.")
        pin.forwardKinematics(self.model, self.data, pin.neutral(self.model))
        self.data = self.model.createData()

    @staticmethod
    def _get_nearest_spd_matrix(M):
        """
        Finds the nearest symmetric positive-definite matrix to M.
        """
        B = (M + M.T) / 2
        eigenvals, eigenvecs = np.linalg.eigh(B)
        eigenvals_clamped = np.maximum(eigenvals, 1e-6)
        return eigenvecs @ np.diag(eigenvals_clamped) @ eigenvecs.T

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
        
        ee_frame_id = self.model.getFrameId(robot_config.END_EFFECTOR_FRAME_NAME)
        pin.computeJointJacobians(self.model, self.data, q_full)
        full_J = pin.getFrameJacobian(self.model, self.data, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
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
