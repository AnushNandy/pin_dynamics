import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import MathUtils
from config import robot_config
import pybullet as p
class RobotDynamics:
    def __init__(self, config):
        self.num_joints = config.NUM_JOINTS
        self.link_names = config.LINK_NAMES_IN_KDL_ORDER
        self.num_links = len(self.link_names)
        self.kdl_chain_full = config.KDL_CHAIN

        self.links = []
        for name in self.link_names:
            dyn_params = config.LINK_DYNAMIC_PARAMETERS[name]
            mass = dyn_params['mass']
            p_com = np.array(dyn_params['com'])
            I_origin = np.array(dyn_params['inertia_tensor'])

            skew_pc = MathUtils.get_skew_symmetric_matrix(p_com)
            I_com = I_origin + mass * (skew_pc @ skew_pc)

            self.links.append({
                'name': name,
                'mass': mass,
                'p_com': p_com,
                'I_com': I_com
            })



    def compute_rnea(self, q, qd, qdd, gravity=np.array([0, 0, -9.81]), f_ext=np.zeros(6)):
        """
        Compute the inverse dynamics using the standard RNEA formulation.

        :param np.ndarray f_ext: External wrench [Fx, Fy, Fz, Nx, Ny, Nz] on the EE, in the base frame.
        """
        _, T_world_links = MathUtils.compute_forward_kinematics(self.kdl_chain_full, q, neocis_convention=True)

        omegas = [np.zeros(3) for _ in range(self.num_links)]
        omega_dots = [np.zeros(3) for _ in range(self.num_links)]
        linear_accels = [np.zeros(3) for _ in range(self.num_links)]  # Acceleration of link origin
        forces = [np.zeros(3) for _ in range(self.num_links)]  # Force exerted ON link i BY link i-1
        torques = [np.zeros(3) for _ in range(self.num_links)]  # Torque exerted ON link i BY link i-1
        joint_torques = np.zeros(self.num_joints)
        z_axis = np.array([0, 0, 1])

        # --- 1. Forward Pass: Propagate Kinematics (from base to end-effector) ---
        R_world_base = T_world_links[0][:3, :3]
        linear_accels[0] = R_world_base.T @ -gravity

        for i in range(1, self.num_links):
            joint_idx = i - 1
            T_parent_to_child = np.linalg.inv(T_world_links[i - 1]) @ T_world_links[i]
            R_parent_to_child = T_parent_to_child[:3, :3]
            p_parent_to_child = T_parent_to_child[:3, 3]

            # Transform parent's angular velocity to child frame and add joint velocity
            omegas[i] = R_parent_to_child.T @ omegas[i - 1] + qd[joint_idx] * z_axis

            # Transform parent's angular accel to child frame and add relative components
            omega_dots[i] = R_parent_to_child.T @ omega_dots[i - 1] + \
                            np.cross(omegas[i], qd[joint_idx] * z_axis) + \
                            qdd[joint_idx] * z_axis

            # Transform parent's linear accel to child frame and add components due to parent's rotation
            linear_accels[i] = R_parent_to_child.T @ (linear_accels[i - 1] + \
                                                      np.cross(omega_dots[i - 1], p_parent_to_child) + \
                                                      np.cross(omegas[i - 1],
                                                               np.cross(omegas[i - 1], p_parent_to_child)))

        # --- 2. Backward Pass: Propagate Forces (from end-effector to base) ---
        R_world_ee = T_world_links[-1][:3, :3]
        f_ext_vec, n_ext_vec = f_ext[:3], f_ext[3:]
        forces[-1] = R_world_ee.T @ f_ext_vec
        torques[-1] = R_world_ee.T @ n_ext_vec

        for i in range(self.num_links - 1, 0, -1):
            link_info = self.links[i]
            p_com = link_info['p_com']  # Vector from link i origin to its CoM, IN FRAME i

            # --- Step A: Calculate inertial forces for link i, in frame i ---
            accel_com = linear_accels[i] + np.cross(omega_dots[i], p_com) + np.cross(omegas[i],
                                                                                     np.cross(omegas[i], p_com))
            F_inertial = link_info['mass'] * accel_com
            N_inertial = link_info['I_com'] @ omega_dots[i] + np.cross(omegas[i], link_info['I_com'] @ omegas[i])

            # Get transformation from parent (i-1) to current link (i)
            T_parent_to_child = np.linalg.inv(T_world_links[i - 1]) @ T_world_links[i]
            R_parent_to_child = T_parent_to_child[:3, :3]
            p_parent_to_child = T_parent_to_child[:3, 3]

            # --- Step B: Calculate total force/torque at link i's origin and propagate to parent ---
            force_from_child = forces[i]
            f_i = force_from_child + F_inertial

            # Torque `n_i` is similar, including moments from all forces.
            torque_from_child = torques[i]  # This is n_{i+1} propagated to frame i's origin.
            n_i = torque_from_child + N_inertial + np.cross(p_com, F_inertial)

            # --- Step C: Propagate wrench to parent frame (i-1) ---
            # `forces[i-1]` and `torques[i-1]` are the wrench exerted BY link i-2 ON link i-1.
            forces[i - 1] = R_parent_to_child @ f_i
            torques[i - 1] = R_parent_to_child @ n_i + np.cross(p_parent_to_child, forces[i - 1])

            # --- Step D: Project torque onto joint axis to find actuation torque ---
            joint_idx = i - 1
            joint_torques[joint_idx] = np.dot(torques[i - 1], z_axis)

        return joint_torques

    def set_parameters_from_vector(self, P_vec):
        """
        Update the link dynamic parameters from a flat parameter vector P.
        Used after system identification.
        """
        num_link_params = 10
        for i in range(self.num_joints):
            link_name = self.link_names[i + 1]  # Skip base link
            link_idx_in_dyn = self.link_names.index(link_name)

            start_idx = i * num_link_params
            end_idx = start_idx + num_link_params
            p_link = P_vec[start_idx:end_idx]

            ixx, ixy, ixz, iyy, iyz, izz, mcx, mcy, mcz, mass = p_link

            if np.abs(mass) < 1e-9:
                com = np.zeros(3)
            else:
                com = np.array([mcx, mcy, mcz]) / mass

            I_com = np.array([[ixx, ixy, ixz],
                              [ixy, iyy, iyz],
                              [ixz, iyz, izz]])

            self.links[link_idx_in_dyn]['mass'] = mass
            self.links[link_idx_in_dyn]['p_com'] = com
            self.links[link_idx_in_dyn]['I_com'] = I_com