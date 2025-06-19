# File: computed_torque_control_sim_fixed.py

import numpy as np
import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics 
from src.systemid.pinocchio_friction_regressor import smooth_sign
from config import robot_config 

# --- Constants and Configuration ---
URDF_PATH = robot_config.URDF_PATH
IDENTIFIED_PARAMS_PATH = "/home/robot/dev/dyn/src/systemid/identified_params.npz"
SIM_DURATION = 15.0
TIME_STEP = 1. / 240.

# --- New Trajectory Parameters from User ---
JOINT_AMPLITUDES = np.deg2rad([15, 20, 12, 18, 10, 8, 6])
JOINT_FREQUENCIES = np.array([0.5, 0.6, 0.4, 0.7, 0.3, 0.4, 0.2])
JOINT_PHASE_OFFSETS = np.array([0, np.pi / 4, np.pi / 3, np.pi / 2, np.pi / 6, np.pi / 8, np.pi / 12])

# --- New Torque Limits from User ---
TORQUE_LIMITS = np.array([100.0, 100.0, 50.0, 50.0, 40.0, 40.0, 10.0])


class ComputedTorqueController:
    """
    Implements a Computed Torque Controller using a Pinocchio model
    that has been updated with identified parameters.
    """
    def __init__(self, urdf_path: str, identified_params_path: str, kp: float, kd: float):
        print("--- Initializing Computed Torque Controller ---")
        self.model_dynamics = PinocchioRobotDynamics(urdf_path)
        self.num_joints = self.model_dynamics.num_actuated_joints

        try:
            params_data = np.load(identified_params_path)
            self.identified_params = params_data['P']
            print(f"Successfully loaded identified parameters from '{identified_params_path}'")
        except FileNotFoundError:
            print(f"ERROR: Identified parameters file not found. Using default URDF values.")
            self.identified_params = None

        if self.identified_params is not None:
            num_rnea_params = (self.model_dynamics.model.nbodies - 1) * 10
            self.rnea_params = self.identified_params[:num_rnea_params]
            self.friction_params = self.identified_params[num_rnea_params:]
            self.model_dynamics.set_parameters_from_vector(self.rnea_params)
        else:
            self.friction_params = np.zeros(self.num_joints * 2)

        self.Kp = np.diag([kp] * self.num_joints)
        self.Kd = np.diag([kd] * self.num_joints)
        print(f"Controller gains: Kp={kp}, Kd={kd} (Critically Damped)")
        print("---------------------------------------------")

    def compute_control_torques(self, q_meas: np.ndarray, qd_meas: np.ndarray,
                                q_des: np.ndarray, qd_des: np.ndarray, qdd_des: np.ndarray,
                                torque_limits: np.ndarray) -> np.ndarray:
        """
        Calculates the full computed torque control law and applies torque limits.
        """
        error_q = q_des - q_meas
        error_qd = qd_des - qd_meas
        
        qdd_cmd = qdd_des + self.Kp @ error_q + self.Kd @ error_qd

        M_hat = self.model_dynamics.compute_mass_matrix(q_meas)
        n_hat = self.model_dynamics.compute_nonlinear_effects(q_meas, qd_meas)

        tau_friction_hat = np.zeros(self.num_joints)
        for i in range(self.num_joints):
            fv_hat = self.friction_params[i * 2]
            fc_hat = self.friction_params[i * 2 + 1]
            tau_friction_hat[i] = fv_hat * qd_meas[i] + fc_hat * smooth_sign(qd_meas[i])

        tau_unlimited = M_hat @ qdd_cmd + n_hat + tau_friction_hat
        
        # --- IMPLEMENT TORQUE LIMITS ---
        # This is a crucial safety feature for any real robot
        tau_cmd = np.clip(tau_unlimited, -torque_limits, torque_limits)
        
        return tau_cmd

def setup_pybullet_simulation(urdf_path: str):
    """Initializes PyBullet, loads the robot, and sets up the environment."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)

    robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
    
    num_joints = p.getNumJoints(robot_id)
    joint_indices = [i for i in range(num_joints) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
    
    p.setJointMotorControlArray(
        bodyIndex=robot_id,
        jointIndices=joint_indices,
        controlMode=p.VELOCITY_CONTROL,
        forces=np.zeros(len(joint_indices))
    )
    return robot_id, joint_indices

def generate_target_trajectory(t: float, num_joints: int):
    """Generates a trajectory using the user-specified parameters."""
    q_des = np.zeros(num_joints)
    qd_des = np.zeros(num_joints)
    qdd_des = np.zeros(num_joints)
    
    for i in range(num_joints):
        angle = 2 * np.pi * JOINT_FREQUENCIES[i] * t + JOINT_PHASE_OFFSETS[i]
        q_des[i] = JOINT_AMPLITUDES[i] * np.sin(angle)
        qd_des[i] = JOINT_AMPLITUDES[i] * (2 * np.pi * JOINT_FREQUENCIES[i]) * np.cos(angle)
        qdd_des[i] = -JOINT_AMPLITUDES[i] * (2 * np.pi * JOINT_FREQUENCIES[i])**2 * np.sin(angle)
        
    return q_des, qd_des, qdd_des

def main():
    """Main simulation loop."""
    if not os.path.exists(IDENTIFIED_PARAMS_PATH):
        print(f"FATAL: Cannot find '{IDENTIFIED_PARAMS_PATH}'. Please run the identification script.")
        return

    robot_id, joint_indices = setup_pybullet_simulation(URDF_PATH)
    num_actuated_joints = len(joint_indices)

    # --- FIX THE GAINS ---
    # Restore high, critically damped gains to ensure robust performance
    KP = 50.0
    KD = 2 * np.sqrt(KP) # This is 20.0
    controller = ComputedTorqueController(
        urdf_path=URDF_PATH,
        identified_params_path=IDENTIFIED_PARAMS_PATH,
        kp=KP,
        kd=KD
    )

    log_t, log_q_des, log_q_meas, log_tau_cmd = [], [], [], []

    print("\n--- Starting Fixed Computed Torque Control Simulation ---")
    for t in np.arange(0, SIM_DURATION, TIME_STEP):
        q_des, qd_des, qdd_des = generate_target_trajectory(t, num_actuated_joints)

        joint_states = p.getJointStates(robot_id, joint_indices)
        q_meas = np.array([state[0] for state in joint_states])
        qd_meas = np.array([state[1] for state in joint_states])

        # Pass the torque limits to the controller
        tau_cmd = controller.compute_control_torques(
            q_meas, qd_meas, q_des, qd_des, qdd_des, TORQUE_LIMITS
        )

        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=tau_cmd
        )
        p.stepSimulation()
        
        log_t.append(t)
        log_q_des.append(q_des)
        log_q_meas.append(q_meas)
        log_tau_cmd.append(tau_cmd)

    p.disconnect()

    # --- Plotting Results ---
    log_q_des = np.array(log_q_des)
    log_q_meas = np.array(log_q_meas)
    tracking_error = log_q_des - log_q_meas

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Fixed Computed Torque Control Performance")
    
    for i in range(min(3, num_actuated_joints)):
        axs[0].plot(log_t, log_q_des[:, i], '--', label=f'q{i}_des')
        axs[0].plot(log_t, log_q_meas[:, i], label=f'q{i}_meas')
    axs[0].set_ylabel("Position (rad)")
    axs[0].legend()
    axs[0].grid(True)
    
    for i in range(min(3, num_actuated_joints)):
        axs[1].plot(log_t, tracking_error[:, i], label=f'e_q{i}')
    axs[1].set_ylabel("Tracking Error (rad)")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(log_t, np.array(log_tau_cmd)[:, :min(3, num_actuated_joints)])
    axs[2].set_ylabel("Torque (Nm)")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True)
    axs[2].legend([f'τ{i}_cmd' for i in range(min(3, num_actuated_joints))])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()