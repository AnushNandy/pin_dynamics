# File: 03Test_pinocchio_int_corrected.py

import pybullet as p
import pybullet_data
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import sys
# Ensure paths are correct for your project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics
# We will compute friction directly, so JointModel is no longer needed here.
# from src.dynamics.friction_transmission import JointModel
from config import robot_config

# --- Configuration ---
URDF_PATH = r"/home/robot/dev/dyn/ArmModels/urdfs/P4/P4_Contra-Angle_right.urdf"
IDENTIFIED_PARAMS_PATH = r"/home/robot/dev/dyn/src/systemid/identified_params.npz"

TIME_STEP = 1. / 240.
GRAVITY_VECTOR = np.array([0, 0, -9.81])

# --- Controller Gains and Limits ---
KP = np.array([600.0, 120.0, 100.0, 70.0, 40.0, 30.0, 20.0])
KD = np.array([150.0, 20.0, 18.0, 12.0, 7.0, 5.0, 3.0])
MAX_TORQUES = np.array([200.0, 200.0, 150.0, 150.0, 100.0, 80.0, 80.0])

# --- Helper Functions ---

def get_joint_indices_by_name(robot_id, joint_names):
    """Helper function to get PyBullet joint indices from names."""
    joint_map = {p.getJointInfo(robot_id, i)[1].decode('UTF-8'): i for i in range(p.getNumJoints(robot_id))}
    return [joint_map[name] for name in joint_names]

def quintic_poly_trajectory(q_start, q_end, duration, t):
    """Generates a smooth quintic polynomial trajectory point (q, qd, qdd)."""
    if t < 0: t = 0
    if t > duration: t = duration
    
    T = duration
    h = q_end - q_start
    
    q_des = q_start + h * (10*(t/T)**3 - 15*(t/T)**4 + 6*(t/T)**5)
    qd_des = (h/T) * (30*(t/T)**2 - 60*(t/T)**3 + 30*(t/T)**4)
    qdd_des = (h/T**2) * (60*(t/T) - 180*(t/T)**2 + 120*(t/T)**3)
    
    return q_des, qd_des, qdd_des

def parse_identified_params(P_vec, num_joints):
    """
    Parses the flat parameter vector P into rigid body and friction params.
    This is the inverse of what PinocchioAndFrictionRegressorBuilder does.
    """
    num_link_params = 10
    total_link_params = num_joints * num_link_params
    
    P_rnea = P_vec[:total_link_params]
    P_friction = P_vec[total_link_params:]
    
    # Friction params are stored as [Fv_j0, Fc_j0, Fv_j1, Fc_j1, ...]
    fv_identified = P_friction[0::2] # Viscous friction coeffs
    fc_identified = P_friction[1::2] # Coulomb friction coeffs
    
    print("\n--- Parsed Identified Parameters ---")
    print(f"Identified Fv: {np.round(fv_identified, 4)}")
    print(f"Identified Fc: {np.round(fc_identified, 4)}")
    print("------------------------------------\n")
    
    return P_rnea, fv_identified, fc_identified

def plot_torques(time_log, ff_log, fb_log, total_log):
    """Plots the feedforward, feedback, and total torques."""
    plt.figure(figsize=(15, 3.5 * robot_config.NUM_JOINTS))
    for i in range(robot_config.NUM_JOINTS):
        plt.subplot(robot_config.NUM_JOINTS, 1, i + 1)
        plt.plot(time_log, ff_log[:, i], 'g--', label=f'Feedforward Torque (τ_ff)')
        plt.plot(time_log, fb_log[:, i], 'r:', label=f'Feedback Torque (τ_fb)')
        plt.plot(time_log, total_log[:, i], 'b-', label=f'Total Torque (τ_total)')
        plt.axhline(y=MAX_TORQUES[i], color='k', linestyle='--', label='Max Torque')
        plt.axhline(y=-MAX_TORQUES[i], color='k', linestyle='--')
        plt.title(f'Torque Components for Joint {i}')
        plt.ylabel('Torque (Nm)')
        plt.grid(True)
        plt.legend()
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.suptitle('Controller Torque Analysis', fontsize=16, y=1.02)
    plt.show()

# --- Main Simulation ---

def run_interactive_simulation(robot_dyn_model: PinocchioRobotDynamics, fv_identified: np.ndarray, fc_identified: np.ndarray):
    """
    Runs an interactive simulation with corrected trajectory logic and FF torque.
    """
    p.connect(p.GUI)
    p.setTimeStep(TIME_STEP)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(*GRAVITY_VECTOR)
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
    joint_indices = get_joint_indices_by_name(robot_id, robot_config.ACTUATED_JOINT_NAMES)
    
    ee_link_index = joint_indices[-1]
    home_pos, _ = p.getLinkState(robot_id, ee_link_index)[:2]
    sliders = {
        'x': p.addUserDebugParameter("Target X", -5, 5, home_pos[0]),
        'y': p.addUserDebugParameter("Target Y", -5, 5, home_pos[1]),
        'z': p.addUserDebugParameter("Target Z", -5, 5, home_pos[2]),
    }
    logs = {'t': [], 'q_des': [], 'q_act': [], 'tau_ff': [], 'tau_fb': [], 'tau_total': []}

    # --- Trajectory State ---
    traj_start_time = 0.0
    traj_duration = 1.5  # Give a bit more time for longer movements
    q_start_traj = np.array([s[0] for s in p.getJointStates(robot_id, joint_indices)])
    q_target_traj = q_start_traj.copy()
    last_q_ik_target = q_start_traj.copy()

    # --- Simulation Loop ---
    start_time = time.time()
    while time.time() - start_time < 60.0: # Run for 60 seconds
        
        current_sim_time = (len(logs['t']) + 1) * TIME_STEP
        
        # 1. Read Target from Sliders and Plan Trajectory if Needed
        target_pos = [p.readUserDebugParameter(sliders[ax]) for ax in ['x', 'y', 'z']]
        q_ik_target = np.array(p.calculateInverseKinematics(robot_id, ee_link_index, target_pos))
        
        # **IMPROVEMENT**: Plan a new trajectory if the IK target has changed.
        # This makes the robot continuously responsive to the slider.
        if np.linalg.norm(q_ik_target - last_q_ik_target) > 0.01:
            # Start new trajectory from the *last commanded position* to ensure smoothness
            _, q_des_last, _ = quintic_poly_trajectory(q_start_traj, q_target_traj, traj_duration, current_sim_time - traj_start_time)

            traj_start_time = current_sim_time
            q_start_traj = q_des_last
            q_target_traj = q_ik_target
            last_q_ik_target = q_ik_target

        # 2. Get Desired State from the Current Trajectory
        t_in_traj = current_sim_time - traj_start_time
        q_des, qd_des, qdd_des = quintic_poly_trajectory(q_start_traj, q_target_traj, traj_duration, t_in_traj)

        # 3. Get Actual State from PyBullet
        joint_states = p.getJointStates(robot_id, joint_indices)
        q_actual = np.array([state[0] for state in joint_states])
        qd_actual = np.array([state[1] for state in joint_states])

        # 4. Compute Torques (Full CTC with Correct Feedforward)
        
        # Feedback (PD) Torque - drives the error to zero
        tau_fb = KP * (q_des - q_actual) + KD * (qd_des - qd_actual)
        
        # Feedforward (Model-Based) Torque
        # **FIX 1**: Compute RNEA torque from the dynamics model with identified parameters
        tau_rnea = robot_dyn_model.compute_rnea(q_des, qd_des, qdd_des)
        
        # **FIX 2**: Compute friction torque using the IDENTIFIED coefficients
        # This uses the simple friction model that was used for identification (Viscous + Coulomb)
        # Using smooth_sign (tanh) for numerical stability around zero velocity.
        tau_friction_ff = fv_identified * qd_des + fc_identified * np.tanh(qd_des / 0.01)
        
        # The total feedforward torque is the sum of rigid-body and friction effects
        tau_ff = tau_rnea + tau_friction_ff
        
        # Total torque is the sum of feedforward and feedback controllers
        tau_total = np.clip(tau_ff + tau_fb, -MAX_TORQUES, MAX_TORQUES)

        p.setJointMotorControlArray(robot_id, joint_indices, p.TORQUE_CONTROL, forces=tau_total)
        p.stepSimulation()

        # Log data for plotting
        logs['t'].append(current_sim_time)
        logs['q_des'].append(q_des)
        logs['q_act'].append(q_actual)
        logs['tau_ff'].append(tau_ff)
        logs['tau_fb'].append(tau_fb)
        logs['tau_total'].append(tau_total)

    p.disconnect()
    
    # Convert logs to numpy arrays for plotting
    for key in logs:
        logs[key] = np.array(logs[key])
        
    plot_torques(logs['t'], logs['tau_ff'], logs['tau_fb'], logs['tau_total'])


def main():
    """Main function to load models and run the interactive simulation."""
    if not os.path.exists(IDENTIFIED_PARAMS_PATH) or not os.path.exists(URDF_PATH):
        print("FATAL: URDF or Identified Parameters file not found. Check paths.")
        return

    P_identified = np.load(IDENTIFIED_PARAMS_PATH)['P']
    
    P_rnea_identified, fv_identified, fc_identified = parse_identified_params(
        P_identified, robot_config.NUM_JOINTS
    )
    
    # Initialize Pinocchio model and set the identified RIGID BODY parameters
    robot_dyn_pinocchio = PinocchioRobotDynamics(URDF_PATH)
    robot_dyn_pinocchio.set_parameters_from_vector(P_rnea_identified)
    
    # 2. Run the interactive simulation
    # Pass the dynamics model AND the identified friction coeffs to the simulation
    run_interactive_simulation(robot_dyn_pinocchio, fv_identified, fc_identified)


if __name__ == "__main__":
    main()