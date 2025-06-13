import pybullet as p
import pybullet_data
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt

# --- Add project root to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics 
from src.dynamics.friction_transmission import JointModel
from config import robot_config 

# --- Configuration ---
URDF_PATH = r"/home/robot/dev/dyn/ArmModels/urdfs/P4/P4_Contra-Angle_right.urdf"
IDENTIFIED_PARAMS_PATH = r"/home/robot/dev/dyn/src/systemid/identified_params.npz"

# --- Simulation & Control Parameters ---
SIM_DURATION_S = 60.0 # Run for a longer time
TIME_STEP = 1. / 240.
GRAVITY_VECTOR = np.array([0, 0, -9.81])
MAX_TORQUES = np.array([100.0, 100.0, 80.0, 80.0, 50.0, 50.0, 50.0]) # Lowered for safety

# PD gains for the inner loop of the impedance controller
# KP = np.array([40.0, 50.0, 40.0, 30.0, 20.0, 15.0, 10.0]) 
# KD = np.array([3.5,  4.5,  3.5,  2.5,  1.8,  1.2,  0.8])
KP = np.array([600.0, 120.0, 100.0, 70.0, 40.0, 30.0, 20.0])
KD = np.array([150.0, 20.0, 18.0, 12.0, 7.0, 5.0, 3.0])

# --- INTERACTIVE PARAMETERS ---
# Choose control scheme: "IMPEDANCE" or "JACOBIAN_TRANSPOSE"
CONTROL_MODE = "JACOBIAN_TRANSPOSE" 
# Choose input method: "MOUSE_SPRING" or "SLIDER_WRENCH"
INPUT_MODE = "SLIDER_WRENCH" 

# For IMPEDANCE control: desired virtual mass of the end-effector
MD_VIRTUAL_MASS = np.diag([5, 5, 5, 0.5, 0.5, 0.5]) # kg and kg*m^2
# For MOUSE_SPRING input: stiffness of the virtual spring
K_SPRING = np.diag([300, 300, 300, 50, 50, 50])

# --- Helper & Plotting Functions ---

def get_joint_indices_by_name(robot_id, joint_names):
    """Gets PyBullet joint indices from names."""
    joint_map = {p.getJointInfo(robot_id, i)[1].decode('UTF-8'): i for i in range(p.getNumJoints(robot_id))}
    return [joint_map[name] for name in joint_names]

def plot_final_analysis(logs):
    """Generates all comprehensive analysis plots."""
    time_log = logs['t']
    
    # Plot 1: End-Effector Position
    plt.figure(figsize=(16, 8))
    plt.suptitle('End-Effector Position', fontsize=16)
    pos_labels = ['X', 'Y', 'Z']
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(time_log, logs['ee_pos'][:, i], 'b-')
        plt.title(f'Position {pos_labels[i]}')
        plt.ylabel('Meters')
        plt.grid(True)
    plt.xlabel('Time (s)')
    plt.show()

    # Plot 2: Wrench (Desired vs. Achieved)
    plt.figure(figsize=(16, 12))
    plt.suptitle('Wrench Analysis (Desired vs. Achieved)', fontsize=16)
    wrench_labels = ['Fx', 'Fy', 'Fz', 'Nx', 'Ny', 'Nz']
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(time_log, logs['wrench_des'][:, i], 'r--', label='Desired Wrench')
        plt.plot(time_log, logs['wrench_achieved'][:, i], 'b-', label='Achieved Wrench (from τ_ff)')
        plt.title(wrench_labels[i])
        plt.ylabel('Force (N)' if i < 3 else 'Torque (Nm)')
        plt.grid(True)
        plt.legend()
    plt.xlabel('Time (s)')
    plt.show()
    
    # Plot 3: Joint Torques
    plt.figure(figsize=(16, 4 * robot_config.NUM_JOINTS))
    plt.suptitle('Joint Torque Analysis', fontsize=16)
    for i in range(robot_config.NUM_JOINTS):
        plt.subplot(robot_config.NUM_JOINTS, 1, i + 1)
        if 'tau_fb' in logs:
            plt.plot(time_log, logs['tau_fb'][:, i], 'r:', label='Feedback Torque (τ_fb)')
            plt.plot(time_log, logs['tau_ff'][:, i], 'g--', label='Feedforward Torque (τ_ff)')
        plt.plot(time_log, logs['tau_total'][:, i], 'b-', label='Total Commanded Torque')
        plt.title(f'Torques for Joint {i}')
        plt.ylabel('Torque (Nm)')
        plt.grid(True)
        plt.legend()
    plt.xlabel('Time (s)')
    plt.show()

# --- Main Simulation ---

def main():
    """Main function to load models and run the interactive simulation."""
    if not os.path.exists(IDENTIFIED_PARAMS_PATH) or not os.path.exists(URDF_PATH):
        print("FATAL: URDF or Identified Parameters file not found. Check paths.")
        return

    # 1. Load Models
    P_identified = np.load(IDENTIFIED_PARAMS_PATH)['P']
    dyn_model = PinocchioRobotDynamics(URDF_PATH)
    dyn_model.set_parameters_from_vector(P_identified)
    
    # We don't use the friction model for this test to keep it focused on dynamics
    
    # 2. Setup Simulation and Interactive Elements
    p.connect(p.GUI)
    p.setTimeStep(TIME_STEP)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(*GRAVITY_VECTOR)
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
    actuated_joint_indices = get_joint_indices_by_name(robot_id, robot_config.ACTUATED_JOINT_NAMES)
    ee_link_index = actuated_joint_indices[-1]

    wrench_sliders = {}
    if INPUT_MODE == "SLIDER_WRENCH":
        wrench_labels = ['Fx', 'Fy', 'Fz', 'Nx', 'Ny', 'Nz']
        force_limit, torque_limit = 50, 5 # N and Nm
        for i, label in enumerate(wrench_labels):
            limit = force_limit if i < 3 else torque_limit
            wrench_sliders[label] = p.addUserDebugParameter(label, -limit, limit, 0)
    
    # 3. Initialize Logs and State Variables
    logs = {'t': [], 'ee_pos': [], 'wrench_des': [], 'wrench_achieved': [], 'tau_total': [], 'tau_ff':[], 'tau_fb':[]}
    q_des = np.array([s[0] for s in p.getJointStates(robot_id, actuated_joint_indices)])
    qd_des = np.zeros(robot_config.NUM_JOINTS)
    mouse_target_pos = None

    # 4. Simulation Loop
    start_time = time.time()
    while time.time() - start_time < SIM_DURATION_S:
        current_sim_time = time.time() - start_time
        
        # --- Get State ---
        joint_states = p.getJointStates(robot_id, actuated_joint_indices)
        q_actual = np.array([s[0] for s in joint_states])
        qd_actual = np.array([s[1] for s in joint_states])
        ee_state = p.getLinkState(robot_id, ee_link_index, computeLinkVelocity=1)
        ee_pos, ee_orn, ee_vel_linear, ee_vel_angular = ee_state[0], ee_state[1], ee_state[6], ee_state[7]
        ee_vel = np.concatenate([ee_vel_linear, ee_vel_angular])

        # --- Get Desired Wrench from Input ---
        wrench_des = np.zeros(6)
        if INPUT_MODE == "MOUSE_SPRING":
            mouse_events = p.getMouseEvents()
            for e in mouse_events:
                # PyBullet mouse event structure: [eventType, mousePosX, mousePosY, buttonIndex, buttonState]
                # eventType: 2 = mouse move, 3 = mouse button
                # buttonState: 3 = pressed, 4 = released, 6 = triggered
                if len(e) >= 4 and e[0] == 2:  # Mouse move event
                    # Get the mouse position in world coordinates
                    mouse_x, mouse_y = e[1], e[2]
                    
                    # Convert screen coordinates to world coordinates
                    # This is a simplified conversion - you might need to adjust based on your camera setup
                    cam_info = p.getDebugVisualizerCamera()
                    view_matrix = cam_info[2]
                    proj_matrix = cam_info[3]
                    
                    # For simplicity, let's just use the mouse position directly
                    # You might want to implement proper screen-to-world conversion
                    if mouse_target_pos is None:
                        mouse_target_pos = list(ee_pos)  # Initialize with current EE position
                    
                    # Update target position based on mouse movement
                    # This is a simple mapping - adjust sensitivity as needed
                    sensitivity = 0.001
                    mouse_target_pos[0] = ee_pos[0] + (mouse_x - 400) * sensitivity  # Assuming screen center at 400
                    mouse_target_pos[1] = ee_pos[1] + (mouse_y - 300) * sensitivity  # Assuming screen center at 300
                    mouse_target_pos[2] = ee_pos[2]  # Keep Z constant
        else: # SLIDER_WRENCH
            for i, label in enumerate(wrench_labels):
                wrench_des[i] = p.readUserDebugParameter(wrench_sliders[label])
        
        # --- Apply Control Law ---
        J = dyn_model.compute_jacobian(q_actual)
        tau_total = np.zeros(robot_config.NUM_JOINTS)
        tau_ff = np.zeros(robot_config.NUM_JOINTS); tau_fb = np.zeros(robot_config.NUM_JOINTS)

        if CONTROL_MODE == "IMPEDANCE":
            # 1. Calculate desired acceleration based on impedance law
            xdd_des = np.linalg.inv(MD_VIRTUAL_MASS) @ wrench_des
            
            # 2. Map task-space acceleration to joint-space acceleration
            # Simplified: qdd = J_pinv * xdd (ignores J_dot*qd term)
            qdd_des = np.linalg.pinv(J) @ xdd_des
            
            # 3. Integrate to get desired joint state for the inner PD+CTC loop
            qd_des = qd_des + qdd_des * TIME_STEP
            q_des = q_des + qd_des * TIME_STEP

            # 4. Use the full PD+CTC controller
            tau_fb = KP * (q_des - q_actual) + KD * (qd_des - qd_actual)
            tau_rnea = dyn_model.compute_rnea(q_des, qd_des, qdd_des)
            tau_ff = tau_rnea # Simplified FF for this test
            tau_total = tau_ff + tau_fb

        elif CONTROL_MODE == "JACOBIAN_TRANSPOSE":
            # Simple, direct mapping of force to torque
            tau_total = J.T @ wrench_des

        # --- Apply Torques & Step ---
        p.setJointMotorControlArray(robot_id, actuated_joint_indices, p.TORQUE_CONTROL, 
                                     forces=np.clip(tau_total, -MAX_TORQUES, MAX_TORQUES))
        p.stepSimulation()
        
        # --- Logging ---
        logs['t'].append(current_sim_time)
        logs['ee_pos'].append(ee_pos)
        logs['wrench_des'].append(wrench_des)
        logs['tau_total'].append(tau_total); logs['tau_ff'].append(tau_ff); logs['tau_fb'].append(tau_fb)
        # "Achieved" wrench is the one produced by the feedforward torques
        wrench_achieved = np.linalg.pinv(J.T) @ tau_ff if np.linalg.norm(tau_ff) > 1e-3 else np.zeros(6)
        logs['wrench_achieved'].append(wrench_achieved)
        
    p.disconnect()
    plot_final_analysis(logs)

if __name__ == "__main__":
    main()