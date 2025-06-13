import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from Dynamics_full.config import robot_config
from Dynamics_full.src.dynamics.rnea import RobotDynamics

# --- Simulation Parameters ---
SIM_DURATION = 10.0  # seconds
TIME_STEP = 1. / 240.  # PyBullet default simulation step
URDF_PATH = r"C:\dev\control-sw-tools\Dynamics_full\ArmModels\urdfs\P4\P4_Contra-Angle_right.urdf"

# --- Control & Trajectory Parameters ---
# PD Gains for the feedback controller (Tune these after fixing the model)
# Start with lower gains after fixing kinematics to ensure stability
KP = np.array([500.0, 500.0, 500.0, 250.0, 100.0, 100.0, 50.0])
KD = np.array([50.0, 50.0, 50.0, 25.0, 10.0, 10.0, 5.0])

# Sine wave trajectory parameters for Joint 1
JOINT_TO_MOVE_IDX = 1  # Index for Joint_1
AMPLITUDE = np.deg2rad(15)  # 45 degrees
FREQUENCY = 0.8  # Hz
INITIAL_JOINT_STATES = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Define the gravity vector consistently
GRAVITY_VECTOR = np.array([0, 0, -9.81])

def generate_sine_trajectory(t, amp, freq, offset):
    """Generates position, velocity, and acceleration for a sine wave."""
    theta = 2 * np.pi * freq
    q_des = offset + amp * np.sin(theta * t)
    qd_des = amp * theta * np.cos(theta * t)
    qdd_des = -amp * (theta ** 2) * np.sin(theta * t)
    return q_des, qd_des, qdd_des

def get_joint_indices_by_name(robot_id, joint_names):
    """Gets a list of joint indices from a list of joint names, in order."""
    num_joints_pb = p.getNumJoints(robot_id)
    joint_name_to_idx = {p.getJointInfo(robot_id, i)[1].decode('UTF-8'): i for i in range(num_joints_pb)}
    return [joint_name_to_idx[name] for name in joint_names]

def main():
    """Main simulation and plotting function."""
    # --- 1. Setup ---
    robot_dyn = RobotDynamics(robot_config)
    physicsClientId = p.connect(p.GUI)
    p.setTimeStep(TIME_STEP)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # Set PyBullet's gravity
    p.setGravity(GRAVITY_VECTOR[0], GRAVITY_VECTOR[1], GRAVITY_VECTOR[2])
    p.loadURDF("plane.urdf")

    if not os.path.exists(URDF_PATH):
        print(f"CRITICAL ERROR: URDF file not found at '{URDF_PATH}'")
        p.disconnect()
        return

    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
    pb_joint_indices = get_joint_indices_by_name(robot_id, robot_config.ACTUATED_JOINT_NAMES)

    # Initialize robot pose and disable PyBullet's default motor control
    for i, joint_idx in enumerate(pb_joint_indices):
        p.resetJointState(robot_id, joint_idx, INITIAL_JOINT_STATES[i])
        p.setJointMotorControl2(robot_id, joint_idx, p.VELOCITY_CONTROL, force=0)

    # --- 2. Data Logging ---
    log_t = []
    log_q_des = []
    log_q_actual = []
    log_tau = []

    q_des_vec = np.copy(INITIAL_JOINT_STATES)
    qd_des_vec = np.zeros(robot_config.NUM_JOINTS)
    qdd_des_vec = np.zeros(robot_config.NUM_JOINTS)

    print("--- Starting RNEA-based Simulation ---")

    # --- 3. Simulation Loop ---
    for t in np.arange(0, SIM_DURATION, TIME_STEP):
        # Get actual state from PyBullet
        joint_states_pb = p.getJointStates(robot_id, pb_joint_indices)
        q_actual = np.array([state[0] for state in joint_states_pb])
        qd_actual = np.array([state[1] for state in joint_states_pb])

        # Generate desired trajectory for the moving joint
        q_des, qd_des, qdd_des = generate_sine_trajectory(t, AMPLITUDE, FREQUENCY,
                                                          INITIAL_JOINT_STATES[JOINT_TO_MOVE_IDX])
        q_des_vec[JOINT_TO_MOVE_IDX] = q_des
        qd_des_vec[JOINT_TO_MOVE_IDX] = qd_des
        qdd_des_vec[JOINT_TO_MOVE_IDX] = qdd_des

        # --- THE CONTROL LAW (CORRECTED) ---

        # a) Feed-forward torque from RNEA.
        #    This term cancels out the robot's dynamics (inertia, Coriolis, gravity).
        #    It should be calculated using the DESIRED state variables (q_des, qd_des, qdd_des).
        #    CRITICAL FIX: Pass the correct gravity vector to the RNEA.
        tau_ff = robot_dyn.compute_rnea(q_des_vec, qd_des_vec, qdd_des_vec, gravity=GRAVITY_VECTOR)

        # b) Feedback torque from PD controller.
        #    This term corrects for any errors between desired and actual states.
        error_q = q_des_vec - q_actual
        error_qd = qd_des_vec - qd_actual
        tau_fb = KP * error_q + KD * error_qd

        # c) Total Torque = Feed-forward (model-based) + Feedback (error correction)
        tau_total = tau_ff + tau_fb

        # Apply torques to the robot
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=pb_joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=tau_total
        )

        p.stepSimulation()

        # Log data for plotting
        log_t.append(t)
        log_q_des.append(np.copy(q_des_vec))
        log_q_actual.append(q_actual)
        log_tau.append(tau_total)

        # A short sleep is not strictly necessary as PyBullet's stepSimulation is blocking,
        # but can be kept if running in a different mode.
        # time.sleep(TIME_STEP)

    p.disconnect()
    print("--- Simulation Finished. Plotting results. ---")

    # --- 4. Plotting ---
    log_t = np.array(log_t)
    log_q_des = np.array(log_q_des)
    log_q_actual = np.array(log_q_actual)
    log_tau = np.array(log_tau)

    fig, axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True) # Increased figsize for better readability
    fig.suptitle('RNEA-based Computed Torque Control Performance', fontsize=16)

    # Plot 1: Position Tracking for the moving joint
    axs[0].plot(log_t, np.rad2deg(log_q_des[:, JOINT_TO_MOVE_IDX]), 'r--', label='Desired Position (deg)')
    axs[0].plot(log_t, np.rad2deg(log_q_actual[:, JOINT_TO_MOVE_IDX]), 'b-', label='Actual Position (deg)')
    axs[0].set_ylabel('Joint Position (deg)')
    axs[0].set_title(f'Tracking for {robot_config.ACTUATED_JOINT_NAMES[JOINT_TO_MOVE_IDX]}')
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Position Tracking Error
    error = np.rad2deg(log_q_des[:, JOINT_TO_MOVE_IDX] - log_q_actual[:, JOINT_TO_MOVE_IDX])
    axs[1].plot(log_t, error, 'g-')
    axs[1].set_ylabel('Error (deg)')
    axs[1].set_title('Position Tracking Error')
    axs[1].grid(True)

    # Plot 3: Applied Torques
    for i in range(robot_config.NUM_JOINTS):
        axs[2].plot(log_t, log_tau[:, i], label=f'τ_{i}')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Torque (Nm)')
    axs[2].set_title('Applied Joint Torques')
    axs[2].legend(loc='upper right', ncol=robot_config.NUM_JOINTS) # Better legend layout
    axs[2].grid(True)
    # Set a sensible y-limit to prevent visual distortions from initial spikes
    torque_limit = np.max(np.abs(log_tau[len(log_tau)//10:])) * 1.2 # Limit based on post-startup torques
    if torque_limit > 0:
      axs[2].set_ylim([-torque_limit, torque_limit])


    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjusted rect
    plt.show()

if __name__ == "__main__":
    main()