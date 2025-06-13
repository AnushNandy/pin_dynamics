import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from Dynamics_full.config import robot_config
from Dynamics_full.src.dynamics.rnea import RobotDynamics

SIM_DURATION = 10.0
TIME_STEP = 1. / 240.
URDF_PATH = r"C:\dev\control-sw-tools\Dynamics_full\ArmModels\urdfs\P4\P4_Contra-Angle_right.urdf"

KP = np.array([100.0, 150.0, 120.0, 80.0, 50.0, 40.0, 30.0])
KD = np.array([20.0, 25.0, 22.0, 15.0, 8.0, 6.0, 4.0])

JOINT_AMPLITUDES = np.array([
    np.deg2rad(15),  # Joint 0: 15 degrees
    np.deg2rad(20),  # Joint 1: 20 degrees
    np.deg2rad(12),  # Joint 2: 12 degrees
    np.deg2rad(18),  # Joint 3: 18 degrees
    np.deg2rad(10),  # Joint 4: 10 degrees
    np.deg2rad(8),  # Joint 5: 8 degrees
    np.deg2rad(6)  # Joint 6: 6 degrees
])

JOINT_FREQUENCIES = np.array([0.5, 0.6, 0.4, 0.7, 0.3, 0.4, 0.2])
JOINT_PHASE_OFFSETS = np.array(
    [0, np.pi / 4, np.pi / 3, np.pi / 2, np.pi / 6, np.pi / 8, np.pi / 12])

INITIAL_JOINT_STATES = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

GRAVITY_VECTOR = np.array([0, 0, -9.81])


def generate_multi_joint_trajectory(t, amplitudes, frequencies, phases, offsets):
    """Generates smooth trajectories for multiple joints with different parameters."""
    num_joints = len(amplitudes)
    q_des = np.zeros(num_joints)
    qd_des = np.zeros(num_joints)
    qdd_des = np.zeros(num_joints)

    for i in range(num_joints):
        omega = 2 * np.pi * frequencies[i]
        phase = phases[i]

        q_des[i] = offsets[i] + amplitudes[i] * np.sin(omega * t + phase)
        qd_des[i] = amplitudes[i] * omega * np.cos(omega * t + phase)
        qdd_des[i] = -amplitudes[i] * (omega ** 2) * np.sin(omega * t + phase)

    return q_des, qd_des, qdd_des


def get_joint_indices_by_name(robot_id, joint_names):
    """Gets a list of joint indices from a list of joint names, in order."""
    num_joints_pb = p.getNumJoints(robot_id)
    joint_name_to_idx = {p.getJointInfo(robot_id, i)[1].decode('UTF-8'): i for i in range(num_joints_pb)}
    return [joint_name_to_idx[name] for name in joint_names]


def apply_torque_limits(tau, max_torques):
    """Apply torque limits to prevent excessive actuator commands."""
    return np.clip(tau, -max_torques, max_torques)


def main():
    """Main simulation and plotting function."""
    # --- 1. Setup ---
    robot_dyn = RobotDynamics(robot_config)
    physicsClientId = p.connect(p.GUI)
    p.setTimeStep(TIME_STEP)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(GRAVITY_VECTOR[0], GRAVITY_VECTOR[1], GRAVITY_VECTOR[2])
    p.loadURDF("plane.urdf")

    if not os.path.exists(URDF_PATH):
        print(f"CRITICAL ERROR: URDF file not found at '{URDF_PATH}'")
        p.disconnect()
        return

    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
    pb_joint_indices = get_joint_indices_by_name(robot_id, robot_config.ACTUATED_JOINT_NAMES)

    for i, joint_idx in enumerate(pb_joint_indices):
        p.resetJointState(robot_id, joint_idx, INITIAL_JOINT_STATES[i])
        p.setJointMotorControl2(robot_id, joint_idx, p.VELOCITY_CONTROL, force=0)

    # Define reasonable torque limits (adjust based on your robot specifications)
    max_torques = np.array([1000.0, 1000.0, 800.0, 600.0, 400.0, 300.0, 200.0])

    # --- 2. Data Logging ---
    log_t = []
    log_q_des = []
    log_q_actual = []
    log_tau = []
    log_tau_ff = []
    log_tau_fb = []

    print("--- Starting Multi-Joint RNEA-based Simulation ---")

    # --- 3. Simulation Loop ---
    for t in np.arange(0, SIM_DURATION, TIME_STEP):
        # Get actual state from PyBullet
        joint_states_pb = p.getJointStates(robot_id, pb_joint_indices)
        q_actual = np.array([state[0] for state in joint_states_pb])
        qd_actual = np.array([state[1] for state in joint_states_pb])

        # Generate desired trajectory for ALL joints
        q_des_vec, qd_des_vec, qdd_des_vec = generate_multi_joint_trajectory(
            t, JOINT_AMPLITUDES, JOINT_FREQUENCIES, JOINT_PHASE_OFFSETS, INITIAL_JOINT_STATES
        )

        # a) Feed-forward torque from RNEA (computed torque control)
        try:
            tau_ff = robot_dyn.compute_rnea(q_des_vec, qd_des_vec, qdd_des_vec, gravity=GRAVITY_VECTOR)
        except Exception as e:
            print(f"RNEA computation failed: {e}")
            tau_ff = np.zeros(robot_config.NUM_JOINTS)

        # b) Feedback torque from PD controller
        error_q = q_des_vec - q_actual
        error_qd = qd_des_vec - qd_actual
        tau_fb = KP * error_q + KD * error_qd

        # c) Total Torque with limits
        tau_total = tau_ff + tau_fb
        tau_total = apply_torque_limits(tau_total, max_torques)

        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=pb_joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=tau_total
        )

        p.stepSimulation()

        log_t.append(t)
        log_q_des.append(np.copy(q_des_vec))
        log_q_actual.append(q_actual)
        log_tau.append(tau_total)
        log_tau_ff.append(tau_ff)
        log_tau_fb.append(tau_fb)

    p.disconnect()
    print("--- Simulation Finished. Plotting results. ---")

    # --- 4. Enhanced Plotting ---
    log_t = np.array(log_t)
    log_q_des = np.array(log_q_des)
    log_q_actual = np.array(log_q_actual)
    log_tau = np.array(log_tau)
    log_tau_ff = np.array(log_tau_ff)
    log_tau_fb = np.array(log_tau_fb)

    # fig = plt.figure(figsize=(20, 16))
    #
    # # Plot 1: Multi-joint position tracking (first 4 joints)
    # for i in range(min(4, robot_config.NUM_JOINTS)):
    #     plt.subplot(4, 2, i + 1)
    #     plt.plot(log_t, np.rad2deg(log_q_des[:, i]), 'r--', label='Desired', linewidth=2)
    #     plt.plot(log_t, np.rad2deg(log_q_actual[:, i]), 'b-', label='Actual', linewidth=1.5)
    #     plt.ylabel('Position (deg)')
    #     plt.title(f'{robot_config.ACTUATED_JOINT_NAMES[i]} Tracking')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #
    # # Plot 2: Tracking errors for first 4 joints
    # for i in range(min(4, robot_config.NUM_JOINTS)):
    #     plt.subplot(4, 2, i + 5)
    #     error = np.rad2deg(log_q_des[:, i] - log_q_actual[:, i])
    #     plt.plot(log_t, error, 'g-', linewidth=1.5)
    #     plt.ylabel('Error (deg)')
    #     plt.title(f'{robot_config.ACTUATED_JOINT_NAMES[i]} Error')
    #     plt.grid(True, alpha=0.3)
    #     if i == 3:  # Last subplot
    #         plt.xlabel('Time (s)')

    plt.figure(figsize=(12, 2.5 * robot_config.NUM_JOINTS))

    for i in range(robot_config.NUM_JOINTS):
        # Position tracking
        plt.subplot(robot_config.NUM_JOINTS, 2, 2 * i + 1)
        plt.plot(log_t, np.rad2deg(log_q_des[:, i]), 'r--', label='Desired', linewidth=2)
        plt.plot(log_t, np.rad2deg(log_q_actual[:, i]), 'b-', label='Actual', linewidth=1.5)
        plt.ylabel('Position (deg)')
        plt.title(f'{robot_config.ACTUATED_JOINT_NAMES[i]} Tracking')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if i == robot_config.NUM_JOINTS - 1:
            plt.xlabel('Time (s)')

        # Error plot
        plt.subplot(robot_config.NUM_JOINTS, 2, 2 * i + 2)
        error = np.rad2deg(log_q_des[:, i] - log_q_actual[:, i])
        plt.plot(log_t, error, 'g-', linewidth=1.5)
        plt.ylabel('Error (deg)')
        plt.title(f'{robot_config.ACTUATED_JOINT_NAMES[i]} Error')
        plt.grid(True, alpha=0.3)
        if i == robot_config.NUM_JOINTS - 1:
            plt.xlabel('Time (s)')

    plt.tight_layout()
    plt.suptitle('Multi-Joint RNEA Control Performance', fontsize=16, y=0.98)

    # Second figure for torques
    fig2, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Total torques
    for i in range(robot_config.NUM_JOINTS):
        axs[0].plot(log_t, log_tau[:, i], label=f'τ_{i}', alpha=0.8)
    axs[0].set_ylabel('Total Torque (Nm)')
    axs[0].set_title('Applied Joint Torques')
    axs[0].legend(loc='upper right', ncol=robot_config.NUM_JOINTS)
    axs[0].grid(True, alpha=0.3)

    # Feedforward torques
    for i in range(robot_config.NUM_JOINTS):
        axs[1].plot(log_t, log_tau_ff[:, i], label=f'τ_ff_{i}', alpha=0.8)
    axs[1].set_ylabel('Feedforward Torque (Nm)')
    axs[1].set_title('RNEA Feedforward Torques')
    axs[1].legend(loc='upper right', ncol=robot_config.NUM_JOINTS)
    axs[1].grid(True, alpha=0.3)

    # Feedback torques
    for i in range(robot_config.NUM_JOINTS):
        axs[2].plot(log_t, log_tau_fb[:, i], label=f'τ_fb_{i}', alpha=0.8)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Feedback Torque (Nm)')
    axs[2].set_title('PD Feedback Torques')
    axs[2].legend(loc='upper right', ncol=robot_config.NUM_JOINTS)
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Print performance metrics
    print("\n--- Performance Metrics ---")
    for i in range(robot_config.NUM_JOINTS):
        error = np.rad2deg(log_q_des[:, i] - log_q_actual[:, i])
        rms_error = np.sqrt(np.mean(error ** 2))
        max_error = np.max(np.abs(error))
        print(f"{robot_config.ACTUATED_JOINT_NAMES[i]}: RMS Error = {rms_error:.2f}°, Max Error = {max_error:.2f}°")

    plt.show()


if __name__ == "__main__":
    main()