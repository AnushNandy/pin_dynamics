import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from Dynamics_full.config import robot_config
from Dynamics_full.src.dynamics.rnea import RobotDynamics

# --- Configuration ---
SIM_DURATION = 10.0
TIME_STEP = 1. / 240.
URDF_PATH = r"C:\dev\control-sw-tools\Dynamics_full\ArmModels\urdfs\P4\P4_Contra-Angle_right.urdf"
IDENTIFIED_PARAMS_PATH = r"C:\dev\control-sw-tools\Dynamics_full\src\systemid\identified_params.npz"
GRAVITY_VECTOR = np.array([0, 0, -9.81])

# --- Enhanced Trajectory Configuration ---
TRAJECTORY_TYPE = "MULTI_SEGMENT"  # Options: "MULTI_SEGMENT", "FIGURE_EIGHT", "PICK_PLACE", "CYCLIC_MOTION"

# Joint limits (adjust these based on your robot's actual limits)
JOINT_LIMITS_MIN = np.deg2rad([-170, -120, -170, -120, -170, -120, -350])
JOINT_LIMITS_MAX = np.deg2rad([170, 120, 170, 120, 170, 120, 350])

# --- Controller Gains ---
KP = np.array([80.0, 120.0, 100.0, 70.0, 40.0, 30.0, 20.0]) / 2.0
KD = np.array([15.0, 20.0, 18.0, 12.0, 7.0, 5.0, 3.0])  # Kd is often related to sqrt(Kp)

# --- Torque Limits ---
MAX_TORQUES = np.array([200.0, 200.0, 150.0, 150.0, 100.0, 80.0, 80.0])


def smooth_step(x):
    """Smooth step function (3rd order polynomial)"""
    return x * x * (3 - 2 * x)


def smooth_step_derivative(x):
    """Derivative of smooth step function"""
    return 6 * x * (1 - x)


def generate_multi_segment_trajectory(t):
    """
    Multi-segment trajectory with different motion phases:
    - Phase 1 (0-2s): Slow extension to working position
    - Phase 2 (2-4s): Fast coordinated movement
    - Phase 3 (4-6s): Precision positioning with small movements
    - Phase 4 (6-8s): Return to home with acceleration/deceleration
    - Phase 5 (8-10s): Complex multi-joint coordination
    """

    # Define waypoints for each joint (in radians)
    waypoints = {
        0: [0, np.pi / 6, np.pi / 4, -np.pi / 6, np.pi / 8],  # Joint 0: Base rotation
        1: [0, np.pi / 4, -np.pi / 6, np.pi / 3, -np.pi / 8],  # Joint 1: Shoulder
        2: [0, -np.pi / 3, np.pi / 4, -np.pi / 4, np.pi / 6],  # Joint 2: Elbow
        3: [0, np.pi / 2, -np.pi / 3, np.pi / 4, -np.pi / 4],  # Joint 3: Wrist 1
        4: [0, -np.pi / 4, np.pi / 3, -np.pi / 6, np.pi / 3],  # Joint 4: Wrist 2
        5: [0, np.pi / 3, -np.pi / 4, np.pi / 2, -np.pi / 3],  # Joint 5: Wrist 3
        6: [0, -np.pi, np.pi, -np.pi / 2, np.pi / 2]  # Joint 6: End effector rotation
    }

    # Time segments
    phase_times = [0, 2, 4, 6, 8, 10]

    q_des = np.zeros(7)
    qd_des = np.zeros(7)
    qdd_des = np.zeros(7)

    for joint in range(7):
        # Determine current phase
        phase = min(int(t / 2), 4)  # 5 phases total (0-4)

        if phase < 4:
            # Interpolate between current and next waypoint
            t_phase = (t - phase_times[phase]) / (phase_times[phase + 1] - phase_times[phase])
            t_phase = np.clip(t_phase, 0, 1)

            # Use different interpolation methods for different phases
            if phase == 0:  # Slow smooth start
                s = smooth_step(t_phase)
            elif phase == 1:  # Fast movement
                s = t_phase
            elif phase == 2:  # Precision positioning
                s = smooth_step(t_phase) * 0.3 + 0.7 * smooth_step(t_phase)
            else:  # Return movement
                s = 1 - (1 - t_phase) ** 2  # Ease out

            # Position interpolation
            q_start = waypoints[joint][phase]
            q_end = waypoints[joint][phase + 1]
            q_des[joint] = q_start + s * (q_end - q_start)

            # Velocity calculation
            dt_phase = phase_times[phase + 1] - phase_times[phase]
            if phase == 1:  # Fast phase - higher velocities
                qd_des[joint] = (q_end - q_start) / dt_phase * (1 + 0.5 * np.sin(2 * np.pi * t_phase))
            else:
                qd_des[joint] = (q_end - q_start) / dt_phase * smooth_step_derivative(t_phase) / dt_phase

            # Acceleration (simplified)
            qdd_des[joint] = 0  # Can be computed for more accuracy

        else:
            # Final phase - complex coordination
            t_final = t - 8
            q_des[joint] = waypoints[joint][4] + 0.1 * np.sin(
                2 * np.pi * (joint + 1) * 0.5 * t_final + joint * np.pi / 4)
            qd_des[joint] = 0.1 * 2 * np.pi * (joint + 1) * 0.5 * np.cos(
                2 * np.pi * (joint + 1) * 0.5 * t_final + joint * np.pi / 4)
            qdd_des[joint] = -0.1 * (2 * np.pi * (joint + 1) * 0.5) ** 2 * np.sin(
                2 * np.pi * (joint + 1) * 0.5 * t_final + joint * np.pi / 4)

    # Apply joint limits
    q_des = np.clip(q_des, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX)

    return q_des, qd_des, qdd_des


def generate_figure_eight_trajectory(t):
    """
    Generate a figure-eight pattern in task space that requires coordinated joint motion.
    """
    # Parameters for figure-eight in task space
    period = 8.0  # seconds for one complete figure-eight
    omega = 2 * np.pi / period

    q_des = np.zeros(7)
    qd_des = np.zeros(7)
    qdd_des = np.zeros(7)

    # Joint 0 (base): Slow oscillation
    q_des[0] = 0.3 * np.sin(omega * t)
    qd_des[0] = 0.3 * omega * np.cos(omega * t)
    qdd_des[0] = -0.3 * omega ** 2 * np.sin(omega * t)

    # Joint 1 (shoulder): Figure-eight Y component
    q_des[1] = 0.4 * np.sin(2 * omega * t)
    qd_des[1] = 0.4 * 2 * omega * np.cos(2 * omega * t)
    qdd_des[1] = -0.4 * (2 * omega) ** 2 * np.sin(2 * omega * t)

    # Joint 2 (elbow): Coordinated with shoulder
    q_des[2] = -0.5 * np.sin(omega * t) + 0.2 * np.cos(2 * omega * t)
    qd_des[2] = -0.5 * omega * np.cos(omega * t) - 0.2 * 2 * omega * np.sin(2 * omega * t)
    qdd_des[2] = 0.5 * omega ** 2 * np.sin(omega * t) - 0.2 * (2 * omega) ** 2 * np.cos(2 * omega * t)

    # Joints 3-5 (wrist): Complex coordination
    for i in range(3, 6):
        phase = i * np.pi / 4
        q_des[i] = 0.3 * np.sin(omega * t + phase) + 0.1 * np.sin(3 * omega * t)
        qd_des[i] = 0.3 * omega * np.cos(omega * t + phase) + 0.1 * 3 * omega * np.cos(3 * omega * t)
        qdd_des[i] = -0.3 * omega ** 2 * np.sin(omega * t + phase) - 0.1 * (3 * omega) ** 2 * np.sin(3 * omega * t)

    # Joint 6 (end effector): Continuous rotation
    q_des[6] = omega * t
    qd_des[6] = omega
    qdd_des[6] = 0

    # Apply joint limits
    q_des = np.clip(q_des, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX)

    return q_des, qd_des, qdd_des


def generate_pick_place_trajectory(t):
    """
    Simulate a pick-and-place operation with distinct phases.
    """
    # Define pick-and-place waypoints
    home_pos = np.array([0, 0, 0, 0, 0, 0, 0])
    approach_pos = np.array([np.pi / 4, np.pi / 6, -np.pi / 3, np.pi / 2, -np.pi / 4, np.pi / 3, 0])
    pick_pos = np.array([np.pi / 4, np.pi / 3, -np.pi / 2, np.pi / 2, -np.pi / 4, np.pi / 3, np.pi / 2])
    lift_pos = np.array([np.pi / 4, np.pi / 6, -np.pi / 3, np.pi / 2, -np.pi / 4, np.pi / 3, np.pi / 2])
    place_approach = np.array([-np.pi / 4, np.pi / 6, -np.pi / 3, np.pi / 2, np.pi / 4, -np.pi / 3, np.pi / 2])
    place_pos = np.array([-np.pi / 4, np.pi / 3, -np.pi / 2, np.pi / 2, np.pi / 4, -np.pi / 3, 0])

    waypoints = [home_pos, approach_pos, pick_pos, lift_pos, place_approach, place_pos, home_pos]

    # Time allocation for each segment
    segment_times = [0, 1.5, 3.0, 4.0, 6.0, 7.5, 10.0]

    # Find current segment
    segment = 0
    for i in range(len(segment_times) - 1):
        if segment_times[i] <= t < segment_times[i + 1]:
            segment = i
            break

    if segment >= len(waypoints) - 1:
        segment = len(waypoints) - 2

    # Interpolate between waypoints
    t_seg = (t - segment_times[segment]) / (segment_times[segment + 1] - segment_times[segment])
    t_seg = np.clip(t_seg, 0, 1)

    # Use smooth interpolation for pick/place operations
    s = smooth_step(t_seg)

    q_des = waypoints[segment] + s * (waypoints[segment + 1] - waypoints[segment])

    # Compute velocities and accelerations
    dt = segment_times[segment + 1] - segment_times[segment]
    qd_des = (waypoints[segment + 1] - waypoints[segment]) / dt * smooth_step_derivative(t_seg)
    qdd_des = np.zeros_like(q_des)  # Simplified - could compute second derivative

    return q_des, qd_des, qdd_des


def generate_cyclic_motion_trajectory(t):
    """
    Generate a complex cyclic motion that exercises all joints with varying frequencies.
    """
    # Different frequencies for each joint to create complex coupling
    frequencies = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.2, 0.4])
    amplitudes = np.array([np.pi / 3, np.pi / 4, np.pi / 3, np.pi / 2, np.pi / 4, np.pi / 3, np.pi])
    phase_offsets = np.array([0, np.pi / 4, np.pi / 2, np.pi / 3, np.pi / 6, np.pi / 8, np.pi / 12])

    # Add some coupling between joints
    omega = 2 * np.pi * frequencies

    q_des = np.zeros(7)
    qd_des = np.zeros(7)
    qdd_des = np.zeros(7)

    for i in range(7):
        # Primary motion
        primary = amplitudes[i] * np.sin(omega[i] * t + phase_offsets[i])

        # Add coupling terms from adjacent joints
        coupling = 0
        if i > 0:
            coupling += 0.1 * amplitudes[i - 1] * np.sin(omega[i - 1] * t + phase_offsets[i - 1])
        if i < 6:
            coupling += 0.1 * amplitudes[i + 1] * np.sin(omega[i + 1] * t + phase_offsets[i + 1])

        # Add time-varying amplitude modulation
        modulation = 1 + 0.3 * np.sin(0.05 * 2 * np.pi * t)

        q_des[i] = (primary + coupling) * modulation
        qd_des[i] = (amplitudes[i] * omega[i] * np.cos(omega[i] * t + phase_offsets[i])) * modulation
        qdd_des[i] = -(amplitudes[i] * omega[i] ** 2 * np.sin(omega[i] * t + phase_offsets[i])) * modulation

    # Apply joint limits
    q_des = np.clip(q_des, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX)

    return q_des, qd_des, qdd_des


def generate_trajectory(t):
    """
    Enhanced trajectory generator with multiple options.
    """
    if TRAJECTORY_TYPE == "MULTI_SEGMENT":
        return generate_multi_segment_trajectory(t)
    elif TRAJECTORY_TYPE == "FIGURE_EIGHT":
        return generate_figure_eight_trajectory(t)
    elif TRAJECTORY_TYPE == "PICK_PLACE":
        return generate_pick_place_trajectory(t)
    elif TRAJECTORY_TYPE == "CYCLIC_MOTION":
        return generate_cyclic_motion_trajectory(t)
    else:
        return generate_multi_segment_trajectory(t)


def get_joint_indices_by_name(robot_id, joint_names):
    joint_map = {p.getJointInfo(robot_id, i)[1].decode('UTF-8'): i for i in range(p.getNumJoints(robot_id))}
    return [joint_map[name] for name in joint_names]


def apply_torque_limits(tau, limits):
    return np.clip(tau, -limits, limits)


def run_simulation(control_mode, robot_dyn_model=None, identified_friction_params=None):
    """
    Runs a simulation with a specific control mode.

    :param control_mode: 'PD_ONLY', 'PD_GRAVITY', or 'PD_CTC'
    :param robot_dyn_model: The RobotDynamics object to use for feedforward.
    :param identified_friction_params: Friction parameters for the model.
    :return: Logs for time, desired position, and actual position.
    """
    physicsClientId = p.connect(p.GUI)
    p.setTimeStep(TIME_STEP)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(*GRAVITY_VECTOR)

    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
    joint_indices = get_joint_indices_by_name(robot_id, robot_config.ACTUATED_JOINT_NAMES)

    # Initialize robot to starting position
    q_initial, _, _ = generate_trajectory(0)
    for i, idx in enumerate(joint_indices):
        p.resetJointState(robot_id, idx, q_initial[i])
        p.setJointMotorControl2(robot_id, idx, p.VELOCITY_CONTROL, force=0)

    log_t, log_q_des, log_q_actual = [], [], []

    for t in np.arange(0, SIM_DURATION, TIME_STEP):
        joint_states = p.getJointStates(robot_id, joint_indices)
        q_actual = np.array([state[0] for state in joint_states])
        qd_actual = np.array([state[1] for state in joint_states])

        q_des, qd_des, qdd_des = generate_trajectory(t)

        # a) Feedback torque (always present)
        tau_fb = KP * (q_des - q_actual) + KD * (qd_des - qd_actual)

        # b) Feed-forward torque
        tau_ff = np.zeros(robot_config.NUM_JOINTS)
        if control_mode != 'PD_ONLY' and robot_dyn_model is not None:
            qdd_ff = qdd_des if control_mode == 'PD_CTC' else np.zeros_like(qdd_des)
            qd_ff = qd_des if control_mode == 'PD_CTC' else np.zeros_like(qd_des)

            tau_ff_rnea = robot_dyn_model.compute_rnea(q_des, qd_ff, qdd_ff, gravity=GRAVITY_VECTOR)

            tau_ff_friction = np.zeros_like(tau_ff_rnea)
            if identified_friction_params is not None:
                for i in range(robot_config.NUM_JOINTS):
                    fv, fc = identified_friction_params[i * 2], identified_friction_params[i * 2 + 1]
                    tau_ff_friction[i] = fv * qd_des[i] + fc * np.sign(qd_des[i])

            tau_ff = tau_ff_rnea + tau_ff_friction

        # c) Total Torque with CRITICAL safety limits
        tau_total = apply_torque_limits(tau_ff + tau_fb, MAX_TORQUES)

        p.setJointMotorControlArray(robot_id, joint_indices, p.TORQUE_CONTROL, forces=tau_total)
        p.stepSimulation()

        log_t.append(t)
        log_q_des.append(q_des)
        log_q_actual.append(q_actual)

    p.disconnect()
    return np.array(log_t), np.array(log_q_des), np.array(log_q_actual)


def main():
    if not os.path.exists(IDENTIFIED_PARAMS_PATH):
        print(f"ERROR: Identified parameters file not found at '{IDENTIFIED_PARAMS_PATH}'.")
        return

    print(f"Using trajectory type: {TRAJECTORY_TYPE}")

    params_data = np.load(IDENTIFIED_PARAMS_PATH)
    P_identified = params_data['P']
    robot_dyn_identified = RobotDynamics(robot_config)
    robot_dyn_identified.set_parameters_from_vector(P_identified)
    num_link_params = robot_config.NUM_JOINTS * 10
    friction_params = P_identified[num_link_params:]

    print("Running simulation 1: PD Control Only...")
    t_pd, q_des_pd, q_act_pd = run_simulation('PD_ONLY')

    print("Running simulation 2: PD + Gravity Compensation...")
    t_g, q_des_g, q_act_g = run_simulation('PD_GRAVITY', robot_dyn_identified, friction_params)

    print("Running simulation 3: PD + Full Computed Torque Control...")
    t_ctc, q_des_ctc, q_act_ctc = run_simulation('PD_CTC', robot_dyn_identified, friction_params)

    print("Plotting results...")
    plt.figure(figsize=(15, 3 * robot_config.NUM_JOINTS))

    for i in range(robot_config.NUM_JOINTS):
        plt.subplot(robot_config.NUM_JOINTS, 1, i + 1)

        err_pd = np.rad2deg(q_des_pd[:, i] - q_act_pd[:, i])
        err_g = np.rad2deg(q_des_g[:, i] - q_act_g[:, i])
        err_ctc = np.rad2deg(q_des_ctc[:, i] - q_act_ctc[:, i])

        rms_pd = np.sqrt(np.mean(err_pd ** 2))
        rms_g = np.sqrt(np.mean(err_g ** 2))
        rms_ctc = np.sqrt(np.mean(err_ctc ** 2))

        plt.plot(t_pd, err_pd, label=f'PD Only (RMS: {rms_pd:.3f}°)', color='red', linestyle=':')
        plt.plot(t_g, err_g, label=f'PD + Gravity (RMS: {rms_g:.3f}°)', color='green', linestyle='--')
        plt.plot(t_ctc, err_ctc, label=f'PD + Full CTC (RMS: {rms_ctc:.3f}°)', color='blue', linestyle='-')

        plt.title(f'Tracking Error for Joint {i} under Different Controllers')
        plt.ylabel('Error (deg)')
        plt.grid(True)
        plt.legend()

    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.suptitle(f'Controller Performance with Identified Model - {TRAJECTORY_TYPE} Trajectory', fontsize=16, y=1.02)
    plt.show()

    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    overall_rms_pd = np.sqrt(
        np.mean([np.mean((np.rad2deg(q_des_pd[:, i] - q_act_pd[:, i])) ** 2) for i in range(robot_config.NUM_JOINTS)]))
    overall_rms_g = np.sqrt(
        np.mean([np.mean((np.rad2deg(q_des_g[:, i] - q_act_g[:, i])) ** 2) for i in range(robot_config.NUM_JOINTS)]))
    overall_rms_ctc = np.sqrt(np.mean(
        [np.mean((np.rad2deg(q_des_ctc[:, i] - q_act_ctc[:, i])) ** 2) for i in range(robot_config.NUM_JOINTS)]))

    print(f"Overall RMS Tracking Error ({TRAJECTORY_TYPE} trajectory):")
    print(f"  PD Only:           {overall_rms_pd:.3f}°")
    print(f"  PD + Gravity:      {overall_rms_g:.3f}°")
    print(f"  PD + Full CTC:     {overall_rms_ctc:.3f}°")
    print(f"\nImprovement with gravity compensation: {((overall_rms_pd - overall_rms_g) / overall_rms_pd) * 100:.1f}%")
    print(f"Improvement with full CTC:             {((overall_rms_pd - overall_rms_ctc) / overall_rms_pd) * 100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()