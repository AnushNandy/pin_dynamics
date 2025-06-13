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

# --- Controller Gains ---
KP = np.array([80.0, 120.0, 100.0, 70.0, 40.0, 30.0, 20.0]) / 2.0
KD = np.array([15.0, 20.0, 18.0, 12.0, 7.0, 5.0, 3.0])  # Kd is often related to sqrt(Kp)

# --- Torque Limits ---
MAX_TORQUES = np.array([200.0, 200.0, 150.0, 150.0, 100.0, 80.0, 80.0])

# --- Trajectory ---
JOINT_AMPLITUDES = np.deg2rad([15, 20, 12, 18, 10, 8, 6])
JOINT_FREQUENCIES = np.array([0.2, 0.25, 0.2, 0.3, 0.15, 0.2, 0.1])  # Reduced frequencies
JOINT_PHASE_OFFSETS = np.array([0, np.pi / 4, np.pi / 3, np.pi / 2, np.pi / 6, np.pi / 8, np.pi / 12])
INITIAL_JOINT_STATES = np.zeros(robot_config.NUM_JOINTS)


def generate_trajectory(t):
    omega = 2 * np.pi * JOINT_FREQUENCIES
    q_des = INITIAL_JOINT_STATES + JOINT_AMPLITUDES * np.sin(omega * t + JOINT_PHASE_OFFSETS)
    qd_des = JOINT_AMPLITUDES * omega * np.cos(omega * t + JOINT_PHASE_OFFSETS)
    qdd_des = -JOINT_AMPLITUDES * (omega ** 2) * np.sin(omega * t + JOINT_PHASE_OFFSETS)
    return q_des, qd_des, qdd_des


def get_joint_indices_by_name(robot_id, joint_names):
    joint_map = {p.getJointInfo(robot_id, i)[1].decode('UTF-8'): i for i in range(p.getNumJoints(robot_id))}
    return [joint_map[name] for name in joint_names]


def apply_torque_limits(tau, limits):
    return np.clip(tau, -limits, limits)


def run_simulation(control_mode, robot_dyn_model=None, identified_friction_params=None):
    """
    Run simulation with a specific control mode.

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

    for i, idx in enumerate(joint_indices):
        p.resetJointState(robot_id, idx, INITIAL_JOINT_STATES[i])
        p.setJointMotorControl2(robot_id, idx, p.VELOCITY_CONTROL, force=0)

    log_t, log_q_des, log_q_actual = [], [], []

    for t in np.arange(0, SIM_DURATION, TIME_STEP):
        joint_states = p.getJointStates(robot_id, joint_indices)
        q_actual = np.array([state[0] for state in joint_states])
        qd_actual = np.array([state[1] for state in joint_states])

        q_des, qd_des, qdd_des = generate_trajectory(t)

        # a) Feedback torque (always present)
        tau_fb = KP * (q_des - q_actual) + KD * (qd_des - qd_actual)

        # b) Feed-forward torque (depends on mode)
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

    # --- Load Identified Model ---
    params_data = np.load(IDENTIFIED_PARAMS_PATH)
    P_identified = params_data['P']
    robot_dyn_identified = RobotDynamics(robot_config)
    robot_dyn_identified.set_parameters_from_vector(P_identified)
    num_link_params = robot_config.NUM_JOINTS * 10
    friction_params = P_identified[num_link_params:]

    # --- Run Simulations for Each Control Mode ---
    print("Running simulation 1: PD Control Only...")
    t_pd, q_des_pd, q_act_pd = run_simulation('PD_ONLY')

    print("Running simulation 2: PD + Gravity Compensation...")
    t_g, q_des_g, q_act_g = run_simulation('PD_GRAVITY', robot_dyn_identified, friction_params)

    print("Running simulation 3: PD + Full Computed Torque Control...")
    t_ctc, q_des_ctc, q_act_ctc = run_simulation('PD_CTC', robot_dyn_identified, friction_params)

    # --- Plotting and Comparison ---
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
    plt.suptitle('Controller Performance with Identified Model', fontsize=16, y=1.02)
    plt.show()


if __name__ == "__main__":
    main()