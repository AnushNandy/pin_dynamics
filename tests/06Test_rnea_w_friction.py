import pybullet as p
import pybullet_data
import numpy as np
import os
import MathUtils
import matplotlib.pyplot as plt
from Dynamics_full.config import robot_config
from Dynamics_full.src.dynamics.rnea import RobotDynamics
from Dynamics_full.src.dynamics.friction_transmission import JointModel

SIM_DURATION = 1
TIME_STEP = 1. / 240.
URDF_PATH = r"C:\dev\control-sw-tools\Dynamics_full\ArmModels\urdfs\P4\P4_Contra-Angle_right.urdf"

KP = np.array([600.0, 120.0, 100.0, 70.0, 40.0, 30.0, 20.0])
KD = np.array([150.0, 20.0, 18.0, 12.0, 7.0, 5.0, 3.0])

M_PI = np.pi

KDL_CHAIN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.37502, 0.0, 0.07664, 0.0, 0.0, 0.0,
               -0.025, 0.0, 0.1645, 0.0, M_PI / 2.0, 0.0,
               0.0, -0.1088, -0.045, 0.0, 0.0, -M_PI / 2.0,
               -0.25712, 0.045, 0.21612, 0.0, 0.0, M_PI / 2.0,
               0.092, 0.0, -0.052, 0.0, -M_PI / 2.0, -M_PI / 2.0,
               -0.052, 0.0, 0.13375, 0.0, M_PI / 2.0, -M_PI / 2.0,
               0.0, 0.0, 0.0987, 2.7951, 0.0, 0.0]

JOINT_AMPLITUDES = np.deg2rad([15, 20, 12, 18, 10, 8, 6])
JOINT_FREQUENCIES = np.array([0.5, 0.6, 0.4, 0.7, 0.3, 0.4, 0.2])
JOINT_PHASE_OFFSETS = np.array([0, np.pi / 4, np.pi / 3, np.pi / 2, np.pi / 6, np.pi / 8, np.pi / 12])
INITIAL_JOINT_STATES = np.zeros(robot_config.NUM_JOINTS)
GRAVITY_VECTOR = np.array([0, 0, -9.81])


ORDERED_LINK_NAMES_KDL_CORRESPONDENCE = [
    "Base",
    "Link_0",
    "Link_1",
    "Link_2",
    "Link_3",
    "Link_4",
    "Link_5",
    "End_Effector"
]

def generate_trajectory(t):
    omega = 2 * np.pi * JOINT_FREQUENCIES
    q_des = INITIAL_JOINT_STATES + JOINT_AMPLITUDES * np.sin(omega * t + JOINT_PHASE_OFFSETS)
    qd_des = JOINT_AMPLITUDES * omega * np.cos(omega * t + JOINT_PHASE_OFFSETS)
    qdd_des = -JOINT_AMPLITUDES * (omega ** 2) * np.sin(omega * t + JOINT_PHASE_OFFSETS)
    return q_des, qd_des, qdd_des

def get_joint_indices_by_name(robot_id, joint_names):
    joint_map = {p.getJointInfo(robot_id, i)[1].decode('UTF-8'): i for i in range(p.getNumJoints(robot_id))}
    return [joint_map[name] for name in joint_names]

def apply_torque_limits(tau, max_torques):
    return np.clip(tau, -max_torques, max_torques)

def compute_inverse_kinematics(robot_id, q_initial, robot_target_pos, robot_target_orientation):
    start_end_effector_pose, _ = MathUtils.compute_forward_kinematics(KDL_CHAIN, q_initial, neocis_convention=True)

    num_joints = p.getNumJoints(robot_id)
    end_effector_link_index = num_joints - 1

    ik_solution = p.calculateInverseKinematics(robot_id, end_effector_link_index, robot_target_pos, robot_target_orientation)
    return ik_solution


def get_link_index_by_name_pb(robot_id, target_link_name):
    # Base link is special (-1)
    base_name = p.getBodyInfo(robot_id)[0].decode('UTF-8')
    if target_link_name == base_name or target_link_name == "Base":
        return -1  # Base link

    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        link_name = info[12].decode('UTF-8')
        if link_name == target_link_name:
            return i

    print(f"Warning: Link '{target_link_name}' not found in PyBullet model.")
    print("Available links:")
    print(f"  Base: {base_name}")
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        print(f"  Joint {i}: {info[12].decode('UTF-8')}")
    return None

def generate_trajectory_ik(q_actual, q_des, total_time=1.0, dt=0.01):
    """
    Generate a smooth trajectory from q_actual to q_des using quintic interpolation.

    Parameters:
    - q_actual: np.ndarray of shape (7,) - starting joint angles
    - q_des: np.ndarray of shape (7,) - target joint angles
    - total_time: float - duration of the trajectory (seconds)
    - dt: float - time step (seconds)

    Returns:
    - q_des_steps: np.ndarray of shape (N, 7) - joint positions over time
    - qd_des_steps: np.ndarray of shape (N, 7) - joint velocities over time
    - qdd_des_steps: np.ndarray of shape (N, 7) - joint accelerations over time
    """
    q_actual = np.array(q_actual)
    q_des = np.array(q_des)
    timesteps = np.arange(0, total_time + dt, dt)
    N = len(timesteps)

    q_des_steps = np.zeros((N, 7))
    qd_des_steps = np.zeros((N, 7))
    qdd_des_steps = np.zeros((N, 7))

    for i in range(7):
        q0 = q_actual[i]
        qf = q_des[i]
        dq0 = dqf = ddq0 = ddqf = 0.0  # Start and end velocities and accelerations are zero

        T = total_time

        # Quintic polynomial coefficients
        a0 = q0
        a1 = dq0
        a2 = 0.5 * ddq0
        a3 = (20 * (qf - q0) - (8 * dqf + 12 * dq0) * T - (3 * ddq0 - ddqf) * T**2) / (2 * T**3)
        a4 = (30 * (q0 - qf) + (14 * dqf + 16 * dq0) * T + (3 * ddq0 - 2 * ddqf) * T**2) / (2 * T**4)
        a5 = (12 * (qf - q0) - (6 * dqf + 6 * dq0) * T - (ddq0 - ddqf) * T**2) / (2 * T**5)

        for j, t in enumerate(timesteps):
            q = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
            qd = a1 + 2 * a2 * t + 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
            qdd = 2 * a2 + 6 * a3 * t + 12 * a4 * t**2 + 20 * a5 * t**3

            q_des_steps[j, i] = q
            qd_des_steps[j, i] = qd
            qdd_des_steps[j, i] = qdd

    return q_des_steps, qd_des_steps, qdd_des_steps

def main():
    robot_dyn = RobotDynamics(robot_config)

    joint_models = []
    for i in range(robot_config.NUM_JOINTS):
        base_coulomb = 2.5 - i * 0.2 # All this stuff is hard-coded
        base_stiction = 3.0 - i * 0.2
        joint_models.append(JointModel(
            gear_ratio=100.0,
            motor_inertia=0.0001 + i * 0.00001,
            coulomb_pos=base_coulomb,
            coulomb_neg=-(base_coulomb * 0.9),
            stiction_pos=base_stiction,
            stiction_neg=-(base_stiction * 0.9),
            viscous_coeff=0.15 - i * 0.01,
            stribeck_vel_pos=0.1,
            stribeck_vel_neg=-0.1,
            stiffness=20000.0,
            hysteresis_shape_A=1.0,
            hysteresis_shape_beta=0.5,
            hysteresis_shape_gamma=0.5,
            hysteresis_shape_n=1.0,
            hysteresis_scale_alpha=0.0,  #off
            dt=TIME_STEP
        ))

    physicsClientId = p.connect(p.GUI)
    p.setTimeStep(TIME_STEP)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(*GRAVITY_VECTOR)
    p.loadURDF("plane.urdf")

    if not os.path.exists(URDF_PATH):
        print(f"URDF not found at {URDF_PATH}")
        return

    robot_id = p.loadURDF(URDF_PATH, [0, 0, 1], useFixedBase=True)
    joint_indices = get_joint_indices_by_name(robot_id, robot_config.ACTUATED_JOINT_NAMES)

    # for i, idx in enumerate(joint_indices):
    #     p.resetJointState(robot_id, idx, INITIAL_JOINT_STATES[i])
    #     p.setJointMotorControl2(robot_id, idx, p.VELOCITY_CONTROL, force=0)

    max_torques = np.array([100.0] * robot_config.NUM_JOINTS)

    log_t, log_q_des, log_q_actual = [], [], []

    for t in np.arange(0, SIM_DURATION, TIME_STEP):
        joint_states = p.getJointStates(robot_id, joint_indices)
        q_actual = np.array([state[0] for state in joint_states])
        qd_actual = np.array([state[1] for state in joint_states])

        for i, link_name_kdl_corr in enumerate(ORDERED_LINK_NAMES_KDL_CORRESPONDENCE):
            pb_link_idx = get_link_index_by_name_pb(robot_id, link_name_kdl_corr)
            if pb_link_idx is None:
                print(f"CRITICAL: Could not find link '{link_name_kdl_corr}' for PyBullet FK. Exiting.")
                p.disconnect()
                return

            if pb_link_idx == -1:
                pos, orn = p.getBasePositionAndOrientation(robot_id)
            else:  # Other links
                link_state = p.getLinkState(robot_id, pb_link_idx, computeForwardKinematics=1)
                pos, orn = link_state[4], link_state[5]

        q_desired = compute_inverse_kinematics(robot_id, q_actual, pos, orn)
        q_des, qd_des, qdd_des = generate_trajectory_ik(q_actual, q_desired)

        for jj in range(len(q_des)):
            tau_rnea = robot_dyn.compute_rnea(q_des[jj], qd_des[jj], qdd_des[jj], gravity=GRAVITY_VECTOR)

            tau_motor_ff = np.zeros(robot_config.NUM_JOINTS)
            for i in range(robot_config.NUM_JOINTS):
                model = joint_models[i]
                tau_motor_ff[i] = model.compute_motor_torque(
                    tau_rnea[i],
                    link_pos=q_des[jj][i],
                    link_vel=qd_des[jj][i],
                    motor_pos=model.N * q_des[jj][i],
                    motor_vel=model.N * qd_des[jj][i],
                    motor_acc=model.N * qdd_des[jj][i]
                )

            # Read current joint states from PyBullet
            joint_states = p.getJointStates(robot_id, joint_indices)
            q_actual = np.array([s[0] for s in joint_states])
            qd_actual = np.array([s[1] for s in joint_states])

            tau_fb = KP * (q_des[jj] - q_actual) + KD * (qd_des[jj] - qd_actual)
            tau_total = tau_motor_ff + tau_fb

            p.setJointMotorControlArray(robot_id, joint_indices, p.TORQUE_CONTROL, forces=tau_total)
            p.stepSimulation()

            log_t.append(t)  # assuming dt is defined
            log_q_des.append(q_des[jj])
            log_q_actual.append(q_actual)

            array_log_t = np.array(log_t)
            array_log_q_des = np.array(log_q_des)
            array_log_q_actual = np.array(log_q_actual)


    p.disconnect()

    plt.figure(figsize=(12, 2.5 * robot_config.NUM_JOINTS))
    for i in range(robot_config.NUM_JOINTS):
        plt.subplot(robot_config.NUM_JOINTS, 2, 2 * i + 1)
        plt.plot(array_log_t, np.rad2deg(array_log_q_des[:, i]), 'r--', label='Desired')
        plt.plot(array_log_t, np.rad2deg(array_log_q_actual[:, i]), 'b-', label='Actual')
        plt.title(f'Joint {i} Tracking'), plt.ylabel('Pos (deg)')
        plt.legend(), plt.grid(True)

        plt.subplot(robot_config.NUM_JOINTS, 2, 2 * i + 2)
        err = np.rad2deg(array_log_q_des[:, i] - array_log_q_actual[:, i])
        plt.plot(array_log_t, err, 'g-', label='Error')
        plt.title(f'Joint {i} Error'), plt.ylabel('Error (deg)')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()