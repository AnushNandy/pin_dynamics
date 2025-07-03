import pybullet as p
import pybullet_data
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics 
from src.dynamics.friction_transmission import JointModel
from config import robot_config 

# --- Configuration ---
URDF_PATH = r"/home/robot/dev/dyn/ArmModels/urdfs/P4/P4_Contra-Angle_right.urdf"
IDENTIFIED_PARAMS_PATH = r"/home/robot/dev/dyn/src/systemid/identified_params_pybullet.npz"

SIM_DURATION = 20.0
TIME_STEP = 1. / 240.
GRAVITY_VECTOR = np.array([0, 0, -9.81])

KP = np.array([100.0, 100.0, 200.0, 450.0, 200.0, 200.0, 0.7])
KD = np.array([2 * np.sqrt(k) for k in KP]) 

# MAX_TORQUES = np.array([200.0, 200.0, 150.0, 150.0, 100.0, 80.0, 80.0])
# MAX_TORQUES = np.array([140, 140, 51, 100, 51, 51, 7.7])
MAX_TORQUES = np.array([140, 140, 51, 51, 14, 14, 7.7])

JOINT_AMPLITUDES = np.deg2rad([15, 20, 12, 18, 10, 8, 0])
JOINT_FREQUENCIES = np.array([0.2, 0.25, 0.2, 0.3, 0.15, 0.2, 0.0])
JOINT_PHASE_OFFSETS = np.array([0, np.pi / 4, np.pi / 3, np.pi / 2, np.pi / 6, np.pi / 8, np.pi / 12])
INITIAL_JOINT_STATES = np.zeros(robot_config.NUM_JOINTS)

def generate_trajectory(t):
    """Generates a sinusoidal trajectory for validation."""
    omega = 2 * np.pi * JOINT_FREQUENCIES
    q_des = INITIAL_JOINT_STATES + JOINT_AMPLITUDES * np.sin(omega * t + JOINT_PHASE_OFFSETS)
    qd_des = JOINT_AMPLITUDES * omega * np.cos(omega * t + JOINT_PHASE_OFFSETS)
    qdd_des = -JOINT_AMPLITUDES * (omega ** 2) * np.sin(omega * t + JOINT_PHASE_OFFSETS)
    return q_des, qd_des, qdd_des

def get_joint_indices_by_name(robot_id, joint_names):
    """Helper function to get PyBullet joint indices from names."""
    joint_map = {p.getJointInfo(robot_id, i)[1].decode('UTF-8'): i for i in range(p.getNumJoints(robot_id))}
    return [joint_map[name] for name in joint_names]

def run_simulation(control_mode: str, 
                   robot_dyn_model: PinocchioRobotDynamics, 
                   joint_models: list = None,
                   identified_friction_params: np.ndarray = None):
    """
    Runs a simulation with a specific control mode.

    :param str control_mode: 'PD_ONLY', 'PD_GRAVITY', or 'PD_CTC'.
    :param PinocchioRobotDynamics robot_dyn_model: The Pinocchio dynamics object.
    :param list joint_models: A list of JointModel objects for detailed friction/transmission.
    :param np.ndarray identified_friction_params: Friction parameters (used for simpler models).
    :return: Logs for time, desired position, and actual position.
    """
    # --- PyBullet Setup ---
    physicsClientId = p.connect(p.GUI)
    p.setTimeStep(TIME_STEP)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(*GRAVITY_VECTOR)
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
    joint_indices = get_joint_indices_by_name(robot_id, robot_config.ACTUATED_JOINT_NAMES)

    # Reset robot to initial state
    for i, idx in enumerate(joint_indices):
        p.resetJointState(robot_id, idx, INITIAL_JOINT_STATES[i])
        p.setJointMotorControl2(robot_id, idx, p.VELOCITY_CONTROL, force=0)

    log_t, log_q_des, log_q_actual = [], [], []

    # --- Simulation Loop ---
    for t in np.arange(0, SIM_DURATION, TIME_STEP):
        joint_states = p.getJointStates(robot_id, joint_indices)
        q_actual = np.array([state[0] for state in joint_states])
        qd_actual = np.array([state[1] for state in joint_states])

        q_des, qd_des, qdd_des = generate_trajectory(t)

        # --- Torque Calculation ---
        # 1. Feedback Torque (always present for stability)
        tau_fb = KP * (q_des - q_actual) + KD * (qd_des - qd_actual)

        # 2. Feed-forward Torque (depends on the control mode)
        tau_ff = np.zeros(robot_config.NUM_JOINTS)
        if control_mode == 'PD_GRAVITY':
            # Use Pinocchio to compute gravity compensation torque
            # RNEA with zero velocity and acceleration gives the gravity term G(q).
            tau_ff = robot_dyn_model.compute_rnea(q_des, np.zeros_like(qd_des), np.zeros_like(qdd_des))

        elif control_mode == 'PD_CTC':
            # Step A: Compute the ideal rigid-body torques at the LINK side using Pinocchio
            tau_rnea_link_side = robot_dyn_model.compute_rnea(q_des, qd_des, qdd_des)

            # Step B: Use the advanced JointModel to calculate the required MOTOR torque.
            if joint_models:
                for i in range(robot_config.NUM_JOINTS):
                    # We use the refactored `compute_feedforward_torque` from Phase 3
                    tau_ff[i] = joint_models[i].compute_feedforward_torque(
                        tau_rnea=tau_rnea_link_side[i],
                        link_pos_des=q_des[i],
                        link_vel_des=qd_des[i],
                        link_acc_des=qdd_des[i]
                    )
            else:
                # Fallback to a simpler model if the full JointModel isn't provided
                tau_ff_friction = np.zeros_like(tau_rnea_link_side)
                if identified_friction_params is not None:
                    for i in range(robot_config.NUM_JOINTS):
                        fv, fc = identified_friction_params[i*2], identified_friction_params[i*2 + 1]
                        tau_ff_friction[i] = fv * qd_des[i] + fc * np.sign(qd_des[i])
                tau_ff = tau_rnea_link_side + tau_ff_friction


        # 3. Total Torque with Safety Limits
        tau_total = np.clip(tau_ff + tau_fb, -MAX_TORQUES, MAX_TORQUES)

        p.setJointMotorControlArray(robot_id, joint_indices, p.TORQUE_CONTROL, forces=tau_total)
        p.stepSimulation()

        log_t.append(t)
        log_q_des.append(q_des)
        log_q_actual.append(q_actual)

    p.disconnect()
    return np.array(log_t), np.array(log_q_des), np.array(log_q_actual)


def main():
    """Main function to load models, run simulations, and plot results."""
    if not os.path.exists(IDENTIFIED_PARAMS_PATH):
        print(f"FATAL: Identified parameters file not found at '{IDENTIFIED_PARAMS_PATH}'.")
        print("Please run the system identification pipeline first.")
        return
    if not os.path.exists(URDF_PATH):
        print(f"FATAL: URDF file not found at '{URDF_PATH}'.")
        return

    # --- 1. Load Identified Parameters ---
    print("Loading identified dynamic parameters...")
    params_data = np.load(IDENTIFIED_PARAMS_PATH)
    P_identified = params_data['P']

    print("\n--- VERIFYING IDENTIFIED PARAMETERS ---")
    print("Shape of P_identified:", P_identified.shape)
    print("All parameters:", P_identified)
    print("Sum of absolute values of P_identified:", np.sum(np.abs(P_identified)))
    print("---------------------------------------\n")
    
    # Extract the friction parameters part for the simpler friction model
    num_link_params = robot_config.NUM_JOINTS * 10
    friction_params = P_identified[num_link_params:]

    # --- 2. Initialize Models ---
    print("Initializing Pinocchio dynamics model...")
    # This single object replaces your rnea.py
    robot_dyn_pinocchio = PinocchioRobotDynamics(URDF_PATH)
    robot_dyn_pinocchio.set_parameters_from_vector(P_identified)
    
    print("Initializing high-fidelity joint models...")
    joint_models = []
    for i in range(robot_config.NUM_JOINTS):
        base_coulomb = 2.5 - i * 0.2
        base_stiction = 3.0 - i * 0.2
        joint_models.append(JointModel(
            gear_ratio=100.0, motor_inertia=0.0001 + i * 0.00001,
            coulomb_pos=base_coulomb, coulomb_neg=-(base_coulomb * 0.9),
            stiction_pos=base_stiction, stiction_neg=-(base_stiction * 0.9),
            viscous_coeff=0.15 - i * 0.01,
            stribeck_vel_pos=0.1, stribeck_vel_neg=-0.1, stiffness=20000.0,
            hysteresis_shape_A=1.0, hysteresis_shape_beta=0.5,
            hysteresis_shape_gamma=0.5, hysteresis_shape_n=1.0,
            hysteresis_scale_alpha=0.0, 
            dt=TIME_STEP
        ))


    # --- 3. Run Simulations ---
    print("\n--- Running Simulation 1: PD Control Only ---")
    t_pd, q_des_pd, q_act_pd = run_simulation('PD_ONLY', robot_dyn_pinocchio)

    print("\n--- Running Simulation 2: PD + Gravity Compensation (via Pinocchio) ---")
    t_g, q_des_g, q_act_g = run_simulation('PD_GRAVITY', robot_dyn_pinocchio)

    print("\n--- Running Simulation 3: PD + Full CTC (Pinocchio RNEA + Joint Models) ---")
    # This is the most advanced controller, combining both models
    t_ctc, q_des_ctc, q_act_ctc = run_simulation('PD_CTC', robot_dyn_pinocchio, joint_models=joint_models)

    # --- 4. Plotting and Comparison ---
    print("\nPlotting results...")
    plt.figure(figsize=(15, 3.5 * robot_config.NUM_JOINTS))

    for i in range(robot_config.NUM_JOINTS):
        plt.subplot(robot_config.NUM_JOINTS, 1, i + 1)

        err_pd = np.rad2deg(q_des_pd[:, i] - q_act_pd[:, i])
        err_g = np.rad2deg(q_des_g[:, i] - q_act_g[:, i])
        err_ctc = np.rad2deg(q_des_ctc[:, i] - q_act_ctc[:, i])

        rms_pd = np.sqrt(np.mean(err_pd ** 2))
        rms_g = np.sqrt(np.mean(err_g ** 2))
        rms_ctc = np.sqrt(np.mean(err_ctc ** 2))

        plt.plot(t_pd, err_pd, label=f'PD Only (RMS: {rms_pd:.3f}°)', color='red', linestyle=':')
        plt.plot(t_g, err_g, label=f'PD + Gravity Comp (RMS: {rms_g:.3f}°)', color='green', linestyle='--')
        plt.plot(t_ctc, err_ctc, label=f'PD + Full CTC (RMS: {rms_ctc:.3f}°)', color='blue', linestyle='-')

        plt.title(f'Tracking Error for Joint {i} under Different Controllers (Pinocchio Backend)')
        plt.ylabel('Error (deg)')
        plt.grid(True)
        plt.legend()

    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.suptitle('Controller Performance Comparison using Identified Pinocchio Model', fontsize=16, y=1.02)
    # plt.show()
    plt.savefig('../test/Fig_02_errplot.png')


if __name__ == "__main__":
    main()