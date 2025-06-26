import numpy as np
import time, os, sys
import pybullet as p
import pybullet_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import robot_config
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics

# --- Constants and Configuration ---
URDF_PATH = robot_config.URDF_PATH
SAVE_PATH = "/home/robot/dev/dyn/src/systemid/sysid_data_pybullet.npz"
SIM_DURATION = 150.0
TIME_STEP = 1. / 240.
NUM_JOINTS = robot_config.NUM_JOINTS

# joint_limits = [
#     (-2.5, 2.5), (-2.3, 2.3), (-2.2, 2.2),
#     (-2.1, -2.1), (-1.8, 1.8), (-1, 1),
#     (-1, 1)
# ]

joint_limits = [
    (-2.5, 2.5), (-2.3, 2.3), (-2.2, 2.2),
    (-2.5, 2.5),(-2.5, 2.5), (-2.5, 2.5),
    (-1, 1)
]
low_bounds = np.array([limit[0] for limit in joint_limits])
high_bounds = np.array([limit[1] for limit in joint_limits])

# --- PD Controller Gains ---
KP = np.array([100.0, 100.0, 200.0, 500.0, 150.0, 150.0, 0.7])
KD = np.array([2 * np.sqrt(k) for k in KP])
MAX_TORQUES = np.array([140, 140, 51, 51, 14, 14, 7.7])

def setup_pybullet_simulation(urdf_path: str):
    """Initializes PyBullet, loads the robot, and configures the joints."""
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(TIME_STEP)

    # Load robot, ensuring it doesn't have a fixed base initially
    # robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
    robot_start_pos = [0, 0, 0.5]
    robot_start_orn = p.getQuaternionFromEuler([0, 0, 0])
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(URDF_PATH, robot_start_pos, useFixedBase=False)
    p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=-1, 
        childBodyUniqueId=-1,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=robot_start_pos,
        childFrameOrientation=robot_start_orn
    )

    # Get the indices of the actuated joints
    all_joint_indices = range(p.getNumJoints(robot_id))
    actuated_joint_indices = [j for j in all_joint_indices if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED]

    if len(actuated_joint_indices) != NUM_JOINTS:
        raise ValueError(f"URDF has {len(actuated_joint_indices)} movable joints, but NUM_JOINTS is {NUM_JOINTS}")

    # IMPORTANT: Disable PyBullet's default motor control
    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=actuated_joint_indices,
        controlMode=p.VELOCITY_CONTROL,
        forces=np.zeros(NUM_JOINTS)
    )
    return robot_id, actuated_joint_indices

def generate_minimum_jerk_trajectory(t, duration_per_target=6.0):
    """Generates a minimum jerk trajectory between random targets."""
    target_idx = int(t / duration_per_target)
    t_local = t % duration_per_target
    
    # Use seeds to make waypoint generation repeatable for a given time
    np.random.seed(target_idx)
    # q_start = np.random.uniform(-1.5, 1.5, NUM_JOINTS)
    q_start = np.random.uniform(low=low_bounds, high=high_bounds, size=NUM_JOINTS)
    np.random.seed(target_idx + 1000)
    # q_end = np.random.uniform(-1.5, 1.5, NUM_JOINTS)
    q_end = np.random.uniform(low=low_bounds, high=high_bounds, size=NUM_JOINTS)
    
    # Clamp to reasonable joint limits to avoid self-collision
    # q_start = np.clip(q_start, -np.pi/2, np.pi/2)
    # q_end = np.clip(q_end, -np.pi/2, np.pi/2)

    T = duration_per_target
    tau = t_local / T
    
    # Quintic polynomial: 6t^5 - 15t^4 + 10t^3
    s = 10*tau**3 - 15*tau**4 + 6*tau**5
    s_dot = (30*tau**2 - 60*tau**3 + 30*tau**4) / T
    s_ddot = (60*tau - 180*tau**2 + 120*tau**3) / (T**2)
    
    q_des = q_start + (q_end - q_start) * s
    qd_des = (q_end - q_start) * s_dot
    qdd_des = (q_end - q_start) * s_ddot
    
    return q_des, qd_des, qdd_des

def generate_fourier_series_trajectory(t, num_harmonics=5):
    """
    Generates an exciting trajectory using a sum of sinusoids for each joint.
    Reduced harmonics for stability.
    """
    q_des = np.zeros(NUM_JOINTS)
    qd_des = np.zeros(NUM_JOINTS)
    qdd_des = np.zeros(NUM_JOINTS)
    w = np.pi   # Reduced base frequency for stability

    for i in range(NUM_JOINTS):
        np.random.seed(i)
        # Reduced amplitude for stability
        a_n = np.random.uniform(0.02, 0.5, num_harmonics)  # Reduced from 0.05-0.2
        b_n = np.random.uniform(0.02, 0.5, num_harmonics)
        phase_shifts = np.random.uniform(0, 2 * np.pi, num_harmonics)

        for n in range(num_harmonics):
            angle = (n + 1) * w * t + phase_shifts[n]
            q_des[i] += a_n[n] * np.sin(angle) + b_n[n] * np.cos(angle)
            qd_des[i] += (n + 1) * w * (a_n[n] * np.cos(angle) - b_n[n] * np.sin(angle))
            qdd_des[i] += -((n + 1) * w)**2 * (a_n[n] * np.sin(angle) + b_n[n] * np.cos(angle))
            
    return q_des, qd_des, qdd_des

def main():
    """
    Generates and saves dynamically-consistent system identification data
    using a closed-loop PyBullet simulation.
    """
    print("--- Generating System Identification Data using PyBullet & Pinocchio ---")

    # 1. Initialize PyBullet and the Pinocchio dynamics model
    robot_id, joint_indices = setup_pybullet_simulation(URDF_PATH)
    dynamics_model = PinocchioRobotDynamics(URDF_PATH)

    # Set initial robot state to be neutral
    neutral_q = np.zeros(NUM_JOINTS)
    for i, joint_idx in enumerate(joint_indices):
        p.resetJointState(robot_id, joint_idx, targetValue=neutral_q[i])
    
    log_q, log_qd, log_qdd, log_tau = [], [], [], []

    print("Starting closed-loop simulation for data generation...")
    for t in np.arange(0, SIM_DURATION, TIME_STEP):
        # --- I. Trajectory Generation ---
        # q_des, qd_des, qdd_des = generate_minimum_jerk_trajectory(t)
        q_des, qd_des, qdd_des = generate_fourier_series_trajectory(t)

        # --- II. Get Actual State from Simulator ---
        joint_states = p.getJointStates(robot_id, joint_indices)
        q_actual = np.array([state[0] for state in joint_states])
        qd_actual = np.array([state[1] for state in joint_states])

        # --- III. Controller Calculation ---
        # a) Feedforward torque from Pinocchio 
        tau_ff = dynamics_model.compute_rnea(q_des, qd_des, qdd_des)

        # b) Feedback torque (PD controller)
        pos_error = q_des - q_actual
        vel_error = qd_des - qd_actual
        tau_fb = KP * pos_error + KD * vel_error
        
        # c) Total command torque
        tau_cmd = tau_ff + tau_fb

        tau_limited = np.clip(tau_cmd, -MAX_TORQUES, MAX_TORQUES)

        # --- IV. Apply Torque and Step Simulation ---
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=tau_limited
        )
        p.stepSimulation()

        # --- V. Log Dynamically Consistent Data ---
        # To get a perfectly consistent dataset for the regressor, we calculate the
        # acceleration that *must* have resulted from the commanded torque,
        # according to the dynamics model.
        # τ = M(q)q̈ + n(q, q̇), so q̈ = M⁻¹(τ - n)
        M_act = dynamics_model.compute_mass_matrix(q_actual)
        n_act = dynamics_model.compute_nonlinear_effects(q_actual, qd_actual)
        
        try:
            M_inv = np.linalg.inv(M_act)
            qdd_actual = M_inv @ (tau_cmd - n_act)
        except np.linalg.LinAlgError:
            # Skip this data point if mass matrix is singular for some reason
            print(f"Warning: Singular mass matrix at t={t:.2f}. Skipping data point.")
            continue

        # Log the actual state and the torque that produced it
        log_q.append(q_actual)
        log_qd.append(qd_actual)
        # log_qdd.append(qdd_actual) # The calculated, consistent acceleration
        log_tau.append(tau_limited)

        if int(t) % 5 == 0 and abs(t - int(t)) < TIME_STEP / 2:
            print(f"Generated data up to {int(t)}s / {int(SIM_DURATION)}s")

    p.disconnect()

    log_q = np.array(log_q)
    log_qd = np.array(log_qd)
    log_tau = np.array(log_tau)
    log_qdd = np.gradient(log_qd, axis=0) / TIME_STEP

    # --- VI. Save the Dataset ---
    # np.savez(
    #     SAVE_PATH,
    #     q=np.array(log_q),
    #     qd=np.array(log_qd),
    #     qdd=np.array(log_qdd),
    #     tau=np.array(log_tau)
    # )
    np.savez(
        SAVE_PATH,
        q=log_q,
        qd=log_qd,
        qdd=log_qdd,
        tau=log_tau
    )

    print(f"--- Data generation complete. ---")
    print(f"Saved {len(log_q)} dynamically consistent samples to '{SAVE_PATH}'")

if __name__ == "__main__":
    main()