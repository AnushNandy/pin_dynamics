import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics 
from src.systemid.pinocchio_friction_regressor import smooth_sign
from config import robot_config 
import pinocchio as pin

SIM_DURATION = 700.0
TIME_STEP = 1. / 240.
URDF_PATH = robot_config.URDF_PATH
URDF_PATH_LEFT = robot_config.URDF_PATH_LEFT
IDENTIFIED_PARAMS_PATH = "/home/robot/dev/dyn/src/systemid/identified_params_pybullet.npz"

SAVE_DIR = "./tests/simulation_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# KP = np.array([100.0, 100.0, 200.0, 450.0, 200.0, 200.0, 0.7])
KP = np.array([100.0, 100.0, 200.0, 500.0, 150.0, 150.0, 0.7])
KD = np.array([2 * np.sqrt(k) for k in KP]) 

# Physics and Robot Constants
GRAVITY_VECTOR = np.array([0, 0, -9.81])
MAX_TORQUES = np.array([140, 140, 51, 51, 14, 14, 7.7])
NUM_JOINTS = robot_config.NUM_JOINTS

class GroundTruthFeedforwardController:
    def __init__(self, urdf_path: str):
        print("--- Initializing GROUND TRUTH Feedforward Controller ---")
        self.model_dynamics = PinocchioRobotDynamics(urdf_path)
        self.num_joints = self.model_dynamics.num_actuated_joints
        print("Using known ground-truth parameters for RNEA and Friction.")
        print("-----------------------------------------------------")

    def compute_feedforward_torque(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """
        Computes the feedforward torque using KNOWN parameters.
        τ_ff = RNEA_truth(q, qd, qdd) + τ_friction_truth(qd)
        """
        tau_rnea = self.model_dynamics.compute_rnea(q, qd, qdd, gravity=GRAVITY_VECTOR)
        return tau_rnea
    

class PinocchioFeedforwardController:
    """
    This controller uses a system-identified Pinocchio model to compute
    the feedforward torques (RNEA + identified friction).
    """
    def __init__(self, urdf_path: str, identified_params_path: str):
        print("--- Initializing Pinocchio Feedforward Controller ---")
        self.model_dynamics = PinocchioRobotDynamics(urdf_path)
        self.num_joints = self.model_dynamics.num_actuated_joints

        self._load_and_set_identified_params(identified_params_path)
        print("-----------------------------------------------------")

    def _load_and_set_identified_params(self, file_path: str):
        try:
            params_data = np.load(file_path)
            identified_params = params_data['P']
            
            num_rnea_params = (self.model_dynamics.model.nbodies - 1) * 10
            rnea_params = identified_params[:num_rnea_params]
            
            self.model_dynamics.set_parameters_from_vector(rnea_params)
            print(f"Loaded and set identified parameters from '{file_path}'")

        except FileNotFoundError:
            print(f"FATAL: Identified params file not found at '{file_path}'.")
            raise

    def compute_feedforward_torque(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """
        Computes the feedforward torque.
        τ_ff = RNEA(q, qd, qdd) + τ_friction(qd)
        """
        tau_rnea = self.model_dynamics.compute_rnea(q, qd, qdd, gravity=GRAVITY_VECTOR)
        return tau_rnea

    def compute_end_effector_pose(self, q: np.ndarray):
        """
        Compute end-effector position and orientation from joint angles
        """
        return self.model_dynamics.compute_forward_kinematics(q)
    
def get_joint_indices_by_name(robot_id, joint_names):
    joint_map = {p.getJointInfo(robot_id, i)[1].decode('UTF-8'): i for i in range(p.getNumJoints(robot_id))}
    return [joint_map[name] for name in joint_names]

def generate_quintic_spline(q0, qf, qd0, qdf, qdd0, qddf, T):
    M = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5],
        [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
        [0, 0, 2, 6*T, 12*T**2, 20*T**3]
    ])
    
    b = np.array([q0, qd0, qdd0, qf, qdf, qddf])
    coeffs = np.linalg.solve(M, b)
    
    def trajectory(t):
        t_vec = np.array([1, t, t**2, t**3, t**4, t**5])
        pos = coeffs @ t_vec
        
        t_vec_d = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
        vel = coeffs @ t_vec_d
        
        t_vec_dd = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3])
        acc = coeffs @ t_vec_dd
        
        return pos, vel, acc
        
    return trajectory

def generate_smooth_trajectory_for_all_joints(start_pos, end_pos, duration, num_joints):
    trajectories = []
    for i in range(num_joints):
        q0, qf = start_pos[i], end_pos[i]
        qd0, qdf = 0, 0
        qdd0, qddf = 0, 0
        
        trajectories.append(generate_quintic_spline(q0, qf, qd0, qdf, qdd0, qddf, duration))
        
    return trajectories

def evaluate_waypoint_trajectory(joint_trajectories, t, joint_idx):
    """Evaluate the piecewise trajectory at time t for a specific joint"""
    for traj_func, start_time, duration in joint_trajectories[joint_idx]:
        if start_time <= t < start_time + duration:
            local_t = t - start_time
            return traj_func(local_t)
    
    last_traj, last_start, last_duration = joint_trajectories[joint_idx][-1]
    return last_traj(last_duration)

def plot_joint_tracking_individual(time_vec, q_des_traj, q_actual_traj, joint_names=None, save_dir="./test/simulation_results"):
    """Plot joint tracking for each joint individually"""
    if joint_names is None:
        joint_names = [f'Joint {i+1}' for i in range(q_des_traj.shape[1])]
    
    # Create subdirectory for joint tracking plots
    tracking_dir = os.path.join(save_dir, "joint_tracking")
    os.makedirs(tracking_dir, exist_ok=True)
    
    for i in range(q_des_traj.shape[1]):
        plt.figure(figsize=(10, 6))
        plt.plot(time_vec, np.rad2deg(q_des_traj[:, i]), 'r-', linewidth=2, label='Desired')
        plt.plot(time_vec, np.rad2deg(q_actual_traj[:, i]), 'b--', linewidth=2, label='Actual')
        plt.title(f'{joint_names[i]} Position Tracking', fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Position (degrees)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(tracking_dir, f"joint_{i+1}_tracking.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_joint_torques_individual(time_vec, tau_ff, tau_fb, tau_total, max_torques, joint_names=None, save_dir="./test/simulation_results"):
    """Plot torques for each joint individually"""
    if joint_names is None:
        joint_names = [f'Joint {i+1}' for i in range(tau_ff.shape[1])]
    
    # Create subdirectory for joint torque plots
    torques_dir = os.path.join(save_dir, "joint_torques")
    os.makedirs(torques_dir, exist_ok=True)
    
    for i in range(tau_ff.shape[1]):
        plt.figure(figsize=(10, 6))
        plt.plot(time_vec, tau_ff[:, i], 'r-', linewidth=2, label='Feedforward')
        plt.plot(time_vec, tau_fb[:, i], 'g-', linewidth=2, label='Feedback')
        plt.plot(time_vec, tau_total[:, i], 'b-', linewidth=2, label='Total Applied')
        # plt.axhline(y=max_torques[i], color='k', linestyle='--', alpha=0.7, label=f'Limit (±{max_torques[i]} Nm)')
        # plt.axhline(y=-max_torques[i], color='k', linestyle='--', alpha=0.7)
        plt.title(f'{joint_names[i]} Torques', fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Torque (Nm)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(torques_dir, f"joint_{i+1}_torques.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_end_effector_cartesian(time_vec, ee_positions, ee_orientations, save_dir="./test/simulation_results"):
    """Plot end-effector Cartesian position and orientation (RPY)"""
    
    # Create main save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Position plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_vec, ee_positions[:, 0], 'r-', linewidth=2, label='X')
    plt.plot(time_vec, ee_positions[:, 1], 'g-', linewidth=2, label='Y')
    plt.plot(time_vec, ee_positions[:, 2], 'b-', linewidth=2, label='Z')
    plt.title('End-Effector Position', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Position (m)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Orientation plot (RPY)
    plt.subplot(2, 1, 2)
    plt.plot(time_vec, np.rad2deg(ee_orientations[:, 0]), 'r-', linewidth=2, label='Roll')
    plt.plot(time_vec, np.rad2deg(ee_orientations[:, 1]), 'g-', linewidth=2, label='Pitch')
    plt.plot(time_vec, np.rad2deg(ee_orientations[:, 2]), 'b-', linewidth=2, label='Yaw')
    plt.title('End-Effector Orientation (RPY)', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Angle (degrees)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Fixed: Use proper filename instead of joint index
    plt.savefig(os.path.join(save_dir, "end_effector_pose.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_simulation_results(time_vec, q_des, q_actual, tau_ff, tau_fb, tau_total, 
                          ee_pos, ee_rpy, max_torques, joint_names=None, 
                          save_dir="./test/simulation_results"):
    """Save all simulation data and plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save raw data
    np.savez(os.path.join(save_dir, 'simulation_data.npz'),
             time=time_vec, q_desired=q_des, q_actual=q_actual,
             tau_feedforward=tau_ff, tau_feedback=tau_fb, tau_total=tau_total,
             ee_position=ee_pos, ee_orientation=ee_rpy)
    
    # Save plots with proper directory handling
    plot_joint_tracking_individual(time_vec, q_des, q_actual, joint_names, save_dir)
    plot_joint_torques_individual(time_vec, tau_ff, tau_fb, tau_total, max_torques, joint_names, save_dir)
    plot_end_effector_cartesian(time_vec, ee_pos, ee_rpy, save_dir)
    
    print(f"All results saved to: {save_dir}/")

def main():
    try:
        controller = PinocchioFeedforwardController(URDF_PATH, IDENTIFIED_PARAMS_PATH)
        # controller = GroundTruthFeedforwardController(URDF_PATH_LEFT)
    except FileNotFoundError:
        return

    # PyBullet Setup
    p.connect(p.GUI)
    p.setTimeStep(TIME_STEP)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(*GRAVITY_VECTOR)
    p.loadURDF("plane.urdf")
    robot_start_pos = [0, 0, 0.5]
    robot_start_orn = p.getQuaternionFromEuler([0, 0, 0])
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
    print("Constrained robot base to world to prevent falling during test.")
    p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "robot_simulation.mp4")
    
    joint_indices = get_joint_indices_by_name(robot_id, robot_config.ACTUATED_JOINT_NAMES)
    print("\nDisabling PyBullet's default joint damping to isolate controller.")
    for idx in joint_indices:
        p.changeDynamics(robot_id, idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
        p.setJointMotorControl2(robot_id, idx, p.VELOCITY_CONTROL, force=0)

    # Generate trajectory with waypoints
    print(f"Generating a {SIM_DURATION}s smooth polynomial trajectory...")
    time_vec = np.arange(0, SIM_DURATION, TIME_STEP)

    waypoints = [
        np.zeros(NUM_JOINTS),  # Start position
        np.deg2rad([15] * NUM_JOINTS),  # Waypoint 1
        np.deg2rad([30] * NUM_JOINTS),  # Waypoint 2
        np.deg2rad([10] * NUM_JOINTS),  # Waypoint 3 
        np.deg2rad([-20] * NUM_JOINTS),  # Waypoint 4
        np.deg2rad([-10] * NUM_JOINTS),  # Waypoint 5
        np.zeros(NUM_JOINTS)  
    ]

    segment_durations = [100.0, 100.0, 150.0, 150.0, 100, 100]

    joint_trajectories = []
    for i in range(NUM_JOINTS):
        joint_trajectories.append([])

    current_time = 0.0
    for seg_idx in range(len(waypoints) - 1):
        q_start_seg = waypoints[seg_idx]
        q_end_seg = waypoints[seg_idx + 1]
        duration = segment_durations[seg_idx]
        
        segment_trajectories = generate_smooth_trajectory_for_all_joints(
            q_start_seg, q_end_seg, duration, NUM_JOINTS
        )
        
        for i in range(NUM_JOINTS):
            joint_trajectories[i].append((segment_trajectories[i], current_time, duration))
        
        current_time += duration

    q_des_traj, qd_des_traj, qdd_des_traj = [], [], []

    for t in time_vec:
        q_des, qd_des, qdd_des = np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS)
        for i in range(NUM_JOINTS):
            q_des[i], qd_des[i], qdd_des[i] = evaluate_waypoint_trajectory(joint_trajectories, t, i)
        q_des_traj.append(q_des)
        qd_des_traj.append(qd_des)
        qdd_des_traj.append(qdd_des)

    q_des_traj = np.array(q_des_traj)
    qd_des_traj = np.array(qd_des_traj)
    qdd_des_traj = np.array(qdd_des_traj)
    print("Trajectory generation complete.")

    initial_q = q_des_traj[0]
    for i, j_idx in enumerate(joint_indices):
        p.resetJointState(robot_id, j_idx, targetValue=initial_q[i], targetVelocity=0)

    # Data logging
    log_t, log_q_des, log_q_actual = [], [], []
    log_tau_ff, log_tau_fb, log_tau_total = [], [], []
    log_ee_pos, log_ee_rpy = [], []

    print("\n--- Starting Simulation ---")
    # Main Control Loop
    for k in range(len(time_vec)):
        q_des = q_des_traj[k]
        qd_des = qd_des_traj[k]
        qdd_des = qdd_des_traj[k]
        joint_states = p.getJointStates(robot_id, joint_indices)
        q_actual = np.array([state[0] for state in joint_states])
        qd_actual = np.array([state[1] for state in joint_states])

        # Compute Torques
        tau_ff = controller.compute_feedforward_torque(q_des, qd_des, qdd_des)
        
        # Feedback torque (PD controller)
        tau_fb = KP * (q_des - q_actual) + KD * (qd_des - qd_actual)
        tau_fb = np.clip(tau_fb, -MAX_TORQUES, MAX_TORQUES)
        
        tau_total = tau_ff + tau_fb
        tau_limited = np.clip(tau_total, -MAX_TORQUES, MAX_TORQUES)

        p.setJointMotorControlArray(
            robot_id, 
            joint_indices, 
            p.TORQUE_CONTROL, 
            forces=tau_limited
        )

        p.stepSimulation()
        # time.sleep(0.000001)

        # Compute end-effector pose using controller's forward kinematics
        try:
            ee_pose = controller.compute_end_effector_pose(q_actual)
            if isinstance(ee_pose, tuple) and len(ee_pose) == 2:
                ee_pos, ee_rot_matrix = ee_pose
                ee_rpy = rotation_matrix__to_rpy(ee_rot_matrix)
            else:
                # Fallback: use PyBullet to get end-effector pose
                ee_link_idx = len(joint_indices) - 1  # Assuming last joint is near end-effector
                ee_state = p.getLinkState(robot_id, ee_link_idx)
                ee_pos = np.array(ee_state[0])
                ee_orn_quat = ee_state[1]
                ee_rpy = np.array(p.getEulerFromQuaternion(ee_orn_quat))
        except:
            # Fallback: use PyBullet
            ee_link_idx = len(joint_indices) - 1
            ee_state = p.getLinkState(robot_id, ee_link_idx)
            ee_pos = np.array(ee_state[0])
            ee_orn_quat = ee_state[1]
            ee_rpy = np.array(p.getEulerFromQuaternion(ee_orn_quat))

        # Log data
        log_t.append(time_vec[k])
        log_q_des.append(q_des)
        log_q_actual.append(q_actual)
        log_tau_ff.append(tau_ff)
        log_tau_fb.append(tau_fb)
        log_tau_total.append(tau_limited)
        log_ee_pos.append(ee_pos)
        log_ee_rpy.append(ee_rpy)

    p.disconnect()
    print("--- Simulation Complete ---")

    # Convert to numpy arrays
    array_log_t = np.array(log_t)
    array_log_q_des = np.array(log_q_des)
    array_log_q_actual = np.array(log_q_actual)
    array_log_tau_ff = np.array(log_tau_ff)
    array_log_tau_fb = np.array(log_tau_fb)
    array_log_tau_total = np.array(log_tau_total)
    array_log_ee_pos = np.array(log_ee_pos)
    array_log_ee_rpy = np.array(log_ee_rpy)

    # Generate joint names (customize based on your robot)
    joint_names = [f'Joint {i+1}' for i in range(NUM_JOINTS)]
    
    # Plot results
    print("\n--- Generating Plots ---")
    # plot_joint_tracking_individual(array_log_t, array_log_q_des, array_log_q_actual, joint_names)
    # plot_joint_torques_individual(array_log_t, array_log_tau_ff, array_log_tau_fb, array_log_tau_total, MAX_TORQUES, joint_names)
    # plot_end_effector_cartesian(array_log_t, array_log_ee_pos, array_log_ee_rpy)
    save_simulation_results(array_log_t, array_log_q_des, array_log_q_actual,
                       array_log_tau_ff, array_log_tau_fb, array_log_tau_total,
                       array_log_ee_pos, array_log_ee_rpy, MAX_TORQUES, joint_names)

if __name__ == "__main__":
    main()