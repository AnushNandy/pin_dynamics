import pybullet as p
import pybullet_data
import numpy as np
import os
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
IDENTIFIED_PARAMS_PATH = "/home/robot/dev/dyn/src/systemid/identified_params.npz"

KP = np.array([100.0, 100.0, 200.0, 600.0, 100.0, 100.0, 10.0])
# KP = np.array([600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0])
KD = np.array([2 * np.sqrt(k) for k in KP]) 
# KD[6] *=10 

# Physics and Robot Constants
GRAVITY_VECTOR = np.array([0, 0, -9.81])
# MAX_TORQUES = np.array([140, 140, 51, 51, 14, 14, 7.7])
MAX_TORQUES = np.array([140, 140, 100, 100, 100, 51, 11])
NUM_JOINTS = robot_config.NUM_JOINTS

class GroundTruthFeedforwardController:
    def __init__(self, urdf_path: str):
        print("--- Initializing GROUND TRUTH Feedforward Controller ---")
        self.model_dynamics = PinocchioRobotDynamics(urdf_path)
        self.num_joints = self.model_dynamics.num_actuated_joints
        # self.true_friction_coeffs = {
        #     'viscous': [0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06],
        #     'coulomb': [0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
        # }
        print("Using known ground-truth parameters for RNEA and Friction.")
        print("-----------------------------------------------------")

    def compute_feedforward_torque(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """
        Computes the feedforward torque using KNOWN parameters.
        τ_ff = RNEA_truth(q, qd, qdd) + τ_friction_truth(qd)
        """
        # 1. Compute RNEA using the default URDF inertial parameters.
        tau_rnea = self.model_dynamics.compute_rnea(q, qd, qdd, gravity=GRAVITY_VECTOR)
        
        # 2. Compute the known ground-truth friction torque.
        # tau_friction = np.zeros(self.num_joints)
        # for i in range(self.num_joints):
        #     fv_true = self.true_friction_coeffs['viscous'][i]
        #     fc_true = self.true_friction_coeffs['coulomb'][i]
        #     tau_friction[i] = fv_true * qd[i] + fc_true * smooth_sign(qd[i])
            
        # return tau_rnea + tau_friction
        return tau_rnea
    

# --- Feedforward Controller ---
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
            # self.friction_params = identified_params[num_rnea_params:]
            
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
        
        # tau_friction = np.zeros(self.num_joints)
        # for i in range(self.num_joints):
        #     fv_hat = self.friction_params[i * 2]
        #     fc_hat = self.friction_params[i * 2 + 1]
        #     tau_friction[i] = fv_hat * qd[i] + fc_hat * smooth_sign(qd[i])
        #     tau_friction = np.zeros(self.num_joints)
            
        # return tau_rnea + tau_friction
        return tau_rnea
    
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
    
    # If beyond all segments, return the last position
    last_traj, last_start, last_duration = joint_trajectories[joint_idx][-1]
    return last_traj(last_duration)

def main():
    try:
        controller = PinocchioFeedforwardController(URDF_PATH, IDENTIFIED_PARAMS_PATH)
        # controller = GroundTruthFeedforwardController(URDF_PATH_LEFT)
    except FileNotFoundError:
        return

    # 1. --- PyBullet Setup ---
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
    
    joint_indices = get_joint_indices_by_name(robot_id, robot_config.ACTUATED_JOINT_NAMES)
    print("\nDisabling PyBullet's default joint damping to isolate controller.")
    for idx in joint_indices:
        p.changeDynamics(robot_id, idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
        p.setJointMotorControl2(robot_id, idx, p.VELOCITY_CONTROL, force=0)

    # 3. --- Generate The ENTIRE Trajectory Upfront ---
    print(f"Generating a {SIM_DURATION}s smooth polynomial trajectory...")
    time_vec = np.arange(0, SIM_DURATION, TIME_STEP)
    # q_start = np.zeros(NUM_JOINTS)
    # q_end = np.deg2rad([30] * NUM_JOINTS) 
    # # q_end = np.deg2rad([0, 0, 0, 0, 0, 0, 30])
    # joint_trajectories = generate_smooth_trajectory_for_all_joints(q_start, q_end, SIM_DURATION, NUM_JOINTS)

    ##---WAYPOINTS---------
    waypoints = [
        np.zeros(NUM_JOINTS),  # Start position
        np.deg2rad([15] * NUM_JOINTS),  # Waypoint 1
        np.deg2rad([30] * NUM_JOINTS),  # Waypoint 2
        np.deg2rad([10] * NUM_JOINTS),  # Waypoint 3 
        np.deg2rad([-20] * NUM_JOINTS),  # Waypoint 4
        np.deg2rad([-10] * NUM_JOINTS),  # Waypoint 5
        np.zeros(NUM_JOINTS)  
    ]

    # Time for each segments
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
    # for t in time_vec:
    #     q_des, qd_des, qdd_des = np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS)
    #     for i in range(NUM_JOINTS):
    #         q_des[i], qd_des[i], qdd_des[i] = joint_trajectories[i](t)
    #     q_des_traj.append(q_des)
    #     qd_des_traj.append(qd_des)
    #     qdd_des_traj.append(qdd_des)

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

    log_t, log_q_des, log_q_actual = [], [], []
    log_tau_ff, log_tau_fb, log_tau_total = [], [], []

    print("\n--- Starting Corrected Simulation ---")
    # 4. --- Main Control Loop ---
    for k in range(len(time_vec)):
        q_des = q_des_traj[k]
        qd_des = qd_des_traj[k]
        qdd_des = qdd_des_traj[k]
        joint_states = p.getJointStates(robot_id, joint_indices)
        q_actual = np.array([state[0] for state in joint_states])
        qd_actual = np.array([state[1] for state in joint_states])

        # --- Compute Torques ---
        tau_ff = controller.compute_feedforward_torque(q_des, qd_des, qdd_des)
        
        # Feedback torque (PD controller) to correct for any errors
        tau_fb = KP * (q_des - q_actual) + KD * (qd_des - qd_actual)
        tau_fb = np.clip(tau_fb, -MAX_TORQUES, MAX_TORQUES)
        
        tau_total = tau_ff + tau_fb
        # tau_total = tau_ff
        
        tau_limited = np.clip(tau_total, -MAX_TORQUES, MAX_TORQUES)

        p.setJointMotorControlArray(
            robot_id, 
            joint_indices, 
            p.TORQUE_CONTROL, 
            forces=tau_limited
        )

        p.stepSimulation()

        # Log data for plotting
        log_t.append(time_vec[k])
        log_q_des.append(q_des)
        log_q_actual.append(q_actual)
        log_tau_ff.append(tau_ff)
        log_tau_fb.append(tau_fb)
        log_tau_total.append(tau_limited)
    

    p.disconnect()
    print("--- Simulation Complete ---")

    # --- Plotting Results ---
    array_log_t = np.array(log_t)
    array_log_q_des = np.array(log_q_des)
    array_log_q_actual = np.array(log_q_actual)

    plt.figure(figsize=(15, 3 * NUM_JOINTS))
    for i in range(NUM_JOINTS):
        plt.subplot(NUM_JOINTS, 2, 2 * i + 1)
        plt.plot(array_log_t, np.rad2deg(array_log_q_des[:, i]), 'r--', label='Desired')
        plt.plot(array_log_t, np.rad2deg(array_log_q_actual[:, i]), 'b-', label='Actual')
        plt.title(f'Joint {i} Tracking'), plt.ylabel('Pos (deg)')
        plt.legend(), plt.grid(True)

        plt.subplot(NUM_JOINTS, 2, 2 * i + 2)
        err = np.rad2deg(array_log_q_des[:, i] - array_log_q_actual[:, i])
        plt.plot(array_log_t, err, 'g-')
        plt.title(f'Joint {i} Tracking Error'), plt.ylabel('Error (deg)')
        plt.grid(True)
    plt.tight_layout()

    # --- MODIFIED: New plotting section for torques ---
    array_log_tau_ff = np.array(log_tau_ff)
    array_log_tau_fb = np.array(log_tau_fb)
    array_log_tau_total = np.array(log_tau_total)

    plt.figure(figsize=(12, 3 * NUM_JOINTS))
    for i in range(NUM_JOINTS):
        plt.subplot(NUM_JOINTS, 1, i + 1)
        plt.plot(array_log_t, array_log_tau_ff[:, i], 'r:', label='Feedforward τ')
        plt.plot(array_log_t, array_log_tau_fb[:, i], 'g:', label='Feedback τ')
        plt.plot(array_log_t, array_log_tau_total[:, i], 'b-', label='Total Applied τ')
        plt.axhline(y=MAX_TORQUES[i], color='k', linestyle='--', label=f'Limit ({MAX_TORQUES[i]} Nm)')
        plt.axhline(y=-MAX_TORQUES[i], color='k', linestyle='--')
        plt.title(f'Joint {i} Torques'), plt.ylabel('Torque (Nm)'), plt.legend(), plt.grid(True)
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()