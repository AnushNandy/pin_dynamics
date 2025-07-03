import pybullet as p
import pybullet_data
import numpy as np
import os, time
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics 
from src.systemid.pinocchio_friction_regressor import smooth_sign
from config import robot_config 
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from pinocchio.robot_wrapper import RobotWrapper

# --- Config ---
SIM_DURATION = 50.0
TIME_STEP = 1. / 240.
URDF_PATH = robot_config.URDF_PATH
IDENTIFIED_PARAMS_PATH = "/home/robot/dev/dyn/src/systemid/identified_params_pybullet.npz"
NUM_JOINTS = robot_config.NUM_JOINTS
MAX_TORQUES = np.array([140, 140, 51, 51, 14, 14, 7.7])
KP = np.array([100.0, 100.0, 200.0, 500.0, 150.0, 150.0, 0.7])
KD = np.array([2 * np.sqrt(k) for k in KP])

# --- Controller Class ---
class PinocchioFeedforwardController:
    def __init__(self, urdf_path: str, identified_params_path: str):
        print("--- Initializing Pinocchio Feedforward Controller ---")
        self.model_dynamics = PinocchioRobotDynamics(urdf_path)
        self.num_joints = self.model_dynamics.num_actuated_joints
        params = np.load(identified_params_path)
        rnea_params = params['P'][:(self.model_dynamics.model.nbodies - 1) * 10]
        self.model_dynamics.set_parameters_from_vector(rnea_params)
        print("-----------------------------------------------------")

    def compute_feedforward_torque(self, q, qd, qdd):
        return self.model_dynamics.compute_rnea(q, qd, qdd)

# --- Trajectory Generation Utilities ---
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
        t_vec_d = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
        t_vec_dd = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3])
        return coeffs @ t_vec, coeffs @ t_vec_d, coeffs @ t_vec_dd

    return trajectory

def generate_segmented_trajectory(waypoints, durations):
    joint_trajectories = [[] for _ in range(NUM_JOINTS)]
    current_time = 0.0
    for seg_idx in range(len(waypoints) - 1):
        q0 = waypoints[seg_idx]
        qf = waypoints[seg_idx + 1]
        duration = durations[seg_idx]
        for j in range(NUM_JOINTS):
            traj = generate_quintic_spline(q0[j], qf[j], 0, 0, 0, 0, duration)
            joint_trajectories[j].append((traj, current_time, duration))
        current_time += duration
    return joint_trajectories

def evaluate_trajectory(joint_trajectories, t, j):
    for traj, t0, dur in joint_trajectories[j]:
        if t0 <= t < t0 + dur:
            return traj(t - t0)
    traj, t0, dur = joint_trajectories[j][-1]
    return traj(dur)

# --- Main Simulation ---
def main():
    controller = PinocchioFeedforwardController(URDF_PATH, IDENTIFIED_PARAMS_PATH)
    robot = RobotWrapper.BuildFromURDF(URDF_PATH, package_dirs=[], root_joint=None)
    model = robot.model
    data = robot.data
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    print("Meshcat initialized at http://127.0.0.1:7000")

    # --- Print joint information like in the main function ---
    print(f"Total configuration dimension (model.nq): {model.nq}")
    print(f"Total velocity dimension (model.nv): {model.nv}")
    print(f"Number of joints in model: {model.njoints}")
    print("------------------------------------")

    print("Joint Names and Indices:")
    actuated_joint_names = robot_config.ACTUATED_JOINT_NAMES
    actuated_joint_ids = [model.getJointId(name) for name in actuated_joint_names]
    
    # Get detailed joint information
    actuated_joint_info = []
    for i, joint_id in enumerate(actuated_joint_ids):
        joint = model.joints[joint_id]
        joint_name = model.names[joint_id]
        q_idx = joint.idx_q
        nq = joint.nq  # Number of configuration variables for this joint
        nv = joint.nv  # Number of velocity variables for this joint
        
        actuated_joint_info.append({
            'id': joint_id,
            'name': joint_name,
            'q_idx': q_idx,
            'nq': nq,
            'nv': nv
        })
        
        print(f"Joint {joint_id} ({joint_name}) -> q_idx: {q_idx}, nq: {nq}, nv: {nv}")
    
    print(f"Model nq: {model.nq}, nv: {model.nv}")

    time_vec = np.arange(0, SIM_DURATION, TIME_STEP)
    waypoints = [
        np.zeros(NUM_JOINTS),
        np.deg2rad([15] * NUM_JOINTS),
        np.deg2rad([30] * NUM_JOINTS),
        np.deg2rad([10] * NUM_JOINTS),
        np.deg2rad([-20] * NUM_JOINTS),
        np.deg2rad([-10] * NUM_JOINTS),
        np.zeros(NUM_JOINTS),
    ]
    durations = [10.0, 10.0, 5.0, 5.0, 10.0, 10.0]
    trajectories = generate_segmented_trajectory(waypoints, durations)

    log_q_des, log_q_actual, log_tau_ff, log_tau_fb, log_tau_cmd = [], [], [], [], []

    for t in time_vec:
        q_des = np.zeros(NUM_JOINTS)
        qd_des = np.zeros(NUM_JOINTS)
        qdd_des = np.zeros(NUM_JOINTS)
        for j in range(NUM_JOINTS):
            q_des[j], qd_des[j], qdd_des[j] = evaluate_trajectory(trajectories, t, j)

        q_actual = q_des.copy()
        qd_actual = qd_des.copy()

        tau_ff = controller.compute_feedforward_torque(q_actual, qd_actual, qdd_des)
        tau_fb = KP * (q_des - q_actual) + KD * (qd_des - qd_actual)
        tau_cmd = tau_ff + tau_fb

        # --- IMPROVED VISUALIZATION (like in main function) ---
        # For joints with nq=2, nv=1, we need to use pin.integrate to properly
        # map from velocity space to configuration space
        
        # Start with neutral configuration
        q_full = pin.neutral(model)
        
        # Create velocity vector - each actuated joint has nv=1
        v_full = np.zeros(model.nv)
        v_full[:NUM_JOINTS] = q_actual  # Assuming actuated joints come first in velocity space
        
        # Use Pinocchio's integrate function to properly map velocities to configuration
        q_full = pin.integrate(model, q_full, v_full)
        
        # Debug: Print some values occasionally
        if int(t * 1000) % 5000 == 0:  # Every 5 seconds
            print(f"t={t:.2f}: q_actual[0]={q_actual[0]:.3f}, q_full[0:14]={q_full[:14]}")
        
        # Update visualization
        viz.display(q_full)

        log_q_des.append(q_des)
        log_q_actual.append(q_actual)
        log_tau_ff.append(tau_ff)
        log_tau_fb.append(tau_fb)
        log_tau_cmd.append(tau_cmd)

        # Add a small delay to make visualization smoother
        if int(t * 1000) % 10 == 0:
            time.sleep(0.0001)

        if int(t) % 5 == 0 and abs(t - int(t)) < TIME_STEP / 2:
            print(f"Simulation time: {int(t)}s / {int(SIM_DURATION)}s")
            print(f"Joint_0 pos: {q_actual[0]:.3f}")

    # --- Plot ---
    log_q_des = np.array(log_q_des)
    log_q_actual = np.array(log_q_actual)
    log_tau_ff = np.array(log_tau_ff)
    log_tau_fb = np.array(log_tau_fb)
    log_tau_cmd = np.array(log_tau_cmd)

    fig, axs = plt.subplots(NUM_JOINTS, 1, figsize=(12, 3 * NUM_JOINTS), sharex=True)
    fig.suptitle("Torque Components", fontsize=18)
    for j in range(NUM_JOINTS):
        axs[j].plot(time_vec, log_tau_ff[:, j], 'r-', label='τ_ff')
        axs[j].plot(time_vec, log_tau_fb[:, j], 'g--', label='τ_fb')
        axs[j].plot(time_vec, log_tau_cmd[:, j], 'b-.', label='τ_cmd')
        axs[j].set_ylabel(f'Joint {j+1} Torque (Nm)')
        axs[j].grid(True)
        axs[j].legend()
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    plt.figure(figsize=(14, 3 * NUM_JOINTS))
    for i in range(NUM_JOINTS):
        plt.subplot(NUM_JOINTS, 2, 2 * i + 1)
        plt.plot(time_vec, np.rad2deg(log_q_des[:, i]), 'r--', label='Desired')
        plt.plot(time_vec, np.rad2deg(log_q_actual[:, i]), 'b-', label='Actual')
        plt.title(f'Joint {i+1} Position Tracking')
        plt.ylabel('Position [deg]')
        plt.legend()
        plt.grid(True)

        plt.subplot(NUM_JOINTS, 2, 2 * i + 2)
        error = np.rad2deg(log_q_des[:, i] - log_q_actual[:, i])
        plt.plot(time_vec, error, 'g')
        plt.title(f'Joint {i+1} Tracking Error')
        plt.ylabel('Error [deg]')
        plt.grid(True)

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()