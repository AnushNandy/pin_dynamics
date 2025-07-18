import numpy as np
import time, os, sys
from scipy.signal import butter, filtfilt
import pinocchio as pin
import matplotlib.pyplot as plt
from pinocchio.visualize import MeshcatVisualizer
from pinocchio.robot_wrapper import RobotWrapper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import robot_config
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics

URDF_PATH = robot_config.URDF_PATH
SAVE_PATH = "/home/robot/dev/dyn/src/systemid/sysid_data_pybullet.npz"
SIM_DURATION = 500.0 #150 500
TIME_STEP = 1. / 240.
NUM_JOINTS = robot_config.NUM_JOINTS
MAX_TORQUES = robot_config.MAX_TORQUES

joint_limits = [
    (-2.5, 2.5), (-2.3, 2.3), (-2.2, 2.2),
    (-2.5, 2.5),(-2.5, 2.5), (-2.5, 2.5),
    (-1, 1)
]

low_bounds = np.array([limit[0] for limit in joint_limits])
high_bounds = np.array([limit[1] for limit in joint_limits])

# --- PD Controller Gains ---
# KP = np.array([100.0, 100.0, 200.0, 500.0, 150.0, 150.0, 0.7])
# KD = np.array([2 * np.sqrt(k) for k in KP])
KP = np.array([100.0, 100.0, 200.0])
KD = np.array([2 * np.sqrt(k) for k in KP])

def setup_meshcat_visualization(urdf_path: str):
    """Initializes Meshcat visualization using Pinocchio's high-level MeshcatVisualizer."""
    robot = RobotWrapper.BuildFromURDF(urdf_path, package_dirs=[], root_joint=None)
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()

    print("Meshcat visualization initialized. Open http://127.0.0.1:7000 in your browser.")
    
    return robot.model, robot.data, viz

def generate_fourier_series_trajectory(t, num_harmonics = 8, base_freq = 0.15): # 5 , 0.5 
    np.random.seed(42)
    
    joint_amplitudes = np.array([
            1.2,   # Joint 2
            1.8,   # Joint 3
            0.5,   # Joint 4
        ])
    
    phi = np.array([
        0,        # Joint 1
        0.2*np.pi,# Joint 2
        0.8*np.pi,# Joint 3
    ])

    A = np.zeros((NUM_JOINTS, num_harmonics))
    for joint_idx in range(NUM_JOINTS):
        A[joint_idx, :] = np.random.uniform(-joint_amplitudes[joint_idx], 
                                          joint_amplitudes[joint_idx], 
                                          num_harmonics)
    PHI = np.zeros((NUM_JOINTS, num_harmonics))
    for joint_idx in range(NUM_JOINTS):
        PHI[joint_idx, :] = np.random.uniform(-phi[joint_idx], 
                                          phi[joint_idx], 
                                          num_harmonics)

    q_offset = np.zeros((NUM_JOINTS, num_harmonics))
    for joint_idx in range(NUM_JOINTS):
        q_offset[joint_idx, :] = np.random.uniform(-joint_amplitudes[joint_idx] * 0.8, 
                                                 joint_amplitudes[joint_idx] * 0.8, 
                                                 num_harmonics)
    q_des = np.zeros(NUM_JOINTS)
    qd_des = np.zeros(NUM_JOINTS)
    qdd_des = np.zeros(NUM_JOINTS)

    for i in range(NUM_JOINTS):
        for n in range(num_harmonics):
            w = base_freq * (n + 1)
            angle = w * t + PHI[i, n]
            q_des[i] += A[i, n] * np.sin(angle) + q_offset[i, n]/num_harmonics
            qd_des[i] += w * A[i, n] * np.cos(angle)
            qdd_des[i] += -w**2 * A[i, n] * np.sin(angle)

    # center_of_range = (high_bounds + low_bounds) / 2.0
    # q_des += center_of_range
    
    # q_des = np.clip(q_des, low_bounds, high_bounds)
    
    return q_des, qd_des, qdd_des

def main():
    """
    Generates and saves dynamically-consistent system identification data
    using Meshcat visualization with Pinocchio.
    """
    print("--- Generating System Identification Data using Pinocchio & Meshcat ---")

    # 1. Initialize Meshcat visualization and the Pinocchio dynamics model
    model, data, vis_pin = setup_meshcat_visualization(URDF_PATH)
    dynamics_model = PinocchioRobotDynamics(URDF_PATH)

    print(f"Total configuration dimension (model.nq): {model.nq}")
    print(f"Total velocity dimension (model.nv): {model.nv}")
    print(f"Number of joints in model: {model.njoints}")
    print("------------------------------------")

    print("Joint Names and Indices:")
    actuated_joint_names = robot_config.ACTUATED_JOINT_NAMES
    actuated_joint_ids = [model.getJointId(name) for name in actuated_joint_names]

    actuated_joint_info = []
    for i, joint_id in enumerate(actuated_joint_ids):
        joint = model.joints[joint_id]
        joint_name = model.names[joint_id]
        q_idx = joint.idx_q
        nq = joint.nq 
        nv = joint.nv 
        
        actuated_joint_info.append({
            'id': joint_id,
            'name': joint_name,
            'q_idx': q_idx,
            'nq': nq,
            'nv': nv
        })
        
        print(f"Joint {joint_id} ({joint_name}) -> q_idx: {q_idx}, nq: {nq}, nv: {nv}")
    
    print(f"Model nq: {model.nq}, nv: {model.nv}")

    # Initialize states
    q_actual = np.zeros(NUM_JOINTS)
    qd_actual = np.zeros(NUM_JOINTS)
    
    log_q, log_qd, log_qdd, log_tau = [], [], [], []
    log_tau_ff, log_tau_fb, log_tau_cmd, log_tau_limited = [], [], [], []

    print("Starting closed-loop simulation for data generation...")
    for t in np.arange(0, SIM_DURATION, TIME_STEP):
        # --- I. Trajectory Generation (for actuated joints) ---
        q_des, qd_des, qdd_des = generate_fourier_series_trajectory(t)

        # --- II. Simulate Robot Dynamics ---
        q_actual = q_des.copy()
        qd_actual = qd_des.copy()

        # --- III. Controller Calculation ---
        tau_ff = dynamics_model.compute_rnea(q_actual, qd_actual, qdd_des)
        pos_error = q_des - q_actual
        vel_error = qd_des - qd_actual
        tau_fb = KP * pos_error + KD * vel_error
        tau_cmd = tau_ff + tau_fb
        
        log_tau_ff.append(tau_ff)
        log_tau_fb.append(tau_fb)
        log_tau_cmd.append(tau_cmd)

        tau_limited = np.clip(tau_cmd, -MAX_TORQUES, MAX_TORQUES)
        log_tau_limited.append(tau_limited)

        # --- IV. VISUALIZATION ---
        # For joints with nq=2, nv=1, we need to use pin.integrate to properly
        # map from velocity space to configuration space

        q_full = pin.neutral(model)
        
        # Velocity vector - each actuated joint has nv=1
        v_full = np.zeros(model.nv)
        v_full[:NUM_JOINTS] = q_actual  
        
        # Use Pinocchio's integrate function to properly map velocities to configuration
        q_full = pin.integrate(model, q_full, v_full)
 
        if int(t * 1000) % 5000 == 0: 
            print(f"t={t:.2f}: q_actual[0]={q_actual[0]:.3f}, q_full[0:14]={q_full[:14]}")

        vis_pin.display(q_full)

        if int(t * 1000) % 10 == 0:  
            time.sleep(0.0001)

        # --- V. Log Data ---
        log_q.append(q_actual)
        log_qd.append(qd_actual)
        log_tau.append(tau_limited)

        if int(t) % 5 == 0 and abs(t - int(t)) < TIME_STEP / 2:
            print(f"Generated data up to {int(t)}s / {int(SIM_DURATION)}s")
            print(f"Joint_0 pos: {q_actual[0]:.3f}")

    # --- VI. Post-Processing and Saving ---
    log_q = np.array(log_q)
    log_qd = np.array(log_qd)
    log_tau = np.array(log_tau)

    log_tau_ff = np.array(log_tau_ff)
    log_tau_fb = np.array(log_tau_fb)
    log_tau_cmd = np.array(log_tau_cmd)
    log_tau_limited = np.array(log_tau_limited)
    time2 = np.arange(0, len(log_tau_ff)) * TIME_STEP
    num_joints = log_tau_ff.shape[1]

    fig, axs = plt.subplots(num_joints, 1, figsize=(14, 3 * num_joints), sharex=True)
    fig.suptitle("Torque Components (Feedforward, Feedback, Commanded)", fontsize=18)

    for j in range(num_joints):
        axs[j].plot(time2, log_tau_ff[:, j], label='τ_ff (RNEA)', linestyle='-')
        axs[j].plot(time2, log_tau_fb[:, j], label='τ_fb (PD)', linestyle='--')
        axs[j].plot(time2, log_tau_cmd[:, j], label='τ_cmd (Total)', linestyle='-.')
        axs[j].set_ylabel(f'Joint {j+1} Torque [Nm]')
        axs[j].grid(True)
        axs[j].legend(loc='upper right', fontsize=10)

    axs[-1].set_xlabel("Time [s]")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
    fs = 1. / TIME_STEP
    cutoff_freq = 15.0  
    nyquist_freq = 0.5 * fs
    order = 4
    b, a = butter(order, cutoff_freq / nyquist_freq, btype='low', analog=False)
    
    log_qdd = np.gradient(log_qd, axis=0) / TIME_STEP
    qdd_filtered = np.apply_along_axis(lambda x: filtfilt(b, a, x, padlen=150), 0, log_qdd)

    np.savez(
        SAVE_PATH,
        q=log_q,
        qd=log_qd,
        qdd=qdd_filtered,
        tau=log_tau
    )
    print(f"--- Data generation complete. ---")
    print(f"Saved {len(log_q)} dynamically consistent samples to '{SAVE_PATH}'")

if __name__ == "__main__":
    main()