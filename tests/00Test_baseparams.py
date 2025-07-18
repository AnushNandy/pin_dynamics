import numpy as np
import os, time
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics 
from src.systemid.pinocchio_friction_regressor import smooth_sign, PinocchioAndFrictionRegressorBuilder
from config import robot_config 
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from pinocchio.robot_wrapper import RobotWrapper
from utils.trajectories import generate_chirp_trajectory, generate_aggressive_trajectory, generate_bang_bang_trajectory, generate_step_trajectory

# --- Config ---
SIM_DURATION = 30
# TIME_STEP = 1. / 240.
TIME_STEP = 0.02
URDF_PATH = robot_config.URDF_PATH
IDENTIFIED_PARAMS_PATH = "/home/robot/dev/dyn/src/systemid/identified_base_params.npz"
NUM_JOINTS = robot_config.NUM_JOINTS
MAX_TORQUES = robot_config.MAX_TORQUES
KP = np.array([100.0, 100.0, 200.0])
KD = np.array([2 * np.sqrt(k) for k in KP])

class PinocchioFeedforwardController:
    def __init__(self, urdf_path: str, identified_params_path: str):
        print("--- Initializing Robust Pinocchio Feedforward Controller ---")
        try:
            params_data = np.load(identified_params_path, allow_pickle=True)
            self.base_params = params_data['base_params']
            self.base_indices = params_data['base_indices']
        except FileNotFoundError:
            print(f"ERROR: Parameter file not found at {identified_params_path}")
            raise

        self.num_joints = robot_config.NUM_JOINTS
        self.regressor_builder = PinocchioAndFrictionRegressorBuilder(urdf_path)
        self.model_dynamics = PinocchioRobotDynamics(urdf_path)
        total_params = self.regressor_builder.total_params
        
        valid_indices = [idx for idx in self.base_indices if idx < total_params]
        invalid_indices = [idx for idx in self.base_indices if idx >= total_params]
        
        if invalid_indices:
            print(f"WARNING: Found {len(invalid_indices)} invalid parameter indices: {invalid_indices}")
            print(f"Total parameters in regressor: {total_params}")
            print(f"Filtering out invalid indices...")
            valid_mask = np.array([idx < total_params for idx in self.base_indices])
            self.base_indices = np.array(self.base_indices)[valid_mask]
            self.base_params = self.base_params[valid_mask]
        
        num_moving_bodies = self.model_dynamics.model.nbodies - 1
        total_link_params = num_moving_bodies * 10
        full_params_inertial = np.zeros(total_link_params)

        num_inertial_base_params = 0
        for i, idx in enumerate(self.base_indices):
            if idx < total_link_params:
                full_params_inertial[idx] = self.base_params[i]
                num_inertial_base_params += 1
        
        self.model_dynamics.set_parameters_from_vector(full_params_inertial)

        print(f"Loaded {len(self.base_params)} valid base parameters.")
        print(f" -> {num_inertial_base_params} inertial parameters")
        print(f" -> {len(self.base_params) - num_inertial_base_params} friction parameters")
        print(f"Valid base parameter indices: {self.base_indices}")
        print(f"Regressor has {total_params} total parameters")
        print("----------------------------------------------------------")

    def compute_feedforward_torque(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """
        Computes the feedforward torque using the base parameter formulation.
        τ = Y(q, qd, qdd) * P
        """
        # 1. Compute the full regressor matrix for the current state.
        Y_full = self.regressor_builder.compute_regressor_matrix(q, qd, qdd)

        # 2. Select only the columns corresponding to our valid base parameters.
        Y_base = Y_full[:, self.base_indices]

        # 3. Compute the final torque using the core identification equation.
        tau_ff = Y_base @ self.base_params
        
        return tau_ff
    
# --- Trajectory Generation Utilities ---
def generate_quintic_spline(q0,     qf, qd0, qdf, qdd0, qddf, T):
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

def generate_straight_line_trajectory(waypoints, durations):
    joint_trajectories = [[] for _ in range(len(waypoints[0]))]
    current_time = 0.0
    
    for seg_idx in range(len(waypoints) - 1):
        q0 = np.array(waypoints[seg_idx])
        qf = np.array(waypoints[seg_idx + 1])
        duration = durations[seg_idx]
        velocity = (qf - q0) / duration
        
        for j in range(len(waypoints[0])):
            def make_linear_traj(q_start, q_end, vel, dur):
                def linear_trajectory(t):
                    t = np.clip(t, 0, dur)
                    
                    # Add acceleration at the beginning and end of segments
                    accel_time = 0.1  # 100ms acceleration/deceleration
                    if t < accel_time:
                        # Accelerating from 0 to vel
                        qdd = vel / accel_time
                        qd = qdd * t
                        q = q_start + 0.5 * qdd * t**2
                    elif t > dur - accel_time:
                        # Decelerating from vel to 0
                        t_decel = t - (dur - accel_time)
                        qdd = -vel / accel_time
                        q = q_start + vel * (dur - accel_time) + vel * t_decel + 0.5 * qdd * t_decel**2
                        qd = vel + qdd * t_decel
                    else:
                        # Constant velocity
                        q = q_start + vel * t
                        qd = vel
                        qdd = 0.0
                    
                    return q, qd, qdd
                return linear_trajectory
            
            traj = make_linear_traj(q0[j], qf[j], velocity[j], duration)
            joint_trajectories[j].append((traj, current_time, duration))
        
        current_time += duration
    
    return joint_trajectories

def evaluate_trajectory(joint_trajectories, t, j):
    for traj, t0, dur in joint_trajectories[j]:
        if t0 <= t < t0 + dur:
            return traj(t - t0)
    traj, t0, dur = joint_trajectories[j][-1]
    return traj(dur)

def main():
    # --- 1. Initialization ---
    print("--- Initializing Controller and Visualizer ---")

    controller = PinocchioFeedforwardController(URDF_PATH, IDENTIFIED_PARAMS_PATH)

    robot_for_viz = RobotWrapper.BuildFromURDF(URDF_PATH, 
                                               package_dirs=[])
    
    model = robot_for_viz.model
    data = robot_for_viz.data
    
    # Verify that the models match
    ctrl_model = controller.model_dynamics.model
    print(f"\nVerification:")
    print(f"Controller Model -> nq={ctrl_model.nq}, nv={ctrl_model.nv}")
    print(f"Visualizer Model -> nq={model.nq}, nv={model.nv}")
    if ctrl_model.nq != model.nq or ctrl_model.nv != model.nv:
        raise RuntimeError("Model mismatch between controller and visualizer!")
    print("Models are consistent. Proceeding.\n")

    viz = MeshcatVisualizer(robot_for_viz.model, robot_for_viz.collision_model, robot_for_viz.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    print("Meshcat initialized. Open the URL from the console.")

    # --- 2. Trajectory Generation ---
    time_vec = np.arange(0, SIM_DURATION, TIME_STEP)
    TRAJECTORY_TYPE = "waypoint" 

    if TRAJECTORY_TYPE == "aggressive":
        trajectory_func = generate_aggressive_trajectory(time_vec, NUM_JOINTS)
    elif TRAJECTORY_TYPE == "step":
        trajectory_func = generate_step_trajectory(time_vec, NUM_JOINTS)
    elif TRAJECTORY_TYPE == "chirp":
        trajectory_func = generate_chirp_trajectory(time_vec, NUM_JOINTS)
    elif TRAJECTORY_TYPE == "bang_bang":
        trajectory_func = generate_bang_bang_trajectory(time_vec, NUM_JOINTS)
    elif TRAJECTORY_TYPE == "waypoint":
        # waypoints = [
        #     np.zeros(NUM_JOINTS),
        #     np.deg2rad([40] * NUM_JOINTS),
        #     np.deg2rad([60] * NUM_JOINTS),
        #     np.deg2rad([30] * NUM_JOINTS),
        #     np.deg2rad([-30] * NUM_JOINTS),
        #     np.deg2rad([-40] * NUM_JOINTS),
        #     np.zeros(NUM_JOINTS),
        # ]
        waypoints = [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.8],
            [0, 0, 0]
        ]
        durations = [10, 10, 10]
        # durations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        # trajectories = generate_segmented_trajectory(waypoints, durations)
        trajectories = generate_straight_line_trajectory(waypoints, durations)
        
        # Wrapper function to match the interface
        def trajectory_func(t):
            q_des = np.zeros(NUM_JOINTS)
            qd_des = np.zeros(NUM_JOINTS)
            qdd_des = np.zeros(NUM_JOINTS)
            for j in range(NUM_JOINTS):
                q_des[j], qd_des[j], qdd_des[j] = evaluate_trajectory(trajectories, t, j)
            return q_des, qd_des, qdd_des

    print(f"Using {TRAJECTORY_TYPE} trajectory")
    # trajectory_func = generate_aggressive_trajectory(time_vec, NUM_JOINTS)

    # --- 3. Simulation Loop ---
    # Prepare lists to log data for plotting
    log_q_des, log_q_actual, log_tau_ff, log_tau_fb, log_tau_cmd = [], [], [], [], []

    # q_actuated_start_idx = model.joints[1].nq
    q_actuated_start_idx = 0

    print("\n--- Starting Simulation Loop ---")
    for t in time_vec:
        # Get desired joint states from the trajectory at time t
        q_des = np.zeros(NUM_JOINTS)
        qd_des = np.zeros(NUM_JOINTS)
        qdd_des = np.zeros(NUM_JOINTS)
        # q_des, qd_des, qdd_des = trajectory_func(t)
        for j in range(NUM_JOINTS):
            q_des[j], qd_des[j], qdd_des[j] = evaluate_trajectory(trajectories, t, j)

        # In this ideal simulation, actual state perfectly tracks desired state
        # In a real robot, this would come from sensors.
        q_actual = q_des.copy()
        qd_actual = qd_des.copy()

        # Compute torques
        # The controller's internal dynamics model correctly handles the floating base
        tau_ff = controller.compute_feedforward_torque(q_actual, qd_actual, qdd_des)
        
        # Feedback torque would be zero in this ideal case, but we calculate for completeness
        tau_fb = KP * (q_des - q_actual) + KD * (qd_des - qd_actual)
        tau_cmd = tau_ff + tau_fb
        
        # 1. Start with the neutral configuration (base at origin, joints at zero).
        q_full_for_viz = pin.neutral(model)
        
        # 2. Place the actual joint angles into the correct slice of the full vector.
        q_full_for_viz[q_actuated_start_idx : q_actuated_start_idx + NUM_JOINTS] = q_actual
        
        # 3. Display this complete configuration.
        viz.display(q_full_for_viz)

        # Log data for plotting
        log_q_des.append(q_des)
        log_q_actual.append(q_actual)
        log_tau_ff.append(tau_ff)
        log_tau_fb.append(tau_fb)
        log_tau_cmd.append(tau_cmd)
        
        # Slow down for visualization
        time.sleep(TIME_STEP / 10.0)

        if int(t) % 10 == 0 and abs(t - int(t)) < TIME_STEP / 2:
            print(f"Simulation time: {int(t)}s / {int(SIM_DURATION)}s")

    # --- 4. Plotting Results ---
    print("--- Simulation Complete. Generating Plots. ---")
    log_q_des = np.array(log_q_des)
    log_q_actual = np.array(log_q_actual)
    log_tau_ff = np.array(log_tau_ff)
    log_tau_fb = np.array(log_tau_fb)
    log_tau_cmd = np.array(log_tau_cmd)

    fig, axs = plt.subplots(NUM_JOINTS, 1, figsize=(12, 3 * NUM_JOINTS), sharex=True)
    fig.suptitle("Torque Components", fontsize=18)
    for j in range(NUM_JOINTS):
        axs[j].plot(time_vec, log_tau_ff[:, j], 'r-', label='τ_ff (Feedforward)')
        axs[j].plot(time_vec, log_tau_fb[:, j], 'g--', label='τ_fb (Feedback)')
        axs[j].plot(time_vec, log_tau_cmd[:, j], 'b-.', label='τ_cmd (Total)')
        axs[j].set_ylabel(f'Joint {j} Torque (Nm)')
        axs[j].grid(True)
        axs[j].legend()
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.figure(figsize=(14, 4 * NUM_JOINTS))
    for i in range(NUM_JOINTS):
        plt.subplot(NUM_JOINTS, 2, 2 * i + 1)
        plt.plot(time_vec, np.rad2deg(log_q_des[:, i]), 'r--', label='Desired')
        plt.plot(time_vec, np.rad2deg(log_q_actual[:, i]), 'b-', label='Actual')
        plt.title(f'Joint {i} Position Tracking')
        plt.ylabel('Position [deg]')
        plt.legend()
        plt.grid(True)

        plt.subplot(NUM_JOINTS, 2, 2 * i + 2)
        error = np.rad2deg(log_q_des[:, i] - log_q_actual[:, i])
        plt.plot(time_vec, error, 'g')
        plt.title(f'Joint {i} Tracking Error')
        plt.ylabel('Error [deg]')
        plt.grid(True)

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()