import pybullet as p
import pybullet_data
import numpy as np
import time
import os, sys
import matplotlib.pyplot as plt
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics
from config import robot_config

# --- Constants ---
IDENTIFIED_PARAMS_PATH = "/home/robot/dev/dyn/src/systemid/identified_params.npz"
URDF_PATH = robot_config.URDF_PATH
NUM_JOINTS = robot_config.NUM_JOINTS
TIME_STEP = 1. / 240.

# PD Control gains - much more conservative to prevent instability
KP = np.array([600.0, 120.0, 100.0, 70.0, 40.0, 30.0, 20.0])
KD = np.array([150.0, 20.0, 18.0, 12.0, 7.0, 5.0, 3.0])

def setup_robot():
    """Setup PyBullet with properly configured robot."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(TIME_STEP)
    
    # Load plane and robot
    p.loadURDF("plane.urdf")
    robot_start_pos = [0, 0, 0.5]
    robot_id = p.loadURDF(URDF_PATH, robot_start_pos, useFixedBase=True)
    
    # Get joint indices
    joint_indices = []
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[2] == p.JOINT_REVOLUTE:
            joint_indices.append(i)
    
    print(f"Found {len(joint_indices)} joints: {joint_indices}")
    
    # CRITICAL: Properly disable default motor control
    for joint_idx in joint_indices:
        p.setJointMotorControl2(
            robot_id,
            joint_idx,
            p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=0
        )
    
    return robot_id, joint_indices

def smooth_derivative(values, dt, window_size=5):
    """Compute smoothed derivative using moving average."""
    if len(values) < 2:
        return 0.0
    
    # Use last few values for smoothing
    recent_values = list(values)[-window_size:]
    if len(recent_values) < 2:
        return (recent_values[-1] - recent_values[0]) / (dt * (len(recent_values) - 1))
    
    # Simple backward difference with smoothing
    return (recent_values[-1] - recent_values[-2]) / dt

def pd_control_with_gravity(q_current, qd_current, q_target, dynamics_model, gravity_mult=0.5):
    """PD controller with optional gravity compensation."""
    # Position error
    e_pos = q_target - q_current
    
    # Velocity error (target velocity is 0)
    e_vel = 0.0 - qd_current
    
    # PD control
    tau_pd = KP * e_pos + KD * e_vel
    
    # Try gravity compensation - but be careful!
    try:
        tau_gravity = dynamics_model.compute_rnea(
            q_current, 
            np.zeros_like(q_current), 
            np.zeros_like(q_current)
        )
        # Limit gravity compensation to prevent instability
        tau_gravity = np.clip(tau_gravity, -50.0, 50.0)
    except:
        print("Warning: Gravity compensation failed, using PD only")
        tau_gravity = np.zeros_like(q_current)
    
    # Use adjustable gravity compensation
    return tau_pd + gravity_mult * tau_gravity

def collect_data_interactive():
    """Interactive data collection with improved PD control."""
    print("\n=== IMPROVED INTERACTIVE MODE ===")
    print("1. Click and drag robot links to move them")
    print("2. Robot should now hold positions much better!")
    print("3. Press SPACE to start/stop recording")
    print("4. Press 'q' to quit")
    
    robot_id, joint_indices = setup_robot()
    
    # Load dynamics model
    dynamics_model = PinocchioRobotDynamics(URDF_PATH)
    if os.path.exists(IDENTIFIED_PARAMS_PATH):
        identified_params = np.load(IDENTIFIED_PARAMS_PATH)['P']
        rnea_params = identified_params[:dynamics_model.num_actuated_joints * 10]
        dynamics_model.set_parameters_from_vector(rnea_params)
        print("Loaded identified parameters")
    else:
        print("Using default parameters")
    
    # Initialize
    joint_states = p.getJointStates(robot_id, joint_indices)
    q_target = np.array([state[0] for state in joint_states])
    
    # Data storage
    data = {
        'time': [],
        'q': [],
        'qd': [],
        'qdd': [],
        'tau_control': [],
        'tau_predicted': []
    }
    
    # Smoothing buffers for derivative calculation
    q_history = [deque(maxlen=10) for _ in range(NUM_JOINTS)]
    qd_history = [deque(maxlen=5) for _ in range(NUM_JOINTS)]
    
    recording = False
    t = 0
    last_manual_update = 0
    
    # UI controls
    record_button = p.addUserDebugParameter("Recording", 0, 1, 0)
    stiffness_slider = p.addUserDebugParameter("Stiffness", 0.1, 3.0, 1.0)
    damping_slider = p.addUserDebugParameter("Damping", 0.1, 3.0, 1.0)
    gravity_comp_slider = p.addUserDebugParameter("Gravity Compensation", 0.0, 1.0, 0.5)
    
    print("Controls ready! Drag the robot around - it should hold positions now.")
    
    try:
        while True:
            # Update gains from sliders
            stiffness_mult = p.readUserDebugParameter(stiffness_slider)
            damping_mult = p.readUserDebugParameter(damping_slider)
            gravity_mult = p.readUserDebugParameter(gravity_comp_slider)
            
            # Update global gains for the control function
            global KP, KD
            KP = np.array([100.0, 80.0, 60.0, 40.0, 20.0, 15.0, 10.0]) * stiffness_mult
            KD = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.5, 1.0]) * damping_mult
            
            recording = bool(p.readUserDebugParameter(record_button))
            
            # Get current state
            joint_states = p.getJointStates(robot_id, joint_indices)
            q = np.array([state[0] for state in joint_states])
            qd = np.array([state[1] for state in joint_states])
            
            # Update history for smoothing
            for i in range(NUM_JOINTS):
                q_history[i].append(q[i])
                qd_history[i].append(qd[i])
            
            # Detect if robot is being manually moved (large position change)
            if len(q_history[0]) > 1:
                q_change = np.linalg.norm(q - np.array([list(q_history[i])[-2] for i in range(NUM_JOINTS)]))
                large_velocity = np.linalg.norm(qd) > 0.5
                
                # Update target if being moved manually
                if q_change > 0.02 or large_velocity:
                    q_target = q.copy()
                    last_manual_update = t
            
            # Apply improved PD control with limited torques
            tau_control = pd_control_with_gravity(q, qd, q_target, dynamics_model, gravity_mult)
            
            # SAFETY: Limit torques to prevent instability
            tau_max = np.array([100.0, 80.0, 60.0, 40.0, 20.0, 15.0, 10.0])
            tau_control = np.clip(tau_control, -tau_max, tau_max)
            
            # Apply torques
            p.setJointMotorControlArray(
                robot_id,
                joint_indices,
                p.TORQUE_CONTROL,
                forces=tau_control
            )
            
            # Calculate smoothed acceleration
            qdd = np.zeros(NUM_JOINTS)
            if len(qd_history[0]) >= 2:
                for i in range(NUM_JOINTS):
                    qdd[i] = smooth_derivative(qd_history[i], TIME_STEP)
            
            # Record data if enabled and settled
            if recording and t > 0.5 and (t - last_manual_update) > 0.2:
                # Predict torques using dynamics model
                tau_predicted = dynamics_model.compute_rnea(q, qd, qdd)
                
                # Store data
                data['time'].append(t)
                data['q'].append(q.copy())
                data['qd'].append(qd.copy())
                data['qdd'].append(qdd.copy())
                data['tau_control'].append(tau_control.copy())
                data['tau_predicted'].append(tau_predicted.copy())
                
                if len(data['time']) % 120 == 0:
                    print(f"Recording... {len(data['time'])} samples")
            
            # Check for quit
            keys = p.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] == p.KEY_WAS_TRIGGERED:
                break
            if ord(' ') in keys and keys[ord(' ')] == p.KEY_WAS_TRIGGERED:
                recording = not recording
                print(f"Recording {'ON' if recording else 'OFF'}")
                
            p.stepSimulation()
            time.sleep(TIME_STEP)
            t += TIME_STEP
            
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    p.disconnect()
    return data

def plot_results(data):
    """Plot the collected dynamics comparison."""
    if not data['time']:
        print("No data collected!")
        return
    
    print(f"Plotting results from {len(data['time'])} samples...")
    
    time_arr = np.array(data['time'])
    tau_control = np.array(data['tau_control'])
    tau_predicted = np.array(data['tau_predicted'])
    q_arr = np.array(data['q'])
    qd_arr = np.array(data['qd'])
    qdd_arr = np.array(data['qdd'])
    
    # Create plots
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    
    # Plot joint positions
    axes[0].set_title("Joint Positions")
    for i in range(min(NUM_JOINTS, 6)):
        axes[0].plot(time_arr, q_arr[:, i], label=f'Joint {i+1}', linewidth=1.5)
    axes[0].set_ylabel("Position (rad)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot joint velocities
    axes[1].set_title("Joint Velocities")
    for i in range(min(NUM_JOINTS, 6)):
        axes[1].plot(time_arr, qd_arr[:, i], label=f'Joint {i+1}', linewidth=1.5)
    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot accelerations (smoothed)
    axes[2].set_title("Joint Accelerations (Smoothed)")
    for i in range(min(NUM_JOINTS, 4)):
        axes[2].plot(time_arr, qdd_arr[:, i], label=f'Joint {i+1}', alpha=0.8)
    axes[2].set_ylabel("Acceleration (rad/s²)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Torque comparison - show difference
    axes[3].set_title("Control vs Predicted Torques")
    for i in range(min(NUM_JOINTS, 3)):
        axes[3].plot(time_arr, tau_control[:, i], '--', 
                    label=f'Control J{i+1}', alpha=0.7, linewidth=2)
        axes[3].plot(time_arr, tau_predicted[:, i], '-', 
                    label=f'Predicted J{i+1}', alpha=0.9)
    axes[3].set_ylabel("Torque (Nm)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print("\n=== DYNAMICS VALIDATION RESULTS ===")
    for i in range(NUM_JOINTS):
        error = tau_predicted[:, i] - tau_control[:, i]
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        max_control = np.max(np.abs(tau_control[:, i]))
        print(f"Joint {i+1}: RMSE = {rmse:.2f} Nm, MAE = {mae:.2f} Nm, Max Control = {max_control:.2f} Nm")
        if max_control > 0:
            print(f"         Relative RMSE = {100*rmse/max_control:.1f}%")

def main():
    """Main function."""
    print("=== FIXED ROBOT DYNAMICS TESTING ===")
    print("Key improvements:")
    print("- Disabled default PyBullet motors")
    print("- Added gravity compensation") 
    print("- Improved target position updates")
    print("- Smoothed acceleration calculation")
    
    data = collect_data_interactive()
    
    if data['time']:
        plot_results(data)
        print("\nDone! The robot should now hold positions properly.")
    else:
        print("No data recorded - make sure to enable recording!")

if __name__ == '__main__':
    main()