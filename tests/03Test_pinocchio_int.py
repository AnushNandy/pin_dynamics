import pybullet as p
import pybullet_data
import numpy as np
import time
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics
from config import robot_config

# --- Constants ---
IDENTIFIED_PARAMS_PATH = "/home/robot/dev/dyn/src/systemid/identified_params.npz"
URDF_PATH = robot_config.URDF_PATH
NUM_JOINTS = robot_config.NUM_JOINTS
TIME_STEP = 1. / 240.  # Slower for manual interaction

# PD Control gains - adjust these if robot is too stiff or too loose
KP = np.array([600.0, 120.0, 100.0, 70.0, 40.0, 30.0, 20.0])
KD = np.array([150.0, 20.0, 18.0, 12.0, 7.0, 5.0, 3.0])

def setup_robot():
    """Setup PyBullet with PD controlled robot."""
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
    
    return robot_id, joint_indices

def pd_control(q_current, qd_current, q_target):
    """Simple PD controller for each joint."""
    # Position error
    e_pos = q_target - q_current
    
    # Velocity error (target velocity is 0 - we want to hold position)
    e_vel = 0.0 - qd_current
    
    # PD control law
    tau_control = KP * e_pos + KD * e_vel
    
    return tau_control

def collect_data_interactive():
    """Interactive data collection with PD control to hold positions."""
    print("\n=== INTERACTIVE MODE WITH PD CONTROL ===")
    print("1. Click and drag the robot links to move them")
    print("2. The robot will 'stick' to wherever you move it (PD control)")
    print("3. Press SPACE or use slider to start/stop recording")
    print("4. Press 'q' to quit and analyze data")
    print("5. The robot should feel 'heavy' but controllable now!")
    
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
    
    # Initialize target positions to current positions
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
    
    recording = False
    qd_prev = np.zeros(NUM_JOINTS)
    q_prev = q_target.copy()
    t = 0
    
    # Add control parameters
    record_button = p.addUserDebugParameter("Recording (0=off, 1=on)", 0, 1, 0)
    kp_slider = p.addUserDebugParameter("Kp Gain", 10, 200, KP[0])
    kd_slider = p.addUserDebugParameter("Kd Gain", 1, 20, KD[0])
    update_target_button = p.addUserDebugParameter("Update Target (0=hold, 1=update)", 0, 1, 1)
    
    print("Ready! Adjust Kp/Kd sliders to change stiffness/damping")
    print("Set 'Update Target' to 1 to let robot follow your movements")
    
    try:
        while True:
            # Update control gains from sliders
            kp_val = p.readUserDebugParameter(kp_slider)
            kd_val = p.readUserDebugParameter(kd_slider)
            KP[:] = kp_val
            KD[:] = kd_val
            
            # Check if we should update target positions
            update_target = bool(p.readUserDebugParameter(update_target_button))
            recording = bool(p.readUserDebugParameter(record_button))
            
            # Get current robot state
            joint_states = p.getJointStates(robot_id, joint_indices)
            q = np.array([state[0] for state in joint_states])
            qd = np.array([state[1] for state in joint_states])
            
            # Update target if enabled (this lets you drag the robot around)
            if update_target:
                # If robot moved significantly, update target
                if np.linalg.norm(q - q_prev) > 0.01:  # Small threshold to avoid noise
                    q_target = q.copy()
            
            # Apply PD control
            tau_control = pd_control(q, qd, q_target)
            
            # Apply control torques
            p.setJointMotorControlArray(
                robot_id,
                joint_indices,
                p.TORQUE_CONTROL,
                forces=tau_control
            )
            
            # Calculate acceleration for dynamics comparison
            qdd = (qd - qd_prev) / TIME_STEP
            qd_prev = qd.copy()
            q_prev = q.copy()
            
            if recording and t > 0.1:  # Skip first few steps
                # Predict torques using your dynamics model
                tau_predicted = dynamics_model.compute_rnea(q, qd, qdd)
                
                # Store data
                data['time'].append(t)
                data['q'].append(q.copy())
                data['qd'].append(qd.copy())
                data['qdd'].append(qdd.copy())
                data['tau_control'].append(tau_control.copy())
                data['tau_predicted'].append(tau_predicted.copy())
                
                # Print status occasionally
                if len(data['time']) % 60 == 0:
                    print(f"Recording... {len(data['time'])} samples collected")
                    print(f"Current pose: {q[:3].round(2)}")  # Show first 3 joints
            
            # Check for quit
            keys = p.getKeyboardEvents()
            if ord('q') in keys:
                break
                
            p.stepSimulation()
            time.sleep(TIME_STEP)
            t += TIME_STEP
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
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
    
    # Plot 1: Joint positions
    axes[0].set_title("Joint Positions")
    for i in range(min(NUM_JOINTS, 6)):  # Limit to 6 joints for readability
        axes[0].plot(time_arr, q_arr[:, i], label=f'Joint {i+1}')
    axes[0].set_ylabel("Position (rad)")
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot 2: Joint velocities
    axes[1].set_title("Joint Velocities")
    for i in range(min(NUM_JOINTS, 6)):
        axes[1].plot(time_arr, qd_arr[:, i], label=f'Joint {i+1}')
    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].grid(True)
    axes[1].legend()
    
    # Plot 3: Joint accelerations
    axes[2].set_title("Joint Accelerations")
    for i in range(min(NUM_JOINTS, 6)):
        axes[2].plot(time_arr, qdd_arr[:, i], label=f'Joint {i+1}', alpha=0.7)
    axes[2].set_ylabel("Acceleration (rad/s²)")
    axes[2].grid(True)
    axes[2].legend()
    
    # Plot 4: Torque comparison
    axes[3].set_title("Torque Comparison: Control vs Predicted Dynamics")
    for i in range(min(NUM_JOINTS, 3)):  # Show fewer joints for torque
        axes[3].plot(time_arr, tau_control[:, i], '--', 
                    label=f'Control J{i+1}', alpha=0.7, linewidth=2)
        axes[3].plot(time_arr, tau_predicted[:, i], '-', 
                    label=f'Predicted J{i+1}', alpha=0.9)
    axes[3].set_ylabel("Torque (Nm)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True)
    axes[3].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print error statistics
    print("\n=== DYNAMICS COMPARISON RESULTS ===")
    print("This compares the control torques needed vs predicted dynamics torques")
    print("(Note: They won't match exactly since control compensates for model errors)")
    
    for i in range(NUM_JOINTS):
        error = tau_predicted[:, i] - tau_control[:, i]
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        print(f"Joint {i+1}: RMSE = {rmse:.3f} Nm, MAE = {mae:.3f} Nm")
    
    # Also show torque statistics
    print(f"\nControl torque magnitudes:")
    for i in range(NUM_JOINTS):
        tau_rms = np.sqrt(np.mean(tau_control[:, i]**2))
        tau_max = np.max(np.abs(tau_control[:, i]))
        print(f"Joint {i+1}: RMS = {tau_rms:.3f} Nm, Max = {tau_max:.3f} Nm")

def main():
    """Main function - collect data then plot."""
    print("=== ROBOT DYNAMICS TESTING WITH PD CONTROL ===")
    print("This will let you manually move the robot with proper control")
    print("The robot will 'stick' to positions you drag it to")
    
    # Collect data interactively
    data = collect_data_interactive()
    
    # Plot results
    if data['time']:
        plot_results(data)
        print("Done! The plots show how control torques compare to predicted dynamics.")
        print("Use this to validate your dynamics model under different motions.")
    else:
        print("No data was recorded. Make sure to turn on recording and move the robot!")

if __name__ == '__main__':
    main()