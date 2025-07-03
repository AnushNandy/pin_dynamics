import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics 
from src.systemid.pinocchio_friction_regressor import PinocchioAndFrictionRegressorBuilder, smooth_sign
from config import robot_config 

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from pinocchio.robot_wrapper import RobotWrapper

SIM_DURATION = 50
TIME_STEP = 1. / 240.
URDF_PATH = robot_config.URDF_PATH
IDENTIFIED_PARAMS_PATH = "/home/robot/dev/dyn/src/systemid/identified_params_pybullet.npz"

# Gains are kept as vectors, as in the original script
KP = np.array([600.0, 600.0, 500.0, 500.0, 550.0, 550.0, 100])
KD = np.array([2 * np.sqrt(k) for k in KP]) 
KI = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) 
GRAVITY_VECTOR = np.array([0, 0, -9.81])
MAX_TORQUES = np.array([140, 140, 51, 51, 14, 14, 7.7])

class PinocchioFeedforwardController:
    """
    This controller uses a system-identified Pinocchio model to compute
    the feedforward torques (RNEA + identified friction).
    """
    def __init__(self, urdf_path: str, identified_params_path: str):
        print("--- Initializing Pinocchio Feedforward Controller ---")
        # 1. Load the dynamic model using Pinocchio
        self.model_dynamics = PinocchioRobotDynamics(urdf_path)
        self.num_joints = self.model_dynamics.num_actuated_joints

        # 2. Load and set the identified parameters
        self._load_and_set_identified_params(identified_params_path)
        print("-----------------------------------------------------")

    def _load_and_set_identified_params(self, file_path: str):
        try:
            params_data = np.load(file_path)
            identified_params = params_data['P']
            
            # Separate the parameters for the rigid body dynamics and friction
            num_rnea_params = (self.model_dynamics.model.nbodies - 1) * 10
            rnea_params = identified_params[:num_rnea_params]
            self.friction_params = identified_params[num_rnea_params:]
            
            # Update the Pinocchio model's inertial parameters
            self.model_dynamics.set_parameters_from_vector(rnea_params)
            print(f"Loaded and set identified parameters from '{file_path}'")

        except FileNotFoundError:
            print(f"FATAL: Identified params file not found at '{file_path}'.")
            print("Cannot proceed without the identified model.")
            raise

    def compute_feedforward_torque(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """
        Computes the feedforward torque.
        τ_ff = RNEA(q, qd, qdd) + τ_friction(qd)
        """
        # 1. Compute the rigid-body torques using Pinocchio's RNEA
        tau_rnea = self.model_dynamics.compute_rnea(q, qd, qdd, gravity=GRAVITY_VECTOR)
        
        # 2. Compute the identified friction torque
        # tau_friction = np.zeros(self.num_joints)
        # for i in range(self.num_joints):
        #     fv_hat = self.friction_params[i * 2]      # Identified viscous friction
        #     fc_hat = self.friction_params[i * 2 + 1]  # Identified Coulomb friction
            # tau_friction[i] = fv_hat * qd[i] + fc_hat * smooth_sign(qd[i])
            
        # 3. The total feedforward torque is the sum
        # return tau_rnea + tau_friction
        return tau_rnea 

def get_end_effector_frame_id(model, end_effector_name="panda_hand"):
    """Get the frame ID for the end effector with proper error handling"""
    try:
        # First, try to get the specific end effector frame
        frame_id = model.getFrameId(end_effector_name)
        print(f"Found end effector frame '{end_effector_name}' with ID: {frame_id}")
        return frame_id
    except:
        print(f"Frame '{end_effector_name}' not found, trying alternative approaches...")
        
        # Try common end effector frame names
        common_names = ["panda_hand", "panda_link8", "panda_tcp", "tool0", "ee_link"]
        for name in common_names:
            try:
                frame_id = model.getFrameId(name)
                print(f"Found alternative end effector frame '{name}' with ID: {frame_id}")
                return frame_id
            except:
                continue
        
        # If no specific end effector frame found, use the last joint frame
        try:
            actuated_joint_names = robot_config.ACTUATED_JOINT_NAMES
            last_joint_name = actuated_joint_names[-1]
            frame_id = model.getFrameId(last_joint_name)
            print(f"Using last joint '{last_joint_name}' as end effector with ID: {frame_id}")
            return frame_id
        except:
            # As a last resort, use the last body frame
            print(f"Using last body frame as end effector with ID: {model.nframes - 1}")
            return model.nframes - 1

def compute_inverse_kinematics(model, data, end_effector_frame_id, target_pos, target_orientation, q_init):
    """Compute inverse kinematics using Pinocchio"""
    # Convert target orientation to SE3 if needed
    if len(target_orientation) == 4:  # quaternion
        target_rot = pin.Quaternion(target_orientation[3], target_orientation[0], target_orientation[1], target_orientation[2]).toRotationMatrix()
    else:  # assume rotation matrix
        target_rot = target_orientation
    
    target_se3 = pin.SE3(target_rot, target_pos)
    
    # Use Pinocchio's inverse kinematics solver
    q_result = q_init.copy()
    
    # Simple iterative IK solver
    for i in range(100):  # max iterations
        pin.forwardKinematics(model, data, q_result)
        pin.updateFramePlacements(model, data)
        
        # Check if frame ID is valid
        if end_effector_frame_id >= len(data.oMf):
            print(f"Warning: Frame ID {end_effector_frame_id} is out of range (max: {len(data.oMf)-1})")
            return q_result
        
        current_se3 = data.oMf[end_effector_frame_id]
        error_se3 = target_se3.inverse() * current_se3
        error_6d = pin.log(error_se3)
        
        if np.linalg.norm(error_6d) < 1e-6:
            break
            
        # Compute Jacobian
        J = pin.computeFrameJacobian(model, data, q_result, end_effector_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        # Damped least squares
        damping = 1e-6
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
        
        # Update configuration
        dq = J_pinv @ error_6d
        q_result = pin.integrate(model, q_result, dq)
    
    return q_result

def generate_trajectory_ik(q_actual, q_des, total_time=0.1, dt=TIME_STEP):
    q_actual = np.array(q_actual)
    q_des = np.array(q_des)
    timesteps = np.arange(0, total_time, dt)
    if len(timesteps) == 0: return np.array([]), np.array([]), np.array([])
    N = len(timesteps)

    q_des_steps = np.zeros((N, 7))
    qd_des_steps = np.zeros((N, 7))
    qdd_des_steps = np.zeros((N, 7))

    for i in range(7):
        q0, qf = q_actual[i], q_des[i]
        dq0 = dqf = ddq0 = ddqf = 0.0
        T = total_time
        a0 = q0; a1 = dq0; a2 = 0.5 * ddq0
        a3 = (20*(qf-q0)-(8*dqf+12*dq0)*T-(3*ddq0-ddqf)*T**2)/(2*T**3)
        a4 = (30*(q0-qf)+(14*dqf+16*dq0)*T+(3*ddq0-2*ddqf)*T**2)/(2*T**4)
        a5 = (12*(qf-q0)-(6*dqf+6*dq0)*T-(ddq0-ddqf)*T**2)/(2*T**5)

        for j, t in enumerate(timesteps):
            q = a0+a1*t+a2*t**2+a3*t**3+a4*t**4+a5*t**5
            qd = a1+2*a2*t+3*a3*t**2+4*a4*t**3+5*a5*t**4
            qdd = 2*a2+6*a3*t+12*a4*t**2+20*a5*t**3
            q_des_steps[j, i], qd_des_steps[j, i], qdd_des_steps[j, i] = q, qd, qdd

    return q_des_steps, qd_des_steps, qdd_des_steps

def main():
    controller = PinocchioFeedforwardController(URDF_PATH, IDENTIFIED_PARAMS_PATH)
    
    # Initialize Pinocchio robot and visualization
    robot = RobotWrapper.BuildFromURDF(URDF_PATH, package_dirs=[], root_joint=None)
    model = robot.model
    data = robot.data
    
    # Initialize Meshcat visualization
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    print("Meshcat initialized at http://127.0.0.1:7000")
    
    # --- Print joint information ---
    print(f"Total configuration dimension (model.nq): {model.nq}")
    print(f"Total velocity dimension (model.nv): {model.nv}")
    print(f"Number of joints in model: {model.njoints}")
    print(f"Number of frames in model: {model.nframes}")
    print("------------------------------------")

    print("Joint Names and Indices:")
    actuated_joint_names = robot_config.ACTUATED_JOINT_NAMES
    actuated_joint_ids = [model.getJointId(name) for name in actuated_joint_names]
    
    # Get detailed joint information
    for i, joint_id in enumerate(actuated_joint_ids):
        joint = model.joints[joint_id]
        joint_name = model.names[joint_id]
        q_idx = joint.idx_q
        nq = joint.nq  # Number of configuration variables for this joint
        nv = joint.nv  # Number of velocity variables for this joint
        print(f"Joint {joint_id} ({joint_name}) -> q_idx: {q_idx}, nq: {nq}, nv: {nv}")
    
    print(f"Model nq: {model.nq}, nv: {model.nv}")
    
    # Print all available frames for debugging
    print("\nAvailable frames:")
    for i in range(model.nframes):
        frame_name = model.frames[i].name
        print(f"Frame {i}: {frame_name}")
    
    # Get end effector frame with better error handling
    end_effector_frame_id = get_end_effector_frame_id(model)
    print(f"Selected end effector frame ID: {end_effector_frame_id}")
    
    # Validate frame ID
    if end_effector_frame_id >= model.nframes:
        print(f"ERROR: Frame ID {end_effector_frame_id} is out of range (max: {model.nframes-1})")
        print("Using last available frame instead...")
        end_effector_frame_id = model.nframes - 1
    
    # Initialize robot state
    q_actual = np.zeros(robot_config.NUM_JOINTS)
    qd_actual = np.zeros(robot_config.NUM_JOINTS)
    integral_error = np.zeros(robot_config.NUM_JOINTS)
    
    # Set initial configuration
    q_full = pin.neutral(model)
    
    # Properly map actuated joints to full configuration
    if model.nq > robot_config.NUM_JOINTS:
        # Assume the first joints are the actuated ones (common for fixed-base robots)
        q_full[:robot_config.NUM_JOINTS] = q_actual
    else:
        q_full[:len(q_actual)] = q_actual
    
    # Update kinematics and display
    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)
    viz.display(q_full)
    
    log_t, log_q_des, log_q_actual = [], [], []

    print("\n--- Starting Meshcat Simulation with Pinocchio Dynamics ---")
    print("Note: This script uses Pinocchio for all kinematics and dynamics.")

    # Initialize circle center
    circle_center = None

    for step, t in enumerate(np.arange(0, SIM_DURATION, TIME_STEP)):
        # Update kinematics first
        pin.forwardKinematics(model, data, q_full)
        pin.updateFramePlacements(model, data)
        
        # Get current end-effector pose with bounds checking
        if end_effector_frame_id < len(data.oMf):
            current_ee_pose = data.oMf[end_effector_frame_id]
            pos = current_ee_pose.translation
            orn = current_ee_pose.rotation
        else:
            print(f"Warning: Frame ID {end_effector_frame_id} out of range, using identity pose")
            pos = np.zeros(3)
            orn = np.eye(3)
        
        # Initialize circle center on first step
        if circle_center is None:
            circle_center = pos.copy()
        
        # For this example, we'll create a simple circular motion target
        circle_radius = 0.1
        target_pos = circle_center + np.array([
            circle_radius * np.cos(t * 0.5),
            circle_radius * np.sin(t * 0.5),
            0.05 * np.sin(t * 0.3)  # Small vertical oscillation
        ])
        
        target_orn = orn  # Keep same orientation
        
        # Compute inverse kinematics
        q_desired = compute_inverse_kinematics(model, data, end_effector_frame_id, target_pos, target_orn, q_full)
        
        # Extract only the actuated joints
        if model.nq > robot_config.NUM_JOINTS:
            q_desired_actuated = q_desired[:robot_config.NUM_JOINTS]
        else:
            q_desired_actuated = q_desired[:robot_config.NUM_JOINTS]
        
        # Generate trajectory
        q_des_steps, qd_des_steps, qdd_des_steps = generate_trajectory_ik(q_actual, q_desired_actuated)

        if q_des_steps.shape[0] == 0:
            continue

        # Execute trajectory steps
        for jj in range(len(q_des_steps)):
            q_des = q_des_steps[jj]
            qd_des = qd_des_steps[jj]
            qdd_des = qdd_des_steps[jj]

            # Compute feedforward torque using identified dynamics
            tau_ff = controller.compute_feedforward_torque(q_des, qd_des, qdd_des)
            
            # PID control
            integral_error += (q_des - q_actual) * TIME_STEP
            if np.linalg.norm(q_des - q_actual) > 0.5: 
                integral_error *= 0.1  
            
            tau_fb = KP * (q_des - q_actual) + KD * (qd_des - qd_actual) + KI * integral_error
            tau_total = tau_ff + tau_fb
            tau_limited = np.clip(tau_total, -MAX_TORQUES, MAX_TORQUES)
            
            # Simple dynamics integration (you would replace this with actual robot control)
            # Assuming unit mass for simplicity
            qdd_actual = tau_limited * 0.01  # Scale down for stability
            qd_actual += qdd_actual * TIME_STEP
            q_actual += qd_actual * TIME_STEP
            
            # Update visualization with proper configuration space mapping
            q_full = pin.neutral(model)
            
            # Properly map actuated joints to full configuration
            if model.nq > robot_config.NUM_JOINTS:
                q_full[:robot_config.NUM_JOINTS] = q_actual
            else:
                q_full[:len(q_actual)] = q_actual
            
            # Update visualization
            viz.display(q_full)
            
            # Add small delay for smoother visualization
            if step % 10 == 0:
                time.sleep(0.001)

            # Log data
            log_t.append(t)
            log_q_des.append(q_des)
            log_q_actual.append(q_actual)
            
            # Debug output
            if step % 1200 == 0:  # Every 5 seconds
                print(f"t={t:.2f}: q_actual[0]={q_actual[0]:.3f}, target_pos={target_pos}")

    # Convert logs to arrays
    array_log_t = np.array(log_t)
    array_log_q_des = np.array(log_q_des)
    array_log_q_actual = np.array(log_q_actual)

    # Plot results
    plt.figure(figsize=(12, 2.5 * robot_config.NUM_JOINTS))
    for i in range(robot_config.NUM_JOINTS):
        plt.subplot(robot_config.NUM_JOINTS, 2, 2 * i + 1)
        plt.plot(array_log_t, np.rad2deg(array_log_q_des[:, i]), 'r--', label='Desired')
        plt.plot(array_log_t, np.rad2deg(array_log_q_actual[:, i]), 'b-', label='Actual')
        plt.title(f'Joint {i+1} Tracking')
        plt.ylabel('Position (deg)')
        plt.legend()
        plt.grid(True)

        plt.subplot(robot_config.NUM_JOINTS, 2, 2 * i + 2)
        err = np.rad2deg(array_log_q_des[:, i] - array_log_q_actual[:, i])
        plt.plot(array_log_t, err, 'g-', label='Error')
        plt.title(f'Joint {i+1} Error')
        plt.ylabel('Error (deg)')
        plt.grid(True)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()