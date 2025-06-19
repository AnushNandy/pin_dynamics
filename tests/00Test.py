import pybullet as p
import pybullet_data
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Ensure correct paths for your project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics 
from src.systemid.pinocchio_friction_regressor import smooth_sign
from config import robot_config 

# --- Simulation & Controller Parameters ---
SIM_DURATION = 50.0 # Shorter duration is fine for validation
TIME_STEP = 1. / 240.
URDF_PATH = robot_config.URDF_PATH
IDENTIFIED_PARAMS_PATH = "/home/robot/dev/dyn/src/systemid/identified_params.npz"

# Controller Gains (can be tuned, but start here)
# KP = np.array([100.0, 100.0, 80.0, 70.0, 40.0, 30.0, 20.0])
KP = np.array([600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0])
KD = np.array([2 * np.sqrt(k) for k in KP]) # Critically damped
KI = np.zeros(7) # Start with zero integral gain to see feedforward performance

# Physics and Robot Constants
GRAVITY_VECTOR = np.array([0, 0, -9.81])
MAX_TORQUES = np.array([140, 140, 51, 51, 14, 14, 7.7])
NUM_JOINTS = robot_config.NUM_JOINTS

class GroundTruthFeedforwardController:
    """
    A "perfect" controller that uses the known ground-truth dynamic parameters.
    This serves as a sanity check to validate the control framework.
    """
    def __init__(self, urdf_path: str):
        print("--- Initializing GROUND TRUTH Feedforward Controller ---")
        # 1. Load the dynamic model using Pinocchio. The parameters from the URDF
        # are considered the "ground truth" for the rigid body dynamics.
        self.model_dynamics = PinocchioRobotDynamics(urdf_path)
        self.num_joints = self.model_dynamics.num_actuated_joints

        # 2. Define the "ground-truth" friction coefficients, matching the data generation script.
        self.true_friction_coeffs = {
            'viscous': [0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06],
            'coulomb': [0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
        }
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
        tau_friction = np.zeros(self.num_joints)
        for i in range(self.num_joints):
            fv_true = self.true_friction_coeffs['viscous'][i]
            fc_true = self.true_friction_coeffs['coulomb'][i]
            tau_friction[i] = fv_true * qd[i] + fc_true * smooth_sign(qd[i])
            
        return tau_rnea + tau_friction

# --- Feedforward Controller (Identical to your original, it's correct) ---
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
            self.friction_params = identified_params[num_rnea_params:]
            
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
        
        tau_friction = np.zeros(self.num_joints)
        for i in range(self.num_joints):
            fv_hat = self.friction_params[i * 2]
            fc_hat = self.friction_params[i * 2 + 1]
            tau_friction[i] = fv_hat * qd[i] + fc_hat * smooth_sign(qd[i])
            
        return tau_rnea + tau_friction

# --- Trajectory Generation (Moved from your data generation script) ---
def generate_fourier_series_trajectory(t, num_harmonics=5):
    """
    Generates a smooth, exciting trajectory using a sum of sinusoids.
    """
    q_des = np.zeros(NUM_JOINTS)
    qd_des = np.zeros(NUM_JOINTS)
    qdd_des = np.zeros(NUM_JOINTS)
    w = 2 * np.pi * 0.2  # Base frequency

    for i in range(NUM_JOINTS):
        np.random.seed(i*10) # Use a different seed for variety
        a_n = np.random.uniform(0.05, 0.5, num_harmonics)
        b_n = np.random.uniform(0.05, 0.5, num_harmonics)
        phase_shifts = np.random.uniform(0, np.pi, num_harmonics)

        for n in range(num_harmonics):
            angle = (n + 1) * w * t + phase_shifts[n]
            q_des[i] += a_n[n] * np.sin(angle) + b_n[n] * np.cos(angle)
            qd_des[i] += (n + 1) * w * (a_n[n] * np.cos(angle) - b_n[n] * np.sin(angle))
            qdd_des[i] += -((n + 1) * w)**2 * (a_n[n] * np.sin(angle) + b_n[n] * np.cos(angle))
            
    return q_des, qd_des, qdd_des

def get_joint_indices_by_name(robot_id, joint_names):
    """Helper function to get joint indices from PyBullet."""
    joint_map = {p.getJointInfo(robot_id, i)[1].decode('UTF-8'): i for i in range(p.getNumJoints(robot_id))}
    return [joint_map[name] for name in joint_names]

# --- Main Simulation and Control Loop ---
def main():
    try:
        controller = PinocchioFeedforwardController(URDF_PATH, IDENTIFIED_PARAMS_PATH)
        # controller = GroundTruthFeedforwardController(URDF_PATH)
    except FileNotFoundError:
        return

    # 1. --- PyBullet Setup ---
    p.connect(p.GUI)
    p.setTimeStep(TIME_STEP)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(*GRAVITY_VECTOR)
    p.loadURDF("plane.urdf")
    # robot_id = p.loadURDF(URDF_PATH, [0, 0, 0.5], useFixedBase=True)
    # joint_indices = get_joint_indices_by_name(robot_id, robot_config.ACTUATED_JOINT_NAMES)
    robot_start_pos = [0, 0, 0.5]
    robot_start_orn = p.getQuaternionFromEuler([0, 0, 0])

    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0.5], useFixedBase=False)
    p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=-1,  # -1 for the base
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

    # 2. --- CRITICAL: Disable PyBullet's Default Dynamics ---
    # This ensures the only forces are gravity and our applied torques,
    # creating a perfect test-bench for our feedforward controller.
    print("\nDisabling PyBullet's default joint damping to isolate controller.")
    for idx in joint_indices:
        p.changeDynamics(robot_id, idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
        # Also disable the default velocity controller by setting a zero gain
        p.setJointMotorControl2(robot_id, idx, p.VELOCITY_CONTROL, force=0)

    # 3. --- Generate The ENTIRE Trajectory Upfront ---
    print(f"Generating a {SIM_DURATION}s trajectory...")
    time_vec = np.arange(0, SIM_DURATION, TIME_STEP)
    q_des_traj, qd_des_traj, qdd_des_traj = [], [], []
    for t in time_vec:
        q, qd, qdd = generate_fourier_series_trajectory(t)
        q_des_traj.append(q)
        qd_des_traj.append(qd)
        qdd_des_traj.append(qdd)
    
    q_des_traj = np.array(q_des_traj)
    qd_des_traj = np.array(qd_des_traj)
    qdd_des_traj = np.array(qdd_des_traj)
    print("Trajectory generation complete.")

    # Reset robot to the initial position of the trajectory
    initial_q = q_des_traj[0]
    for i, j_idx in enumerate(joint_indices):
        p.resetJointState(robot_id, j_idx, targetValue=initial_q[i], targetVelocity=0)

    # Data logging
    log_t, log_q_des, log_q_actual = [], [], []

    print("\n--- Starting Corrected Simulation ---")
    # 4. --- Main Control Loop ---
    for k in range(len(time_vec)):
        # Get desired state from the pre-generated trajectory
        q_des = q_des_traj[k]
        qd_des = qd_des_traj[k]
        qdd_des = qdd_des_traj[k]

        # Get actual state from simulation
        joint_states = p.getJointStates(robot_id, joint_indices)
        q_actual = np.array([state[0] for state in joint_states])
        qd_actual = np.array([state[1] for state in joint_states])

        # --- Compute Torques ---
        # Feedforward torque from our identified model
        tau_ff = controller.compute_feedforward_torque(q_des, qd_des, qdd_des)
        
        # Feedback torque (PD controller) to correct for any errors
        tau_fb = KP * (q_des - q_actual) + KD * (qd_des - qd_actual)
        
        # Total torque
        tau_total = tau_ff + tau_fb
        tau_limited = np.clip(tau_total, -MAX_TORQUES, MAX_TORQUES)
        
        # Apply the computed torque
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
    plt.show()

if __name__ == "__main__":
    main()