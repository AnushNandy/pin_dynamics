import pybullet as p
import pybullet_data
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics 
from src.systemid.pinocchio_friction_regressor import PinocchioAndFrictionRegressorBuilder, smooth_sign
from config import robot_config 

SIM_DURATION = 50
TIME_STEP = 1. / 240.
URDF_PATH = robot_config.URDF_PATH
IDENTIFIED_PARAMS_PATH = "/home/robot/dev/dyn/src/systemid/identified_params_pybullet.npz"

# Gains are kept as vectors, as in the original script
KP = np.array([100.0, 100.0, 200.0, 450.0, 200.0, 200.0, 0.7])
KD = np.array([2 * np.sqrt(k) for k in KP]) 
KI = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) 
GRAVITY_VECTOR = np.array([0, 0, -9.81])
MAX_TORQUES = np.array([140, 140, 51, 51, 14, 14, 7.7])
# MAX_TORQUES = np.array([140, 140, 51, 100, 51, 51, 51])

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
    
def get_joint_indices_by_name(robot_id, joint_names):
    joint_map = {p.getJointInfo(robot_id, i)[1].decode('UTF-8'): i for i in range(p.getNumJoints(robot_id))}
    return [joint_map[name] for name in joint_names]

def get_end_effector_link_index(robot_id):
    # Find the link corresponding to the last actuated joint
    num_joints = p.getNumJoints(robot_id)
    joint_indices = [i for i in range(num_joints) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
    last_joint_info = p.getJointInfo(robot_id, joint_indices[-1])
    return last_joint_info[0] # The link index is the first element of the joint info

def compute_inverse_kinematics(robot_id, end_effector_link_index, robot_target_pos, robot_target_orientation):
    
    ik_solution = p.calculateInverseKinematics(
        robot_id,
        end_effector_link_index,
        robot_target_pos,
        robot_target_orientation
    )
    return np.array(ik_solution)

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
    try:
        controller = PinocchioFeedforwardController(URDF_PATH, IDENTIFIED_PARAMS_PATH)
    except FileNotFoundError:
        return

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
    end_effector_link_index = get_end_effector_link_index(robot_id)
    
    # --- MODIFICATION: New lists for Cartesian logging ---
    log_t = []
    log_pos_des, log_orn_des = [], []
    log_pos_actual, log_orn_actual = [], []
    integral_error = np.zeros(7)

    print("\n--- Starting Refactored Simulation with Pinocchio Dynamics ---")
    print("Note: This script preserves the original's control loop structure.")

    for t in np.arange(0, SIM_DURATION, TIME_STEP):
        joint_states = p.getJointStates(robot_id, joint_indices)
        q_actual = np.array([state[0] for state in joint_states])
        qd_actual = np.array([state[1] for state in joint_states])

        # This is the target Cartesian pose for the *entire* upcoming trajectory segment
        link_state = p.getLinkState(robot_id, end_effector_link_index, computeForwardKinematics=1)
        pos_target_segment, orn_target_segment = link_state[0], link_state[1]

        q_desired = compute_inverse_kinematics(robot_id, end_effector_link_index, pos_target_segment, orn_target_segment)
        q_des_steps, qd_des_steps, qdd_des_steps = generate_trajectory_ik(q_actual, q_desired)

        if q_des_steps.shape[0] == 0:
            p.stepSimulation()
            continue

        for jj in range(len(q_des_steps)):
            q_des, qd_des, qdd_des = q_des_steps[jj], qd_des_steps[jj], qdd_des_steps[jj]

            tau_ff = controller.compute_feedforward_torque(q_des, qd_des, qdd_des)
            current_joint_states = p.getJointStates(robot_id, joint_indices)
            q_current_actual = np.array([s[0] for s in current_joint_states])
            qd_current_actual = np.array([s[1] for s in current_joint_states])

            integral_error += (q_des - q_current_actual) * TIME_STEP
            if np.linalg.norm(q_des - q_current_actual) > 0.5: 
                integral_error *= 0.1  
            
            tau_fb = KP * (q_des - q_current_actual) + KD * (qd_des - qd_current_actual) + KI * integral_error
            tau_total = tau_ff + tau_fb
            tau_limited = np.clip(tau_total, -MAX_TORQUES, MAX_TORQUES)
            p.setJointMotorControlArray(robot_id, joint_indices, p.TORQUE_CONTROL, forces=tau_limited)
            p.stepSimulation()
            
            # --- MODIFICATION: Log Cartesian data at each step ---
            current_ee_state = p.getLinkState(robot_id, end_effector_link_index)
            pos_actual_current = current_ee_state[0]
            orn_actual_current = current_ee_state[1]
            
            log_t.append(p.getPhysicsEngineParameters()['fixedTimeStep'] * (len(log_t) + 1))
            log_pos_des.append(pos_target_segment)
            log_orn_des.append(orn_target_segment)
            log_pos_actual.append(pos_actual_current)
            log_orn_actual.append(orn_actual_current)

    p.disconnect()

    # --- MODIFICATION: New plotting section for Cartesian error ---
    print("--- Simulation Complete. Generating Cartesian Error Plots. ---")

    # Convert logs to numpy arrays for vectorized calculations
    array_log_t = np.array(log_t)
    array_pos_des = np.array(log_pos_des)
    array_pos_actual = np.array(log_pos_actual)
    array_orn_des = np.array(log_orn_des)
    array_orn_actual = np.array(log_orn_actual)

    # 1. Calculate Translational Error
    pos_error_vec = array_pos_des - array_pos_actual
    pos_error_mag = np.linalg.norm(pos_error_vec, axis=1)

    # 2. Calculate Angular Error
    angular_error_mag = []
    for i in range(len(array_orn_des)):
        # Calculate the difference quaternion: q_diff = q_des * q_actual_inverse
        orn_act = array_orn_actual[i]
        diff_quat = p.getDifferenceQuaternion(orn_act, array_orn_des[i])
        # Convert the difference quaternion to an axis-angle representation
        axis, angle = p.getAxisAngleFromQuaternion(diff_quat)
        angular_error_mag.append(abs(angle))
    
    angular_error_mag = np.array(angular_error_mag)

    # --- EDIT: Define a cutoff to remove the initial spike from plots ---
    CUTOFF_TIME = 1.0 # seconds to ignore at the start
    
    # Find the first index in the time array that is greater than or equal to the cutoff
    if len(array_log_t) > 0 and array_log_t[-1] > CUTOFF_TIME:
        cutoff_index = np.searchsorted(array_log_t, CUTOFF_TIME)
    else:
        cutoff_index = 0 # Plot all data if simulation is too short or empty
        print("Warning: Simulation duration is less than cutoff time. Plotting all data.")

    # Slice all data arrays to create a "post-settling" view
    plot_t = array_log_t[cutoff_index:]
    plot_pos_error_vec = pos_error_vec[cutoff_index:]
    plot_pos_error_mag = pos_error_mag[cutoff_index:]
    plot_angular_error_mag = angular_error_mag[cutoff_index:]

    # --- Plotting Translational Error ---
    plt.figure(figsize=(14, 8))
    plt.suptitle("End-Effector Translational Error Analysis (Post-Settling)", fontsize=16, weight='bold')

    # Plot 1: Error components (X, Y, Z)
    plt.subplot(2, 1, 1)
    plt.plot(plot_t, plot_pos_error_vec[:, 0] * 1000, label='Error X')
    plt.plot(plot_t, plot_pos_error_vec[:, 1] * 1000, label='Error Y')
    plt.plot(plot_t, plot_pos_error_vec[:, 2] * 1000, label='Error Z')
    plt.title("Error Components")
    plt.ylabel("Error (mm)")
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Error magnitude
    plt.subplot(2, 1, 2)
    plt.plot(plot_t, plot_pos_error_mag * 1000, 'r-', label='Magnitude')
    plt.title("Error Magnitude")
    plt.xlabel(f"Time (s) - Starting from t={CUTOFF_TIME}s")
    plt.ylabel("Error Magnitude (mm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # --- Plotting Angular Error ---
    plt.figure(figsize=(14, 5))
    plt.suptitle("End-Effector Angular Error Analysis (Post-Settling)", fontsize=16, weight='bold')
    plt.plot(plot_t, np.rad2deg(plot_angular_error_mag), 'g-', label='Angular Error')
    plt.title("Error Magnitude")
    plt.xlabel(f"Time (s) - Starting from t={CUTOFF_TIME}s")
    plt.ylabel("Angular Error (degrees)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


if __name__ == "__main__":
    main()