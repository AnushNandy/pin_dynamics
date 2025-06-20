# File: 06_refactored_with_pinocchio.py

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
IDENTIFIED_PARAMS_PATH = "/home/robot/dev/dyn/src/systemid/identified_params.npz"

# Gains are kept as vectors, as in the original script
# KP = np.array([100.0, 100.0, 80.0, 70.0, 40.0, 30.0, 20.0])
# KP = np.array([600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0])
KP = np.array([100.0, 100.0, 200.0, 200.0, 80.0, 100.0, 100.0])
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
    controller = PinocchioFeedforwardController(URDF_PATH, IDENTIFIED_PARAMS_PATH)

    p.connect(p.GUI)
    p.setTimeStep(TIME_STEP)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(*GRAVITY_VECTOR)
    p.loadURDF("plane.urdf")
    robot_start_pos = [0, 0, 0.5]
    robot_start_orn = p.getQuaternionFromEuler([0, 0, 0])
    # robot_id = p.loadURDF(URDF_PATH, [0, 0, 0.5], useFixedBase=True)
    robot_id = p.loadURDF(URDF_PATH, robot_start_pos, useFixedBase=False)
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
    end_effector_link_index = get_end_effector_link_index(robot_id)

    log_t, log_q_des, log_q_actual = [], [], []
    integral_error = np.zeros(7)

    print("\n--- Starting Refactored Simulation with Pinocchio Dynamics ---")
    print("Note: This script preserves the original's control loop structure.")

    for t in np.arange(0, SIM_DURATION, TIME_STEP):
        # Get actual state from simulation
        joint_states = p.getJointStates(robot_id, joint_indices)
        q_actual = np.array([state[0] for state in joint_states])
        qd_actual = np.array([state[1] for state in joint_states])

        # Get end-effector pose for IK target (original logic)
        link_state = p.getLinkState(robot_id, end_effector_link_index, computeForwardKinematics=1)
        pos, orn = link_state[0], link_state[1] # World position/orientation of the link frame

        q_desired = compute_inverse_kinematics(robot_id, end_effector_link_index, pos, orn)
        
        q_des_steps, qd_des_steps, qdd_des_steps = generate_trajectory_ik(q_actual, q_desired)

        if q_des_steps.shape[0] == 0:
            p.stepSimulation()
            continue

        for jj in range(len(q_des_steps)):
            q_des = q_des_steps[jj]
            qd_des = qd_des_steps[jj]
            qdd_des = qdd_des_steps[jj]

            tau_ff = controller.compute_feedforward_torque(q_des, qd_des, qdd_des)
            current_joint_states = p.getJointStates(robot_id, joint_indices)
            q_current_actual = np.array([s[0] for s in current_joint_states])
            qd_current_actual = np.array([s[1] for s in current_joint_states])


            integral_error += (q_des - q_current_actual) * TIME_STEP
            if np.linalg.norm(q_des - q_current_actual) > 0.5: 
                integral_error *= 0.1  
            
            # tau_fb = KP * (q_des - q_current_actual) + KD * (qd_des - qd_current_actual) + KI * integral_error
            
            tau_fb = KP * (q_des - q_current_actual) + KD * (qd_des - qd_current_actual)
            tau_total = tau_ff + tau_fb

            tau_limited = np.clip(tau_total, -MAX_TORQUES, MAX_TORQUES)
            p.setJointMotorControlArray(robot_id, joint_indices, p.TORQUE_CONTROL, forces=tau_limited)
            p.stepSimulation()

            log_t.append(p.getPhysicsEngineParameters()['fixedTimeStep'] * (len(log_t) + 1))
            log_q_des.append(q_des)
            log_q_actual.append(q_current_actual)

    p.disconnect()

    array_log_t = np.array(log_t)
    array_log_q_des = np.array(log_q_des)
    array_log_q_actual = np.array(log_q_actual)

    plt.figure(figsize=(12, 2.5 * robot_config.NUM_JOINTS))
    for i in range(robot_config.NUM_JOINTS):
        plt.subplot(robot_config.NUM_JOINTS, 2, 2 * i + 1)
        plt.plot(array_log_t, np.rad2deg(array_log_q_des[:, i]), 'r--', label='Desired')
        plt.plot(array_log_t, np.rad2deg(array_log_q_actual[:, i]), 'b-', label='Actual')
        plt.title(f'Joint {i} Tracking'), plt.ylabel('Pos (deg)')
        plt.legend(), plt.grid(True)

        plt.subplot(robot_config.NUM_JOINTS, 2, 2 * i + 2)
        err = np.rad2deg(array_log_q_des[:, i] - array_log_q_actual[:, i])
        plt.plot(array_log_t, err, 'g-', label='Error')
        plt.title(f'Joint {i} Error'), plt.ylabel('Error (deg)')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
