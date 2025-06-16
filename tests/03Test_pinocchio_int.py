import pybullet as p
import pybullet_data
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics 
from src.dynamics.friction_transmission import JointModel
from config import robot_config 

URDF_PATH = r"/home/robot/dev/dyn/ArmModels/urdfs/P4/P4_Contra-Angle_right.urdf"
IDENTIFIED_PARAMS_PATH = r"/home/robot/dev/dyn/src/systemid/identified_params.npz"


TIME_STEP = 1. / 240.
GRAVITY_VECTOR = np.array([0, 0, -9.81])

KP = np.array([600.0, 120.0, 100.0, 70.0, 40.0, 30.0, 20.0])
KD = np.array([150.0, 20.0, 18.0, 12.0, 7.0, 5.0, 3.0])

MAX_TORQUES = np.array([200.0, 200.0, 150.0, 150.0, 100.0, 80.0, 80.0])

def get_joint_indices_by_name(robot_id, joint_names):
    """Helper function to get PyBullet joint indices from names."""
    joint_map = {p.getJointInfo(robot_id, i)[1].decode('UTF-8'): i for i in range(p.getNumJoints(robot_id))}
    return [joint_map[name] for name in joint_names]

def quintic_poly_trajectory(q_start, q_end, duration, t):
    """
    Generates a smooth quintic polynomial trajectory point (q, qd, qdd).
    Ensures zero velocity and acceleration at the start and end.
    """
    if t < 0: t = 0
    if t > duration: t = duration
    
    T = duration
    a0 = q_start
    a1 = np.zeros_like(q_start)
    a2 = np.zeros_like(q_start)
    a3 = (20 * (q_end - q_start)) / (2 * T**3)
    a4 = (30 * (q_start - q_end)) / (2 * T**4)
    a5 = (12 * (q_end - q_start)) / (2 * T**5)
    q_des = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    qd_des = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    qdd_des = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
    
    return q_des, qd_des, qdd_des

def plot_torques(time_log, ff_log, fb_log, total_log):
    """Plots the feedforward, feedback, and total torques."""
    plt.figure(figsize=(15, 3.5 * robot_config.NUM_JOINTS))
    for i in range(robot_config.NUM_JOINTS):
        plt.subplot(robot_config.NUM_JOINTS, 1, i + 1)
        plt.plot(time_log, ff_log[:, i], 'g--', label=f'Feedforward Torque (τ_ff)')
        plt.plot(time_log, fb_log[:, i], 'r:', label=f'Feedback Torque (τ_fb)')
        plt.plot(time_log, total_log[:, i], 'b-', label=f'Total Torque (τ_total)')
        plt.axhline(y=MAX_TORQUES[i], color='k', linestyle='--', label='Max Torque')
        plt.axhline(y=-MAX_TORQUES[i], color='k', linestyle='--')
        plt.title(f'Torque Components for Joint {i}')
        plt.ylabel('Torque (Nm)')
        plt.grid(True)
        plt.legend()
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.suptitle('Controller Torque Analysis', fontsize=16, y=1.02)
    plt.show()

def run_interactive_simulation(robot_dyn_model: PinocchioRobotDynamics, joint_models: JointModel):
    """
    Runs an interactive simulation where from Dynamics_full.config import robot_config the user controls the end-effector pose.
    """
    p.connect(p.GUI)
    p.setTimeStep(TIME_STEP)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(*GRAVITY_VECTOR)
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
    joint_indices = get_joint_indices_by_name(robot_id, robot_config.ACTUATED_JOINT_NAMES)
    
    # Get end-effector link index
    ee_link_index = joint_indices[-1]
    home_pos, _ = p.getLinkState(robot_id, ee_link_index)[:2]
    sliders = {
        'x': p.addUserDebugParameter("Target X", -5, 5, home_pos[0]),
        'y': p.addUserDebugParameter("Target Y", -5, 5, home_pos[1]),
        'z': p.addUserDebugParameter("Target Z", -5, 5, home_pos[2]),
    }
    logs = {'t': [], 'q_des': [], 'q_act': [], 'tau_ff': [], 'tau_fb': [], 'tau_total': []}

    # --- Trajectory State ---
    traj_start_time = 0
    traj_duration = 1.0  # seconds
    q_start_traj = np.array([s[0] for s in p.getJointStates(robot_id, joint_indices)])
    q_target_traj = q_start_traj.copy()
    last_target_pos = home_pos

    # --- Simulation Loop ---
    start_time = time.time()
    while time.time() - start_time < 30.0: # Run for 30 seconds
        
        current_sim_time = len(logs['t']) * TIME_STEP
        
        # 1. Read Target from Sliders and Plan Trajectory
        target_pos = [p.readUserDebugParameter(sliders[ax]) for ax in ['x', 'y', 'z']]
        
        # If target changed significantly, plan a new trajectory
        if np.linalg.norm(np.array(target_pos) - np.array(last_target_pos)) > 0.01:
            q_actual = np.array([s[0] for s in p.getJointStates(robot_id, joint_indices)])
            q_ik_target = p.calculateInverseKinematics(robot_id, ee_link_index, target_pos)
            
            # Start a new trajectory from the current actual position
            traj_start_time = current_sim_time
            q_start_traj = q_actual
            q_target_traj = np.array(q_ik_target)
            last_target_pos = target_pos

        # 2. Get Desired State from Current Trajectory
        t_in_traj = current_sim_time - traj_start_time
        q_des, qd_des, qdd_des = quintic_poly_trajectory(q_start_traj, q_target_traj, traj_duration, t_in_traj)

        # 3. Get Actual State
        joint_states = p.getJointStates(robot_id, joint_indices)
        q_actual = np.array([state[0] for state in joint_states])
        qd_actual = np.array([state[1] for state in joint_states])

        # 4. Compute Torques (Full CTC)
        tau_fb = KP * (q_des - q_actual) + KD * (qd_des - qd_actual)
        tau_rnea = robot_dyn_model.compute_rnea(q_des, qd_des, qdd_des)
        tau_ff = np.zeros_like(tau_rnea)
        for i in range(robot_config.NUM_JOINTS):
            tau_ff[i] = joint_models[i].compute_feedforward_torque(
                tau_rnea[i], q_des[i], qd_des[i], qdd_des[i]
            )
        
        tau_total = np.clip(tau_ff + tau_fb, -MAX_TORQUES, MAX_TORQUES)

        p.setJointMotorControlArray(robot_id, joint_indices, p.TORQUE_CONTROL, forces=tau_total)
        p.stepSimulation()

        logs['t'].append(current_sim_time)
        logs['q_des'].append(q_des)
        logs['q_act'].append(q_actual)
        logs['tau_ff'].append(tau_ff)
        logs['tau_fb'].append(tau_fb)
        logs['tau_total'].append(tau_total)

    p.disconnect()
    
    for key in logs:
        logs[key] = np.array(logs[key])
        
    plot_torques(logs['t'], logs['tau_ff'], logs['tau_fb'], logs['tau_total'])


def main():
    """Main function to load models and run the interactive simulation."""
    if not os.path.exists(IDENTIFIED_PARAMS_PATH) or not os.path.exists(URDF_PATH):
        print("FATAL: URDF or Identified Parameters file not found. Check paths.")
        return

    # 1. Load Identified Parameters and Initialize Models
    P_identified = np.load(IDENTIFIED_PARAMS_PATH)['P']
    robot_dyn_pinocchio = PinocchioRobotDynamics(URDF_PATH)
    robot_dyn_pinocchio.set_parameters_from_vector(P_identified)
    
    joint_models = []
    for i in range(robot_config.NUM_JOINTS):
        base_coulomb = 2.5 - i * 0.2
        base_stiction = 3.0 - i * 0.2
        joint_models.append(JointModel(
            gear_ratio=100.0, motor_inertia=0.0001 + i * 0.00001,
            coulomb_pos=base_coulomb, coulomb_neg=-(base_coulomb * 0.9),
            stiction_pos=base_stiction, stiction_neg=-(base_stiction * 0.9),
            viscous_coeff=0.15 - i * 0.01,
            stribeck_vel_pos=0.1, stribeck_vel_neg=-0.1, stiffness=20000.0,
            hysteresis_shape_A=1.0, hysteresis_shape_beta=0.5,
            hysteresis_shape_gamma=0.5, hysteresis_shape_n=1.0,
            hysteresis_scale_alpha=0.0, dt=TIME_STEP
        ))

    # 2. Run the interactive simulation
    run_interactive_simulation(robot_dyn_pinocchio, joint_models)


if __name__ == "__main__":
    main()