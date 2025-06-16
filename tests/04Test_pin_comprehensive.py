import pybullet as p
import pybullet_data
import numpy as np
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics 
from src.dynamics.friction_transmission import JointModel
from config import robot_config 

URDF_PATH = r"/home/robot/dev/dyn/ArmModels/urdfs/P4/P4_Contra-Angle_right.urdf"
IDENTIFIED_PARAMS_PATH = r"/home/robot/dev/dyn/src/systemid/identified_params.npz"

# Options: "DISTURBANCE", "CONSTANT_VELOCITY", "HIGH_FREQUENCY"
TEST_TO_RUN = "CONSTANT_VELOCITY" 

TIME_STEP = 1. / 240.
GRAVITY_VECTOR = np.array([0, 0, -9.81])
MAX_TORQUES = np.array([200.0, 200.0, 150.0, 150.0, 100.0, 80.0, 80.0])

KP = np.array([50.0, 60.0, 50.0, 40.0, 25.0, 20.0, 15.0]) 
KD = np.array([4.0,  5.0,  4.0,  3.0,  2.0,  1.5,  1.0])

_pybullet_connected = False

def setup_simulation(reconnect=False):
    """Connects to PyBullet, loads robot, and returns IDs."""
    global _pybullet_connected
    
    if not _pybullet_connected:
        p.connect(p.GUI)
        _pybullet_connected = True
    elif reconnect:
        p.resetSimulation()
    
    p.setTimeStep(TIME_STEP)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(*GRAVITY_VECTOR)
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
    actuated_joint_indices = get_joint_indices_by_name(robot_id, robot_config.ACTUATED_JOINT_NAMES)

    for i in range(p.getNumJoints(robot_id)):
        if i not in actuated_joint_indices:
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, force=0)

    return robot_id, actuated_joint_indices

def plot_results(time_log, data_logs, title, y_label):
    """Generic plotting function."""
    num_plots = len(data_logs[next(iter(data_logs))][0])
    plt.figure(figsize=(16, 4 * num_plots))
    for i in range(num_plots):
        plt.subplot(num_plots, 1, i + 1)
        for label, data in data_logs.items():
            plt.plot(time_log, data[:, i], label=label)
        plt.title(f'{title} for Joint {i}')
        plt.ylabel(y_label)
        plt.grid(True)
        plt.legend()
    plt.xlabel('Time (s)')
    plt.show()

def get_joint_indices_by_name(robot_id, joint_names):
    """Helper function to get PyBullet joint indices from a list of names."""
    joint_map = {p.getJointInfo(robot_id, i)[1].decode('UTF-8'): i for i in range(p.getNumJoints(robot_id))}
    return [joint_map[name] for name in joint_names]

def generate_constant_velocity_traj(t, joint_idx, duration=4.0):
    """Generates a triangular wave for position, resulting in a square wave for velocity."""
    q_des = np.zeros(robot_config.NUM_JOINTS)
    qd_des = np.zeros(robot_config.NUM_JOINTS)
    
    cycle_time = t % duration
    velocity = 1.0 / (duration / 2.0)
    
    if cycle_time < duration / 2.0:
        q_des[joint_idx] = velocity * cycle_time
        qd_des[joint_idx] = velocity
    else:
        q_des[joint_idx] = velocity * (duration - cycle_time)
        qd_des[joint_idx] = -velocity
        
    return q_des, qd_des, np.zeros_like(q_des)

def generate_figure_eight_traj(t, robot_id, ee_link_index):
    """Generates a dynamic figure-eight trajectory for the end-effector."""
    center_pos = [0.4, 0.0, 0.5]
    axis1_len = 0.15
    axis2_len = 0.1
    omega = 2 * np.pi / 5.0 # 5 seconds per loop
    
    x = center_pos[0] + axis1_len * np.sin(omega * t)
    y = center_pos[1] + axis2_len * np.sin(2 * omega * t)
    z = center_pos[2]
    target_pos = [x, y, z]
    q_des_t = np.array(p.calculateInverseKinematics(robot_id, ee_link_index, target_pos))
    
    return q_des_t[:7], np.zeros(7), np.zeros(7)

def run_test_loop(robot_id, joint_indices, duration, dyn_model, joint_models, traj_generator, control_mode, external_force=None):
    """A generic loop for running a simulation test."""
    logs = {'t': [], 'err': [], 'tau_ff': [], 'tau_fb': []}
    ee_link_index = joint_indices[-1]
    
    for t_step in range(int(duration / TIME_STEP)):
        t = t_step * TIME_STEP
        
        # Apply external force if specified
        if external_force and external_force['start'] <= t < external_force['end']:
            p.applyExternalForce(robot_id, external_force['link'], external_force['force'], 
                                 [0,0,0], p.WORLD_FRAME)
        
        q_des, qd_des, qdd_des = traj_generator(t, robot_id, ee_link_index)
        
        joint_states = p.getJointStates(robot_id, joint_indices)
        q_actual = np.array([s[0] for s in joint_states])
        qd_actual = np.array([s[1] for s in joint_states])

        # Controller
        tau_fb = KP * (q_des - q_actual) + KD * (qd_des - qd_actual)
        tau_ff = np.zeros_like(tau_fb)
        
        if control_mode == 'PD_GRAVITY':
            tau_ff = dyn_model.compute_rnea(q_des, np.zeros_like(qd_des), np.zeros_like(qdd_des))
        elif control_mode == 'PD_CTC':
            tau_rnea = dyn_model.compute_rnea(q_des, qd_des, qdd_des)
            for i in range(robot_config.NUM_JOINTS):
                tau_ff[i] = joint_models[i].compute_feedforward_torque(
                    tau_rnea[i], q_des[i], qd_des[i], qdd_des[i]
                )

        tau_total = np.clip(tau_ff + tau_fb, -MAX_TORQUES, MAX_TORQUES)
        p.setJointMotorControlArray(robot_id, joint_indices, p.TORQUE_CONTROL, forces=tau_total)
        p.stepSimulation()

        # Logging
        logs['t'].append(t)
        logs['err'].append(np.rad2deg(q_des - q_actual))
        logs['tau_ff'].append(tau_ff)
        logs['tau_fb'].append(tau_fb)

    return {key: np.array(val) for key, val in logs.items()}


def main():
    if not os.path.exists(IDENTIFIED_PARAMS_PATH) or not os.path.exists(URDF_PATH):
        print("FATAL: URDF or Identified Parameters file not found. Check paths.")
        return

    # 1. Load Models
    P_identified = np.load(IDENTIFIED_PARAMS_PATH)['P']
    dyn_model = PinocchioRobotDynamics(URDF_PATH)
    dyn_model.set_parameters_from_vector(P_identified)
    
    joint_models = [JointModel(gear_ratio=100.0, motor_inertia=0.0001, coulomb_pos=2.5, coulomb_neg=-2.2,
                                stiction_pos=3.0, stiction_neg=-2.7, viscous_coeff=0.15, stribeck_vel_pos=0.1,
                                stribeck_vel_neg=-0.1, stiffness=20000, hysteresis_shape_A=1.0, hysteresis_shape_beta=0.5,
                                hysteresis_shape_gamma=0.5, hysteresis_shape_n=1.0, hysteresis_scale_alpha=0.0, dt=TIME_STEP)
                    for _ in range(robot_config.NUM_JOINTS)]

    robot_id, joint_indices = setup_simulation()

    if TEST_TO_RUN == "DISTURBANCE":
        print("Running Disturbance Rejection Test...")
        force = {'link': 6, 'force': [0, 500, 0], 'start': 2.0, 'end': 2.2}
        logs = run_test_loop(robot_id, joint_indices, 5.0, dyn_model, joint_models, 
                             lambda t, rid, ee_idx: (np.zeros(7), np.zeros(7), np.zeros(7)), 'PD_CTC', external_force=force)
        p.disconnect()
        plot_results(logs['t'], {'Tracking Error': logs['err']}, 'Disturbance Rejection Error', 'Error (deg)')
        plot_results(logs['t'], {'Feedback Torque': logs['tau_fb'], 'Feedforward Torque': logs['tau_ff']}, 
                     'Disturbance Rejection Torques', 'Torque (Nm)')

    elif TEST_TO_RUN == "CONSTANT_VELOCITY":
        print("Running Constant Velocity Friction Test for Joint 3...")
        logs = run_test_loop(robot_id, joint_indices, 8.0, dyn_model, joint_models, 
                             lambda t, rid, ee_idx: generate_constant_velocity_traj(t, 3), 'PD_CTC')
        p.disconnect()
        plot_results(logs['t'], {'Tracking Error': logs['err']}, 'Constant Velocity Error', 'Error (deg)')
        plot_results(logs['t'], {'Feedback Torque': logs['tau_fb'], 'Feedforward Torque': logs['tau_ff']}, 
                     'Constant Velocity Torques', 'Torque (Nm)')

    elif TEST_TO_RUN == "HIGH_FREQUENCY":
        print("Running High-Frequency Dynamics Test...")
        traj_gen = lambda t, rid, ee_idx: generate_figure_eight_traj(t, rid, ee_idx)
        
        logs_pd = run_test_loop(robot_id, joint_indices, 10.0, dyn_model, joint_models, traj_gen, 'PD_ONLY')
        robot_id, joint_indices = setup_simulation(reconnect=True)
        logs_g = run_test_loop(robot_id, joint_indices, 10.0, dyn_model, joint_models, traj_gen, 'PD_GRAVITY')
        
        robot_id, joint_indices = setup_simulation(reconnect=True)
        logs_ctc = run_test_loop(robot_id, joint_indices, 10.0, dyn_model, joint_models, traj_gen, 'PD_CTC')
        
        p.disconnect()
        
        plot_results(logs_pd['t'], {'PD Only': logs_pd['err'], 'PD+Gravity': logs_g['err'], 'Full CTC': logs_ctc['err']},
                     'High-Frequency Tracking Comparison', 'Error (deg)')

if __name__ == "__main__":
    main()