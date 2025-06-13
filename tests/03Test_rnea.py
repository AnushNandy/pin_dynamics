import pybullet as p
import numpy as np
import pybullet_data
import time
import os

import MathUtils
from Dynamics_full.config import robot_config
from Dynamics_full.src.dynamics.rnea import RobotDynamics

# Use the same configuration as the FK test
JOINT_STATES = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
URDF_PATH = r"C:\dev\control-sw-tools\Dynamics_full\ArmModels\urdfs\P4\P4_Contra-Angle_right.urdf"


def draw_frame(frame_matrix, length=0.05, line_width=3, text=None):
    """Draws a coordinate frame in PyBullet."""
    origin = frame_matrix[0:3, 3]
    x_axis = origin + frame_matrix[0:3, 0] * length
    y_axis = origin + frame_matrix[0:3, 1] * length
    z_axis = origin + frame_matrix[0:3, 2] * length
    p.addUserDebugLine(origin, x_axis, [1, 0, 0], lineWidth=line_width)
    p.addUserDebugLine(origin, y_axis, [0, 1, 0], lineWidth=line_width)
    p.addUserDebugLine(origin, z_axis, [0, 0, 1], lineWidth=line_width)
    if text:
        p.addUserDebugText(text, origin + np.array([0, 0, 0.01]), textColorRGB=[0, 0, 0], textSize=0.8)


def get_joint_indices_by_name(robot_id, joint_names):
    """Gets a list of joint indices from a list of joint names."""
    joint_indices = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        if info[1].decode('UTF-8') in joint_names:
            joint_indices.append(i)
    # Ensure the order matches the input joint_names
    ordered_indices = [None] * len(joint_names)
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode('UTF-8')
        if name in joint_names:
            idx = joint_names.index(name)
            ordered_indices[idx] = i
    return ordered_indices


def main():
    """Main execution function."""
    # --- 1. RNEA Calculation ---
    print("--- RNEA Inverse Dynamics Calculation ---")
    robot_dyn = RobotDynamics(robot_config)

    # Define a static test case (zero velocity and acceleration)
    q = JOINT_STATES
    qd = np.zeros(robot_config.NUM_JOINTS)
    qdd = np.zeros(robot_config.NUM_JOINTS)
    gravity = np.array([0, 0, -9.81])

    print(f"Calculating gravity compensation torques for joint state:\n q = {q}")

    # Compute the torques
    gravity_torques = robot_dyn.compute_rnea(q, qd, qdd, gravity)

    print("\n--- Computed Gravity Compensation Torques (Nm) ---")
    for i, name in enumerate(robot_config.ACTUATED_JOINT_NAMES):
        print(f"  {name}: {gravity_torques[i]:.4f}")

    # --- 2. PyBullet Visualization ---
    physicsClientId = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)  # Turn off PyBullet gravity to see our own dynamics
    p.loadURDF("plane.urdf")

    if not os.path.exists(URDF_PATH):
        print(f"CRITICAL ERROR: URDF file not found at '{URDF_PATH}'")
        p.disconnect()
        return

    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)

    # Make robot semi-transparent to better see the frames
    for i in range(p.getNumJoints(robot_id)):
        p.changeVisualShape(robot_id, i, rgbaColor=[0.5, 0.5, 0.5, 0.5])
    p.changeVisualShape(robot_id, -1, rgbaColor=[0.5, 0.5, 0.5, 0.5])

    # Get joint indices in the correct order
    pb_joint_indices = get_joint_indices_by_name(robot_id, robot_config.ACTUATED_JOINT_NAMES)

    # Set the robot's pose in PyBullet
    for i, joint_idx in enumerate(pb_joint_indices):
        p.resetJointState(robot_id, joint_idx, JOINT_STATES[i])

    # --- 3. Draw KDL Frames for Verification ---
    print("\nVisualizing KDL-based frames in PyBullet...")
    # We must use our trusted FK, not PyBullet's, for drawing frames
    _, T_world_links_kdl = MathUtils.compute_forward_kinematics(
        robot_config.KDL_CHAIN, JOINT_STATES, neocis_convention=True
    )

    for i, transform in enumerate(T_world_links_kdl):
        link_name = robot_config.LINK_NAMES_IN_KDL_ORDER[i]
        draw_frame(transform, text=f"KDL_{link_name}")

    print("\nSimulation running. Press Ctrl+C in the terminal to exit.")
    try:
        while True:
            p.stepSimulation()
            time.sleep(1. / 240.)
    except KeyboardInterrupt:
        print("Exiting.")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()