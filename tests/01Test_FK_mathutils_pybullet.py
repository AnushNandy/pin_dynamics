import pybullet as p
import numpy as np
import pybullet_data
import os
import sys
import time
import MathUtils

M_PI = np.pi

NOMINAL_KDL = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.37502, 0.0, 0.07664, 0.0, 0.0, 0.0,
               -0.025, 0.0, 0.1645, 0.0, M_PI / 2.0, 0.0,
               0.0, -0.1088, -0.045, 0.0, 0.0, -M_PI / 2.0,
               -0.25712, 0.045, 0.21612, 0.0, 0.0, M_PI / 2.0,
               0.092, 0.0, -0.052, 0.0, -M_PI / 2.0, -M_PI / 2.0,
               -0.052, 0.0, 0.13375, 0.0, M_PI / 2.0, -M_PI / 2.0,
               0.0, 0.0, 0.0987, 2.7951, 0.0, 0.0]

ORDERED_LINK_NAMES_KDL_CORRESPONDENCE = [
    "Base",
    "Link_0",
    "Link_1",
    "Link_2",
    "Link_3",
    "Link_4",
    "Link_5",
    "End_Effector"
]

ORDERED_JOINT_NAMES_URDF = [
    "Joint_0", "Joint_1", "Joint_2", "Joint_3", "Joint_4", "Joint_5", "Joint_6"
]

URDF_PATH = r"C:\dev\control-sw-tools\Dynamics_full\ArmModels\urdfs\P4\P4_Contra-Angle_right.urdf"

JOINT_STATES = [0.0, 1.54, 0.0, 0.0, 0.0, 0.0, 0.0]


def get_matrix_from_pose_pb(pose_pb):
    """Convert PyBullet pose (pos, quat) to a 4x4 transformation matrix."""
    pos, quat = pose_pb
    rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    transformation_matrix = np.eye(4)
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = pos
    return transformation_matrix


def draw_frame(frame_matrix, length=0.05, line_width=3, text=None, unique_id=-1):
    """Draws a coordinate frame in PyBullet."""
    origin = frame_matrix[0:3, 3]
    x_axis_tip = origin + frame_matrix[0:3, 0] * length
    y_axis_tip = origin + frame_matrix[0:3, 1] * length
    z_axis_tip = origin + frame_matrix[0:3, 2] * length

    item_ids = []
    item_ids.append(p.addUserDebugLine(origin, x_axis_tip, [1, 0, 0], lineWidth=line_width))
    item_ids.append(p.addUserDebugLine(origin, y_axis_tip, [0, 1, 0], lineWidth=line_width))
    item_ids.append(p.addUserDebugLine(origin, z_axis_tip, [0, 0, 1], lineWidth=line_width))

    if text:
        item_ids.append(p.addUserDebugText(text, origin + np.array([0, 0, 0.01]),
                                           textColorRGB=[0, 0, 0], textSize=1.0))
    return item_ids


def get_link_index_by_name_pb(robot_id, target_link_name):
    # Base link is special (-1)
    base_name = p.getBodyInfo(robot_id)[0].decode('UTF-8')
    if target_link_name == base_name or target_link_name == "Base":
        return -1  # Base link

    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        link_name = info[12].decode('UTF-8')
        if link_name == target_link_name:
            return i

    print(f"Warning: Link '{target_link_name}' not found in PyBullet model.")
    print("Available links:")
    print(f"  Base: {base_name}")
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        print(f"  Joint {i}: {info[12].decode('UTF-8')}")
    return None


def get_joint_index_by_name_pb(robot_id, target_joint_name):
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('UTF-8')
        if joint_name == target_joint_name:
            return i

    print(f"Warning: Joint '{target_joint_name}' not found in PyBullet model.")
    print("Available joints:")
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        print(f"  Joint {i}: {info[1].decode('UTF-8')}")
    return None

def hide_robot_visual(robot_id):
    p.changeVisualShape(robot_id, -1, rgbaColor=[0, 0, 0, 0])

    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        p.changeVisualShape(robot_id, i, rgbaColor=[0, 0, 0, 0.3])


def main():
    print("--- KDL Custom FK Calculation ---")
    print("Nominal KLDL - ", NOMINAL_KDL)
    T_ee_kdl, per_link_transforms_kdl = MathUtils.compute_forward_kinematics(NOMINAL_KDL, JOINT_STATES,
                                                                             True)
    # T_ee_kdl2,_ = MathUtils.compute_forward_kinematics(NOMINAL_KDL, JOINT_STATES, False)


    # print("============")
    # print("T_ee from KDL True : ", T_ee_kdl)
    # print("============")
    # print("============")
    # print("T_ee from KDL False : ", T_ee_kdl2)
    # print("============")

    if T_ee_kdl is None:
        print("Failed to compute KDL FK. Exiting.")
        return

    print("\nKDL FK Results (World Frame Transforms):")
    for i, link_name in enumerate(ORDERED_LINK_NAMES_KDL_CORRESPONDENCE):
        print(f"\n  {link_name} (from KDL) T_world_{link_name}:")
        print(np.round(per_link_transforms_kdl[i], 5))



    physicsClientId = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")
    p.changeVisualShape(planeId, -1, rgbaColor=[0.8, 0.8, 0.8, 0.3])

    robot_start_pos = [0, 0, 0]
    # robot_start_pos = [0.00821, -0.00252, 0.14078]
    robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])


    robot_id = p.loadURDF(URDF_PATH, robot_start_pos, robot_start_orientation, useFixedBase=True)

    hide_robot_visual(robot_id)

    print(f"\n--- PyBullet FK Calculation (Robot ID: {robot_id}) ---")

    pb_joint_indices = []
    for joint_name in ORDERED_JOINT_NAMES_URDF:
        idx = get_joint_index_by_name_pb(robot_id, joint_name)
        if idx is None:
            print(f"CRITICAL: Could not find joint '{joint_name}' in URDF. Exiting.")
            p.disconnect()
            return
        pb_joint_indices.append(idx)

    # Set joint states in PyBullet
    print(f"\nSetting PyBullet joint states for joints {ORDERED_JOINT_NAMES_URDF} to: {JOINT_STATES}")
    for i, joint_idx_pb in enumerate(pb_joint_indices):
        p.resetJointState(robot_id, joint_idx_pb, JOINT_STATES[i])

    # Get link poses from PyBullet
    pybullet_link_poses_matrices = []
    print("\nPyBullet FK Results (World Frame Transforms):")

    for i, link_name_kdl_corr in enumerate(ORDERED_LINK_NAMES_KDL_CORRESPONDENCE):
        pb_link_idx = get_link_index_by_name_pb(robot_id, link_name_kdl_corr)
        if pb_link_idx is None:
            print(f"CRITICAL: Could not find link '{link_name_kdl_corr}' for PyBullet FK. Exiting.")
            p.disconnect()
            return

        if pb_link_idx == -1:
            pos, orn = p.getBasePositionAndOrientation(robot_id)
        else:  # Other links
            link_state = p.getLinkState(robot_id, pb_link_idx, computeForwardKinematics=1)
            pos, orn = link_state[4], link_state[5]

        T_world_link_pb = get_matrix_from_pose_pb((pos, orn))
        pybullet_link_poses_matrices.append(T_world_link_pb)

        print(f"\n  {link_name_kdl_corr} (from PyBullet) T_world_{link_name_kdl_corr}:")
        print(np.round(T_world_link_pb, 5))

        # Draw frames
        # PyBullet frame in blue (slightly thicker)
        draw_frame(T_world_link_pb, length=0.05, line_width=5, text=f"PB_{link_name_kdl_corr}")
        # KDL frame in red
        draw_frame(per_link_transforms_kdl[i], length=0.055, line_width=2, text=f"KDL_{link_name_kdl_corr}")

        # Numerical comparison
        diff_matrix = np.abs(T_world_link_pb - per_link_transforms_kdl[i])
        max_diff = np.max(diff_matrix)
        avg_diff = np.mean(diff_matrix)
        print(f"    Max abs difference for {link_name_kdl_corr}: {max_diff:.6f}, Avg abs difference: {avg_diff:.6f}")

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