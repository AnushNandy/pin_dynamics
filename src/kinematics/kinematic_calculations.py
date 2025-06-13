import numpy as np
import MathUtils
from Dynamics_full.config import robot_config

def compute_geometric_jacobian(kdl_chain_params, joint_angles, neocis_convention_fk = False):
    """
    computes geometric jacobian for our serial manip
    assumes all joints are revolute and axes were [0,0,1] in KDL setup
    Args:
        kdl_chain_params (list): The KDL chain for MathUtils.compute_forward_kinematics.
                                 (num_joints + 1) entries of [Tx,Ty,Tz,Rz,Ry,Rx].
        joint_angles (list): Current joint angles (radians). Length num_joints.

    Returns:
        np.ndarray: The 6xN geometric Jacobian matrix (N = num_joints).
                    Top 3 rows are Jv (linear velocity), bottom 3 are Jw (angular velocity).
    """

    num_joints = len(joint_angles)
    # if len(kdl_chain_params) != num_joints+ 1:
    #     raise ValueError("KDL chain length must be #Joints + 1")

    T_ee, per_joint_transforms = MathUtils.compute_forward_kinematics(kdl_chain_params, joint_angles, neocis_convention = neocis_convention_fk)

    p_E = T_ee[:3, 3] #position

    jacobian = np.zeros((6, num_joints))

    for j in range(num_joints):
        T_0_j = per_joint_transforms[j]
        z_j = T_0_j[:3, 2]
        p_j = T_0_j[:3, 3]
        # For a revolute joint:
        # Jv_j = z_j x (p_E - p_j)
        # Jw_j = z_j
        # Linear velocity component
        jacobian[:3, j] = np.cross(z_j, (p_E - p_j))
        # Angular velocity component
        jacobian[3:, j] = z_j

    return jacobian

if __name__ == '__main__':
    KDL_CHAIN_TEST = robot_config.KDL_CHAIN
    NUM_JOINTS_TEST = robot_config.NUM_JOINTS
    print("Loaded KDL_CHAIN and NUM_JOINTS from robot_config.py")
    print("KDL and Joints: ", len(KDL_CHAIN_TEST)," & " , NUM_JOINTS_TEST)

    test_joint_angles = [0.1] * NUM_JOINTS_TEST # Small non-zero angles
    # test_joint_angles = [np.pi/4, np.pi/4]


    print(f"KDL for test ({len(KDL_CHAIN_TEST)//6 -1} DOF): {KDL_CHAIN_TEST}")
    print(f"Test Joint Angles: {test_joint_angles}")

    neocis_fk_convention_for_jacobian = True
    print(f"Using neocis_convention={neocis_fk_convention_for_jacobian} for FK in Jacobian calculation.")

    jacobian_matrix = compute_geometric_jacobian(
        KDL_CHAIN_TEST,
        test_joint_angles,
        neocis_convention_fk=neocis_fk_convention_for_jacobian
    )
    print("\nComputed Jacobian Matrix (Jv top, Jw bottom):")
    print(np.round(jacobian_matrix, 5))