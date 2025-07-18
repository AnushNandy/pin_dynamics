import numpy as np

ROBOT_NAME = "3DOF"
URDF_PATH = r"/home/robot/dev/dyn/3DOF/urdf/3DOF Yomi Assembly_Anush.urdf"
MAX_TORQUES = np.array([140, 51, 51])

LINK_NAMES_IN_KDL_ORDER = [
    "world",
    "base_link",
    "Link1",
    "Link2",
    "Link3",
    "Link4",
    "End_effector"
]

ACTUATED_JOINT_NAMES = [
    "Joint1", "Joint2", "Joint3"
]

NUM_JOINTS = len(ACTUATED_JOINT_NAMES)

END_EFFECTOR_FRAME_NAME = "End_effector"