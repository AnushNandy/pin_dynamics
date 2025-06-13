import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

URDF_FILE_NAME = r"P4 Guide Arm Contra Handpiece.urdf"
URDF_FILE_PATH = os.path.join(PROJECT_ROOT, r"P4 Guide Arm Contra Handpiece\urdf", URDF_FILE_NAME)

PYBULLET_GUI = True
GRAVITY = [0, 0, -9.81]
TIME_STEP = 1.0/240.0

END_EFFECTOR_LINK_NAME = "Link7"
NUM_ACTUATED_JOINTS = 7

