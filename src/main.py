import numpy as np
import os
import sys
dynamics_full_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if dynamics_full_root not in sys.path:
    sys.path.insert(0, dynamics_full_root)

from config import robot_config
from src.kinematics import kinematic_calculations
