import pybullet as p
import numpy as np
import time
import pybullet_data
import time, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import robot_config

URDF_PATH = r"/home/robot/dev/dyn/ArmModels/urdfs/P4/P4_Contra-Angle_right.urdf"
SAVE_PATH = "/home/robot/dev/dyn/src/systemid/sysid_data.npz"
SIM_DURATION = 50.0  # Longer duration for more data
TIME_STEP = 1. / 240.
NUM_JOINTS = robot_config.NUM_JOINTS


def generate_fourier_series_trajectory(t, num_harmonics=6):
    """
    Generat exciting trajectory using a sum of sinusoids for each joint.
    Crucial for system identification to ensure all dynamic effects are present.
    """
    q_des = np.zeros(NUM_JOINTS)
    qd_des = np.zeros(NUM_JOINTS)
    qdd_des = np.zeros(NUM_JOINTS)

    # Base frequency
    w = 2 * np.pi * 0.5

    for i in range(NUM_JOINTS):
        a_n = [(0.5 / (n + 1)) for n in range(num_harmonics)]
        b_n = [(0.5 / (n + 1)) for n in range(num_harmonics)]

        for n in range(1, num_harmonics + 1):
            # Add a phase shift per joint and harmonic to prevent synced movements
            phase_shift = (i * np.pi / 3) + (n * np.pi / 4)

            # Position
            q_des[i] += a_n[n - 1] * np.sin(n * w * t + phase_shift) + b_n[n - 1] * np.cos(n * w * t + phase_shift)

            # Velocity
            qd_des[i] += n * w * (
                        a_n[n - 1] * np.cos(n * w * t + phase_shift) - b_n[n - 1] * np.sin(n * w * t + phase_shift))

            # Acceleration
            qdd_des[i] += (n * w) ** 2 * (
                        -a_n[n - 1] * np.sin(n * w * t + phase_shift) - b_n[n - 1] * np.cos(n * w * t + phase_shift))

    return q_des, qd_des, qdd_des


def main():
    """
    Generates and saves ground-truth system identification data using PyBullet.
    """
    physicsClientId = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    friction_coeffs = {
        'viscous': [0.1, 0.15, 0.08, 0.12, 0.09, 0.11, 0.06],
        'coulomb': [0.05, 0.08, 0.04, 0.06, 0.05, 0.07, 0.03]
    }

    if not os.path.exists(URDF_PATH):
        print(f"CRITICAL ERROR: URDF file not found at '{URDF_PATH}'")
        return

    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
    # robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBas)

    log_q, log_qd, log_qdd, log_tau = [], [], [], []

    print("--- Generating System Identification Data ---")

    for t in np.arange(0, SIM_DURATION, TIME_STEP):
        q_des, qd_des, qdd_des = generate_fourier_series_trajectory(t)

        tau_ground_truth = p.calculateInverseDynamics(
            bodyUniqueId=robot_id,
            objPositions=q_des.tolist(),
            objVelocities=qd_des.tolist(),
            objAccelerations=qdd_des.tolist()
        )

        tau_friction = np.zeros(NUM_JOINTS)
        for i in range(NUM_JOINTS):
            tau_friction[i] = (friction_coeffs['viscous'][i] * qd_des[i] + 
                             friction_coeffs['coulomb'][i] * np.sign(qd_des[i]))
            
        tau_total = np.array(tau_ground_truth) + tau_friction

        log_q.append(q_des)
        log_qd.append(qd_des)
        log_qdd.append(qdd_des)
        log_tau.append(tau_total)

        # Progress indicator
        if int(t) % 5 == 0 and abs(t - int(t)) < TIME_STEP / 2:
            print(f"Generated data up to {int(t)}s / {int(SIM_DURATION)}s")

    p.disconnect()

    np.savez(
        SAVE_PATH,
        q=np.array(log_q),
        qd=np.array(log_qd),
        qdd=np.array(log_qdd),
        tau=np.array(log_tau)
    )

    print(f"--- Data generation complete. ---")
    print(f"Saved {len(log_q)} samples to '{SAVE_PATH}'")


if __name__ == "__main__":
    main()