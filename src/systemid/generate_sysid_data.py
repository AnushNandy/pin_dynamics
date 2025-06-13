import pybullet as p
import numpy as np
import time
import pybullet_data
import os
from Dynamics_full.config import robot_config

URDF_PATH = r"C:\dev\control-sw-tools\Dynamics_full\ArmModels\urdfs\P4\P4_Contra-Angle_right.urdf"
SAVE_PATH = "sysid_data.npz"
SIM_DURATION = 20.0  # Longer duration for more data
TIME_STEP = 1. / 240.
NUM_JOINTS = robot_config.NUM_JOINTS


def generate_fourier_series_trajectory(t, num_harmonics=5):
    """
    Generat exciting trajectory using a sum of sinusoids for each joint.
    Crucial for system identification to ensure all dynamic effects are present.
    """
    q_des = np.zeros(NUM_JOINTS)
    qd_des = np.zeros(NUM_JOINTS)
    qdd_des = np.zeros(NUM_JOINTS)

    # Base frequency
    w = 2 * np.pi * 0.1

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

    if not os.path.exists(URDF_PATH):
        print(f"CRITICAL ERROR: URDF file not found at '{URDF_PATH}'")
        return

    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)

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

        log_q.append(q_des)
        log_qd.append(qd_des)
        log_qdd.append(qdd_des)
        log_tau.append(tau_ground_truth)

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