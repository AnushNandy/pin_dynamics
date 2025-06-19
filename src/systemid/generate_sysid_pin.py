# In a new file: generate_sysid_data_pinocchio.py

import numpy as np
import time, os, sys
# Make sure the paths are correct for your project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import robot_config
# Import your Pinocchio dynamics class
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics
from src.systemid.pinocchio_friction_regressor import smooth_sign

# Constants
URDF_PATH = robot_config.URDF_PATH
SAVE_PATH = "/home/robot/dev/dyn/src/systemid/sysid_data_pinocchio.npz"
SIM_DURATION = 100.0
TIME_STEP = 1. / 240.
NUM_JOINTS = robot_config.NUM_JOINTS

def generate_fourier_series_trajectory(t, num_harmonics=6):
    """
    Generates an exciting trajectory using a sum of sinusoids for each joint.
    (This function is excellent, no changes needed)
    """
    q_des = np.zeros(NUM_JOINTS)
    qd_des = np.zeros(NUM_JOINTS)
    qdd_des = np.zeros(NUM_JOINTS)
    w = 2 * np.pi * 0.5

    for i in range(NUM_JOINTS):
        # Use different amplitudes/phases per joint to ensure rich, decoupled motion
        np.random.seed(i) # Ensure a, b, phase are consistent across runs for joint i
        a_n = np.random.uniform(0.05, 0.2, num_harmonics)
        b_n = np.random.uniform(0.05, 0.2, num_harmonics)
        phase_shifts = np.random.uniform(0, 2 * np.pi, num_harmonics)

        for n in range(num_harmonics):
            angle = (n + 1) * w * t + phase_shifts[n]
            q_des[i] += a_n[n] * np.sin(angle) + b_n[n] * np.cos(angle)
            qd_des[i] += (n + 1) * w * (a_n[n] * np.cos(angle) - b_n[n] * np.sin(angle))
            qdd_des[i] += -((n + 1) * w)**2 * (a_n[n] * np.sin(angle) + b_n[n] * np.cos(angle))
            
    return q_des, qd_des, qdd_des

def main():
    """
    Generates and saves ground-truth system identification data using PINOCCHIO.
    This ensures the data is consistent with the regressor model.
    """
    print("--- Generating System Identification Data using Pinocchio ---")

    # 1. Initialize the dynamics model from the URDF.
    # The parameters in the URDF will be our "ground truth" for the rigid body dynamics.
    dynamics_model = PinocchioRobotDynamics(URDF_PATH)

    # 2. Define the "ground-truth" friction coefficients that we want to identify.
    # These should be POSITIVE.
    true_friction_coeffs = {
        'viscous': [0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06],
        'coulomb': [0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
    }
    print("Ground Truth Friction Params (Fv, Fc):")
    for i in range(NUM_JOINTS):
        print(f"  Joint {i}: Fv={true_friction_coeffs['viscous'][i]:.2f}, Fc={true_friction_coeffs['coulomb'][i]:.2f}")


    log_q, log_qd, log_qdd, log_tau = [], [], [], []

    for t in np.arange(0, SIM_DURATION, TIME_STEP):
        q_des, qd_des, qdd_des = generate_fourier_series_trajectory(t)

        # 3. Calculate ground-truth torques using the Pinocchio model (RNEA).
        # This uses the floating-base model, matching the regressor.
        tau_rnea = dynamics_model.compute_rnea(q_des, qd_des, qdd_des)

        # 4. Add the ground-truth friction torques.
        tau_friction = np.zeros(NUM_JOINTS)
        for i in range(NUM_JOINTS):
            tau_friction[i] = (true_friction_coeffs['viscous'][i] * qd_des[i] +
                             true_friction_coeffs['coulomb'][i] * smooth_sign(qd_des[i]))

        # The total torque is the sum of RNEA and friction.
        tau_total = tau_rnea + tau_friction

        # 5. Log the data
        log_q.append(q_des)
        log_qd.append(qd_des)
        log_qdd.append(qdd_des)
        log_tau.append(tau_total)

        if int(t) % 5 == 0 and abs(t - int(t)) < TIME_STEP / 2:
            print(f"Generated data up to {int(t)}s / {int(SIM_DURATION)}s")

    # 6. Save the new, consistent dataset.
    np.savez(
        SAVE_PATH,
        q=np.array(log_q),
        qd=np.array(log_qd),
        qdd=np.array(log_qdd),
        tau=np.array(log_tau)
    )

    print(f"--- Data generation complete. ---")
    print(f"Saved {len(log_q)} consistent samples to '{SAVE_PATH}'")


if __name__ == "__main__":
    main()