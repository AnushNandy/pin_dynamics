import numpy as np
import time, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import robot_config
# from Dynamics_full.src.systemid.dynamics_regressor import RegressorBuilder
from pinocchio_friction_regressor import PinocchioAndFrictionRegressorBuilder

DATA_PATH = "/home/robot/dev/dyn/src/systemid/sysid_data.npz"
SAVE_PATH = "/home/robot/dev/dyn/src/systemid/identified_params.npz"
L2_REGULARIZATION = 1e-6 #small regularization to stabilize solution!!


def main():
    """
    Perform system identification using the collected data and the numerical regressor.
    """
    print("--- Starting System Identification ---")

    data = np.load(DATA_PATH)
    q_data = data['q']
    qd_data = data['qd']
    qdd_data = data['qdd']
    tau_data = data['tau']
    num_samples = len(q_data)
    print(f"Loaded {num_samples} samples from '{DATA_PATH}'")

    # 2. Initialize the regressor builder
    regressor_builder = PinocchioAndFrictionRegressorBuilder(robot_config)
    num_joints = regressor_builder.num_joints
    num_params = regressor_builder.total_params

    # 3. Construct the stacked regressor matrix (Y_stack) and torque vector (tau_stack)
    # Y_stack will have shape (num_samples * num_joints, num_params)
    # tau_stack will have shape (num_samples * num_joints, 1)
    Y_stack = np.zeros((num_samples * num_joints, num_params))
    tau_stack = np.zeros((num_samples * num_joints, 1))

    print("Building the stacked regressor matrix... This may take a moment.")
    start_time = time.time()

    for i in range(num_samples):
        q = q_data[i]
        qd = qd_data[i]
        qdd = qdd_data[i]
        tau = tau_data[i]

        # Compute the regressor for this time step
        Y_i = regressor_builder.compute_regressor_matrix(q, qd, qdd)

        # Place it in the correct block of the stacked matrix
        row_start = i * num_joints
        row_end = row_start + num_joints
        Y_stack[row_start:row_end, :] = Y_i
        tau_stack[row_start:row_end, 0] = tau

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_samples} samples...")

    duration = time.time() - start_time
    print(f"Regressor matrix built in {duration:.2f} seconds.")

    # 4. Solve the linear least-squares problem: Y_stack * P = tau_stack
    print("Solving for dynamic parameters using least-squares...")

    # Solve for P in min || Y*P - T ||^2 + alpha * ||P||^2 (Tikhonov Regularization)
    # == (Y.T*Y + alpha*I) * P = Y.T*T
    Y_T_Y = Y_stack.T @ Y_stack
    Y_T_tau = Y_stack.T @ tau_stack

    # Add regularization to the diagonal
    regularization_matrix = L2_REGULARIZATION * np.eye(num_params)

    # Solve the regularized system
    identified_params = np.linalg.solve(Y_T_Y + regularization_matrix, Y_T_tau)
    identified_params = identified_params.flatten()

    print("--- Identification Complete ---")

    # 5. Save the identified parameters
    np.savez(SAVE_PATH, P=identified_params)
    print(f"Identified parameters saved to '{SAVE_PATH}'")

    print("\nIdentified Friction Coefficients (Viscous, Coulomb):")
    friction_params = identified_params[regressor_builder.total_link_params:]
    for i in range(num_joints):
        fv = friction_params[i * 2]
        fc = friction_params[i * 2 + 1]
        print(f"  Joint {i}: Fv = {fv:.4f}, Fc = {fc:.4f}")


if __name__ == "__main__":
    main()