import numpy as np
import time, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import robot_config
from pinocchio_friction_regressor import PinocchioAndFrictionRegressorBuilder

DATA_PATH = "/home/robot/dev/dyn/src/systemid/sysid_data.npz"
SAVE_PATH = "/home/robot/dev/dyn/src/systemid/identified_params.npz"
L2_REGULARIZATION = 1e-6

def main():
    """
    Perform system identification using the collected data and the Pinocchio regressor.
    """
    print("--- Starting System Identification ---")

    # 1. Load the data
    data = np.load(DATA_PATH)
    q_data = data['q']
    qd_data = data['qd']
    qdd_data = data['qdd']
    tau_data = data['tau']
    num_samples = len(q_data)
    print(f"Loaded {num_samples} samples from '{DATA_PATH}'")

    # 2. Initialize the Pinocchio regressor builder
    # Assuming robot_config has URDF_PATH attribute
    urdf_path = robot_config.URDF_PATH
    regressor_builder = PinocchioAndFrictionRegressorBuilder(urdf_path)
    
    num_joints = regressor_builder.num_joints
    num_params = regressor_builder.total_params
    print(f"Robot has {num_joints} joints, {num_params} total parameters to identify")

    # 3. Build stacked regressor matrix and torque vector
    Y_stack = np.zeros((num_samples * num_joints, num_params))
    tau_stack = np.zeros((num_samples * num_joints, 1))

    print("Building the stacked regressor matrix...")
    start_time = time.time()

    for i in range(num_samples):
        q = q_data[i]
        qd = qd_data[i]
        qdd = qdd_data[i]
        tau = tau_data[i]

        # Compute regressor matrix for this sample
        Y_i = regressor_builder.compute_regressor_matrix(q, qd, qdd)

        # Stack into the big matrix
        row_start = i * num_joints
        row_end = row_start + num_joints
        Y_stack[row_start:row_end, :] = Y_i
        tau_stack[row_start:row_end, 0] = tau

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_samples} samples...")

    duration = time.time() - start_time
    print(f"Regressor matrix built in {duration:.2f} seconds.")

    # 4. Solve least squares with regularization
    print("Solving for dynamic parameters...")
    
    Y_T_Y = Y_stack.T @ Y_stack
    Y_T_tau = Y_stack.T @ tau_stack
    regularization_matrix = L2_REGULARIZATION * np.eye(num_params)
    
    identified_params = np.linalg.solve(Y_T_Y + regularization_matrix, Y_T_tau)
    identified_params = identified_params.flatten()

    print("--- Identification Complete ---")

    # 5. Save results
    np.savez(SAVE_PATH, P=identified_params)
    print(f"Identified parameters saved to '{SAVE_PATH}'")

    # 6. Display friction coefficients
    print("\nIdentified Friction Coefficients (Viscous, Coulomb):")
    friction_params = identified_params[regressor_builder.total_link_params:]
    for i in range(num_joints):
        fv = friction_params[i * 2]
        fc = friction_params[i * 2 + 1]
        print(f"  Joint {i}: Fv = {fv:.4f}, Fc = {fc:.4f}")

if __name__ == "__main__":
    main()