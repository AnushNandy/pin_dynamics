import numpy as np
import time, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import robot_config
from pinocchio_friction_regressor import PinocchioAndFrictionRegressorBuilder
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# DATA_PATH = "/home/robot/dev/dyn/src/systemid/sysid_data_pybullet.npz"
DATA_PATH = "/home/robot/dev/dyn/src/systemid/system_id_data_3joint_final.npz"
SAVE_PATH = "/home/robot/dev/dyn/src/systemid/identified_params_pybullet.npz"
L2_REGULARIZATION = 1e-3
TIME_STEP_FOR_PLOTTING = 0.02
# TIME_STEP_FOR_PLOTTING = 1/240.0 
SKIP_TIME_SECONDS = 10.0

def analyze_regressor(Y_stack):
    """Computes and prints diagnostics for a regressor matrix."""
    # print(f"\n--- Analysis for {name} ---")
    
    # 1. Condition Number
    cond_num = np.linalg.cond(Y_stack)
    print(f"Condition Number: {cond_num:,.2e}")

    # 2. Singular Values
    print("Computing SVD...")
    U, s, Vh = np.linalg.svd(Y_stack, full_matrices=False)
    print("SVD computed.")
        
    return s

def plot_sysid_data(q, qd, qdd, tau, time_step, title="System ID Input Data"):
    """
    Generates a comprehensive plot of the system identification data.

    Args:
        q (np.ndarray): Array of joint positions, shape (num_samples, num_joints).
        qd (np.ndarray): Array of joint velocities, shape (num_samples, num_joints).
        qdd (np.ndarray): Array of joint accelerations, shape (num_samples, num_joints).
        tau (np.ndarray): Array of joint torques, shape (num_samples, num_joints).
        time_step (float): The time step between samples, used for the x-axis.
        title (str): The main title for the plot.
    """
    num_samples, num_joints = q.shape
    time_vector = np.arange(num_samples) * time_step

    # Create a 2x2 grid of subplots for a compact dashboard view
    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    fig.suptitle(title, fontsize=18, weight='bold')

    # Define plot properties for each data type
    plot_configs = {
        'Positions (q)': {'ax': axs[0, 0], 'data': q, 'unit': 'rad'},
        'Velocities (qd)': {'ax': axs[0, 1], 'data': qd, 'unit': 'rad/s'},
        'Accelerations (qdd)': {'ax': axs[1, 0], 'data': qdd, 'unit': 'rad/s²'},
        'Torques (τ)': {'ax': axs[1, 1], 'data': tau, 'unit': 'Nm'},
    }
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_joints))

    for name, config in plot_configs.items():
        ax = config['ax']
        data = config['data']
        
        for j in range(num_joints):
            ax.plot(time_vector, data[:, j], label=f'Joint {j}', color=colors[j], linewidth=1.5)
            
        ax.set_title(name, fontsize=14)
        ax.set_ylabel(f"[{config['unit']}]", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)


    # Set common x-axis label for the bottom plots
    axs[1, 0].set_xlabel("Time [s]", fontsize=12)
    axs[1, 1].set_xlabel("Time [s]", fontsize=12)

    # Adjust layout to prevent titles from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    """
    Perform system identification using the collected data and the Pinocchio regressor.
    """
    print("--- Starting System Identification ---")

    # 1. Load the data
    data = np.load(DATA_PATH)
    q_data_full = data['q']
    qd_data_full = data['qd']
    qdd_data_full = data['qdd']
    tau_data_full = data['tau']
    
    samples_to_skip = int(SKIP_TIME_SECONDS / TIME_STEP_FOR_PLOTTING)

    q_data = q_data_full[samples_to_skip:]
    qd_data = qd_data_full[samples_to_skip:]
    qdd_data = qdd_data_full[samples_to_skip:]
    tau_data = tau_data_full[samples_to_skip:]

    print("Visualizing input data...")
    plot_sysid_data(q_data, qd_data, qdd_data, tau_data, 
                    time_step=TIME_STEP_FOR_PLOTTING,
                    title=f"Input Data from '{os.path.basename(DATA_PATH)}'")
    print(tau_data)
    np.savetxt( "tau_sim.txt", tau_data)
    num_samples = len(q_data)
    print(f"Loaded {num_samples} samples from '{DATA_PATH}'")

    # 2. Initialize the Pinocchio regressor builder
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

    an_reg = analyze_regressor(Y_stack)
    print("Analyzing regressor : ", an_reg)

    # 4. Solve least squares with regularization
    print("Solving for dynamic parameters...")
    ridge_model = Ridge(alpha=L2_REGULARIZATION, fit_intercept=False, solver='svd')

    ridge_model.fit(Y_stack, tau_stack.ravel())

    identified_params = ridge_model.coef_
    
    # -- Least squares method -- 
    # Y_T_Y = Y_stack.T @ Y_stack
    # Y_T_tau = Y_stack.T @ tau_stack
    # regularization_matrix = L2_REGULARIZATION * np.eye(num_params)
    
    # identified_params = np.linalg.solve(Y_T_Y + regularization_matrix, Y_T_tau)
    # identified_params = identified_params.flatten()

    print("--- Identification Complete ---")
    print("Identified Params: ", identified_params)

    # 5. Save results
    np.savez(SAVE_PATH, P=identified_params)
    print(f"Identified parameters saved to '{SAVE_PATH}'")

    # 6. Display friction coefficients
    # print("\nIdentified Friction Coefficients (Viscous, Coulomb):")
    # friction_params = identified_params[regressor_builder.total_link_params:]
    # for i in range(num_joints):
    #     fv = friction_params[i * 2]
    #     fc = friction_params[i * 2 + 1]
    #     print(f"  Joint {i}: Fv = {fv:.4f}, Fc = {fc:.4f}")

if __name__ == "__main__":
    main()