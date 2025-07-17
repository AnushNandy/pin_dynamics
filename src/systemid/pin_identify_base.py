import numpy as np
import time, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import robot_config
from pinocchio_friction_regressor import PinocchioAndFrictionRegressorBuilder
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# DATA_PATH = "/home/robot/dev/dyn/src/systemid/system_id_data_3joint_final.npz"
DATA_PATH = "/home/robot/dev/dyn/src/systemid/sysid_data_pybullet.npz"
SAVE_PATH = "/home/robot/dev/dyn/src/systemid/identified_base_params.npz"
L2_REGULARIZATION = 1e-3 
TIME_STEP_FOR_PLOTTING = 0.02
SKIP_TIME_SECONDS = 10.0

def compute_base_parameters_svd(Y_stack, tau_stack, tolerance=1e-12):
    """
    Compute the base parameters using SVD decomposition.
    
    Args:
        Y_stack: Full regressor matrix (n_samples*n_joints, n_params)
        tau_stack: Torque vector (n_samples*n_joints, 1)
        tolerance: Threshold for considering singular values as zero
    
    Returns:
        Y_base: Base regressor matrix
        base_indices: Indices of base parameters
        base_params: Identified base parameters
    """
    print("Computing base parameters using SVD...")
    
    # Compute SVD of the regressor matrix
    U, s, Vh = np.linalg.svd(Y_stack, full_matrices=False)
    
    # Find the rank by counting singular values above tolerance
    rank = np.sum(s > tolerance)
    print(f"Matrix rank: {rank} (out of {Y_stack.shape[1]} total parameters)")
    print(f"Largest singular value: {s[0]:.2e}")
    print(f"Smallest non-zero singular value: {s[rank-1]:.2e}")
    print(f"Condition number of base parameters: {s[0]/s[rank-1]:.2e}")
    
    # Select base parameters using the right singular vectors
    # The first 'rank' columns of V^T correspond to the base parameters
    V_base = Vh[:rank, :].T  # Shape: (n_params, rank)
    
    # Find which original parameters contribute most to each base parameter
    base_indices = []
    for i in range(rank):
        # Find the parameter with maximum absolute contribution to this base parameter
        max_idx = np.argmax(np.abs(V_base[:, i]))
        base_indices.append(max_idx)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_base_indices = []
    for idx in base_indices:
        if idx not in seen:
            seen.add(idx)
            unique_base_indices.append(idx)
    
    # If we have fewer unique indices than rank, find additional ones
    if len(unique_base_indices) < rank:
        for i in range(Y_stack.shape[1]):
            if i not in seen and len(unique_base_indices) < rank:
                # Check if this parameter adds significant information
                temp_indices = unique_base_indices + [i]
                Y_temp = Y_stack[:, temp_indices]
                if np.linalg.cond(Y_temp) < 1e10:  # Reasonable condition number
                    unique_base_indices.append(i)
                    seen.add(i)
    
    base_indices = unique_base_indices[:rank]
    
    # Create base regressor matrix
    Y_base = Y_stack[:, base_indices]
    
    print(f"Selected {len(base_indices)} base parameters with indices: {base_indices}")
    
    # Solve for base parameters
    ridge_model = Ridge(alpha=L2_REGULARIZATION, fit_intercept=False, solver='svd')
    ridge_model.fit(Y_base, tau_stack.ravel())
    base_params = ridge_model.coef_
    
    return Y_base, base_indices, base_params

def analyze_regressor(Y_stack, name="Regressor"):
    """Computes and prints diagnostics for a regressor matrix."""
    print(f"\n--- Analysis for {name} ---")
    
    # 1. Condition Number
    cond_num = np.linalg.cond(Y_stack)
    print(f"Condition Number: {cond_num:,.2e}")
    
    # 2. Matrix properties
    print(f"Matrix shape: {Y_stack.shape}")
    print(f"Rank: {np.linalg.matrix_rank(Y_stack)}")
    
    # 3. Singular Values
    print("Computing SVD...")
    U, s, Vh = np.linalg.svd(Y_stack, full_matrices=False)
    print("SVD computed.")
    
    print(f"Largest singular value: {s[0]:.2e}")
    print(f"Smallest singular value: {s[-1]:.2e}")
    print(f"Number of significant singular values (>1e-12): {np.sum(s > 1e-12)}")
    
    return s

def plot_singular_values(s_full, s_base, title="Singular Values Comparison"):
    """Plot singular values for full and base parameter matrices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Full matrix singular values
    ax1.semilogy(s_full, 'b.-', label='Full Parameters')
    ax1.set_title('Full Parameter Matrix')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Singular Value')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1e-12, color='r', linestyle='--', label='Tolerance')
    ax1.legend()
    
    ax2.semilogy(s_base, 'r.-', label='Base Parameters')
    ax2.set_title('Base Parameter Matrix')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Singular Value')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()

def plot_sysid_data(q, qd, qdd, tau, time_step, title="System ID Input Data"):
    """
    Generates a comprehensive plot of the system identification data.
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

def validate_identification(Y_base, base_params, tau_stack):
    """Validate the identification by computing prediction error."""
    tau_predicted = Y_base @ base_params
    error = tau_stack.ravel() - tau_predicted
    
    mse = np.mean(error**2)
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(error))
    
    print(f"\n--- Validation Results ---")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Max absolute error: {max_error:.6f}")
    print(f"Error std: {np.std(error):.6f}")
    
    # Plot error distribution
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(error)
    plt.title('Prediction Error Over Time')
    plt.xlabel('Sample')
    plt.ylabel('Torque Error [Nm]')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(error, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Error Distribution')
    plt.xlabel('Torque Error [Nm]')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Perform system identification using base parameters only.
    """
    print("--- Starting Base Parameter Identification ---")

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
    
    num_samples = len(q_data)
    print(f"Loaded {num_samples} samples from '{DATA_PATH}'")

    # 2. Initialize the Pinocchio regressor builder
    urdf_path = robot_config.URDF_PATH
    regressor_builder = PinocchioAndFrictionRegressorBuilder(urdf_path)
    
    num_joints = regressor_builder.num_joints
    num_params = regressor_builder.total_params
    print(f"Robot has {num_joints} joints, {num_params} total parameters")

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

    # 4. Analyze full regressor matrix
    s_full = analyze_regressor(Y_stack, "Full Parameter Matrix")

    # 5. Compute base parameters
    Y_base, base_indices, base_params = compute_base_parameters_svd(Y_stack, tau_stack)
    
    # 6. Analyze base regressor matrix
    s_base = analyze_regressor(Y_base, "Base Parameter Matrix")

    # 7. Plot singular values comparison
    plot_singular_values(s_full, s_base, "Singular Values: Full vs Base Parameters")

    # 8. Validate identification
    validate_identification(Y_base, base_params, tau_stack)

    print("\n--- Base Parameter Identification Complete ---")
    print(f"Number of base parameters: {len(base_params)}")
    print(f"Base parameter indices: {base_indices}")
    print(f"Base parameter values: {base_params}")

    # 9. Save results
    np.savez(SAVE_PATH, 
             base_params=base_params,
             base_indices=base_indices,
             Y_base=Y_base,
             condition_number=np.linalg.cond(Y_base))
    print(f"Base parameters saved to '{SAVE_PATH}'")

    # 10. Create mapping from base parameters back to full parameter space
    full_params_reconstructed = np.zeros(num_params)
    full_params_reconstructed[base_indices] = base_params
    
    print(f"\nReconstructed full parameter vector (only base parameters are non-zero):")
    print(f"Shape: {full_params_reconstructed.shape}")
    print(f"Non-zero elements: {np.sum(full_params_reconstructed != 0)}")

if __name__ == "__main__":
    main()