import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import time

def cross_validate_base_parameters(DATA_PATH, regressor_builder, base_indices, 
                                 test_size=0.2, random_state=42, 
                                 TIME_STEP_FOR_PLOTTING=0.02, 
                                 SKIP_TIME_SECONDS=10.0,
                                 L2_REGULARIZATION=1e-3):
    """
    Cross-validate the base parameter identification using train/test split.
    
    Args:
        DATA_PATH: Path to the system ID data
        regressor_builder: PinocchioAndFrictionRegressorBuilder instance
        base_indices: Indices of identified base parameters
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        results: Dictionary containing validation metrics
    """
    
    print("=== CROSS-VALIDATION FOR BASE PARAMETER IDENTIFICATION ===")
    
    # 1. Load and prepare data
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
    
    num_samples = len(q_data)
    num_joints = regressor_builder.num_joints
    
    print(f"Total samples: {num_samples}")
    print(f"Training samples: {int(num_samples * (1 - test_size))}")
    print(f"Testing samples: {int(num_samples * test_size)}")
    
    # 2. Split data into train/test
    indices = np.arange(num_samples)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, 
                                          random_state=random_state, shuffle=False)
    
    # Sort indices to maintain temporal order within each set
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    
    print(f"Training indices: {train_idx[0]} to {train_idx[-1]}")
    print(f"Testing indices: {test_idx[0]} to {test_idx[-1]}")
    
    # 3. Build regressor matrices for train and test sets
    def build_regressor_for_indices(sample_indices):
        """Build regressor matrix for given sample indices"""
        n_samples = len(sample_indices)
        Y_stack = np.zeros((n_samples * num_joints, regressor_builder.total_params))
        tau_stack = np.zeros((n_samples * num_joints, 1))
        
        for i, sample_idx in enumerate(sample_indices):
            q = q_data[sample_idx]
            qd = qd_data[sample_idx]
            qdd = qdd_data[sample_idx]
            tau = tau_data[sample_idx]
            
            Y_i = regressor_builder.compute_regressor_matrix(q, qd, qdd)
            
            row_start = i * num_joints
            row_end = row_start + num_joints
            Y_stack[row_start:row_end, :] = Y_i
            tau_stack[row_start:row_end, 0] = tau
        
        return Y_stack, tau_stack
    
    print("\nBuilding training regressor matrix...")
    start_time = time.time()
    Y_train, tau_train = build_regressor_for_indices(train_idx)
    train_time = time.time() - start_time
    
    print("\nBuilding testing regressor matrix...")
    start_time = time.time()
    Y_test, tau_test = build_regressor_for_indices(test_idx)
    test_time = time.time() - start_time
    
    print(f"Training regressor built in {train_time:.2f} seconds")
    print(f"Testing regressor built in {test_time:.2f} seconds")
    
    # 4. Extract base parameter columns
    Y_train_base = Y_train[:, base_indices]
    Y_test_base = Y_test[:, base_indices]
    
    print(f"\nBase regressor shapes:")
    print(f"Training: {Y_train_base.shape}")
    print(f"Testing: {Y_test_base.shape}")
    
    # 5. Train on training set
    print("\nTraining base parameters...")
    ridge_model = Ridge(alpha=L2_REGULARIZATION, fit_intercept=False, solver='svd')
    ridge_model.fit(Y_train_base, tau_train.ravel())
    base_params_identified = ridge_model.coef_
    
    # 6. Evaluate on both training and test sets
    print("\nEvaluating performance...")
    
    # Training set performance
    tau_train_pred = Y_train_base @ base_params_identified
    train_mse = mean_squared_error(tau_train.ravel(), tau_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(tau_train.ravel(), tau_train_pred)
    train_max_error = np.max(np.abs(tau_train.ravel() - tau_train_pred))
    
    # Test set performance
    tau_test_pred = Y_test_base @ base_params_identified
    test_mse = mean_squared_error(tau_test.ravel(), tau_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(tau_test.ravel(), tau_test_pred)
    test_max_error = np.max(np.abs(tau_test.ravel() - tau_test_pred))
    
    # 7. Compute residuals
    train_residuals = tau_train.ravel() - tau_train_pred
    test_residuals = tau_test.ravel() - tau_test_pred
    
    # 8. Print results
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"Number of base parameters: {len(base_params_identified)}")
    print(f"Base parameter indices: {base_indices}")
    print(f"L2 regularization: {L2_REGULARIZATION}")
    
    print(f"\nTRAINING SET PERFORMANCE:")
    print(f"  MSE: {train_mse:.6f}")
    print(f"  RMSE: {train_rmse:.6f}")
    print(f"  R²: {train_r2:.6f}")
    print(f"  Max error: {train_max_error:.6f}")
    print(f"  Error std: {np.std(train_residuals):.6f}")
    
    print(f"\nTEST SET PERFORMANCE:")
    print(f"  MSE: {test_mse:.6f}")
    print(f"  RMSE: {test_rmse:.6f}")
    print(f"  R²: {test_r2:.6f}")
    print(f"  Max error: {test_max_error:.6f}")
    print(f"  Error std: {np.std(test_residuals):.6f}")
    
    # 9. Compute generalization metrics
    generalization_gap_rmse = test_rmse - train_rmse
    generalization_gap_r2 = train_r2 - test_r2
    relative_performance = test_rmse / train_rmse
    
    print(f"\nGENERALIZATION ANALYSIS:")
    print(f"  RMSE gap (test - train): {generalization_gap_rmse:.6f}")
    print(f"  R² gap (train - test): {generalization_gap_r2:.6f}")
    print(f"  Relative performance (test/train RMSE): {relative_performance:.4f}")
    
    if relative_performance < 1.2:
        print("  ✓ Good generalization (test RMSE < 1.2 × train RMSE)")
    elif relative_performance < 1.5:
        print("  ⚠ Moderate generalization (test RMSE < 1.5 × train RMSE)")
    else:
        print("  ✗ Poor generalization (test RMSE > 1.5 × train RMSE)")
    
    # 10. Statistical significance test
    from scipy import stats
    
    # Test if residuals are normally distributed
    train_shapiro = stats.shapiro(train_residuals[:5000])  # Limit for computational efficiency
    test_shapiro = stats.shapiro(test_residuals[:5000])
    
    print(f"\nRESIDUAL NORMALITY TESTS (Shapiro-Wilk):")
    print(f"  Training set: p-value = {train_shapiro.pvalue:.6f}")
    print(f"  Test set: p-value = {test_shapiro.pvalue:.6f}")
    
    # 11. Create comprehensive plots
    plot_cross_validation_results(train_residuals, test_residuals, 
                                tau_train.ravel(), tau_train_pred,
                                tau_test.ravel(), tau_test_pred,
                                base_params_identified, base_indices)
    
    # 12. Return results dictionary
    results = {
        'base_params': base_params_identified,
        'base_indices': base_indices,
        'train_metrics': {
            'mse': train_mse,
            'rmse': train_rmse,
            'r2': train_r2,
            'max_error': train_max_error,
            'residuals': train_residuals
        },
        'test_metrics': {
            'mse': test_mse,
            'rmse': test_rmse,
            'r2': test_r2,
            'max_error': test_max_error,
            'residuals': test_residuals
        },
        'generalization': {
            'rmse_gap': generalization_gap_rmse,
            'r2_gap': generalization_gap_r2,
            'relative_performance': relative_performance
        }
    }
    
    return results

def plot_cross_validation_results(train_residuals, test_residuals, 
                                 tau_train_actual, tau_train_pred,
                                 tau_test_actual, tau_test_pred,
                                 base_params, base_indices):
    """Create comprehensive plots for cross-validation results"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Residual plots
    ax1 = plt.subplot(2, 4, 1)
    plt.plot(train_residuals, 'b-', alpha=0.7, label='Training')
    plt.plot(test_residuals, 'r-', alpha=0.7, label='Test')
    plt.title('Residuals Over Time')
    plt.xlabel('Sample')
    plt.ylabel('Torque Error [Nm]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Residual histograms
    ax2 = plt.subplot(2, 4, 2)
    plt.hist(train_residuals, bins=50, alpha=0.7, label='Training', color='blue', density=True)
    plt.hist(test_residuals, bins=50, alpha=0.7, label='Test', color='red', density=True)
    plt.title('Residual Distributions')
    plt.xlabel('Torque Error [Nm]')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Q-Q plots for normality
    ax3 = plt.subplot(2, 4, 3)
    from scipy import stats
    stats.probplot(train_residuals, dist="norm", plot=plt)
    plt.title('Training Residuals Q-Q Plot')
    plt.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 4, 4)
    stats.probplot(test_residuals, dist="norm", plot=plt)
    plt.title('Test Residuals Q-Q Plot')
    plt.grid(True, alpha=0.3)
    
    # 4. Prediction vs actual scatter plots
    ax5 = plt.subplot(2, 4, 5)
    plt.scatter(tau_train_actual, tau_train_pred, alpha=0.5, s=1, label='Training')
    min_val = min(tau_train_actual.min(), tau_train_pred.min())
    max_val = max(tau_train_actual.max(), tau_train_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Torque [Nm]')
    plt.ylabel('Predicted Torque [Nm]')
    plt.title('Training Set: Predicted vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(2, 4, 6)
    plt.scatter(tau_test_actual, tau_test_pred, alpha=0.5, s=1, label='Test', color='red')
    min_val = min(tau_test_actual.min(), tau_test_pred.min())
    max_val = max(tau_test_actual.max(), tau_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Torque [Nm]')
    plt.ylabel('Predicted Torque [Nm]')
    plt.title('Test Set: Predicted vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Base parameter values
    ax7 = plt.subplot(2, 4, 7)
    plt.bar(range(len(base_params)), base_params, alpha=0.7)
    plt.title('Identified Base Parameters')
    plt.xlabel('Base Parameter Index')
    plt.ylabel('Parameter Value')
    plt.xticks(range(len(base_params)), [f'{i}' for i in base_indices], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6. Residual autocorrelation
    ax8 = plt.subplot(2, 4, 8)
    from scipy.signal import correlate
    
    # Compute autocorrelation for a subset of residuals
    n_corr = min(1000, len(train_residuals))
    lags = np.arange(-n_corr//2, n_corr//2)
    train_autocorr = correlate(train_residuals[:n_corr], train_residuals[:n_corr], mode='same')
    train_autocorr = train_autocorr / train_autocorr[n_corr//2]
    
    test_autocorr = correlate(test_residuals[:n_corr], test_residuals[:n_corr], mode='same')
    test_autocorr = test_autocorr / test_autocorr[n_corr//2]
    
    plt.plot(lags, train_autocorr, 'b-', alpha=0.7, label='Training')
    plt.plot(lags, test_autocorr, 'r-', alpha=0.7, label='Test')
    plt.title('Residual Autocorrelation')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Cross-Validation Results for Base Parameter Identification', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()

# Usage example:
def run_cross_validation_test():
    """Example usage of the cross-validation function"""
    
    # These would be your actual values
    DATA_PATH = "/home/robot/dev/dyn/src/systemid/system_id_data_3joint_final.npz"
    
    # Load your regressor builder (this would be your actual instance)
    # regressor_builder = PinocchioAndFrictionRegressorBuilder(urdf_path)
    
    # Your identified base parameter indices
    base_indices = [42, 43, 41, 22, 37, 35, 19, 38, 44, 39, 28, 25, 27, 24, 29]
    
    # Run cross-validation
    # results = cross_validate_base_parameters(
    #     DATA_PATH=DATA_PATH,
    #     regressor_builder=regressor_builder,
    #     base_indices=base_indices,
    #     test_size=0.2,
    #     random_state=42
    # )
    
    # return results
    
    print("This is a template function. Replace with your actual regressor_builder instance.")
    print("Base parameter indices to test:", base_indices)

if __name__ == "__main__":
    run_cross_validation_test()