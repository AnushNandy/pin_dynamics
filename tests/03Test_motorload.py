# --- REWRITTEN FILE: 03Test_motorload.py ---

import numpy as np
import pandas as pd
import os
import sys
from typing import Tuple

# Ensure the source directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.systemid.pinocchio_friction_regressor import PinocchioAndFrictionRegressorBuilder
from config import robot_config

# --- Configuration ---
URDF_PATH = robot_config.URDF_PATH
IDENTIFIED_PARAMS_PATH = "/home/robot/dev/dyn/src/systemid/identified_base_params.npz"
NUM_JOINTS = robot_config.NUM_JOINTS
NUM_SAMPLES = 5000

# Gear ratios for each joint (motor-to-joint)
GEAR_RATIOS = np.array([120, 120, 100, 100, 80, 80, 50])

# Motor specifications from datasheets
MOTOR_SPECS = {
    'Joint': [f'J{i}' for i in range(NUM_JOINTS)],
    'Max Motor Load (Continuous) [Nm]': [140, 140, 51, 51, 14, 14, 7.7],
    'Max Motor Load (Instant) [Nm]': [395, 395, 143, 143, 70, 70, 35],
}

def analyze_motor_loads(
    regressor_builder: PinocchioAndFrictionRegressorBuilder,
    base_params: np.ndarray,
    num_samples: int = 5000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs a Monte Carlo simulation to estimate peak and continuous motor loads.

    This function simulates the robot in many random configurations and states
    (q, qd, qdd) to estimate the torques required at the motors. The underlying
    model is the standard equation of motion in regressor form: τ = Y(q, qd, qdd) * p.

    Args:
        regressor_builder: An initialized instance of the regressor builder, which
                           constructs the dynamics regressor matrix Y.
        base_params: The identified numerical vector of the robot's base dynamic
                     parameters (p). Shape: (num_base_params,).
        num_samples: The number of random configurations to simulate.
        seed: A seed for the random number generator for reproducibility.

    Returns:
        A tuple containing:
        - max_motor_torques (np.ndarray): The peak instantaneous torque seen by each
                                          motor across all samples. Shape: (NUM_JOINTS,).
        - rms_motor_torques (np.ndarray): The RMS torque for each motor, representing
                                          the continuous load. Shape: (NUM_JOINTS,).
    """
    print(f"Analyzing motor loads with {num_samples} random configurations...")
    np.random.seed(seed)
    
    # Define the state space for sampling.
    # Note: These ranges should ideally reflect the expected operational workspace
    # and velocities/accelerations of the target task.
    q_samples = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(num_samples, NUM_JOINTS))
    qd_samples = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, NUM_JOINTS))
    qdd_samples = np.random.uniform(low=-2.0, high=2.0, size=(num_samples, NUM_JOINTS))
    
    all_motor_torques = np.zeros((num_samples, NUM_JOINTS))
    
    for i in range(num_samples):
        if (i + 1) % 500 == 0:
            print(f"  Processing sample {i+1}/{num_samples}...")
            
        q, qd, qdd = q_samples[i], qd_samples[i], qdd_samples[i]
        
        # 1. Compute the full dynamics regressor matrix Y for the current state.
        #    Y has shape (num_joints, num_base_params)
        Y = regressor_builder.compute_regressor_matrix(q, qd, qdd)
        
        # 2. Calculate the required JOINT torque using τ = Y * p
        #    This is the core dynamics calculation.
        tau_joint = Y @ base_params
        
        # 3. Convert JOINT torque to MOTOR torque.
        #    Assumes a perfect gearbox (100% efficiency). In reality, you might
        #    include a gearbox efficiency term, e.g., τ_motor = τ_joint / (N * η).
        tau_motor = tau_joint / GEAR_RATIOS
        
        all_motor_torques[i, :] = tau_motor

    # Calculate peak torque (L-infinity norm) for instantaneous load analysis
    max_motor_torques = np.max(np.abs(all_motor_torques), axis=0)

    # Calculate Root Mean Square (RMS) torque for continuous/thermal load analysis
    rms_motor_torques = np.sqrt(np.mean(all_motor_torques**2, axis=0))
    
    return max_motor_torques, rms_motor_torques


def create_results_table(max_torques: np.ndarray, rms_torques: np.ndarray) -> pd.DataFrame:
    """
    Creates a pandas DataFrame to display the motor load analysis results.

    Args:
        max_torques: Calculated peak motor torques.
        rms_torques: Calculated RMS motor torques.

    Returns:
        A pandas DataFrame summarizing the results and safety factors.
    """
    df = pd.DataFrame(MOTOR_SPECS)
    
    df['Calculated Peak Motor Load [Nm]'] = max_torques
    df['Calculated RMS Motor Load [Nm]'] = rms_torques
    
    # Safety Factor (SF) = Capacity / Demand
    # SF > 1.0 is desired.
    df['SF_Instant'] = df['Max Motor Load (Instant) [Nm]'] / df['Calculated Peak Motor Load [Nm]']
    df['SF_Continuous'] = df['Max Motor Load (Continuous) [Nm]'] / df['Calculated RMS Motor Load [Nm]']
    
    return df


def main():
    """
    Main execution function for the motor load analysis pipeline.
    """
    print("--- Motor Load Analysis from Identified Dynamics ---")

    if not os.path.exists(IDENTIFIED_PARAMS_PATH):
        print(f"FATAL: Identified parameters file not found at '{IDENTIFIED_PARAMS_PATH}'")
        sys.exit(1)
        
    # 1. Initialize the regressor builder, which encapsulates the robot's model.
    regressor_builder = PinocchioAndFrictionRegressorBuilder(URDF_PATH)
    
    # 2. Load the identified parameters from the .npz file.
    identified_params_data = np.load(IDENTIFIED_PARAMS_PATH)
    base_params_vec = identified_params_data['base_params']
    
    print(f"Successfully loaded {len(base_params_vec)} identified base parameters.")
    
    # 3. Run the Monte Carlo analysis.
    max_torques, rms_torques = analyze_motor_loads(
        regressor_builder, 
        base_params_vec,
        num_samples=NUM_SAMPLES
    )
    
    # 4. Create and display the final results table.
    results_table = create_results_table(max_torques, rms_torques)
    
    # Save results to a CSV file for record-keeping
    output_path = 'motor_load_analysis_results.csv'
    results_table.to_csv(output_path, index=False)
    print(f"\nResults saved to '{output_path}'")
    
    print("\n--- MOTOR LOAD ANALYSIS RESULTS ---")
    print(results_table.to_string(float_format="%.3f"))
    
    print("\n--- SAFETY ANALYSIS SUMMARY ---")
    overloaded_instant = results_table[results_table['SF_Instant'] < 1.0]
    overloaded_continuous = results_table[results_table['SF_Continuous'] < 1.0]

    if not overloaded_instant.empty:
        print("\n⚠️  INSTANTANEOUS OVERLOAD WARNING (Risk of stalling/damage):")
        print("The following joints may exceed their peak torque limits.")
        print(overloaded_instant[['Joint', 'Calculated Peak Motor Load [Nm]', 'Max Motor Load (Instant) [Nm]', 'SF_Instant']].to_string(index=False))
        
    if not overloaded_continuous.empty:
        print("\n⚠️  CONTINUOUS OVERLOAD WARNING (Risk of overheating):")
        print("The following joints may exceed their continuous torque limits.")
        print(overloaded_continuous[['Joint', 'Calculated RMS Motor Load [Nm]', 'Max Motor Load (Continuous) [Nm]', 'SF_Continuous']].to_string(index=False))

    if overloaded_instant.empty and overloaded_continuous.empty:
        print("\n✅ All motors operate within their specified load limits for the simulated motions.")

if __name__ == "__main__":
    main()