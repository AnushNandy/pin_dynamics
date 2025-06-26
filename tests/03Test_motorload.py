import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.systemid.pinocchio_friction_regressor import PinocchioAndFrictionRegressorBuilder
from config import robot_config

URDF_PATH = robot_config.URDF_PATH
IDENTIFIED_PARAMS_PATH = "/home/robot/dev/dyn/src/systemid/identified_params_pybullet.npz"
NUM_JOINTS = robot_config.NUM_JOINTS
NUM_SAMPLES = 5000 

GEAR_RATIOS = np.array([
    120,  # Joint 0
    120,  # Joint 1
    100,  # Joint 2
    100,  # Joint 3
    80,   # Joint 4
    80,   # Joint 5
    50    # Joint 6
])

MOTOR_SPECS = {
    'Joint': [f'J{i}' for i in range(NUM_JOINTS)],
    'Max Motor Load (Continuous) [Nm]': [140, 140, 51, 51, 14, 14, 7.7],
    'Max Motor Load (Instant) [Nm]': [395, 395, 143, 143, 70, 70, 35],
}


def analyze_motor_loads(regressor_builder, P_params):
    """
    Performs a Monte Carlo simulation to estimate peak and continuous motor loads.

    This function simulates the robot in many random configurations to find the
    torques that the motors will experience during operation.

    Args:
        regressor_builder: An instance of the regressor builder.
        P_params: The identified parameter vector for the robot.

    Returns:
        A tuple containing:
        - max_motor_torques: The peak instantaneous torque seen by each motor.
        - rms_motor_torques: The RMS torque, representing the continuous load.
    """
    print(f"Analyzing motor loads with {NUM_SAMPLES} random configurations...")
    np.random.seed(42)
    q_samples = np.random.uniform(0, np.pi/3, size=(NUM_SAMPLES, NUM_JOINTS))
    qd_samples = np.random.uniform(-0.1, 0.1, size=(NUM_SAMPLES, NUM_JOINTS))
    qdd_samples = np.random.uniform(-0.01, 0.01, size=(NUM_SAMPLES, NUM_JOINTS))
    
    all_motor_torques = np.zeros((NUM_SAMPLES, NUM_JOINTS))
    
    for i in range(NUM_SAMPLES):
        if (i + 1) % 200 == 0:
            print(f"  Processing sample {i+1}/{NUM_SAMPLES}...")
            
        q, qd, qdd = q_samples[i], qd_samples[i], qdd_samples[i]
        
        # 1. Compute the full dynamics regressor matrix for the current state.
        Y = regressor_builder.compute_regressor_matrix(q, qd, qdd)
        
        # 2. Calculate the required JOINT torque using the identified parameters.
        tau_joint = Y @ P_params
        
        # 3. Convert JOINT torque to MOTOR torque using the gear ratios.
        tau_motor = tau_joint / GEAR_RATIOS
        
        all_motor_torques[i, :] = tau_motor

    max_motor_torques = np.max(np.abs(all_motor_torques), axis=0)

    rms_motor_torques = np.sqrt(np.mean(all_motor_torques**2, axis=0))
    
    return max_motor_torques, rms_motor_torques


def create_results_table(max_torques, rms_torques):
    """
    Creates a pandas DataFrame to display the motor load analysis results.
    """
    df = pd.DataFrame(MOTOR_SPECS)
    
    df['Calculated Peak Motor Load [Nm]'] = max_torques
    df['Calculated RMS Motor Load [Nm]'] = rms_torques
    
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
        return
        
    # 1. Initialize the regressor builder, which knows the robot's structure.
    regressor_builder = PinocchioAndFrictionRegressorBuilder(URDF_PATH)
    
    # 2. Load the parameters we found during system identification.
    P_identified = np.load(IDENTIFIED_PARAMS_PATH)['P']
    print(f"Successfully loaded {len(P_identified)} identified parameters.")
    
    # 3. Run the Monte Carlo analysis.
    max_torques, rms_torques = analyze_motor_loads(regressor_builder, P_identified)
    
    # 4. Create and display the final results table.
    results_table = create_results_table(max_torques, rms_torques)
    results_table.to_csv('tests/output.csv', index=False)
    
    print("\n--- MOTOR LOAD ANALYSIS RESULTS ---")
    print(results_table.to_string(float_format="%.3f"))
    
    print("\n--- SAFETY ANALYSIS SUMMARY ---")
    overloaded_instant = results_table[results_table['SF_Instant'] < 1.0]
    overloaded_continuous = results_table[results_table['SF_Continuous'] < 1.0]

    if not overloaded_instant.empty:
        print("⚠️  INSTANT OVERLOAD WARNING (Risk of stalling/damage):")
        print(overloaded_instant[['Joint', 'Calculated Peak Motor Load [Nm]', 'Max Motor Load (Instant) [Nm]', 'SF_Instant']])
        
    if not overloaded_continuous.empty:
        print("⚠️  CONTINUOUS OVERLOAD WARNING (Risk of overheating):")
        print(overloaded_continuous[['Joint', 'Calculated RMS Motor Load [Nm]', 'Max Motor Load (Continuous) [Nm]', 'SF_Continuous']])

    if overloaded_instant.empty and overloaded_continuous.empty:
        print("✅ All motors operate within their specified load limits for the simulated motions.")

if __name__ == "__main__":
    main()