import numpy as np
import pinocchio as pin
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamics.pinocchio_dynamics import PinocchioRobotDynamics 
from src.systemid.pinocchio_friction_regressor import PinocchioAndFrictionRegressorBuilder, smooth_sign
from config import robot_config 

URDF_PATH = robot_config.URDF_PATH
IDENTIFIED_PARAMS_PATH = "/home/robot/dev/dyn/src/systemid/identified_params.npz"
NUM_JOINTS = robot_config.NUM_JOINTS

def get_ground_truth_params(model: pin.Model) -> np.ndarray:
    """
    Constructs the ground-truth parameter vector P_true from the URDF and
    the known friction coefficients used in the data generator.

    ## FINAL CORRECTION:
    1. Uses the correct Parallel Axis Theorem: I_origin = I_com - m*skew(c)*skew(c)
    2. Assembles the parameter vector in the order Pinocchio's regressor expects:
       [m, mcx, mcy, mcz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    """
    p_rnea_true = []
    
    for i in range(2, NUM_JOINTS + 2):
        inertia = model.inertias[i]
        m = inertia.mass
        c = inertia.lever      # Center of mass in local frame
        I_com = inertia.inertia  # Inertia tensor about COM
        c_skew = pin.skew(c)
        I_origin = I_com - m * (c_skew @ c_skew)
        
        p_link = np.zeros(10)
        p_link[0] = m
        p_link[1:4] = m * c

        # p_link[4] = I_com[0, 0]  # Ixx_com
        # p_link[5] = I_com[0, 1]  # Ixy_com
        # p_link[6] = I_com[0, 2]  # Ixz_com
        # p_link[7] = I_com[1, 1]  # Iyy_com
        # p_link[8] = I_com[1, 2]  # Iyz_com
        # p_link[9] = I_com[2, 2]  # Izz_com

        p_link[4] = I_origin[0, 0]  # Ixx_com
        p_link[5] = I_origin[0, 1]  # Ixy_com
        p_link[6] = I_origin[0, 2]  # Ixz_com
        p_link[7] = I_origin[1, 1]  # Iyy_com
        p_link[8] = I_origin[1, 2]  # Iyz_com
        p_link[9] = I_origin[2, 2]  # Izz_com
        
        p_rnea_true.extend(p_link)
        
        print(f"Body {i}: m={m:.4f}, c={np.round(c, 4)}, I_com_diag=[{I_com[0,0]:.4f}, {I_com[1,1]:.4f}, {I_com[2,2]:.4f}]")

    true_friction_coeffs = {
        'viscous': [0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06],
        'coulomb': [0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
    }
    p_friction_true = []
    for i in range(NUM_JOINTS):
        p_friction_true.append(true_friction_coeffs['viscous'][i])
        p_friction_true.append(true_friction_coeffs['coulomb'][i])
        
    print(f"Ground truth parameter vector length: {len(p_rnea_true)} + {len(p_friction_true)} = {len(p_rnea_true) + len(p_friction_true)}")
    return np.concatenate([p_rnea_true, p_friction_true])


def main():
    print("--- Verifying System Identification Pipeline ---")

    if not os.path.exists(IDENTIFIED_PARAMS_PATH):
        print(f"FATAL: Identified parameters file not found at '{IDENTIFIED_PARAMS_PATH}'")
        return
        
    regressor_builder = PinocchioAndFrictionRegressorBuilder(URDF_PATH)
    dynamics_model_true = PinocchioRobotDynamics(URDF_PATH)
    dynamics_model_identified = PinocchioRobotDynamics(URDF_PATH) 
    
    P_identified = np.load(IDENTIFIED_PARAMS_PATH)['P']
    P_true = get_ground_truth_params(regressor_builder.model)
    
    if P_true.shape != P_identified.shape:
        print(f"ERROR: Parameter vector shapes don't match! P_true={P_true.shape}, P_id={P_identified.shape}")
        return
    np.random.seed(42)
    q_test = np.random.uniform(-np.pi, np.pi, size=NUM_JOINTS)
    qd_test = np.random.uniform(-0.1, 0.1, size=NUM_JOINTS)
    qdd_test = np.random.uniform(-0.01, 0.01, size=NUM_JOINTS)

    print("\n--- Generated Test State (q, qd, qdd) ---")
    print(f"q:   {np.round(q_test, 3)}")
    print(f"qd:  {np.round(qd_test, 3)}")
    print(f"qdd: {np.round(qdd_test, 3)}")

    # --- 3. Compute Ground Truth Torque & Regressor ---
    tau_rnea_true = dynamics_model_true.compute_rnea(q_test, qd_test, qdd_test)
    tau_friction_true = np.array([
        P_true[-NUM_JOINTS*2 + 2*i] * qd_test[i] + P_true[-NUM_JOINTS*2 + 2*i+1] * smooth_sign(qd_test[i])
        for i in range(NUM_JOINTS)
    ])
    tau_true = tau_rnea_true + tau_friction_true
    
    print(f"\ntau_rnea_true: {np.round(tau_rnea_true, 3)}")
    print(f"tau_friction_true: {np.round(tau_friction_true, 3)}")
    print(f"tau_total_true: {np.round(tau_true, 3)}")
    
    Y_test = regressor_builder.compute_regressor_matrix(q_test, qd_test, qdd_test)
    print(f"Regressor matrix shape: {Y_test.shape}")

    # --- VERIFICATION 1: Ground Truth Self-Consistency ---
    print("\n--- VERIFICATION 1: Ground Truth Self-Consistency (Y * P_true == tau_true?) ---")
    tau_from_Y_P_true = Y_test @ P_true
    error1 = np.linalg.norm(tau_true - tau_from_Y_P_true)
    print(f"Torque from direct computation (τ_true):     {np.round(tau_true, 6)}")
    print(f"Torque from regressor (Y @ P_true):          {np.round(tau_from_Y_P_true, 6)}")
    print(f"Difference: {np.round(tau_true - tau_from_Y_P_true, 6)}")
    print(f"--> L2 Norm of Error: {error1:.2e}")
    if error1 < 1e-3:
        print("--> SUCCESS: The regressor matrix correctly represents the dynamics.")
    else:
        print("--> FAILURE: The regressor matrix is NOT consistent with the dynamics.")

    # --- VERIFICATION 2: Identified Parameter Consistency ---
    print("\n--- VERIFICATION 2: Identified Parameter Consistency (Y * P_identified ~= tau_true?) ---")
    tau_from_Y_P_identified = Y_test @ P_identified
    error2 = np.linalg.norm(tau_true - tau_from_Y_P_identified)
    print(f"Torque from ground truth data (τ_true):      {np.round(tau_true, 6)}")
    print(f"Torque from identified params (Y @ P_id):    {np.round(tau_from_Y_P_identified, 6)}")
    print(f"--> L2 Norm of Error: {error2:.2e}")
    if error2 < 1e-1:
        print("--> SUCCESS: The identified parameters provide a good fit for this data point.")
    else:
        print("--> WARNING: The identified parameters have a large error for this data point.")

    # --- VERIFICATION 3: Custom RNEA Engine Consistency ---
    print("\n--- VERIFICATION 3: Custom RNEA Engine Consistency ---")
    total_link_params = regressor_builder.total_link_params
    P_rnea_identified = P_identified[:total_link_params]
    Y_rnea_test = Y_test[:, :total_link_params]
    
    tau_rnea_from_Y_P = Y_rnea_test @ P_rnea_identified
    
    dynamics_model_identified.set_parameters_from_vector(P_rnea_identified)
    tau_rnea_from_engine = dynamics_model_identified.compute_rnea(q_test, qd_test, qdd_test)

    error3 = np.linalg.norm(tau_rnea_from_engine - tau_rnea_from_Y_P)
    print(f"RNEA torque from regressor:      {np.round(tau_rnea_from_Y_P, 6)}")
    print(f"RNEA torque from dynamics engine:  {np.round(tau_rnea_from_engine, 6)}")
    print(f"--> L2 Norm of Error: {error3:.2e}")
    if error3 < 1e-3:
        print("--> SUCCESS: The custom dynamics engine is consistent with the regressor.")
    else:
        print("--> FAILURE: The set_parameters_from_vector method may be incorrect.")

if __name__ == "__main__":
    main()