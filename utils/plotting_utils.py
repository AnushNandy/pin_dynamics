import matplotlib.pyplot as plt
import numpy as np

def plot_sysid_validation(t, tau_commanded, tau_predicted, joint_names):
    """
    Plots the comparison between commanded and predicted torques for each joint.
    This validates the RNEA (dynamics model) part of the controller.
    """
    num_joints = tau_commanded.shape[1]
    fig, axes = plt.subplots(num_joints, 1, figsize=(15, 3 * num_joints), sharex=True)
    if num_joints == 1: axes = [axes] # Ensure axes is always iterable
    fig.suptitle('Dynamics Model Validation: Commanded vs. Predicted Torques', fontsize=16)

    # Calculate overall RMSE between commanded and predicted (model output) torque
    overall_rmse = np.sqrt(np.mean((tau_commanded - tau_predicted)**2))

    for i in range(num_joints):
        error = tau_commanded[:, i] - tau_predicted[:, i]
        rmse = np.sqrt(np.mean(error**2))
        axes[i].plot(t, tau_commanded[:, i], 'b-', label='Commanded Torque (Total)', alpha=0.8)
        axes[i].plot(t, tau_predicted[:, i], 'r--', label='Predicted Torque (from Model)')
        axes[i].set_ylabel('Torque (Nm)')
        axes[i].set_title(f'Joint: {joint_names[i]} (RMSE: {rmse:.4f} Nm)')
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.text(0.5, 0.96, f'Overall Torque RMSE: {overall_rmse:.4f} Nm', ha='center', va='top', fontsize=12, color='darkred')
    print("\nDisplaying System ID torque validation plot...")
    plt.show()

def plot_control_performance(t, x_des, x_act, joint_names):
    """
    Plots the tracking performance of the controller (desired vs. actual end-effector position).
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('End-Effector Control Performance: Position Tracking', fontsize=16)
    pos_labels = ['X', 'Y', 'Z']
    
    for i in range(3):
        error = x_des[:, i] - x_act[:, i]
        rmse = np.sqrt(np.mean(error**2)) * 1000 # convert to mm
        axes[i].plot(t, x_des[:, i], 'r--', label='Desired Position')
        axes[i].plot(t, x_act[:, i], 'b-', label='Actual Position')
        axes[i].set_title(f'{pos_labels[i]} Position Tracking (RMSE: {rmse:.2f} mm)')
        axes[i].set_ylabel('Position (m)')
        axes[i].grid(True)
        axes[i].legend()
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("\nDisplaying control tracking performance plot...")
    plt.show()


def plot_control_torques(t, tau_commanded, tau_gravity, joint_names):
    """
    Plots the torques commanded by the controller, separating the gravity component.
    """
    num_joints = tau_commanded.shape[1]
    fig, axes = plt.subplots(num_joints, 1, figsize=(15, 3 * num_joints), sharex=True)
    if num_joints == 1: axes = [axes] # Ensure axes is always iterable
    fig.suptitle('Control Effort: Commanded Torques & Gravity Compensation', fontsize=16)

    for i in range(num_joints):
        axes[i].plot(t, tau_commanded[:, i], 'b-', label='Total Commanded Torque')
        axes[i].plot(t, tau_gravity[:, i], 'g--', label='Gravity Compensation Torque')
        axes[i].set_ylabel('Torque (Nm)')
        axes[i].set_title(f'Joint: {joint_names[i]}')
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("\nDisplaying commanded torque plot...")
    plt.show()