import matplotlib.pyplot as plt
import numpy as np

def plot_sysid_validation(t, tau_measured, tau_predicted, joint_names):
    """
    Plots the comparison between measured and predicted torques for each joint
    after system identification.
    """
    num_joints = tau_measured.shape[1]
    fig, axes = plt.subplots(num_joints, 1, figsize=(15, 3 * num_joints), sharex=True)
    if num_joints == 1: axes = [axes] # Ensure axes is always iterable
    fig.suptitle('System ID Validation: Measured vs. Predicted Torques', fontsize=16)

    overall_rmse = np.sqrt(np.mean((tau_measured - tau_predicted)**2))

    for i in range(num_joints):
        error = tau_measured[:, i] - tau_predicted[:, i]
        rmse = np.sqrt(np.mean(error**2))
        axes[i].plot(t, tau_measured[:, i], 'b-', label=f'Measured (PyBullet Truth)')
        axes[i].plot(t, tau_predicted[:, i], 'r--', label=f'Predicted (Identified Model)')
        axes[i].set_ylabel('Torque (Nm)')
        axes[i].set_title(f'Joint: {joint_names[i]} (RMSE: {rmse:.4f} Nm)')
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.text(0.5, 0.96, f'Overall Torque RMSE: {overall_rmse:.4f} Nm', ha='center', va='top', fontsize=12, color='darkred')
    print("\nDisplaying System ID torque validation plot...")
    plt.show()

def plot_parameter_comparison(link_names, initial_params, identified_params):
    """
    Creates a bar chart to compare the initial and identified mass for each link.
    """
    num_links = len(link_names)
    # The mass is the first of the 10 dynamic parameters for each link
    initial_masses = [initial_params[i * 10] for i in range(num_links)]
    identified_masses = [identified_params[i * 10] for i in range(num_links)]

    x = np.arange(num_links)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, initial_masses, width, label='Initial (from config)')
    rects2 = ax.bar(x + width/2, identified_masses, width, label='Identified (from PyBullet data)')

    ax.set_ylabel('Mass (kg)')
    ax.set_title('Comparison of Link Masses: Initial vs. Identified')
    ax.set_xticks(x)
    ax.set_xticklabels(link_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--')

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    print("\nDisplaying link mass comparison plot...")
    plt.show()


def plot_control_performance(t, q_des, q_act, v_des, v_act, joint_names):
    """
    Plots the tracking performance of the controller (desired vs. actual).
    """
    num_joints = q_des.shape[1]
    fig, axes = plt.subplots(num_joints, 2, figsize=(18, 4 * num_joints), sharex=True)
    fig.suptitle('Control Performance: Trajectory Tracking (PD + Identified Gravity Comp)', fontsize=16)

    # Plot Positions
    for i in range(num_joints):
        ax = axes[i, 0]
        ax.plot(t, q_des[:, i], 'r--', label='Desired Position')
        ax.plot(t, q_act[:, i], 'b-', label='Actual Position')
        ax.set_title(f'{joint_names[i]} - Position Tracking')
        ax.set_ylabel('Angle (rad)')
        ax.grid(True)
        ax.legend()
    axes[-1, 0].set_xlabel('Time (s)')

    # Plot Velocities
    for i in range(num_joints):
        ax = axes[i, 1]
        ax.plot(t, v_des[:, i], 'r--', label='Desired Velocity')
        ax.plot(t, v_act[:, i], 'b-', label='Actual Velocity')
        ax.set_title(f'{joint_names[i]} - Velocity Tracking')
        ax.set_ylabel('Velocity (rad/s)')
        ax.grid(True)
        ax.legend()
    axes[-1, 1].set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("\nDisplaying control tracking performance plot...")
    plt.show()

def plot_control_torques(t, tau_commanded, joint_names):
    """
    Plots the torques commanded by the PD+G controller.
    """
    num_joints = tau_commanded.shape[1]
    fig, axes = plt.subplots(num_joints, 1, figsize=(15, 3 * num_joints), sharex=True)
    if num_joints == 1: axes = [axes] # Ensure axes is always iterable
    fig.suptitle('Control Effort: Commanded Torques from PD+G Controller', fontsize=16)

    for i in range(num_joints):
        axes[i].plot(t, tau_commanded[:, i], 'g-', label='Commanded Torque')
        axes[i].set_ylabel('Torque (Nm)')
        axes[i].set_title(f'Joint: {joint_names[i]}')
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("\nDisplaying commanded torque plot...")
    plt.show()