import numpy as np

class JointModel:
    def __init__(self,
                 gear_ratio: float,
                 motor_inertia: float,
                 # Friction Parameters (Asymmetric Stribeck)
                 coulomb_pos: float, coulomb_neg: float,
                 stiction_pos: float, stiction_neg: float,
                 viscous_coeff: float,
                 stribeck_vel_pos: float, stribeck_vel_neg: float,
                 # Transmission Parameters
                 stiffness: float,
                 # Bouc-Wen Hysteresis Parameters
                 hysteresis_shape_A: float,
                 hysteresis_shape_beta: float,
                 hysteresis_shape_gamma: float,
                 hysteresis_shape_n: float,
                 hysteresis_scale_alpha: float,
                 dt: float):
        """
                Initializes the joint model with its dynamic parameters.

                :param float gear_ratio: Gear ratio (N).
                :param float motor_inertia: Inertia of the motor rotor (Jm).
                :param ...: Friction and transmission parameters.
                :param float dt: Simulation time step for discrete integration.
                """
        self.N = gear_ratio
        self.J_m = motor_inertia
        self.dt = dt

        # friction parameters
        self.tau_c_p, self.tau_c_n = coulomb_pos, coulomb_neg
        self.tau_s_p, self.tau_s_n = stiction_pos, stiction_neg
        self.f_v = viscous_coeff
        self.om_s_p, self.om_s_n = stribeck_vel_pos, stribeck_vel_neg

        # transmission parameters
        self.K = stiffness
        self.alpha_h = hysteresis_scale_alpha
        self.A_h = hysteresis_shape_A
        self.beta_h = hysteresis_shape_beta
        self.gamma_h = hysteresis_shape_gamma
        self.n_h = hysteresis_shape_n

        self.hysteresis_state_h = 0.0  # Bouc-Wen state 'h'

    def compute_friction_torque(self, link_velocity: float) -> float:
        """
        Computes the friction torque based on the asymmetric Stribeck model.

        :param float link_velocity: The angular velocity of the output link (ω_l).
        :return: The friction torque (τ_friction).
        """
        omega = link_velocity
        tau_viscous = self.f_v * omega

        if np.abs(omega) < 1e-6:
            return 0.0

        if omega > 0:
            tau_c = self.tau_c_p + (self.tau_s_p - self.tau_c_p) * \
                    np.exp(-(omega / self.om_s_p) ** 2)
        else:
            tau_c = self.tau_c_n + (self.tau_s_n - self.tau_c_n) * \
                    np.exp(-(omega / self.om_s_n) ** 2)

        return tau_c + tau_viscous

    def _update_hysteresis_state(self, relative_velocity: float):
        """
        Updates the internal hysteresis state 'h' using Euler integration.

        :param float relative_velocity: The velocity difference across the gear (d(delta theta)/dt).
        """
        h = self.hysteresis_state_h
        d_delta_theta_dt = relative_velocity

        h_dot = self.A_h * d_delta_theta_dt - \
                (self.beta_h * abs(d_delta_theta_dt) * (abs(h) ** (self.n_h - 1)) * h +
                 self.gamma_h * d_delta_theta_dt * (abs(h) ** self.n_h))

        # Simple Euler integration for the state update
        self.hysteresis_state_h += h_dot * self.dt

    def compute_motor_torque(self,
                             tau_rnea: float,
                             link_pos: float, link_vel: float,
                             motor_pos: float, motor_vel: float, motor_acc: float) -> float:
        """
        Calculate the required motor torque given the full state and desired link torque.

        :param float tau_rnea: The desired torque at the link side (from RNEA).
        :param float link_pos: Current position of the link (θ_l).
        :param float link_vel: Current velocity of the link (ω_l).
        :param float motor_pos: Current position of the motor (θ_m).
        :param float motor_vel: Current velocity of the motor (ω_m).
        :param float motor_acc: Current acceleration of the motor (α_m).
        :return: The required motor torque (τ_m).
        """
        # 1. Calculate friction at the link side
        tau_friction = self.compute_friction_torque(link_vel)

        # 2. Calculate the total required transmission torque
        # From RNEA equation: τ_load = τ_transmission - τ_friction
        tau_transmission = tau_rnea + tau_friction

        # 3. Calculate the motor-side torque required to generate τ_transmission
        # From motor equation: τ_m = J_m*α_m + (1/N)*τ_transmission
        tau_motor = self.J_m * motor_acc + (1.0 / self.N) * tau_transmission

        # 1. Calculate deflection and relative velocity
        delta_theta = (motor_pos / self.N) - link_pos
        relative_velocity = (motor_vel / self.N) - link_vel

        # 2. Update and calculate hysteresis torque
        self._update_hysteresis_state(relative_velocity)
        tau_hysteresis = self.alpha_h * self.K * self.hysteresis_state_h

        # 3. Calculate stiffness torque
        tau_stiffness = self.K * delta_theta

        # 4. Total transmission torque based on current state
        tau_transmission_current = tau_stiffness + tau_hysteresis

        # 5. Calculate motor torque
        tau_motor_cmd = self.J_m * motor_acc + (1.0 / self.N) * tau_transmission_current

        # 1. Compute friction from desired link velocity
        tau_friction_ff = self.compute_friction_torque(link_vel)  # Using qd_des here

        # 2. Compute deflection from desired states. This is tricky.
        # Often, a simplified model is used for feedforward.
        # Let's assume you can compute a feedforward `tau_transmission_ff`.
        # For simplicity, let's start by adding only friction to RNEA.

        tau_motor_ff = (1.0 / self.N) * (tau_rnea + tau_friction_ff) + self.J_m * motor_acc

        return tau_motor_ff
    
    def compute_feedforward_torque(self,
                                 tau_rnea: float,
                                 link_pos_des: float,
                                 link_vel_des: float,
                                 link_acc_des: float) -> float:
        """
        Calculate the required motor torque for feedforward control.
        This combines the rigid-body torque from RNEA with the detailed
        friction and transmission model.

        τ_motor = J_m * α_m + (1/N) * τ_transmission
        τ_transmission = τ_rnea + τ_friction + τ_flexibility
        
        :param float tau_rnea: The desired torque at the link side (from RNEA).
        :param float link_pos_des: Desired position of the link (θ_l).
        :param float link_vel_des: Desired velocity of the link (ω_l).
        :param float link_acc_des: Desired acceleration of the link (α_l).
        :return: The required motor torque (τ_m) for feedforward.
        """
        # 1. Estimate motor-side kinematics based on desired link-side kinematics
        # This assumes a rigid gearbox for kinematic calculations.
        motor_pos_des = link_pos_des * self.N
        motor_vel_des = link_vel_des * self.N
        motor_acc_des = link_acc_des * self.N

        # 2. Calculate feedforward friction torque based on desired velocity
        tau_friction_ff = self.compute_friction_torque(link_vel_des)
        
        # 3. Calculate flexibility/transmission torque
        # For a feedforward model, we assume we are perfectly tracking, so the
        # deflection `delta_theta` is what is required to produce `tau_rnea`.
        # A simplified model often assumes the hysteresis component is zero for FF.
        # tau_transmission = K * delta_theta + tau_hysteresis
        # For simplicity in feedforward, we can approximate the required transmission
        # torque as the sum of RNEA and friction. More complex models could
        # try to predict the deflection.
        tau_transmission_ff = tau_rnea + tau_friction_ff

        # 4. Calculate the total motor torque
        # τ_motor = J_m * α_m + (1/N) * τ_transmission
        tau_motor_ff = self.J_m * motor_acc_des + (1.0 / self.N) * tau_transmission_ff
        
        return tau_motor_ff