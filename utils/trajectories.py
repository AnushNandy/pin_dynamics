import numpy as np

def generate_aggressive_trajectory(time_vec, num_joints):
    """
    Generate a more aggressive trajectory with higher accelerations and velocities.
    This will produce much higher torque requirements.
    """
    
    # --- Option 1: Multi-frequency sinusoidal trajectory ---
    def sinusoidal_trajectory(t):
        q = np.zeros(num_joints)
        qd = np.zeros(num_joints)
        qdd = np.zeros(num_joints)
        
        # Different frequencies and amplitudes for each joint
        frequencies = [0.5, 0.8, 1.2]  # Hz
        amplitudes = [np.pi/3, np.pi/2, np.pi/4]  # radians (60°, 90°, 45°)
        phase_shifts = [0, np.pi/4, np.pi/2]
        
        for j in range(num_joints):
            omega = 2 * np.pi * frequencies[j]
            amp = amplitudes[j]
            phase = phase_shifts[j]
            
            # Position, velocity, acceleration
            q[j] = amp * np.sin(omega * t + phase)
            qd[j] = amp * omega * np.cos(omega * t + phase)
            qdd[j] = -amp * omega**2 * np.sin(omega * t + phase)
        
        return q, qd, qdd
    
    return sinusoidal_trajectory

def generate_step_trajectory(time_vec, num_joints):
    """
    Generate a step trajectory with rapid transitions - very high accelerations!
    """
    
    def step_trajectory(t):
        q = np.zeros(num_joints)
        qd = np.zeros(num_joints)
        qdd = np.zeros(num_joints)
        
        # Step parameters
        step_duration = 2.0  # seconds
        transition_time = 0.1  # very fast transitions
        
        # Define step positions for each joint
        step_positions = [
            [0, np.pi/2, -np.pi/2, np.pi/3, -np.pi/3, 0],  # Joint 0
            [0, np.pi/3, -np.pi/3, np.pi/2, -np.pi/2, 0],  # Joint 1  
            [0, np.pi/4, -np.pi/4, np.pi/6, -np.pi/6, 0],  # Joint 2
        ]
        
        for j in range(num_joints):
            # Determine which step we're in
            step_index = int(t / step_duration)
            t_in_step = t - step_index * step_duration
            
            if step_index >= len(step_positions[j]) - 1:
                # Hold final position
                q[j] = step_positions[j][-1]
                qd[j] = 0
                qdd[j] = 0
            else:
                current_pos = step_positions[j][step_index]
                next_pos = step_positions[j][step_index + 1]
                
                if t_in_step < transition_time:
                    # Rapid transition using cubic polynomial
                    s = t_in_step / transition_time
                    s2 = s * s
                    s3 = s2 * s
                    
                    # Cubic blend: 3s² - 2s³
                    blend = 3 * s2 - 2 * s3
                    blend_dot = (6 * s - 6 * s2) / transition_time
                    blend_ddot = (6 - 12 * s) / (transition_time**2)
                    
                    q[j] = current_pos + (next_pos - current_pos) * blend
                    qd[j] = (next_pos - current_pos) * blend_dot
                    qdd[j] = (next_pos - current_pos) * blend_ddot
                else:
                    # Hold position
                    q[j] = next_pos
                    qd[j] = 0
                    qdd[j] = 0
        
        return q, qd, qdd
    
    return step_trajectory

def generate_chirp_trajectory(time_vec, num_joints):
    """
    Generate a chirp (frequency sweep) trajectory - tests robot at multiple frequencies
    """
    
    def chirp_trajectory(t):
        q = np.zeros(num_joints)
        qd = np.zeros(num_joints)
        qdd = np.zeros(num_joints)
        
        # Chirp parameters
        f0 = 0.1  # Start frequency (Hz)
        f1 = 2.0  # End frequency (Hz)
        T = 18.0  # Total duration
        
        # Frequency sweep rate
        k = (f1 - f0) / T
        
        # Amplitude for each joint
        amplitudes = [np.pi/3, np.pi/2, np.pi/4]
        
        for j in range(num_joints):
            amp = amplitudes[j]
            
            # Instantaneous frequency
            freq_t = f0 + k * t
            omega_t = 2 * np.pi * freq_t
            
            # Phase (integral of frequency)
            phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
            
            q[j] = amp * np.sin(phase)
            qd[j] = amp * omega_t * np.cos(phase)
            qdd[j] = -amp * omega_t**2 * np.sin(phase) + amp * 2 * np.pi * k * np.cos(phase)
        
        return q, qd, qdd
    
    return chirp_trajectory

def generate_bang_bang_trajectory(time_vec, num_joints):
    """
    Generate bang-bang trajectory - maximum acceleration/deceleration
    """
    
    def bang_bang_trajectory(t):
        q = np.zeros(num_joints)
        qd = np.zeros(num_joints)
        qdd = np.zeros(num_joints)
        
        # Bang-bang parameters
        move_duration = 3.0  # seconds per move
        max_accel = 10.0  # rad/s²
        target_angles = [np.pi/2, np.pi/3, np.pi/4]
        
        for j in range(num_joints):
            # Determine which move we're in
            move_index = int(t / move_duration)
            t_in_move = t - move_index * move_duration
            
            # Alternate between positive and negative targets
            target = target_angles[j] * (1 if move_index % 2 == 0 else -1)
            
            if move_index == 0:
                start_pos = 0
            else:
                start_pos = target_angles[j] * (1 if (move_index-1) % 2 == 0 else -1)
            
            # Bang-bang profile
            distance = target - start_pos
            accel_time = min(move_duration / 2, np.sqrt(abs(distance) / max_accel))
            decel_time = accel_time
            coast_time = move_duration - accel_time - decel_time
            
            if t_in_move < accel_time:
                # Acceleration phase
                q[j] = start_pos + 0.5 * np.sign(distance) * max_accel * t_in_move**2
                qd[j] = np.sign(distance) * max_accel * t_in_move
                qdd[j] = np.sign(distance) * max_accel
            elif t_in_move < accel_time + coast_time:
                # Coast phase
                coast_vel = np.sign(distance) * max_accel * accel_time
                coast_pos = start_pos + 0.5 * np.sign(distance) * max_accel * accel_time**2
                q[j] = coast_pos + coast_vel * (t_in_move - accel_time)
                qd[j] = coast_vel
                qdd[j] = 0
            else:
                # Deceleration phase
                t_decel = t_in_move - accel_time - coast_time
                coast_vel = np.sign(distance) * max_accel * accel_time
                coast_pos = start_pos + 0.5 * np.sign(distance) * max_accel * accel_time**2
                decel_start_pos = coast_pos + coast_vel * coast_time
                
                q[j] = decel_start_pos + coast_vel * t_decel - 0.5 * np.sign(distance) * max_accel * t_decel**2
                qd[j] = coast_vel - np.sign(distance) * max_accel * t_decel
                qdd[j] = -np.sign(distance) * max_accel
        
        return q, qd, qdd
    
    return bang_bang_trajectory