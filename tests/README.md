
# Model Validation and Future Work

## 1. Overview

After identifying a new set of dynamic parameters, it is essential to validate their effectiveness. This describes the validation framework (`validate_model.py`) used to quantitatively assess the performance of the identified model in a closed-loop control simulation. It also outlines the next steps for refining the model and transitioning to a real-world robot.

## 2. The Validation Control Strategy: Computed Torque Control (CTC)

To test the quality of our dynamic model, we use it as a feedforward term in a **Computed Torque Controller**. The control law is:

$$
\tau_{cmd} = \tau_{ff} + \tau_{fb}
$$

#### 2.1. The Feedback Term (`τ_fb`)

This is a standard Proportional-Derivative (PD) controller that corrects for any errors between the desired and actual states. Its goal is to drive the tracking error `e = q_{des} - q` to zero.

$$
\tau_{fb} = K_p (q_{des} - q) + K_d (\dot{q}_{des} - \dot{q}) = K_p e + K_d \dot{e}
$$

Where `Kp` and `Kd` are diagonal matrices of positive feedback gains.

#### 2.2. The Feedforward Term (`τ_ff`)

This term uses our identified dynamic model to predict the torque required to follow the desired trajectory. An accurate `τ_ff` significantly reduces the burden on the feedback controller.

$$
\tau_{ff} = \hat{M}(q)\ddot{q}_{des} + \hat{C}(q, \dot{q})\dot{q}_{des} + \hat{G}(q) + \hat{F}(\dot{q}_{des})
$$

The `^` (hat) notation signifies that these are our *estimated* model parameters.

The core idea of CTC is that if our model is perfect (`M̂=M`, `Ĉ=C`, etc.), the feedforward term cancels the robot's true dynamics, leaving a simple, linear error dynamic governed by the PD gains: `ë + Kdė + Kpe = 0`. The better our model, the closer we get to this ideal behavior.

## 3. The Validation Framework (`validate_model.py`)

To properly analyze the contribution of our identified model, the validation script compares three different controllers:

1.  **PD Only (Baseline):** `τ_ff = 0`. This shows the performance with only feedback control. We expect significant tracking lag, especially due to gravity.
2.  **PD + Gravity Compensation:** `τ_ff = Ĝ(q)`. This uses only the gravity component of our model. It tests the accuracy of the identified mass and center-of-mass parameters.
3.  **PD + Full CTC:** This uses the full feedforward term. It tests the accuracy of the complete dynamic model, including inertia and friction.

For each controller, the script runs a PyBullet simulation and plots the joint tracking errors over time, along with the Root Mean Square (RMS) error.

### 3.1. Interpreting the Validation Plots

The plots provide a clear, quantitative measure of model quality.
-   **The Ideal Result:** For each joint, you should see a clear hierarchy of performance: `RMS_Error(PD Only) > RMS_Error(PD + Gravity) > RMS_Error(PD + Full CTC)`. This indicates that each component of the model we add improves tracking performance.
-   **Common Failure Mode 1 (Bad Inertia):** If `CTC` performance is worse than `Gravity` performance, it means your identified inertia or friction parameters are incorrect and are actively harming the controller.
-   **Common Failure Mode 2 (Bad Gains):** If all three controllers show very large, similar errors for a specific joint, it means the `Kp`/`Kd` gains for that joint are too low to overcome the physical loads. The controller is too "weak", which is a tuning problem, not a modeling problem.

## 4. Next Steps: Towards a Real-World Model

This simulation-based pipeline provides a powerful foundation. To apply it to a physical robot, the following steps are required.

### 4.1. Integrating Advanced Friction Models

The current pipeline identifies a simple viscous-coulomb friction model. To incorporate the more complex `JointModel` (with Stribeck and hysteresis effects), a **two-stage identification process** is necessary:

1.  **Stage 1 (Linear Identification):** Run the current pipeline exactly as is. This will identify the rigid-body parameters (`M`, `C`, `G`) and a rough approximation of friction.
2.  **Stage 2 (Non-Linear Identification):**
    a.  Calculate the *residual torque* from your real-world data:
        $$
        \tau_{residual} = \tau_{measured} - Y_{rigidbody}(q, \dot{q}, \ddot{q}) \cdot \hat{\Phi}_{rigidbody}
        $$
        This `τ_residual` represents all the dynamic effects not captured by the rigid-body model—primarily the complex friction.
    b.  Use a non-linear optimization routine (e.g., `scipy.optimize.minimize`) to find the parameters of your `JointModel` (hysteresis shapes, stiction levels, etc.) that best reproduce the `τ_residual` signal.

### 4.2. Transitioning to Real-World Data

The core change is replacing `generate_sysid_data.py` with a script that collects data from the physical hardware.

1.  **Data Acquisition:** Write a program for your robot controller that commands an exciting trajectory (like the Fourier series) and logs the following data at a high, constant frequency (e.g., >100Hz):
    *   `q_measured` (from joint encoders)
    *   `q̇_measured` (from joint encoders, often filtered)
    *   `τ_commanded` (the torque command sent to the motor drivers)

2.  **Estimating Acceleration (`q̈`):** You cannot directly measure acceleration. It must be estimated by numerically differentiating the velocity signal. This process is sensitive to noise. A common and robust method is to use a **Savitzky-Golay filter**, which fits a local polynomial to the data to compute the derivative smoothly.
    ```python
    # Example using scipy
    from scipy.signal import savgol_filter
    
    # Filter velocity signal first to reduce noise
    qd_filtered = savgol_filter(qd_measured, window_length=31, polyorder=3)
    
    # Differentiate the filtered velocity to get acceleration
    qdd_estimated = savgol_filter(qd_filtered, window_length=31, polyorder=3, deriv=1, delta=TIME_STEP)
    ```

3.  **Using the Data:** Once you have collected real-world `(q, q̇, q̈_est, τ_cmd)` data and saved it as an `.npz` file, the rest of the pipeline (`identify_parameters.py` and `validate_model.py`) can be used without modification to identify and validate a model based on the real system's behavior.