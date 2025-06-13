#  Robot Arm System Identification Pipeline

## 1. Overview

Theory and implementation of a system identification pipeline for a 7-DOF serial-link manipulator. The goal is to derive a high-fidelity dynamic model from data, which is essential for implementing advanced control strategies like Computed Torque Control or Impedance Control. The standard URDF model is often inaccurate due to unmodeled friction, motor dynamics, and imprecise inertial parameters.

This pipeline consists of three main phases, each corresponding to a Python script:

1.  **Data Generation (`generate_sysid_data.py`):** Collects ground-truth state and torque data from a PyBullet simulation using an exciting trajectory.
2.  **Regressor Construction (`dynamics_regressor.py`):** Implements a method to construct the dynamics regressor matrix, which linearly relates robot states to dynamic parameters.
3.  **Parameter Identification (`identify_parameters.py`):** Uses the generated data and the regressor to solve for the unknown dynamic parameters via linear least-squares.

## 2. The Theory of Manipulator Dynamics & Identification

### 2.1. The Equation of Motion

The dynamics of a rigid n-DOF serial manipulator can be described by the following equation:

$$
\tau = M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) + F(\dot{q})
$$

Where:
-   `τ` ∈ ℝⁿ is the vector of joint torques.
-   `q`, `q̇`, `q̈` ∈ ℝⁿ are the vectors of joint position, velocity, and acceleration.
-   `M(q)` ∈ ℝⁿˣⁿ is the symmetric, positive-definite **mass matrix**.
-   `C(q, q̇)` ∈ ℝⁿˣⁿ is the **Coriolis and centrifugal matrix**.
-   `G(q)` ∈ ℝⁿ is the **gravity vector**.
-   `F(q̇)` ∈ ℝⁿ is the **friction torque vector**.

### 2.2. Linearity in the Dynamic Parameters

The most crucial property for system identification is that the equation of motion is **linear** with respect to a specific set of dynamic parameters. Each link `i` can be described by a vector of 10 standard dynamic parameters:

$$
\Phi_i = [I_{xx,i}, I_{xy,i}, I_{xz,i}, I_{yy,i}, I_{yz,i}, I_{zz,i}, m_i c_{x,i}, m_i c_{y,i}, m_i c_{z,i}, m_i]^T
$$

Where `I` terms are the elements of the inertia tensor about the link's origin, `mc` terms are the first moments of mass (mass times center-of-mass vector), and `m` is the link mass.

By rearranging the equation of motion, we can express the torque vector as a linear mapping from a vector containing all these link parameters:

$$
\tau = Y(q, \dot{q}, \ddot{q}) \cdot \Phi
$$

Where:
-   `Φ` ∈ ℝ¹⁰ⁿ is the stacked vector of all link dynamic parameters.
-   `Y(q, \dot{q}, \ddot{q})` ∈ ℝⁿˣ¹⁰ⁿ is the **dynamics regressor matrix**. Its elements are complex, non-linear functions of the robot's state (`q`, `q̇`, `q̈`) but are completely independent of the physical parameters `Φ`.

For this project, we augment the parameter vector `Φ` to also include simple friction terms (viscous `Fv` and Coulomb `Fc` for each joint), making the final parameter vector `Φ_full` and the regressor `Y_full`.

### 2.3. Least-Squares Formulation

If we collect `k` samples of robot states and torques, we can stack the regressor equation into a large linear system:

$$
\begin{bmatrix} \tau(t_1) \\ \tau(t_2) \\ \vdots \\ \tau(t_k) \end{bmatrix} = \begin{bmatrix} Y(t_1) \\ Y(t_2) \\ \vdots \\ Y(t_k) \end{bmatrix} \cdot \Phi_{full} \quad \implies \quad \mathbf{T} = \mathbf{Y} \cdot \Phi_{full}
$$

Our goal is to find the parameter vector `Φ_full` that minimizes the squared error `|| Y ⋅ Φ_full - T ||²`. The well-known solution to this linear least-squares problem is:

$$
\hat{\Phi}_{full} = (\mathbf{Y}^T \mathbf{Y})^{-1} \mathbf{Y}^T \mathbf{T}
$$

To improve numerical stability, especially with noisy data or poor excitation, we use **Tikhonov regularization (Ridge Regression)**, which adds a penalty term for large parameter values:

$$
\hat{\Phi}_{full} = (\mathbf{Y}^T \mathbf{Y} + \lambda I)^{-1} \mathbf{Y}^T \mathbf{T}
$$

Where `λ` is a small regularization coefficient.

## 3. Pipeline Implementation

### Phase 1: Data Generation (`generate_sysid_data.py`)

-   **Purpose:** To generate a rich dataset of `(q, q̇, q̈, τ)` that sufficiently "excites" the robot's dynamics.
-   **Method:** A trajectory composed of a **Fourier series** (a sum of sinusoids with different frequencies and phases) is used for each joint. This is a standard technique for ensuring "persistent excitation," making all dynamic effects observable in the data.
-   **Ground Truth:** The simulation is run in PyBullet. For each point `(q_des, q̇_des, q̈_des)` on the trajectory, the "ground truth" torque `τ` is calculated using PyBullet's built-in inverse dynamics function: `p.calculateInverseDynamics()`. This function uses the (inaccurate) parameters from the URDF file, creating a perfect ground truth for our *simulated* world.
-   **Output:** `sysid_data.npz`, containing arrays for `q`, `qd`, `qdd`, and `tau` over the entire simulation.

### Phase 2: Regressor Construction (`dynamics_regressor.py`)

-   **Problem:** Deriving the regressor `Y` symbolically is computationally infeasible for a 7-DOF arm.
-   **Solution:** A **numerical regressor** is constructed. This method exploits the linearity property directly. To compute the `j`-th column of `Y` (which corresponds to parameter `Φ_j`):
    1.  Create a temporary parameter vector `Φ_test` where `Φ_j = 1` and all other parameters are `0`.
    2.  Update the `RobotDynamics` engine with this `Φ_test`.
    3.  Call the RNEA algorithm (`compute_rnea`). The resulting torque vector *is* the `j`-th column of the regressor matrix `Y`.
-   **Implementation:** The `RegressorBuilder.compute_regressor_matrix()` function iterates through every single parameter (all 10 rigid body parameters for each of the 7 links, plus 2 friction parameters for each joint), running RNEA for each one to build the full regressor matrix column by column.

### Phase 3: Solving for Parameters (`identify_parameters.py`)

-   **Purpose:** To implement the least-squares solution described in Section 2.3.
-   **Method:**
    1.  Loads the `sysid_data.npz` file.
    2.  Initializes the `RegressorBuilder`.
    3.  Iterates through every sample `i` in the dataset.
    4.  For each sample, it calls `regressor_builder.compute_regressor_matrix(q_i, qd_i, qdd_i)` to get the `(n x p)` regressor `Y_i`.
    5.  It stacks these matrices into a giant `(k*n x p)` matrix `Y_stack` and the corresponding torques into a `(k*n x 1)` vector `tau_stack`.
    6.  Finally, it solves the regularized linear system `(Y_stack.T @ Y_stack + λ*I) * P = Y_stack.T @ tau_stack` using `np.linalg.solve`.
-   **Output:** `identified_params.npz`, a file containing the single vector `P` of all identified dynamic parameters.

## 4. How to Run

1.  **Generate Data:**
    ```bash
    python generate_sysid_data.py
    ```
2.  **Identify Parameters:**
    ```bash
    python identify_parameters.py
    ```

After running these two scripts, you will have the `identified_params.npz` file, which is the final product of this pipeline and is ready for use in validation and control.
