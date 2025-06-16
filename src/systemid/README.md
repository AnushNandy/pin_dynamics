# Robot System Identification

A comprehensive system identification toolkit for robotic manipulators that identifies both rigid-body dynamics parameters and joint friction coefficients using Pinocchio and PyBullet.

## Overview

This package provides a complete pipeline for identifying the dynamic parameters of robotic manipulators, including:
- **Rigid-body parameters**: Mass, center of mass, and inertia tensor components (10 parameters per link)
- **Friction parameters**: Viscous and Coulomb friction coefficients (2 parameters per joint)

The identification process uses the dynamics regressor approach, where the robot's equation of motion is linearized with respect to the unknown parameters.

## Key Features

- **Robust URDF handling**: Automatically handles both fixed-base and floating-base robot models
- **Comprehensive parameter identification**: Simultaneously identifies rigid-body and friction parameters
- **Exciting trajectory generation**: Uses Fourier series to generate persistently exciting trajectories
- **Ground-truth data generation**: Leverages PyBullet's inverse dynamics for accurate reference data
- **Regularized least squares**: Includes L2 regularization to handle ill-conditioned problems

## Architecture

```
├── generate_sysid_data.py         # Data generation using PyBullet
├── pinocchio_friction_regressor.py # Regressor matrix computation
└── pin_identify_parameters.py     # Parameter identification
```

## Dependencies

```python
# Core dependencies
import pinocchio as pin
import pybullet as p
import numpy as np
import pybullet_data

# Project-specific
from config import robot_config
```

### Installation Requirements

```bash
pip install pinocchio
pip install pybullet
pip install numpy
```

## Usage

### 1. Data Generation

Generate training data with persistently exciting trajectories:

```bash
python generate_sysid_data.py
```

This script:
- Loads the robot URDF in PyBullet
- Generates Fourier series trajectories for each joint
- Computes ground-truth torques using inverse dynamics
- Saves data to `sysid_data.npz`

**Key parameters:**
- `SIM_DURATION`: Duration of data collection (default: 20s)
- `TIME_STEP`: Simulation timestep (default: 1/240s)
- `num_harmonics`: Number of harmonics in Fourier series (default: 5)

### 2. Parameter Identification

Identify dynamic parameters from the collected data:

```bash
python pin_identify_parameters.py
```

This script:
- Loads the generated trajectory data
- Builds the dynamics regressor matrix using Pinocchio
- Solves the regularized least squares problem
- Saves identified parameters to `identified_params.npz`

**Key parameters:**
- `L2_REGULARIZATION`: Regularization coefficient (default: 1e-6)

## Mathematical Foundation

### Dynamics Regressor

The robot's equation of motion can be written as:
$$
τ = M(q)q̈ + C(q,q̇)q̇ + G(q) + F(q̇)
$$

This is linearized with respect to parameters as:
$$
τ = Y(q,q̇,q̈) × P
$$

Where:
- $Y$: Dynamics regressor matrix
- $P$: Parameter vector [rigid-body params, friction params]

### Parameter Vector Structure

For a 7-DOF manipulator:
```
P = [m₁, c₁ₓ, c₁ᵧ, c₁ᵤ, I₁ₓₓ, I₁ᵧᵧ, I₁ᵤᵤ, I₁ₓᵧ, I₁ₓᵤ, I₁ᵧᵤ,  # Link 1 (10 params)
     ...
     m₇, c₇ₓ, c₇ᵧ, c₇ᵤ, I₇ₓₓ, I₇ᵧᵧ, I₇ᵤᵤ, I₇ₓᵧ, I₇ₓᵤ, I₇ᵧᵤ,  # Link 7 (10 params)
     Fv₁, Fc₁, Fv₂, Fc₂, ..., Fv₇, Fc₇]                      # Friction (14 params)
```

Total: 84 parameters (70 rigid-body + 14 friction)

### Friction Model

Joint friction is modeled as:
```
τ_friction = Fv × q̇ + Fc × sign(q̇)
```

Where:
- `Fv`: Viscous friction coefficient
- `Fc`: Coulomb friction coefficient

## Configuration

Update `robot_config.py` with your robot specifications:

```python
# robot_config.py
NUM_JOINTS = 7
URDF_PATH = "path/to/your/robot.urdf"
```

## File Descriptions

### `generate_sysid_data.py`

Generates ground-truth training data using PyBullet simulation.

**Key functions:**
- `generate_fourier_series_trajectory()`: Creates persistently exciting trajectories
- Uses PyBullet's `calculateInverseDynamics()` for accurate torque computation

**Output:** `sysid_data.npz` containing:
- `q`: Joint positions [N_samples × N_joints]
- `qd`: Joint velocities [N_samples × N_joints] 
- `qdd`: Joint accelerations [N_samples × N_joints]
- `tau`: Joint torques [N_samples × N_joints]

### `pinocchio_friction_regressor.py`

Core regressor computation using Pinocchio.

**Key class:** `PinocchioAndFrictionRegressorBuilder`

**Features:**
- Handles floating-base models by proper state vector construction
- Computes rigid-body regressor using `pin.computeJointTorqueRegressor()`
- Appends friction parameter columns
- Robust parameter slicing for actuated joints only

### `pin_identify_parameters.py`

Parameter identification using regularized least squares.

**Process:**
1. Load trajectory data
2. Build stacked regressor matrix Y
3. Solve: `(YᵀY + λI)P = YᵀT` where λ is regularization
4. Extract and display friction coefficients

## Troubleshooting

### Common Issues

1. **URDF not found**
   - Verify `URDF_PATH` in configuration
   - Ensure file exists and is readable

2. **Shape mismatch errors**
   - Check `NUM_JOINTS` matches your robot
   - Verify URDF has expected joint structure

3. **Poor identification results**
   - Increase `SIM_DURATION` for more data
   - Adjust `num_harmonics` for more excitation
   - Tune `L2_REGULARIZATION` parameter

4. **Numerical issues**
   - Increase regularization if matrix is singular
   - Check for NaN/Inf values in trajectory data

### Validation

Compare identified parameters against:
- CAD model specifications
- Manufacturer datasheets
- Cross-validation with held-out data

## Advanced Usage

### Custom Trajectory Generation

Replace Fourier series with your own exciting trajectories:

```python
def custom_trajectory(t):
    # Your trajectory generation logic
    return q_des, qd_des, qdd_des
```

### Parameter Constraints

Add physical constraints to the optimization:

```python
# Example: Positive mass constraints
from scipy.optimize import lsq_linear

# Set bounds for parameters
bounds = (lower_bounds, upper_bounds)
result = lsq_linear(Y_stack, tau_stack, bounds=bounds)
```

## References

1. Khalil, W., & Dombre, E. (2004). *Modeling, identification and control of robots*
2. Gautier, M., & Khalil, W. (1990). "Direct calculation of minimum set of inertial parameters of serial robots"
3. Pinocchio documentation: https://stack-of-tasks.github.io/pinocchio/
