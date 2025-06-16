# Pinocchio-Based Robot Dynamics Module

This module implements robot dynamics computations using [Pinocchio](https://stack-of-tasks.github.io/pinocchio/) — a fast and flexible C++ library with Python bindings for multi-body dynamics and robotics. It supports:
1. Forward and Inverse dynamics 
2. Jacobian computations
3. Operational space control.

## 📁 File: `pinocchio_dynamics.py`

### Summary

This file defines the `PinocchioRobotDynamics` class, which loads a robot URDF model and enables computation of:

* Inverse dynamics using RNEA (Recursive Newton-Euler Algorithm)
* Geometric Jacobians for the end-effector
* Task-space (operational space) inertia matrices
* Runtime updating of link inertial parameters from a flat vector

---

## 🛠️ Dependencies

* [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
* NumPy
* A `robot_config` module (user-supplied) that provides:

  * `NUM_JOINTS`: Number of actuated joints
  * `END_EFFECTOR_FRAME_NAME`: Name of the robot's end-effector frame

---

## 🧠 Class: `PinocchioRobotDynamics`

### `__init__(urdf_path)`

Initializes the robot model from a URDF file with a floating base (`JointModelFreeFlyer`). Sets up the model and data structures required for further computations.

---

### `set_parameters_from_vector(P_vec)`

**Purpose:**
Updates the dynamic parameters (mass, center of mass, inertia) of each link in the model using a flat parameter vector.

**Math involved:**
Each link's 10 parameters are:

* 6 elements for inertia tensor about the COM: `[ixx, ixy, ixz, iyy, iyz, izz]`
* 3 for mass-weighted COM position: `[mcx, mcy, mcz]`
* 1 for mass

Let:

* $\mathbf{c} = \frac{1}{m} [mcx, mcy, mcz]$ — Center of mass
* $\mathbf{I}_{com}$ — Inertia about COM
* $\mathbf{I}_{origin} = \mathbf{I}_{com} - m \cdot [\mathbf{c}]_\times [\mathbf{c}]_\times$ — Inertia about origin using the **parallel axis theorem**

To ensure physical consistency, the inertia matrix is adjusted to be symmetric positive definite (SPD) using a nearest-SPD projection via SVD.

---

### `compute_rnea(q, qd, qdd, gravity)`

**Purpose:**
Computes the joint torques required to follow a desired motion using **Inverse Dynamics**.

**Math (RNEA):**
Uses:

* $\tau = RNEA(q, \dot{q}, \ddot{q})$

This method fills a full state vector by embedding actuated joint values into a floating-base model (with proper indexing offsets) and applies the gravity vector before calling `pin.rnea`.

---

### `compute_jacobian(q)`

**Purpose:**
Computes the **geometric Jacobian** for the end-effector.

**Math:**

* $\mathbf{v}_{ee} = \mathbf{J}(q) \cdot \dot{q}$
* Uses `pin.computeJointJacobians` and `pin.getFrameJacobian` to get the Jacobian matrix in the local frame.

Only actuated joint columns are returned: `J[:, 6:]`.

---

### `compute_task_space_inertia(J)`

**Purpose:**
Computes the **operational space inertia matrix** (aka task-space inertia), $\Lambda(q)$

**Math:**

* $\Lambda(q) = (J M^{-1} J^T)^{-1}$
* This maps forces in task space to accelerations:
  $F_{ee} = \Lambda(q) \ddot{x}_{ee}$

For numerical stability:

* Solves $M x = J^T$ for $x$
* Then $J M^{-1} J^T = J x$
* Inverts using pseudo-inverse in case of singularities

---

## 📦 Usage Example

```python
from pinocchio_dynamics import PinocchioRobotDynamics
import numpy as np

# Initialize model
robot = PinocchioRobotDynamics("path/to/your_robot.urdf")

# Set identified parameters (flattened 10*N vector)
robot.set_parameters_from_vector(P_vec)

# Inverse dynamics
tau = robot.compute_rnea(q, qd, qdd)

# Jacobian
J = robot.compute_jacobian(q)

# Task-space inertia
Lambda = robot.compute_task_space_inertia(J)
```

---

## 📌 Notes

* Indices are carefully managed to accommodate floating base models.
* Inertia tensors are corrected to maintain physical realism.
* Assumes a fixed order of joint parameters in the flat vector: `[ixx, ixy, ixz, iyy, iyz, izz, mcx, mcy, mcz, mass]`.

---

## 📚 References

* Featherstone, R. *"Rigid Body Dynamics Algorithms"*
* Operational space control: Khatib, O. *"A Unified Approach for Motion and Force Control of Robot Manipulators"*
* [Pinocchio Documentation](https://stack-of-tasks.github.io/pinocchio/)
