# Recursive Newton-Euler Algorithm (RNEA) - Mathematical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Algorithm Structure](#algorithm-structure)
4. [Forward Pass: Kinematic Propagation](#forward-pass-kinematic-propagation)
5. [Backward Pass: Force Propagation](#backward-pass-force-propagation)
6. [Implementation Details](#implementation-details)
7. [Coordinate Frame Conventions](#coordinate-frame-conventions)
8. [Validation and Testing](#validation-and-testing)

## Overview

The Recursive Newton-Euler Algorithm (RNEA) is an efficient O(n) computational method for solving the **inverse dynamics problem** of serial-link manipulators. Given joint positions, velocities, and desired accelerations, RNEA computes the required joint torques while accounting for:

- Gravitational forces
- Inertial forces (linear and angular acceleration)
- Coriolis and centrifugal forces
- Gyroscopic effects
- External forces/torques on the end-effector

## Mathematical Foundation

### Newton-Euler Equations of Motion

For each rigid body link *i*, the Newton-Euler equations in the link's coordinate frame are:

**Force Balance (Newton's Second Law):**
```
F_i = m_i * a_{com,i}
```

**Torque Balance (Euler's Equation):**
```
N_i = I_{com,i} * α_i + ω_i × (I_{com,i} * ω_i)
```

Where:
- `F_i` = net force acting on link *i*
- `N_i` = net torque acting on link *i* about its center of mass
- `m_i` = mass of link *i*
- `a_{com,i}` = linear acceleration of link *i*'s center of mass
- `I_{com,i}` = inertia tensor of link *i* about its center of mass
- `α_i` = angular acceleration of link *i*
- `ω_i` = angular velocity of link *i*

### Kinematic Relationships

**Angular Velocity Recursion:**
```
ω_i = R_{i-1,i}^T * ω_{i-1} + q̇_i * z_i
```

**Angular Acceleration Recursion:**
```
α_i = R_{i-1,i}^T * α_{i-1} + ω_i × (q̇_i * z_i) + q̈_i * z_i
```

**Linear Acceleration Recursion:**
```
a_i = R_{i-1,i}^T * [a_{i-1} + α_{i-1} × p_{i-1,i} + ω_{i-1} × (ω_{i-1} × p_{i-1,i})]
```

**Center of Mass Acceleration:**
```
a_{com,i} = a_i + α_i × p_{com,i} + ω_i × (ω_i × p_{com,i})
```

Where:
- `R_{i-1,i}` = rotation matrix from frame *i-1* to frame *i*
- `p_{i-1,i}` = position vector from origin of frame *i-1* to origin of frame *i*
- `p_{com,i}` = position vector from origin of frame *i* to its center of mass
- `z_i = [0, 0, 1]^T` = joint axis (assuming revolute joints aligned with z-axis)
- `q̇_i`, `q̈_i` = joint velocity and acceleration

## Algorithm Structure

The RNEA algorithm consists of two main passes:

### Pass 1: Forward Kinematic Propagation (Base → End-Effector)
Propagates kinematics from the base to the end-effector:
1. Angular velocities: `ω_0 → ω_1 → ... → ω_n`
2. Angular accelerations: `α_0 → α_1 → ... → α_n`
3. Linear accelerations: `a_0 → a_1 → ... → a_n`

### Pass 2: Backward Force Propagation (End-Effector → Base)
Propagates forces and torques from the end-effector to the base:
1. Apply external forces at end-effector
2. Compute inertial forces for each link
3. Propagate net forces and torques backward
4. Project torques onto joint axes to get actuator torques

## Forward Pass: Kinematic Propagation

### Initialization
```python
# Base frame (world frame)
ω_0 = [0, 0, 0]^T
α_0 = [0, 0, 0]^T
a_0 = -R_world_base^T * g  # Gravity in base frame
```

### Recursive Propagation (i = 1 to n)

**Step 1: Transform to Current Frame**
```python
T_{i-1,i} = T_world_{i-1}^{-1} @ T_world_i
R_{i-1,i} = T_{i-1,i}[0:3, 0:3]
p_{i-1,i} = T_{i-1,i}[0:3, 3]
```

**Step 2: Angular Velocity**
```python
ω_i = R_{i-1,i}^T @ ω_{i-1} + q̇_i * z
```

**Step 3: Angular Acceleration**
```python
α_i = R_{i-1,i}^T @ α_{i-1} + ω_i × (q̇_i * z) + q̈_i * z
```

**Step 4: Linear Acceleration**
```python
a_i = R_{i-1,i}^T @ (a_{i-1} + α_{i-1} × p_{i-1,i} + ω_{i-1} × (ω_{i-1} × p_{i-1,i}))
```

## Backward Pass: Force Propagation

### Initialization at End-Effector
```python
# External wrench on end-effector (transformed to EE frame)
f_n = R_world_ee^T @ f_ext[0:3]  # External force
n_n = R_world_ee^T @ f_ext[3:6]  # External torque
```

### Recursive Propagation (i = n to 1)

**Step 1: Compute Inertial Forces**
```python
# Center of mass acceleration
a_{com,i} = a_i + α_i × p_{com,i} + ω_i × (ω_i × p_{com,i})

# Inertial force (Newton's law)
F_{inertial,i} = m_i * a_{com,i}

# Inertial torque (Euler's equation)
N_{inertial,i} = I_{com,i} @ α_i + ω_i × (I_{com,i} @ ω_i)
```

**Step 2: Force/Torque Balance at Link Origin**
```python
# Total force at link i's origin
f_i = f_{i+1} + F_{inertial,i}

# Total torque at link i's origin
n_i = n_{i+1} + N_{inertial,i} + p_{com,i} × F_{inertial,i}
```

**Step 3: Transform to Parent Frame**
```python
# Transform wrench to parent frame (i-1)
f_{i-1} = R_{i-1,i} @ f_i
n_{i-1} = R_{i-1,i} @ n_i + p_{i-1,i} × f_{i-1}
```

**Step 4: Extract Joint Torque**
```python
# Project torque onto joint axis
τ_i = n_{i-1} · z  # Dot product with joint axis
```

## Implementation Details

### Inertia Tensor Transformation

The implementation transforms inertia tensors from the origin to the center of mass using the parallel axis theorem:

```python
# Skew-symmetric matrix of COM position
S = [[0, -p_z, p_y],
     [p_z, 0, -p_x],
     [-p_y, p_x, 0]]

# Transform inertia to COM frame
I_com = I_origin + m * (S @ S^T)
```

This is mathematically equivalent to:
```
I_com = I_origin + m * (||p_com||² * I₃ - p_com ⊗ p_com)
```

### Gravity Handling

Gravity is incorporated by setting the base linear acceleration:
```python
a_0 = -R_world_base^T @ g
```

This effectively treats gravity as a fictitious acceleration in the opposite direction, which propagates through the kinematic chain.

### External Force Integration

External forces/torques applied to the end-effector are transformed into the end-effector's coordinate frame:
```python
f_ext_ee = R_world_ee^T @ f_ext[0:3]
n_ext_ee = R_world_ee^T @ f_ext[3:6]
```

## Coordinate Frame Conventions

### Frame Definitions
- **World Frame**: Fixed inertial reference frame
- **Link Frame i**: Attached to link *i*, typically at joint *i*
- **Joint Axis**: z-axis of each link frame (revolute joints)

### Transformation Matrices
The implementation uses homogeneous transformation matrices:
```
T_world_i = [[R_world_i, p_world_i],
             [0^T,       1        ]]
```

Where:
- `R_world_i` = 3×3 rotation matrix from world to link *i*
- `p_world_i` = 3×1 position vector from world origin to link *i* origin

### Sign Conventions
- **Positive joint angles**: Follow right-hand rule about z-axis
- **Gravity vector**: Points downward in world frame: `g = [0, 0, -9.81]^T`
- **Joint torques**: Positive values produce positive joint accelerations

## Validation and Testing

### Test Scenario
The implementation is validated using a 7-DOF manipulator with:
- **Complex trajectories**: Multi-frequency sinusoidal motions
- **Different joint parameters**: Varying amplitudes, frequencies, and phases
- **Physics simulation**: PyBullet for ground truth comparison
- **Control integration**: Combined with PD feedback for trajectory tracking

### Performance Metrics
The test demonstrates:
- **Tracking accuracy**: RMS errors typically < 15° for complex trajectories
- **Computational efficiency**: Real-time capable (240 Hz simulation)
- **Physical realism**: Smooth, physically plausible torque profiles
- **Stability**: No numerical instabilities or divergence

### Key Validation Points
1. **Energy conservation**: Computed torques should conserve mechanical energy
2. **Passivity**: System should dissipate energy appropriately
3. **Symmetry**: Identical motions should produce identical torques
4. **Gravity compensation**: Static equilibrium should be maintained with appropriate torques

## Mathematical Properties

### Computational Complexity
- **Time complexity**: O(n) where n = number of joints
- **Space complexity**: O(n) for storing intermediate variables
- **Numerical stability**: Well-conditioned for typical manipulator configurations

### Physical Consistency
The algorithm ensures:
- **Force equilibrium**: ΣF = ma for each link
- **Torque equilibrium**: ΣN = Iα + ω×(Iω) for each link
- **Kinematic consistency**: Velocities and accelerations satisfy kinematic constraints

### Advantages over Lagrangian Methods
1. **Computational efficiency**: O(n) vs O(n³) for Lagrangian approach
2. **Physical insight**: Direct interpretation of forces and torques
3. **Numerical robustness**: Avoids symbolic differentiation of kinetic energy
4. **External force handling**: Natural incorporation of external wrenches

## References

1. Featherstone, R. (2008). *Rigid Body Dynamics Algorithms*. Springer.
2. Craig, J.J. (2017). *Introduction to Robotics: Mechanics and Control*. Pearson.
3. Siciliano, B., et al. (2010). *Robotics: Modelling, Planning and Control*. Springer.
4. Luh, J.Y.S., Walker, M.W., Paul, R.P. (1980). "On-line computational scheme for mechanical manipulators." *ASME Journal of Dynamic Systems, Measurement, and Control*, 102(2), 69-76.