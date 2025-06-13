import math
import numpy as np
import sys
import scipy.optimize as scioptimize
import scipy.spatial.transform as scitransform

sys.dont_write_bytecode = True


class Plane:
    plane_normal: [float] = [1.0, 0.0, 0.0]
    plane_point: [float] = [0.0, 0.0, 0.0]

    def __init__(self, normal, point):
        """ Plane Class

        :param list[float] normal: Normal vector of the plane
        :param list[float] point: Coordinates of a point on the plane
        """
        self.plane_normal = normal / np.linalg.norm(normal)
        self.plane_point = point


def solve_ax_equals_b(a, b, overdetermined):
    """
    :param np.ndarray a: coefficient matrix
    :param np.ndarray b: ordinate
    :param bool overdetermined: true if system is overdetermined, false when well determined
    :return np.ndarray | tuple[np.ndarray, np.ndarray, int, np.ndarray]: solution to ax=xb system
    """

    if len(a) != len(b):
        raise RuntimeError(f"Data Processing Error - a and b length mismatch")

    if overdetermined:
        return np.linalg.lstsq(a, b)
    else:
        return np.linalg.solve(a, b)


def solve_ax_equals_xb_over_determined_system(aa, bb):
    """
    This implementation has been taken from the paper attached in design document of MathUtils
    :param np.ndarray aa: adjacent 4x4 transforms for 1
    :param np.ndarray bb: adjacent 4x4 transforms for 2
    :return np.ndarray: x which is the solution to Ax=xB system
    """
    if len(aa) != len(bb):
        raise RuntimeError(f"Data Processing Error - aa and bb length mismatch")

    if len(aa) == 0:
        raise RuntimeError(f"Data Processing Error - aa and bb are empty")

    num_motions = len(aa)
    a_rows = 9 * num_motions
    a_cols = 9
    a = np.zeros((a_rows, a_cols))

    # Construct matrix A
    for idx in range(num_motions):
        ra = aa[idx, :3, :3]
        rb = bb[idx, :3, :3]
        a[9 * idx:9 * (idx + 1), :] = np.kron(ra, np.eye(3)) + np.kron(-np.eye(3), rb.T)

    # Perform SVD on matrix A
    u, s, vt = np.linalg.svd(a)
    v = vt.T

    # Extract the last column of V as a rotational matrix
    last_v_col = v[:, -1].reshape(3, 3)
    r = last_v_col

    # Normalize R. Why is this needed?
    determinant_r = np.linalg.det(r)
    normalization_denominator = abs(determinant_r ** 1/3)
    normalization_numerator = np.sign(determinant_r)
    normalization_factor = normalization_numerator / normalization_denominator
    r_normed = r * normalization_factor

    # Perform SVD on normalized R
    u_r, _, vt_r = np.linalg.svd(r_normed)

    # Calculate the optimal rotation matrix real_R
    real_r = u_r @ vt_r

    # Adjust real_R if determinant is negative
    if np.linalg.det(real_r) < 0:
        diags = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        real_r = u_r @ diags @ vt_r

    # Solve for translation vector t
    c = np.zeros((3 * num_motions, 3))
    d = np.zeros((3 * num_motions, 1))

    for idx in range(num_motions):
        f = aa[idx, :3, 3]
        c[3 * idx:3 * (idx + 1), :] = np.eye(3) - aa[idx, :3, :3]
        d[3 * idx:3 * (idx + 1), 0] = aa[idx, :3, 3] - np.matmul(real_r, bb[idx, :3, 3].T)

    t = solve_ax_equals_b(c, d, overdetermined=True)[0]

    # Construct the transformation matrix X
    x = np.eye(4)
    x[:3, :3] = real_r
    x[:3, 3] = t.squeeze()

    return x


def compute_forward_kinematics_error(kdl, joint_angles, measured_transform, neocis_convention):
    """
    finds translation and angular error between forward kinematics generated pose of end effector
    compared to measured 6DOF pose of end effector.

    :param list[float] kdl: link parameters of the robot in neocis convention
    :param np.ndarray | list[float] joint_angles: list of angles reported by the encoders of the robot
    :param list[list[float]] measured_transform: list of transforms against which the FK has to be compared.
    Transforms should be input in robot base frame
    :param bool neocis_convention: specify kdl is in neocis convention or not, for performing forward kinematics
    :return tuple[float, float]: angular and translation errors
    """
    ee_transform, _ = compute_forward_kinematics(kdl, list(joint_angles), neocis_convention=neocis_convention)
    angular_diff, translation_diff = compute_difference_between_transforms(ee_transform, measured_transform)
    return angular_diff, translation_diff

def check_orthonormality(rotation_matrix):
    """
    Check orthonormality of a rotation matrix

    :param ndarray rotation_matrix: The rotation matrix to be checked. Axes are normalized.
    :return bool: A flag indicating whether the matrix is orthonormal or not
    """
    # check if R @ R.T == I within a tolerance
    return np.allclose(rotation_matrix @ rotation_matrix.T, np.identity(3), rtol=1e-6, atol=1e-6)


def check_homogeneous_transform(transformation_matrix):
    """
    Check if a provided matrix is a valid Homogeneous Transform

    :param list[list[float]] transformation_matrix: The transformation matrix to be checked. Axes are normalized.
    :return bool: A flag indicating whether the matrix is a valid Homogeneous Transform or not
    """
    # Convert the matrix to an array
    matrix = np.array(transformation_matrix)

    # Check the dimensionality of the matrix
    if matrix.shape != (4, 4):
        return False

    # Check if the bottom row is [0, 0, 0, 1]
    if not np.all(matrix[3, :] == [0, 0, 0, 1]):
        return False

    # Check if the rotation component is orthonormal
    return check_orthonormality(matrix[:3, :3])


def convert_to_homogeneous_transform(pose_parameters, neocis_convention, is_degree=False):
    """
    Compute homogeneous transform given pose parameters in either neocis or standard convention

    :param list[float] pose_parameters: Pose parameters in the form of [Tx, Ty, Tz, Rz, Ry, Rx]
    :param bool neocis_convention: A flag indicating whether parameters are in neocis convention or not (standard)
    :param bool is_degree: A flag indicating whether rotation values are in degrees or not
    :return ndarray: 4x4 homogeneous transform with float type
    """
    Tx, Ty, Tz, Rz, Ry, Rx = pose_parameters

    if is_degree:
        Rz = np.deg2rad(Rz)
        Ry = np.deg2rad(Ry)
        Rx = np.deg2rad(Rx)

    # pre-compute cos and sin values to speed up computation
    sinRx, cosRx = np.sin(Rx), np.cos(Rx)
    sinRy, cosRy = np.sin(Ry), np.cos(Ry)
    sinRz, cosRz = np.sin(Rz), np.cos(Rz)

    R_00 = cosRy*cosRz
    R_01 = cosRz*sinRx*sinRy - cosRx*sinRz
    R_02 = cosRx*cosRz*sinRy + sinRx*sinRz

    R_10 = cosRy*sinRz
    R_11 = cosRx*cosRz + sinRy*sinRx*sinRz
    R_12 = -cosRz*sinRx + cosRx*sinRy*sinRz

    R_20 = -sinRy
    R_21 = cosRy*sinRx
    R_22 = cosRx*cosRy

    if neocis_convention:  # rotation then translation
        transform = np.asarray([
            [R_00, R_01, R_02, Tx*R_00 + Ty*R_01 + Tz*R_02],
            [R_10, R_11, R_12, Tx*R_10 + Ty*R_11 + Tz*R_12],
            [R_20, R_21, R_22, Tx*R_20 + Ty*R_21 + Tz*R_22],
            [0, 0, 0, 1]])
    else:  # standard convention -- translation then rotation
        transform = np.asarray([
            [R_00, R_01, R_02, Tx],
            [R_10, R_11, R_12, Ty],
            [R_20, R_21, R_22, Tz],
            [0, 0, 0, 1]])

    return transform


def convert_to_pose_parameters(transform, neocis_convention, return_degree=False):
    """
    Compute pose parameters given homogeneous transform

    :param list[list[float]] transform: Homogeneous transform
    :param bool neocis_convention: A flag indicating whether output parameters are in neocis convention or not (standard)
    :param bool return_degree: A flag indicating whether output parameter rotation should be in degrees or not
    :return ndarray: Pose parameters in the form of [Tx, Ty, Tz, Rz, Ry, Rx]
    """
    transform = np.array(transform)

    is_orthonormal = check_orthonormality(transform[:3, :3])
    if not is_orthonormal:
        raise RuntimeError("Input rotation matrix is not orthonormal")

    pose_parameters = np.zeros(6)
    rx = math.atan2(transform[2, 1], transform[2, 2])
    ry = math.asin(-transform[2, 0])
    rz = math.atan2(transform[1, 0], transform[0, 0])
    t_vector = transform[:3, 3]

    if neocis_convention:
        R = transform[0:3, 0:3]
        t_vector = np.linalg.inv(R) @ t_vector

    pose_parameters[:3] = t_vector

    if return_degree:
        rz = np.rad2deg(rz)
        ry = np.rad2deg(ry)
        rx = np.rad2deg(rx)

    pose_parameters[3:] = [rz, ry, rx]

    return pose_parameters


def transform_points_between_frames(points, transform):
    """
    Transform one set of points from one frame of reference to another frame

    :param list[list[float]] points: List of points in the original frame
    :param list[list[float]] transform: 4x4 transform that describes the transform from the original frame to another frame
    :return ndarray: List of transformed points in the new frame
    """
    n = len(points)
    transform = np.array(transform)

    # format points for matrix multiplication (each column is one point)
    points = np.array(points).T
    points = np.vstack((points, np.ones(n)))

    transformed_points = transform @ points
    return transformed_points[:3, :].T


def transform_force_torque_between_frames(force_vector, torque_vector, transform):
    """
    Transform a set of forces and torques from one frame of reference to another frame

    :param list[float] force_vector: The force vector in the original frame.
    :param list[float] torque_vector: The torque vector in the original frame.
    :param list[list[float]] transform: 4x4 transform that describes the transform from the original frame to another frame
    :return tuple[np.ndarray, np.ndarray]: A tuple featuring [The force vector in the new frame,
                                                              The torque vector in the new frame]
    """
    # Check that the provided transformation matrix is valid
    transform = np.array(transform)
    if not check_homogeneous_transform(transform):
        raise RuntimeError(f"Data Processing Error - An invalid Homogeneous Transformation Matrix was provided.")

    # Extract the two components from the transform
    rotation_matrix = transform[:3, :3]
    translation_vector = transform[:3, 3]

    # Transform the data
    transformed_forces = rotation_matrix @ force_vector
    transformed_torques = rotation_matrix @ (torque_vector + np.cross(translation_vector, force_vector))

    return transformed_forces, transformed_torques

def convert_axisangle_to_matrix(axis, angle, is_degree=False):
    """
    Convert axis-angle representation of rotation to a rotation matrix

    :param list[float] axis: 3x1 list describing the axis of rotation
    :param float angle: The angle of rotation about the axis 
    :param bool is_degree: A flag indicating whether angles are given in degrees or not
    :return ndarray: 3x3 rotation matrix
    """
    if is_degree:
        angle = np.deg2rad(angle)

    # ensure axis is a unit vector
    x, y, z = np.array(axis) / np.linalg.norm(axis)
    cos, sin = np.cos(angle), np.sin(angle)

    R_00 = x**2 * (1-cos) + cos
    R_01 = x*y*(1-cos) - z*sin
    R_02 = x*z*(1-cos) + y*sin

    R_10 = x*y*(1-cos) + z*sin
    R_11 = y**2 * (1-cos) + cos
    R_12 = y*z*(1-cos) - x*sin

    R_20 = x*z*(1-cos) - y*sin
    R_21 = y*z*(1-cos) + x*sin
    R_22 = z**2 * (1-cos) + cos

    rotation_matrix = np.asarray([
        [R_00, R_01, R_02],
        [R_10, R_11, R_12],
        [R_20, R_21, R_22]])

    return rotation_matrix


def compute_difference_between_transforms(transform1, transform2, return_degree=False):
    """
    Compute the angular and translational difference between two transforms

    :param list[list[float]] | ndarray transform1: 1st transform to compare
    :param list[list[float]] | ndarray transform2: 2nd transform to compare
    :param bool return_degree: A flag indicating whether angular error is given in degrees or not
    :return tuple[float, float]: A tuple with: [angular error, translational error]
    """
    transform1, transform2 = np.array(transform1), np.array(transform2)
    if not check_orthonormality(transform1[:3, :3]):
        raise RuntimeError("Input rotation matrix 1 is not orthonormal")
    if not check_orthonormality(transform2[:3, :3]):
        raise RuntimeError("Input rotation matrix 2 is not orthonormal")

    t_vector_1, t_vector_2 = transform1[:3, 3], transform2[:3, 3]
    translational_diff = np.linalg.norm(t_vector_2 - t_vector_1)

    R_matrix_1, R_matrix_2 = transform1[:3, :3], transform2[:3, :3]
    R_matrix_diff = R_matrix_1 @ R_matrix_2.T
    angular_diff = math.acos(np.clip((np.trace(R_matrix_diff) - 1) / 2, -1, 1))

    if return_degree:
        angular_diff = np.rad2deg(angular_diff)

    return angular_diff, translational_diff


def compute_average_transform(transforms):
    """
    Compute 4x4 transform describing the average of a list of transforms

    :param list[ndarray] transforms: List of 4x4 matrices to average
    :return ndarray: The 4x4 transform that describes the average of the input transforms
    """
    # check orthonormality
    is_orthonormal = [check_orthonormality(np.array(transform)[:3, :3]) for transform in transforms]
    if not all(is_orthonormal):
        non_orthonormal = [idx for idx, value in enumerate(is_orthonormal) if not value]
        raise RuntimeError(f"Input transforms at indices {non_orthonormal} are not orthonormal")
    # average all transforms
    average_transform = np.mean(np.array(transforms), axis=0)
    t_average = average_transform[:3, 3]
    R_average = average_transform[:3, :3]

    # convert R_average to orthonormal matrix
    U, S, Vt = np.linalg.svd(R_average)
    R_matrix = U @ Vt

    average_frame = np.identity(4)
    average_frame[:3, :3] = R_matrix
    average_frame[:3, 3] = t_average

    return average_frame


def compute_transform_list_errors_from_average(transform_list):
    """
    Calculate angular and translational errors relative to the average transform.

   :param list[ndarray] transform_list: A list of 4x4 transformation matrices (numpy arrays).
   :return tuple[list[float], list[float], ndarray]: A tuple containing:
       - A list of angular errors (radians) for each transform relative to the average transform.
       - A list of translational errors (meters) for each transform relative to the average transform.
       - The average transform
   """
    avg_marker_transform = compute_average_transform(transform_list)
    angular_errors, translation_errors = [], []
    # compare each transform to the average and append values to lists
    for marker_transform in transform_list:
        angular_error, translation_error = compute_difference_between_transforms(marker_transform, avg_marker_transform, return_degree=False)
        angular_errors.append(angular_error)
        translation_errors.append(translation_error)
    return angular_errors, translation_errors, avg_marker_transform


def compute_point_cloud_registration(source_points, target_points):
    """
    Carries out point cloud registration to find the rigid transform between two sets of points
    Algorithm from http://nghiaho.com/?page_id=671

    :param list[list[float]] source_points: List of source points
    :param list[list[float]] target_points: List of target points
    :return ndarray: 4x4 transform that describes the least squares distance based rigid transform
    """
    point_count = len(source_points)
    # computes rigid transform for 4 or more corresponding pairs of points
    if point_count < 3:
        raise RuntimeError("Computation Error - Insufficient number of points")
    if len(source_points) != len(target_points):
        raise RuntimeError("Computation Error - Mismatch in number of source vs target points")

    # get centroid of both point clouds
    source_center = np.sum(source_points, axis=0) / point_count
    target_center = np.sum(target_points, axis=0) / point_count

    # subtract the center to prepare for SVD
    mat_s = np.subtract(source_points, source_center)
    mat_t = np.subtract(target_points, target_center)

    # apply SVD to compute the rotation
    mat_t_transpose = mat_t.transpose()
    matrix_h = mat_t_transpose @ mat_s
    U, S, Vt = np.linalg.svd(matrix_h)
    R = (Vt.T @ U.T).transpose()

    # if negative determinant, multiply the 3rd column of the Vt matrix by -1 to fix the rotation
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = (Vt.T @ U.T).transpose()

    # find out the translation using the formula
    # R x centroid_source + t = centroid_target
    t = (-R @ np.array(source_center)) + np.array(target_center)

    # convert R and t to 4x4 matrix form
    result = np.identity(4)
    result[:3, :3] = R
    result[:3, 3] = t

    return result


def calculate_rms(data):
    """
    calculate rms value of input array
    :param np.ndarray data: 1D array to calculate rms over
    :return float: rms value of the input data
    """
    rms = np.sqrt(np.mean(data**2))
    return rms


def compute_forward_kinematics(kdl, joint_angles, neocis_convention=True):
    """
    Compute pose for each joint in the kinematic chain

    :param list[float] kdl: Kdl of arm using Neocis convention
    :param list[float] joint_angles: Specified joint angles. J0 will be automatically added
    :param bool neocis_convention: A flag indicating whether parameters are in neocis convention or not (standard)
    :return tuple[ndarray, list[ndarray]] | tuple[None, None]: A tuple with: [end effector transform, list of 4x4 pose transforms for each joint]
    """
    if len(kdl) % 6 != 0:  # check if length of kdl is a multiple of 6
        return None, None

    # reshape a copy of the kdl as a np.ndarray of type float
    kdl = np.reshape(np.asarray(np.copy(kdl), dtype="float"), (int(len(kdl)/6), 6))

    # apply joint angles to Rz
    for n in range(len(joint_angles)):
        kdl[n+1][3] += joint_angles[n]

    # start with base frame
    base_frame = convert_to_homogeneous_transform(kdl[0], neocis_convention)
    combined_transform = base_frame
    per_joint_transforms = [base_frame]

    for n in range(1, len(kdl)):
        current_transform = convert_to_homogeneous_transform(kdl[n], neocis_convention)
        combined_transform = combined_transform @ current_transform
        per_joint_transforms.append(combined_transform)

    end_effector_transform = per_joint_transforms[-1]
    return end_effector_transform, per_joint_transforms


def generate_common_base_frame(points):
    """
    Generate the coordinate frame of the common base using the locations of S1, S2 and S3 divots
    This function is only applicable to Gen1.

    :param list[list[float]] points: List of the S1, S2, and S3 divot points
    :return ndarray: 4x4 transform that represents the base frame WRT the base coordinate frame of the robot
    """
    s1, s2, s3 = points

    # create y vector
    y_vector = np.array(s3) - np.array(s2)
    y_vector = y_vector / np.linalg.norm(y_vector)

    plane_vector = np.array(s1) - np.array(s2)
    plane_vector = plane_vector / np.linalg.norm(plane_vector)

    # z vector is the cross product of the plane vector and y vector
    z_vector = np.cross(plane_vector, y_vector)
    z_vector = z_vector / np.linalg.norm(z_vector)

    # x vector is cross product of y and z vector
    x_vector = np.cross(y_vector, z_vector)
    x_vector = x_vector / np.linalg.norm(x_vector)

    transform_device2cb = np.identity(4)
    transform_device2cb[:3, 0] = x_vector
    transform_device2cb[:3, 1] = y_vector
    transform_device2cb[:3, 2] = z_vector
    transform_device2cb[:3, 3] = s1

    # calculate inverse to get T_cb2device
    return np.linalg.inv(transform_device2cb)


def compute_error_norms(source_points, target_points, transform):
    """
    Calculate the error norms between target and registered target points after applying a transformation.

    :param list[list[float]] source_points: List of source points.
    :param list[list[float]] target_points: List of target points.
    :param list[list[float]] transform: Transformation matrix.
    :return numpy.ndarray: Vector of error norms between target and registered target points.
    """
    registered_target_points = transform_points_between_frames(source_points, transform)
    error_norms = np.linalg.norm(np.array(target_points) - registered_target_points, axis=1)
    return error_norms


def compute_error_norms_using_transforms(source_transforms, target_transforms, transform):
    """
    Compute angular and translational errors between target and registered target poses after applying a transformation.

    :param list[numpy.ndarray] source_transforms: List of 4x4 transformation matrices for source points.
    :param list[numpy.ndarray] target_transforms: List of 4x4 transformation matrices for target points.
    :param numpy.ndarray transform: 4x4 transformation matrix to register source points.
    :return tuple[list[float], list[float]]: Lists of angular errors (in degrees) and translational errors (in meters).
    """
    angular_errors, translation_errors = [], []

    # Ensure source_transforms and target_transforms have the same length
    if len(source_transforms) != len(target_transforms):
        raise ValueError("Source transforms and Target transforms must have the same number of elements.")

    for source_point_transform, target_transform in zip(source_transforms, target_transforms):
        # Register the source transform using the given transform
        registered_target_transform = transform @ source_point_transform

        # Compute the difference between the registered source transform and the target transform
        angular_error, translation_error = compute_difference_between_transforms(
            registered_target_transform, target_transform, return_degree=False)

        # Append the errors to the respective lists
        angular_errors.append(angular_error)
        translation_errors.append(translation_error)
    return angular_errors, translation_errors


def calculate_distance_between_points(point1, point2):
    """ Calculate distance between two points

    :param list[float] point1: Coordinate of point1
    :param list[float] point2: Coordinate of point2
    :return float: Distance between the two points
    """
    return np.linalg.norm(np.subtract(point1, point2))


def align_vectors(vector_source, vector_target):
    """ Find the rotation matrix that aligns source to target

    :param list[float] vector_source: source vector
    :param list[float] vector_target: target vector
    :return ndarray: 3x3 rotation matrix
    """
    vector_source = np.reshape(vector_source, (1, -1))
    vector_target = np.reshape(vector_target, (1, -1))

    # scipy.spatial.transform.Rotation.align_vector finds the rotation matrix that transforms b -> a
    rotation_matrix, *_ = scitransform.Rotation.align_vectors(vector_target, vector_source)
    rotation_matrix = rotation_matrix.as_matrix()
    return rotation_matrix


def remove_vector_component_along_direction(vector, direction):
    """ Remove the vector component along a given direction

    :param list[float] vector: Input vector
    :param list[float] direction: The direction along which the component should be removed
    :return ndarray: The vector that is perpendicular to the given direction
    """
    vector, direction = np.array(vector), np.array(direction)
    return vector - (np.dot(vector, direction) / (np.linalg.norm(direction) ** 2)) * direction


def offset_plane_by_distance(plane, offset_distance):
    """ Offset plane by a desired distance along its normal

    :param Plane plane: Input plane
    :param float offset_distance: Desired distance to offset the plane
    :return Plane: An instance of the offset plane
    """
    return Plane(plane.plane_normal, plane.plane_point + plane.plane_normal * offset_distance)


def compute_plane_plane_intersection(plane1, plane2):
    """ Compute the intersection between two planes
    This solves the following system of equations to get a common points (t0) on both planes:
    n1.t = n1.t1
    n2.t = n2.t2
    [[n1]T [n2]T] * t = [[n1.t1] [n2.t2]]

    :param Plane plane1: An instance of the first plane
    :param Plane plane2: An instance of the second plane
    :return tuple[list, list] | tuple[ndarray, ndarray]: tuple of [the intersection point, line direction]
    """
    n1 = plane1.plane_normal
    n2 = plane2.plane_normal
    cross_product = np.cross(n1, n2)
    if np.linalg.norm(cross_product) < 1e-6:
        return [], []
    A = np.array([np.transpose(n1), np.transpose(n2)])
    d = np.array([np.dot(n1, plane1.plane_point), np.dot(n2, plane2.plane_point)])
    t0 = np.dot(np.linalg.pinv(A), d)
    return t0, cross_product


def compute_plane_line_intersection(plane, line_point, line_direction):
    """ Compute the point of intersection between a plane and a line
    Plane: plane_normal.(P - plane_point) = 0
    Line: P = line_point + t*line_direction
    Solves the equation for t:
    plane_normal.(line_point + t*line_direction) = plane_normal.plane_point

    :param Plane plane: A Plane instance describing the plane
    :param list[float]: A point that exists on the line
    :param list[float]: A vector describing the line's direction
    :return list | ndarray: The point of intersection between the plane and the line
    """
    n = plane.plane_normal
    line_direction = line_direction / np.linalg.norm(line_direction)
    if abs(np.dot(n, line_direction)) < 1e-6:
        return []
    t = (np.dot(n, plane.plane_point) - np.dot(n, line_point)) / np.dot(n, line_direction)
    return line_point + t * line_direction


def project_point_onto_plane(point, plane):
    """ Find the projected point on a plane

    :param list[float] point: The desired point to project
    :param Plane plane: The instance of the plane to project onto
    :return ndarray: The projected point
    """
    vec = np.subtract(point, plane.plane_point)
    normal_dist = np.dot(vec, plane.plane_normal)
    projected_point = point - normal_dist * plane.plane_normal
    return projected_point


def compute_signed_distance_point_from_plane(point, plane):
    """ Compute the signed distance between a point and plane

    :param list[float] point: A point coordinate
    :param Plane plane: The desired plane instance to compute signed distance
    :return float: the signed distance between the point and plane (positive is when the point is on the side of the plane normal)
    """
    vec = np.subtract(point, plane.plane_point)
    return np.dot(vec, plane.plane_normal)


def get_linearly_interpolated_points(start_point, end_point, num_points):
    """ Get linearly interpolated points

    :param list[float] start_point: Coordinate of start point
    :param list[float] end_point: Coordinate of end point
    :param int num_points: Number of points in the interpolated list
    :return list[ndarray]: List of interpolated points that include the start and end points
    """
    direction_vec = np.subtract(end_point, start_point)
    length = np.linalg.norm(direction_vec)
    direction_vec = direction_vec / length
    t_list = np.linspace(0.0, length, num=num_points)
    interpolated_points = []
    for t in t_list:
        pt = start_point + t * direction_vec
        interpolated_points.append(pt)
    return interpolated_points


def fit_plane_to_points(points):
    """ Fit a plane to a list of points
    source: https://math.stackexchange.com/questions/3501135/fitting-a-plane-to-points-using-svd

    :param list[list[float]] points: List of points
    :return tuple[Plane, list[float]: A tuple featuring [The Plane instance that best fits the given points,
                                                         A list of the shortest normal distance from each point
                                                         to the plane.]
    """
    if len(points) < 3:
        raise RuntimeError("Insufficient number of points to fit a plane")
    data = np.array(points)
    center = data.mean(axis=0)
    u, s, vt = np.linalg.svd(data - center)
    normal = vt[-1, :]

    error_data = np.dot(data - center, normal)

    return Plane(normal, center), error_data


# Define the optimization cost function
def _circle_residuals(c, x, y):
    """
    Private Cost Function for Circle Fitting
    """
    xc, yc, rc = c
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - rc


def fit_circle_to_points(points):
    """
    Fit a circle to a set of 3D points.

    :param list[list[float]] points: List of points
    :return tuple[list[float], list[float], float, list[float]]: A tuple featuring [The vector normal to the circle on
                                                                                    which the points were fit;
                                                                                    The center of the circle;
                                                                                    The radius of the circle;
                                                                                    A list of the shortest distance from
                                                                                    each point to the circle when
                                                                                    projecting on the circle plane]
    """
    if len(points) < 3:
        raise RuntimeError("Insufficient number of points to fit a plane")

    # Extract the plane on which the points lie
    plane, _ = fit_plane_to_points(points)
    circle_normal = plane.plane_normal
    circle_normal = circle_normal / np.linalg.norm(circle_normal)

    # Rotate the points to flat plane frame for easier computation in 2D
    rot1, _ = scitransform.Rotation.align_vectors([np.array([0, 0, 1])], [circle_normal])
    points_2d = rot1.apply(np.array(points))
    x_points, y_points = points_2d[:, 0], points_2d[:, 1]

    # Extract the reference Z from a point known to lie exactly on the plane
    reference_z = rot1.apply(np.array(plane.plane_point))[2]

    # Compute an initial estimate
    x_m, y_m = np.mean(x_points), np.mean(y_points)
    r_guess = np.mean(np.linalg.norm([x_points - x_m, y_points - y_m], axis=0))

    # Run the optimization and extract the solution (value associated with the ".x")
    xc, yc, circle_radius = scioptimize.least_squares(_circle_residuals, [x_m, y_m, r_guess], args=(x_points, y_points)).x

    # Compute the errors
    distances_to_center = np.linalg.norm(points_2d - [xc, yc, reference_z], axis=1)
    error_data = np.abs(distances_to_center - circle_radius)

    # Rotate the points back to the original frame
    rot2 = rot1.inv()
    circle_center = rot2.apply(np.array([xc, yc, reference_z]))

    return circle_normal, circle_center.flatten(), circle_radius, error_data


def compute_angle_between_vectors(vector_1, vector_2, return_degree):
    """ Find the angle between two vectors

    :param list[float] vector_1: The first vector to be used for angle computation
    :param list[float] vector_2: The second vector to be used for angle computation
    :param bool return_degree: A flag to specify whether the result should be in degrees or radians

    :return float: The angle between the two vectors, in specified units
    """
    # Compute the norm of the provided vectors
    vector_1_norm = np.linalg.norm(vector_1)
    vector_2_norm = np.linalg.norm(vector_2)

    # Check if either of the vectors is a zero vector
    if vector_1_norm == 0 or vector_2_norm == 0:
        raise RuntimeError("Math Computation Error - Cannot compute the angle between two vectors when one or both are 0-vectors.")

    # Compute the angle
    cos_theta = np.dot(vector_1, vector_2) / (vector_1_norm * vector_2_norm)
    angle = np.arccos(cos_theta)

    # Convert to degrees if necessary
    if return_degree:
        angle = np.degrees(angle)

    return angle


def get_skew_symmetric_matrix(v):
    """ Returns the skew-symmetric matrix of a 3D vector v. """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def count2deg(resolution, count_value):
    """
    Unit Conversion Function

    A function that converts the provided value in encoder counts into degrees

    :param int resolution: The resolution of the encoder expressed as the maximum number of counts
    :param int count_value: The value of angle in counts of encoder
    :return float: The corresponding value in degrees
    """

    return count_value * 360 / resolution


def deg2count(resolution, deg_value):
    """
    Unit Conversion Function

    A function that converts the provided value in degrees into encoder counts

    :param int resolution: The resolution of the encoder expressed as the maximum number of counts
    :param float deg_value: The value of angle in degrees
    :return int: The corresponding value in counts of encoder
    """

    return int(round(deg_value * resolution / 360))


def count2rad(resolution, counts):
    """
    :param int resolution: encoder resolution (max count)
    :param int counts: encoder counts, for which the corresponding signed angle has to be calculated
    :return float: angle (in radians) corresponding to counts
    """
    return counts * ((2 * np.pi) / resolution)


def rad2count(resolution, angle):
    """
    :param int resolution: encoder resolution (max count)
    :param float angle: angle (in radians), for which the corresponding encoder count has to be calculated
    :return int: encoder count corresponding to angle (in radians)
    """
    return int(round(angle * (resolution / (2 * np.pi))))
