import numpy as np
import random
import math

def euler_to_quaternion(euler):
    yaw, pitch, roll = euler
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr

    return np.array([qx, qy, qz, qw])

def angular_difference(v1, v2):
    q1 = euler_to_quaternion(v1)
    q2 = euler_to_quaternion(v2)

    # Calculate the relative quaternion (rotation from v1 to v2)
    relative_quaternion = np.multiply(q2, np.array([q1[0], q1[1], q1[2], -q1[3]]))

    # Convert the relative quaternion to Euler angles
    relative_euler = np.array([
        np.arctan2(2 * (relative_quaternion[3] * relative_quaternion[0] + relative_quaternion[1] * relative_quaternion[2]),
                   1 - 2 * (relative_quaternion[0]**2 + relative_quaternion[1]**2)),
        np.arcsin(2 * (relative_quaternion[3] * relative_quaternion[1] - relative_quaternion[2] * relative_quaternion[0])),
        np.arctan2(2 * (relative_quaternion[3] * relative_quaternion[2] + relative_quaternion[0] * relative_quaternion[1]),
                   1 - 2 * (relative_quaternion[1]**2 + relative_quaternion[2]**2))
    ])

    # Ensure the angles are in the desired range (0 to pi)
    # relative_euler = np.abs(relative_euler)

    return relative_euler

def angle_between_vectors(self, vector1, vector2):
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector1, vector2)

    # Calculate the magnitudes (norms) of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate the angle between the vectors in radians
    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))

    return angle_rad


def quaternion_to_euler(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    length = np.sqrt(x**2 + y**2 + z**2 + w**2)
    x /= length
    y /= length
    z /= length
    w /= length

    # Calculate the angles
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(2 * (w * y - x * z))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return yaw, pitch, roll

def get_drone_normal(yaw, pitch, roll):
    yaw += np.pi
    pitch = -pitch
    # Compute rotation matrices for yaw, pitch, and roll
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                     [np.sin(yaw), np.cos(yaw), 0],
                     [0, 0, 1]])

    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])

    R_roll = np.array([[1, 0, 0],
                      [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])

    # Compute the composite rotation matrix
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))

    # Extract the normal vector pointing up (z-axis in body frame)
    normal_vector = np.dot(R, np.array([0, 0, 1]))

    # Normalize the normal vector
    normal_vector /= np.linalg.norm(normal_vector)

    return normal_vector

def new_respawn_point(point):
    distance = random.uniform(0, 2)
    theta = random.uniform(0, 2 * np.pi)  # Azimuthal angle
    phi = random.uniform(0, np.pi)        # Polar angle

    # Convert spherical coordinates to Cartesian coordinates
    x = point[0] + distance * np.sin(phi) * np.cos(theta)
    y = point[1] + distance * np.sin(phi) * np.sin(theta)
    z = point[2] + distance * np.cos(phi)

    return np.array([x, y, z])

def new_move_point(point):
    distance = random.uniform(0, 2)
    theta = random.uniform(0, 2 * np.pi)  # Azimuthal angle
    phi = random.uniform(0, np.pi)        # Polar angle

    # Convert spherical coordinates to Cartesian coordinates
    x = point[0] + distance * np.sin(phi) * np.cos(theta)
    y = point[1] + distance * np.sin(phi) * np.sin(theta)
    z = point[2] + distance * np.cos(phi)

    return np.array([x, y, z])

def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1.x, q1.y, q1.z, q1.w
    w2, x2, y2, z2 = q2.x, q2.y, q2.z, q2.w

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return x, y, z, w

def quaternion_conjugate(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return -x, -y, -z, w

def quaternion_difference(q1, q2):
    q2_conjugate = quaternion_conjugate(q2)
    return quaternion_multiply(q1, q2_conjugate)

def quaternion_dot(q1, q2):
    q1 = np.array([q1.x, q1.y, q1.z, q1.w])
    q2 = np.array([q2.x, q2.y, q2.z, q2.w])
    return np.dot(q1, q2)

def quaternion_absolute_difference(q1, q2):
    dot_product = quaternion_dot(q1, q2)
    
    # Ensure the dot product is within the valid range [-1, 1]
    dot_product = min(1.0, max(-1.0, dot_product))
    
    # Calculate the angle between the quaternions using the arccosine
    angle = 2 * np.arccos(abs(dot_product))
    
    return angle