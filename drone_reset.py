import math
import rospy
import numpy as np
from clover import srv
from std_srvs.srv import Trigger
from std_srvs.srv import Empty, EmptyRequest
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3
import os
import subprocess

BASE_HEIGHT = 0.0607 # height offset of drone for spawning at Vec3

rospy.init_node('drone_reset')

get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
set_rates = rospy.ServiceProxy('set_rates', srv.SetRates)
set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
# set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
land = rospy.ServiceProxy('land', Trigger)

pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([x, y, z, w])

# parameters are in units of radians
def euler_to_quaternion(yaw, pitch, roll):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w /= norm
    x /= norm
    y /= norm
    z /= norm

    return Quaternion(x, y, z, w)

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

class DroneState(ModelState):
    def __init__(self):
        super().__init__()
        self.reset_position = Vector3(0, 0, BASE_HEIGHT)
        self.reset_orientation = euler_to_quaternion(0, 0, 0)
        self.reset_velocity_linear = Vector3(0, 0, 0)
        self.reset_velocity_angular = Vector3(0, 0, 0)

        self.model_name = 'clover'
        self.pose.position = self.reset_position
        self.pose.orientation = self.reset_orientation

def get_drone_state(state: DroneState) -> ModelState:
    rospy.wait_for_service('/gazebo/get_model_state')
    try:
        _state = get_state('clover', 'world')
        return _state

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return None

def reset_drone_state(state: DroneState) -> int:
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        state.pose.position = state.reset_position
        state.pose.orientation = state.reset_orientation
        state.twist.linear = state.reset_velocity_linear
        state.twist.angular = state.reset_velocity_angular
        resp = set_state( state )
        return 1

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return 0

def quaternion_rotation(vector, quaternion):
    quaternion = np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z])
    # Ensure the vector is a 3D vector and the quaternion is normalized
    if len(vector) != 3:
        raise ValueError("Input vector must be a 3D vector")
    
    norm = np.linalg.norm(quaternion)
    if abs(norm - 1.0) > 1e-6:
        raise ValueError("Input quaternion must be normalized")

    # Convert the 3D vector to a pure quaternion (0 + xi + yj + zk)
    v_quaternion = np.array([0, vector[0], vector[1], vector[2]])

    # Perform the rotation
    conjugate = np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])
    rotated_vector_quaternion = quaternion_multiply(quaternion, quaternion_multiply(v_quaternion, conjugate))

    # Extract the resulting 3D vector part from the rotated quaternion
    rotated_vector = rotated_vector_quaternion[1:]

    return rotated_vector

def get_normal(yaw, pitch, roll):
    # Calculate the direction cosines (direction ratios) of the normal vector
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)

    # Calculate the components of the normal vector
    x = cos_yaw * cos_pitch
    y = sin_yaw * cos_pitch
    z = sin_pitch  # Positive because the drone's top points upward along the z-axis

    # Create a NumPy array for the components
    normal_vector = np.array([x, y, z])

    # Normalize the vector to get a unit vector
    magnitude = np.linalg.norm(normal_vector)
    if magnitude == 0:
        # Avoid division by zero
        return np.array([0, 0, 0])
    else:
        return normal_vector / magnitude

def euler_to_rotation_matrix(yaw, pitch, roll):
    yaw = yaw
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

if __name__ == "__main__": 
    drone_state = DroneState()

    
    
    rospy.wait_for_service('/set_rates')
    set_rates(pitch_rate=0.0, yaw_rate=0.0, roll_rate=0, thrust=0.6, auto_arm=True)
    # navigate(x=0, y=0, z=1.5, speed=0.5, frame_id='body', auto_arm=True)
    reset_world()
    rospy.sleep(1)
    drone_state.reset_position = Vector3(1., 1., 0 + BASE_HEIGHT)
    euler = [0, 0, 0]
    drone_state.reset_orientation = euler_to_quaternion(euler[0], euler[1], euler[2])
    reset_drone_state(drone_state)
    # unpause_physics(EmptyRequest())
    
    # takeoff()
    
    rospy.sleep(1)
    # navigate(x=0, y=0, z=1.5, frame_id='body', auto_arm=True)
    rospy.sleep(1)

    rospy.wait_for_service('/navigate')
    navigate(x=0, y=0, z=1.5, speed=0.4, frame_id='', auto_arm=True)
    # for i in range(100):
    #     relative_path =  "../../catkin_ws/devel/lib/px4/px4-drone_control"  # For example: '../folder/subfolder'
    #     arr = np.array([0.1, .1, 0.1, 0.1, 1])
    #     command = [relative_path] + [str(a) for a in arr]

    #     process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    #     # Wait for the process to finish and get the return code
    #     return_code = process.wait()

    #     # Check if the process executed successfully
    #     if return_code != 0:
    #         print("Error running C++ executable")
    #     else:
    #         print("success")
    
    # cur_state = get_drone_state(drone_state)
    # q = cur_state.pose.orientation

    # UpVector = np.array([1, 0, 0])
    # newUp = quaternion_rotation(UpVector, drone_state.reset_orientation)

    # # print(UpVector, newUp)

    # yaw, pitch, roll = quaternion_to_euler(drone_state.reset_orientation)
    # print(euler)
    # print(yaw, pitch, roll)

    # new_normal = euler_to_rotation_matrix(yaw + np.pi, pitch, roll)
    # print("new_normal", new_normal)

    # normal = get_normal(yaw, pitch, roll)
    # print('normal:', normal[0], normal[1], normal[2])
    # normal = get_normal(yaw - np.pi/2, pitch, roll)
    # print('normal:', normal[0], normal[1], normal[2])

    # telem = get_telemetry()