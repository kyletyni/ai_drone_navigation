import sys
import math
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import random
import subprocess

import rospy
from clover import srv
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState 
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3

# import drone_reset

BASE_HEIGHT = 0.0607
drone_control_path = "../../catkin_ws/devel/lib/px4/px4-drone_control"

# from clover.srv import SetVelocity, SetVelocityRequest
# from geometry_msgs.msg import Wrench, Vector3, PoseStamped
def euler_to_quaternion(roll, pitch, yaw):
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
    distance = random.uniform(1, 8)
    theta = random.uniform(0, 2 * np.pi)  # Azimuthal angle
    phi = random.uniform(0, np.pi)        # Polar angle

    # Convert spherical coordinates to Cartesian coordinates
    x = point[0] + distance * np.sin(phi) * np.cos(theta)
    y = point[1] + distance * np.sin(phi) * np.sin(theta)
    z = point[2] + distance * np.cos(phi)

    return np.array([x, y, z])


class PositionControlEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PositionControlEnv, self).__init__()

        rospy.init_node('jupyter_velocity_controller', anonymous=True)

        self.set_rates_service = rospy.ServiceProxy('/set_rates', srv.SetRates)
        self.set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_velocity_service = rospy.ServiceProxy('/set_rates', srv.SetVelocity)
        self.reset_world_service = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        # Action: Velocity [-1, 1]
        # self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Action: Set Rates [0, 1]
        rad_rate_low = 0
        rad_rate_high = 1
        thrust_low = 0
        thrust_high = 1

        # roll_rate, pitch_rate, yaw_rate, thrust
        self.action_space = spaces.Box(low=np.array([rad_rate_low, rad_rate_low, rad_rate_low, thrust_low]),
                                       high=np.array([rad_rate_high, rad_rate_high, rad_rate_high, thrust_high]),
                                       dtype=np.float32)

        # State: current position [-10, 10]
        pos_low, pos_high = -100, 100
        orientation_low, orientation_high = -10, 10
        vel_low, vel_high = -10, 10
        ang_vel_low, ang_vel_high = -10, 10

        obs_low = np.array([pos_low, pos_low, pos_low, 
                            orientation_low, orientation_low, orientation_low,
                            vel_low, vel_low, vel_low,
                            ang_vel_low, ang_vel_low, ang_vel_low])
        obs_high = np.array([pos_high, pos_high, pos_high, 
                             orientation_high, orientation_high, orientation_high,
                             vel_high, vel_high, vel_high,
                             ang_vel_high, ang_vel_high, ang_vel_high])

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.current_obs = np.array([0, 0, BASE_HEIGHT, 
                                     0, 0, 0,
                                     0, 0, 0,
                                     0, 0, 0])

        self.target_position = np.array([0, 0, 10])
        self.target_orientation = np.array([0, 0, 0])

        self.prev_action = np.array([0, 0, 0, 0])

        self.current_step = 0
        self.max_step_per_episode = 2000
        self.flipped = False
        self.time_alive = 0


    def update_drone_state(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            _state = self.get_state_service('clover', 'world')
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

        yaw, pitch, roll = quaternion_to_euler(_state.pose.orientation)
        x, y, z = _state.pose.position.x, _state.pose.position.y, _state.pose.position.z
        linear_x, linear_y, linear_z = _state.twist.linear.x, _state.twist.linear.y, _state.twist.linear.z,
        angular_x, angular_y, angular_z = _state.twist.angular.x, _state.twist.angular.y, _state.twist.angular.z
        drone_normal = get_drone_normal(yaw, pitch, roll)

        self.current_position = np.array([x, y, z])

        self.relative_position = self.current_position - self.target_position
        self.current_orientation = np.array([yaw, pitch, roll])

        self.linear_velocity = np.array([_state.twist.linear.x, _state.twist.linear.y, _state.twist.linear.z])
        self.angular_velocity = np.array([_state.twist.angular.x, _state.twist.angular.y, _state.twist.angular.z])

        self.flipped = np.abs(roll) > np.pi/3 or np.abs(pitch) > np.pi/3

        self.current_obs = np.array([self.relative_position[0], self.relative_position[1], self.relative_position[2], 
                                     yaw, pitch, roll, 
                                     linear_x, linear_y, linear_z,
                                     angular_x, angular_y, angular_z])

        #self.position_subscriber = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.position_callback)

    def normalize_reward(self, reward, min_reward, max_reward, desired_min=-1, desired_max=1):
        # Ensure the reward is within the [min_reward, max_reward] range
        reward = max(min_reward, min(reward, max_reward))
        
        # Normalize the reward to the [desired_min, desired_max] range
        normalized_reward = (reward - min_reward) / (max_reward - min_reward) * (desired_max - desired_min) + desired_min
        
        return normalized_reward

    def reset(self):
        rospy.wait_for_service('/set_rates')
        self.set_rates_service(roll_rate=0, pitch_rate=0, yaw_rate=0, thrust=0, auto_arm=True)
        
        self.reset_world_service()

        # respawn_point = new_respawn_point(self.target_position)
        respawn_point = self.target_position

        state = ModelState()
        state.model_name = 'clover'
        state.pose.position = Vector3(respawn_point[0], respawn_point[1], respawn_point[2] + BASE_HEIGHT)
        state.pose.orientation = euler_to_quaternion(0, 0, 0)
        state.twist.linear = Vector3(0, 0, 0.1)
        state.twist.angular = Vector3(0, 0, 0)

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            resp = self.set_state_service( state )
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
    
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            _state = self.get_state_service('clover', 'base_link')
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

        yaw, pitch, roll = quaternion_to_euler(_state.pose.orientation)
        _state.twist.linear.x

        # print("reset: yaw", yaw, "pitch:", pitch, "roll", roll)
        self.liftoff = False
        self.time_alive = 0
        self.current_step = 0

        self.current_position = np.array([_state.pose.position.x, _state.pose.position.y, _state.pose.position.z])

        self.linear_velocity = np.array([_state.twist.linear.x, _state.twist.linear.y, _state.twist.linear.z])
        self.angular_velocity = np.array([_state.twist.angular.x, _state.twist.angular.y, _state.twist.angular.z])

        self.relative_position = self.current_position - self.target_position
        self.current_obs = np.array([self.relative_position[0], self.relative_position[1], self.relative_position[2],
                                     yaw, pitch, roll, 
                                     _state.twist.linear.x, _state.twist.linear.y, _state.twist.linear.z, 
                                     _state.twist.angular.x, _state.twist.angular.y, _state.twist.angular.z])
        
        return self.current_obs

    def step(self, action):

        _action = action.reshape(-1)

        command = [drone_control_path] + [str(a) for a in _action] + ['1']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the process to finish and get the return code
        return_code = process.wait()

        # Check if the process executed successfully
        if return_code != 0:
            print("Error running C++ executable", return_code)
        #self.set_rates_service(roll_rate=_action[0], pitch_rate=_action[1], yaw_rate=_action[2], thrust=_action[3], auto_arm=True)

        done = False
        
        rospy.sleep(0.004)
        self.time_alive += 0.004

        # apply low pass filter for smoothing
        k1 = 0.5
        _action = k1 * _action + (1 - k1) * self.prev_action

        self.update_drone_state()

        # Reward is negative absolute difference between current and target position
        position_diff = np.linalg.norm(self.current_position[:2] - self.target_position[:2])
        angle_diff = np.linalg.norm(self.current_orientation - self.target_orientation)

        # Penalty for quick changes to set rates
        k2 = 0.1
        rates_change_penalty = k2 * np.sum(np.abs(_action - self.prev_action))

        # reward = (10 / position_diff)

        position_reward = 0
        orient_reward = 0
        z_reward = 0
        reward = 0

        if position_diff < 2:
            position_reward = (100 / (position_diff + 0.001)) - 20
        else:
            position_reward = -(position_diff ** 2)

        pos_normal = self.normalize_reward(position_reward, -200, 3000, -2, 5)
        reward += pos_normal

        # if angle_diff < np.pi*2/3:
        #     orient_reward = 20 / (angle_diff + 0.2)
        # else:
        #     orient_reward = -(angle_diff**3)

        # velocity reward
        reward += math.log10(np.linalg.norm(self.linear_velocity) + 0.01) * \
                 math.log10(np.linalg.norm(self.relative_position) + 0.01)
        
        reward += math.log10(np.linalg.norm(self.angular_velocity) + 0.01) * \
                 math.log10(np.linalg.norm(self.relative_position) + 0.01)

        yaw_err = self.current_orientation[0] - self.target_orientation[0]

        if abs(yaw_err) < np.pi/6:
            yaw_reward = (-100 * abs(yaw_err)) + 100
        else:
            yaw_reward = (-100 * abs(yaw_err)) + 50

        yaw_normal = self.normalize_reward(yaw_reward, -800, 100, -4, 1)
        reward += yaw_normal
        
        orient_err = np.linalg.norm(self.current_orientation[1:3] - self.target_orientation[1:3])

        if orient_err < np.pi:
            orient_reward = (-100 * abs(orient_reward)) + 160
        else:
            orient_reward = (-50 * abs(orient_reward))

        orient_normal = self.normalize_reward(orient_reward, -400, 160, -4, 1.6)
        reward += orient_normal

        if abs(self.relative_position[2]) < 2:
            z_reward = (100 / abs(self.relative_position[2] + 0.001)) - 20
        else:
            z_reward = -(abs(self.relative_position[2]) ** 2)

        z_normal = self.normalize_reward(z_reward, -200, 900, -1, 4.5)
        reward += z_normal

        if self.flipped:
            reward = -10
            done = True
        elif position_diff < 1:
            position_reward = 10 / (position_diff**2 + 1e-9)
            done = False
        elif position_diff < 3:
            position_reward = -(position_diff**2) + 10
            done = False
        else:
            position_reward = -(position_diff**3)
            done = False
        
        # reward += min(40, 1 / angle_diff)

        # position_reward = max(-1000, min(position_reward, 10000))
        # min_reward = -1000
        # max_reward = 10000
        # normalized_pos_reward = (position_reward - min_reward) / (max_reward - min_reward) * (1 - (-1)) + (-1)

        #reward += normalized_pos_reward

        # normal_reward = 2 / np.linalg.norm(((self.target_position - self.current_position) / position_diff) - drone_normal)
        
        # reward += normal_reward
        # reward += self.time_alive * 100
        #reward -= rates_change_penalty

        if self.current_step % 10 == 0:
            print(f'reward:{reward:.4f} yaw:{self.current_orientation[0]:.4f}  pitch:{self.current_orientation[1]:.4f}  roll:{self.current_orientation[2]:.4f}')
        
        if np.linalg.norm(self.relative_position) > 14:
            done = True

        self.current_step += 1
        if self.current_step >= 1000:
            done = True

        self.prev_action = _action
        return self.current_obs, reward, done, {}

    def render(self, mode='human'):
        if not hasattr(self, 'figure'):
            self.figure, self.ax = plt.subplots(figsize=(8, 6))
            plt.ion()
        else:
            self.ax.clear()

        self.ax.clear()
        self.ax.bar(['x', 'y', 'z'], self.current_position, label='Current Position', alpha=0.6, edgecolor='red')
        self.ax.bar(['x', 'y', 'z'], self.target_position, label='Target Position', alpha=0.6, edgecolor='black', linewidth=1.2, fill=False)
        self.ax.legend()
        self.ax.set_ylim([-10, 10])
        clear_output(wait=True)

        plt.pause(0.5)

        display(self.figure)


    def close(self):
        if hasattr(self, 'figure'):
            plt.close(self.figure)

    #velocity controller
    def position_callback(self, msg):
        self.current_position = msg.pose
        print("Position received:", self.current_position)

    def get_current_position(self):
        try:
            position_message = rospy.wait_for_message('/mavros/local_position/pose', PoseStamped, timeout=5)
            position = position_message.pose.position
            print("Current position:", position.x, position.y, position.z)
            return  np.array([position.x, position.y, position.z])
        except rospy.ROSException as e:
            print("Failed to get position message within timeout:", str(e))
        except rospy.ROSInterruptException:
            print("Node was shutdown")


    def set_velocity(self, vx, vy, vz):
        velocity_request = SetVelocityRequest()
        velocity_request.vx = vx
        velocity_request.vy = vy
        velocity_request.vz = vz
        velocity_request.auto_arm = 1
        try:
            response = self.set_velocity_service(velocity_request)
            print("Service call successful. Response: ", response)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
