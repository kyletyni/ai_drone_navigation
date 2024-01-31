import sys
import math
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import random
import subprocess
from utility import *

import rospy
from clover import srv
from std_srvs.srv import Empty, EmptyRequest
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState 
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3

BASE_HEIGHT = 0.0607
pi = np.pi
drone_control_path = "../catkin_ws/devel/lib/px4/px4-drone_control"

# from clover.srv import SetVelocity, SetVelocityRequest
# from geometry_msgs.msg import Wrench, Vector3, PoseStamped

class PositionControlEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PositionControlEnv, self).__init__()

        rospy.init_node('jupyter_velocity_controller', anonymous=True)

        self.set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.reset_world_service = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_rates_service = rospy.ServiceProxy('/set_rates', srv.SetRates)
        self.pause_service = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        # Action Space

        # set_rates(roll_rate, pitch_rate, yaw_rate, thrust)   
        # roll_rate, pitch_rate, yaw_rate – pitch, roll, and yaw rates (rad/s);
        # thrust — throttle level, ranges from 0 (no throttle, propellers are stopped) to 1 (full throttle).
        # auto_arm – switch the drone to OFFBOARD and arm automatically (the drone will take off);

        # roll_rate, pitch_rate, yaw_rate, thrust
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -0.5]),
                                       high=np.array([1, 1, 1, 0.5]),
                                       dtype=np.float32)

        # Observation Space

        # position (x, y, z)
        # orientatoin (yaw [-pi, pi], pitch [pi/2, -pi/2], roll[pi, -pi])
        # linear velocity (x, y, z)
        # angular velocity (yaw, pitch, roll)

        obs_low = np.array([-np.inf, -np.inf, -np.inf, -np.pi, -np.pi/2, -np.pi,
                            -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        obs_high = np.array([np.inf, np.inf, np.inf, np.pi, np.pi/2, np.pi,
                             np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.current_obs = np.array([0, 0, BASE_HEIGHT, 
                                     0, 0, 0,
                                     0, 0, 0,
                                     0, 0, 0])

        self.target_position = np.array([0, 0, 6])
        self.target_orientation = np.array([0, 0, 0])

        self.prev_action = np.array([0, 0, 0, 0])
        self.reached_goal = False

        self.current_step = 0
        self.max_step_per_episode = 1500
        self.flipped = False
        self.time_alive = 0
        self.env_bound = 0.5

    def load_state(self):
        # load drone state
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.state = ModelState()
            self.state.pose.position = Vector3(self.current_pos[0], self.current_pos[1], self.current_pos[2])
            self.state.pose.orientation = self.current_quat
            self.state.twist.linear = Vector3(self.linear_velocity[0], self.linear_velocity[1], self.linear_velocity[2])
            self.state.twist.angular = Vector3(self.angular_velocity[0], self.angular_velocity[1], self.angular_velocity[2])

            resp = self.set_state_service(self.state)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
            return

        # unpause physics
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_service(EmptyRequest())
        except rospy.ServiceException as e:
            print("Failed to unpause simulation: %s" % e)


    def unload_state(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_service(EmptyRequest())
        except rospy.ServiceException as e:
            print("Failed to pause simulation: %s" % e)

    def update_drone_state(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            _state = self.get_state_service('clover', 'world')
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

        x, y, z = _state.pose.position.x, _state.pose.position.y, _state.pose.position.z
        x_rel, y_rel, z_rel = x - self.target_position[0], y - self.target_position[1], z - self.target_position[2] 
        yaw, pitch, roll = quaternion_to_euler(_state.pose.orientation)

        linear_x, linear_y, linear_z = _state.twist.linear.x, _state.twist.linear.y, _state.twist.linear.z,
        angular_x, angular_y, angular_z = _state.twist.angular.x, _state.twist.angular.y, _state.twist.angular.z

        
        self.current_pos = np.array([x, y, z])
        self.current_relative_pos = np.array([x_rel, y_rel, z_rel])
        self.current_orientation = np.array([yaw, pitch, roll])
        self.current_quat = _state.pose.orientation
        self.linear_velocity = np.array([_state.twist.linear.x, _state.twist.linear.y, _state.twist.linear.z])
        self.angular_velocity = np.array([_state.twist.angular.x, _state.twist.angular.y, _state.twist.angular.z])

        self.flipped = np.abs(roll) > np.pi/2 or np.abs(pitch) > np.pi/2

        self.current_obs = np.array([x_rel, y_rel, z_rel, 
                                     yaw, pitch, roll, 
                                     linear_x, linear_y, linear_z,
                                     angular_x, angular_y, angular_z])

        #self.position_subscriber = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.position_callback)

    def reset(self):
        rospy.wait_for_service('/set_rates')
        self.set_rates_service(roll_rate=0, pitch_rate=0, yaw_rate=0, thrust=0.5, auto_arm=True)
        
        self.reset_world_service()
        respawn_point = new_respawn_point(self.target_position)
        self.target_position = np.array([0, 0, 5])
        # respawn_point = self.target_position

        state = ModelState()
        state.model_name = 'clover'
        state.pose.position = Vector3(respawn_point[0], respawn_point[1], respawn_point[2] + BASE_HEIGHT)
        q = euler_to_quaternion(np.array([random.uniform(-pi/4, pi/4), random.uniform(-pi/4, pi/4), random.uniform(-pi/4, pi/4)]))
        q = Quaternion(q[0], q[1], q[2], q[3])
        state.pose.orientation = q
        
        linear_vel_min, linear_vel_max = 0.5, 0.5 #-3, 3
        angular_vel_min, angular_vel_max = 0.25, 0.25 #-1, 1

        state.twist.linear = Vector3(np.random.normal(0, 0.25),
                                     np.random.normal(0, 0.25), 
                                     np.random.normal(0, 0.25))         
        state.twist.angular = Vector3(np.random.normal(0, 0.125),
                                      np.random.normal(0, 0.125),
                                      np.random.normal(0, 0.125))

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            resp = self.set_state_service( state )
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

        self.update_drone_state()
        self.current_step = 0
        self.reached_goal = False
        self.reached_goal_step = 0
        return self.current_obs

    
    def step(self, action):
        # self.load_state()
        
        _action = action.reshape(-1)
        self.set_rates_service(roll_rate=_action[0], pitch_rate=_action[1], yaw_rate=_action[2], thrust=_action[3] + 0.5, auto_arm=True)

        done = False
        rospy.sleep(0.004)
        # self.unload_state()

        self.time_alive += 0.004

        self.update_drone_state()
    
        # Old Reward
        #reward = max(0, 1 - (np.linalg.norm(self.current_relative_pos) ** 2)) - \
        #             0.2 *  np.linalg.norm(self.current_orientation) - \
        #             0.1 * np.linalg.norm(self.angular_velocity)

        # New Reward
        reward = self.get_reward()

        # if self.flipped:
        #     reward = -10

        if self.current_step % 100 == 0:
            print(f'Thrust Agent: reward:{reward:.4f} yaw:{self.current_orientation[0]:.4f} pitch:{self.current_orientation[1]:.4f} roll:{self.current_orientation[2]:.4f} Action: 0:{_action[0]:.4f} 1:{_action[1]:.4f} 2:{_action[2]:.4f} 3:{_action[3]:.4f}')
            pass
            # print(f'Thrust Agent: reward:{reward:.4f} yaw:{self.current_orientation[0]:.4f}  pitch:{self.current_orientation[1]:.4f}  roll:{self.current_orientation[2]:.4f} {angle_diff}')
            
        self.current_step += 1
        
        if np.linalg.norm(self.current_relative_pos) > 2 or self.current_step >= self.max_step_per_episode:
            # reward = -10
            done = True


        # if np.linalg.norm(self.current_relative_pos) < 0.5:
        #     if not self.reached_goal:   
        #         self.reached_goal = True
        #         self.reached_goal_step = self.current_step
        #     elif self.current_step - self.reached_goal_step > 150:
        #         reward = 100
        #         self.target_position = new_move_point(self.target_position)
        #         print(self.target_position)
        # elif self.reached_goal:
        #     self.reached_goal = False

        self.prev_action = _action
        return self.current_obs, reward, done, {}

    def get_reward(self):
        # Constants
        self.position_reward_constant = 5.0
        self.orientation_reward_constant = 0.5
        self.linear_velocity_reward_constant = 0.125
        self.angular_velocity_reward_constant = 0.25
        self.max_reward_for_velocity_towards_goal = 2.0
        self.bonus_to_reach_goal = 15.0
        self.reward_for_staying_alive = 1.0

        dist = np.linalg.norm(self.current_relative_pos)
        if dist < self.env_bound / 3:
            reward_position = 15
        else:
            reward_position = dist ** 2 * (-self.position_reward_constant)

        reward_orientation = np.linalg.norm(self.current_orientation) * (-self.orientation_reward_constant)
        reward_linear_velocity = np.linalg.norm(self.linear_velocity) * (-self.linear_velocity_reward_constant)
        reward_angular_velocity = np.linalg.norm(self.angular_velocity) * (-self.angular_velocity_reward_constant)

        alive_bonus = self.reward_for_staying_alive

        # if in goal range we reward agent
        extra_bonus = 0 
        #if self.goal_reached(self.current_relative_pos):
           #extra_bonus = self.bonus_to_reach_goal

        # if outside of boundary we apply penalty
        extra_penalty = 0
        if np.linalg.norm(self.current_relative_pos) > self.env_bound * 1.5:
            extra_penalty = -self.bonus_to_reach_goal

        # reward agent to move towards goal if system is away from goal
        reward_velocity_towards_goal = 0.0
        if not self.goal_reached(self.current_relative_pos):  
            reward_velocity_towards_goal += self.reward_velocity_towards_goal(self.current_relative_pos, self.linear_velocity)

        rewards = (reward_position, reward_orientation, reward_linear_velocity, reward_angular_velocity, alive_bonus, extra_bonus, extra_penalty, reward_velocity_towards_goal)
        reward = sum(rewards)
        return reward

    def reward_velocity_towards_goal(self, error_xyz, velocity):
        """
            error_xyz (numpy.ndarray): Position error of the system along xyz-coordinates.
            velocity (numpy.ndarray): Velocity vector (vx, vy, vz) of the system in body reference frame

        """
        if self.goal_reached(error_xyz):
            return self.max_reward_for_velocity_towards_goal
        unit_xyz = error_xyz / (self.norm(error_xyz) + 1e-6)
        velocity_direction = velocity / (self.norm(velocity) + 1e-6)
        reward = np.dot(unit_xyz, velocity_direction)
        return np.clip(reward, -np.inf, self.max_reward_for_velocity_towards_goal)

    def norm(self, arr):
        return np.linalg.norm(arr)

    def goal_reached(self, xyz):
        if self.norm(xyz) < self.env_bound:
            return True
        return False


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