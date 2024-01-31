import sys
import gym
from gym import spaces
import math

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from utility import *

import rospy
from clover import srv
from clover.srv import SetVelocityRequest
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Wrench, Vector3, PoseStamped, Quaternion
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from gazebo_msgs.srv import GetModelState, SetModelState 
from std_msgs.msg import String
from mavros_msgs.msg import State
from std_srvs.srv import Empty
import tf.transformations as transformations

BASE_HEIGHT = 0.0607


class PositionControlEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(PositionControlEnv, self).__init__()

        rospy.init_node('jupyter_velocity_controller', anonymous=True)

        self.set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.reset_world_service = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_velocity_service = rospy.ServiceProxy('/set_velocity', srv.SetVelocity)
        self.set_rates_service = rospy.ServiceProxy('/set_rates', srv.SetRates)
        self.navigate_service = rospy.ServiceProxy('/navigate', srv.Navigate)

        # Action: Velocity [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # State: current position [-10, 10]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.current_obs = np.array([0, 0, BASE_HEIGHT, 0, 0, 0])

        self.target_position = np.array([0.0, 0.0, 20.0])
        self.current_pos = np.array([0.0, 0.0, BASE_HEIGHT])
        self.reached_goal = False

        self.current_step = 0
        self.max_step_per_episode = 2000
        self.flipped = False


    def reset(self):
        self.current_step = 0
        rospy.wait_for_service('/set_velocity')
        self.set_velocity_service(vx=0, vy=0, vz=1, yaw=0, auto_arm=1, frame_id='map')

        self.reset_world_service()

        self.target_position = np.array([0, 0, 20])
        # respawn_point = new_respawn_point(self.target_position)
        respawn_point = self.target_position

        state = ModelState()
        state.model_name = 'clover'
        state.pose.position = Vector3(respawn_point[0], respawn_point[1], respawn_point[2] + BASE_HEIGHT)
        q = euler_to_quaternion(np.array([0, 0, 0]))
        q = Quaternion(q[0], q[1], q[2], q[3])
        state.pose.orientation = q

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
        self.current_step += 1
        _action = action.reshape(-1)
    
        self.set_velocity_service(vx=_action[0], vy=_action[1], vz=_action[2], yaw=0)
        
        done = False
        rospy.sleep(0.004)
        self.update_drone_state()


        reward = max(0, 1 - (np.linalg.norm(self.current_relative_pos) ** 2)) - \
                     0.05 * np.linalg.norm(self.linear_velocity)

        if self.current_step % 10 == 0:
            pass
        print(f'Thrust Agent: reward:{reward:.4f} vx:{_action[0]:.4f} vy:{_action[1]:.4f} vz:{_action[2]:.4f}')
            # print(f'Thrust Agent: reward:{reward:.4f} yaw:{self.current_orientation[0]:.4f} pitch:{self.current_orientation[1]:.4f} roll:{self.current_orientation[2]:.4f} Action: 0:{_action[0]:.4f} 1:{_action[1]:.4f} 2:{_action[2]:.4f} 3:{_action[3]:.4f}')
            

        if self.flipped:
            reward = -10

        if np.linalg.norm(self.current_relative_pos) > 5:
            # reward = -10
            done = True

        self.current_step += 1
        if self.current_step >= 2000:
            done = True

        if np.linalg.norm(self.current_relative_pos) < 0.2:
            if not self.reached_goal:   
                self.reached_goal = True
                self.reached_goal_step = self.current_step
            elif self.current_step - self.reached_goal_step > 500:
                reward = 100
                self.target_position = new_move_point(self.target_position)
                print("new target:", self.target_position)
        elif self.reached_goal:
            self.reached_goal = False


        if self.current_step >= 2000: 
            done = True

        return self.current_obs, reward, done, {}


    def close(self):
        if hasattr(self, 'figure'):
            plt.close(self.figure)

    def update_drone_state(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            _state = self.get_state_service('clover', 'world')
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

        x, y, z = _state.pose.position.x, _state.pose.position.y, _state.pose.position.z
        x_rel, y_rel, z_rel = x - self.target_position[0], y - self.target_position[1], z - self.target_position[2] 
        linear_x, linear_y, linear_z = _state.twist.linear.x, _state.twist.linear.y, _state.twist.linear.z,
        yaw, pitch, roll = quaternion_to_euler(_state.pose.orientation)

        self.current_pos = np.array([x, y, z])
        self.current_relative_pos = np.array([x_rel, y_rel, z_rel])
        self.linear_velocity = np.array([_state.twist.linear.x, _state.twist.linear.y, _state.twist.linear.z])
        
        self.flipped = np.abs(roll) > np.pi/2 or np.abs(pitch) > np.pi/2

        self.current_obs = np.array([x_rel, y_rel, z_rel,
                                     linear_x, linear_y, linear_z])

    def set_velocity(self, vx, vy, vz):
        velocity_request = SetVelocityRequest()
        velocity_request.vx = vx
        velocity_request.vy = vy
        velocity_request.vz = vz
        velocity_request.frame_id = 'map'
        velocity_request.auto_arm = 1
        try:
            response = self.set_velocity_service(velocity_request)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)