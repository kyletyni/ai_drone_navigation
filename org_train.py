import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import rospy
import sys
from clover.srv import SetVelocity, SetVelocityRequest
import numpy as np
from geometry_msgs.msg import Wrench, Vector3, PoseStamped

class PositionControlEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(PositionControlEnv, self).__init__()

        # Action: Velocity [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # State: current position [-10, 10]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)

        self.current_position = np.array([0.0, 0.0, 0.0])
        self.target_position = np.array([1.0, 1.0, 1.0])

        rospy.init_node('jupyter_velocity_controller', anonymous=True)
        #self.position_subscriber = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.position_callback)
        self.set_velocity_service = rospy.ServiceProxy('/set_velocity', SetVelocity)


    def reset(self):
        self.current_position = np.array([0.0, 0.0, 0.0])
        return np.array([self.current_position])

    def step(self, action):
        velocity = action.reshape(-1)
        self.set_velocity(velocity[0], velocity[1],velocity[2])
        rospy.sleep(0.004)
        self.current_position = self.get_current_position()


        # Reward is negative absolute difference between current and target position
        reward = -np.sum(np.abs(self.target_position - self.current_position))

        done = False  # continuous task

        return self.current_position, reward, done, {}

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
