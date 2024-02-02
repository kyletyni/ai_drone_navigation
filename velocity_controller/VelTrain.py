from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from VelEnv import VelocityControlEnv
import rospy
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from std_srvs.srv import Empty, EmptyRequest
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

class PauseSimulationCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PauseSimulationCallback, self).__init__(verbose)
        self.pause_service = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    def _on_step(self) -> bool:
        return True  # return False if you want to stop the training

    def _on_rollout_start(self) -> None:
        # This method will be called at the start of each rollout
        self.unpause_simulation()

    def _on_rollout_end(self) -> None:
        # This method will be called at the end of each rollout,
        # i.e., after collecting a bunch of experience but before updating the policy.
        self.pause_simulation()

    def pause_simulation(self):
        try:
            self.pause_service(EmptyRequest())
            print("Simulation paused")
        except rospy.ServiceException as e:
            print("Failed to pause simulation: %s" % e)

    def unpause_simulation(self):
        try:
            self.unpause_service(EmptyRequest())
            print("Simulation unpaused")
        except rospy.ServiceException as e:
            print("Failed to unpause simulation: %s" % e)

# Usage
env = DummyVecEnv([lambda: VelocityControlEnv()])
model = PPO("MlpPolicy", env, n_steps=1000, verbose=1)

# Initialize the callback
pause_simulation_callback = PauseSimulationCallback()
checkpoint_callback = CheckpointCallback(save_freq=20000, save_path='../models/', name_prefix='vel_ctrl')#, verbose=1, save_replay_buffer=False)

# Combine the callbacks
callback_list = [pause_simulation_callback, checkpoint_callback]

# Start training
model.learn(total_timesteps=10000000, callback=callback_list)