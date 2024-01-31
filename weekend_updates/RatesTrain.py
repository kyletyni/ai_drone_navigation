from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from RatesEnv import PositionControlEnv
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
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_service(EmptyRequest())
            print("Simulation paused")
        except rospy.ServiceException as e:
            print("Failed to pause simulation: %s" % e)

    def unpause_simulation(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_service(EmptyRequest())
            print("Simulation unpaused")
        except rospy.ServiceException as e:
            print("Failed to unpause simulation: %s" % e)

# Usage
env_names = ['PositionControlEnv-v0', 'PositionControlEnv-v1', 'PositionControlEnv-v2', 'PositionControlEnv-v3']
# envs = [PositionControlEnv() for _ in range(len(env_names))]
# vec_env = DummyVecEnv([lambda: env for env in envs])
vec_env = DummyVecEnv([lambda: PositionControlEnv()])

model = PPO.load("./models/drl_model_60000_steps", env=vec_env, n_steps=2000, verbose=1)
# model = PPO("MlpPolicy", env=vec_env, n_steps=2000, verbose=1, policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh))

# Initialize the callback
pause_simulation_callback = PauseSimulationCallback()
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='drl_model')#, verbose=1, save_replay_buffer=False)

# Combine the callbacks
callback_list = [checkpoint_callback]

# Start training
model.learn(total_timesteps=500000, callback=callback_list)