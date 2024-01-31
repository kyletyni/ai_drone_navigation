from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from RatesEnv import PositionControlEnv
import rospy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from std_srvs.srv import Empty, EmptyRequest
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

eval_env = PositionControlEnv()
model = PPO.load("./models/drl_model_650000_steps", env=eval_env, n_steps=2000)
obs = eval_env.reset()
for _ in range(2000): #1000
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = eval_env.step(action)
    if done:
        print("Game is Over", info)
        break
#display(eval_env.figure)
eval_env.close()
