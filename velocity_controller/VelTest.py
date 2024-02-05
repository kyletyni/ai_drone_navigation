from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import tensorflow as tf
from VelEnv import VelocityControlEnv

# Create DummyVecEnv for the evaluation environment
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.sac.policies import MlpPolicy  # Correct import path
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Create the Stable Baselines3 model
eval_env = DummyVecEnv([lambda: VelocityControlEnv()])
model = PPO(CustomMlpPolicy, eval_env, verbose=1,
           policy_kwargs=dict(features_extractor_class=CustomFeaturesExtractor))

# Set the model weights
model.policy.set_weights(weights)

# Evaluate the model
obs = eval_env.reset()
for _ in range(2000):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = eval_env.step(action)

    # eval_env.target_pos

    if done:
        print("Game is Over", info)
        break

# Close the environment
eval_env.close()