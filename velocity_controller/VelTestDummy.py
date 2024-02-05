import gym
import numpy as np 
import torch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn  # Import torch.nn for the Tanh activation function

from stable_baselines3.common.vec_env import DummyVecEnv
from VelEnv import VelocityControlEnv

data = np.loadtxt("data.csv", delimiter=",")
position = torch.from_numpy(data[:, :3]).float()
setpoint_velocity = torch.from_numpy(data[:, 3:]).float()


class VelocityPredictor(nn.Module):
    def __init__(self):
        super(VelocityPredictor, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Input layer (3 features)
        self.fc2 = nn.Linear(64, 64)  # First hidden layer
        self.fc3 = nn.Linear(64, setpoint_velocity.shape[1])  # Output layer

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)  # No activation for the output layer
        return x


class CustomPPOModel(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define the policy network architecture
        self.mlp_extractor = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
        )
        # Separate head for policy and value
        self.action_net = torch.nn.Linear(64, 3)
        self.value_net = torch.nn.Linear(64, 1)

    def load_weights(self, weights_path):
        # Load pre-trained weights using torch.load
        pretrained_dict = torch.load(weights_path)

        # Map pre-trained layer names to your architecture (adjust accordingly)
        pretrained_to_mine = {
            "fc1.weight": "mlp_extractor.0.weight",
            "fc1.bias": "mlp_extractor.0.bias",
            "fc2.weight": "mlp_extractor.2.weight",
            "fc2.bias": "mlp_extractor.2.bias",
            "fc3.weight": "action_net.weight",
            "fc3.bias": "action_net.bias"
            # ... (map remaining layers, if any)
        }

        # Assign loaded weights to corresponding layers in your model
        model_dict = self.state_dict()
        for name, _ in pretrained_dict.items():
            if name in pretrained_to_mine:
                model_dict[pretrained_to_mine[name]] = pretrained_dict[name]
        self.load_state_dict(model_dict)

    def forward_critic(self, obs, deterministic=False):
        # features = self.mlp_extractor(obs)
        # action_prob = torch.nn.functional.softmax(self.action_net(features), dim=-1)
        # value = self.value_net(features).reshape(-1)
        # log_prob = torch.log(action_prob)  # Add this line
        # return action_prob, value, log_prob  # Return all three values

        features = self.mlp_extractor(obs)
        action_prob = torch.nn.functional.softmax(self.action_net(features), dim=-1)
        value = self.value_net(features).reshape(-1)

        # Calculate log_prob for the chosen action (assuming single observation)
        chosen_action = torch.argmax(action_prob, dim=-1)  # Get the index of the chosen action
        log_prob = torch.log(action_prob[0, chosen_action])  # Extract log probability for that action

        # Adjustment for batch observations (if applicable):
        # If your forward method handles multiple observations in a batch,
        # you might need to apply torch.log along the batch dimension:
        # log_prob = torch.log(action_prob)  # Apply torch.log to the entire action probability tensor

        return action_prob, value, log_prob


# Create the environment
env = DummyVecEnv([lambda: VelocityControlEnv()])
print(stable_baselines3.__version__)


velocity_model = VelocityPredictor()
velocity_model.load_state_dict(torch.load('velocity_model.pth'))
velocity_model.eval()


# Create the PPO agent with the custom MLPPolicy
policy_kwargs = dict(net_arch=[64, 64], activation_fn=nn.Tanh)
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# model = PPO(CustomPPOModel, env, verbose=1)
# model.policy.load_weights("velocity_model.pth")

pretrained_dict = torch.load("velocity_model.pth")

# Map pre-trained layer names to your architecture (adjust accordingly)
pretrained_to_mine = {
    "fc1.weight": "mlp_extractor.policy_net.0.weight",
    "fc1.bias": "mlp_extractor.policy_net.0.bias",
    "fc2.weight": "mlp_extractor.policy_net.2.weight",
    "fc2.bias": "mlp_extractor.policy_net.2.bias",
    "fc3.weight": "action_net.weight",
    "fc3.bias": "action_net.bias"
    # ... (map remaining layers, if any)
}
print(model.policy.state_dict().keys())

# Assign loaded weights to corresponding layers in your model
model_dict = model.policy.state_dict()
for name, _ in pretrained_dict.items():
    if name in pretrained_to_mine:
        model_dict[pretrained_to_mine[name]] = pretrained_dict[name]
model.policy.load_state_dict(model_dict)


print(velocity_model.state_dict().keys())

print(model.policy.state_dict())
print(velocity_model.state_dict())

model.learn(1000000)