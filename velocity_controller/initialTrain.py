import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
import gym
import zipfile
# Load your dataset (replace with your actual data loading code)
data = np.loadtxt("data.csv", delimiter=",")
position = torch.from_numpy(data[:, :3]).float()
setpoint_velocity = torch.from_numpy(data[:, 3:]).float()

# Create the neural network model
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

model = VelocityPredictor()

# Compile the model
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

# Train the model using only position data as input
for epoch in range(1):  # Adjust epochs as needed
    print('epoch', epoch)
    for i in range(0, len(position), 32):  # Adjust batch size as needed
        x_batch = position[i:i+32]
        y_batch = setpoint_velocity[i:i+32]

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'velocity_model.pth')
print(model.state_dict())

# Save the model using the PPO save method
# ppo_model = PPO("MlpPolicy", "CartPole-v1")
# ppo_model.load("ppo_vel_model.zip")

# Make predictions on new position data
new_position = torch.tensor([1.0, 2.0, 3.0]).float()
predicted_setpoint_velocity = model(new_position)
print("Predicted setpoint velocity:", predicted_setpoint_velocity.detach().numpy())
