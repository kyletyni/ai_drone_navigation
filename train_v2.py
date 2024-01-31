from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from env import PositionControlEnv
import rospy
import numpy as np
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

# Initialize the callback
gen_str = ""
pause_simulation_callback = PauseSimulationCallback()
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./models/', name_prefix='drl_model')#, verbose=1, save_replay_buffer=False)

# Combine the callbacks
callback_list = [pause_simulation_callback, checkpoint_callback]

# Start training
# model.learn(total_timesteps=1000000, callback=callback_list)


env = DummyVecEnv([lambda: PositionControlEnv()])
eval_env = PositionControlEnv()
# model = PPO("MlpPolicy", env, n_steps=1000, verbose=1)

# Define hyperparameters
num_generations = 100  # Number of generations
population_size = 20  # Number of agents in each generation
num_timesteps = 20000  # Number of timesteps for each agent training
num_episodes = 10  # Number of episodes for evaluation


# Main training loop
for generation in range(num_generations):
    gen_str = str(generation)
    agents = []
    agent_rewards = []

    for _ in range(population_size):
        agent = PPO("MlpPolicy", env, n_steps=500, verbose=1)
        agent.learn(total_timesteps=num_timesteps, callback=callback_list)
        pause_simulation_callback.unpause_simulation()
        agents.append(agent)

        # Evaluate the agent
        mean_reward, _ = evaluate_policy(agent, agent.get_env(), n_eval_episodes=num_episodes)
        agent_rewards.append(mean_reward)

    # Select the top-performing agents
    elite_indices = sorted(range(len(agent_rewards)), key=lambda i: agent_rewards[i], reverse=True)[:int(population_size * 0.2)]
    elite_agents = [agents[i] for i in elite_indices]

    # Create a new population by cloning and training the elite agents
    new_agents = []
    for _ in range(population_size - len(elite_indices)):
        parent_index = np.random.choice(elite_indices)
        child_agent = elite_agents[parent_index].clone()
        child_agent.learn(total_timesteps=num_timesteps)
        new_agents.append(child_agent)

    # Replace the old agents with the new agents
    agents = [new_agents[i] if i not in elite_indices else agents[i] for i in range(population_size)]

    # Print the average reward of the generation
    avg_reward = np.mean(agent_rewards)
    print(f"Generation {generation + 1}, Average Reward: {avg_reward}")

# Final agent after all generations
best_agent = agents[np.argmax(agent_rewards)]
best_agent.save("best_agent")

# To test the best agent
test_mean_reward, _ = evaluate_policy(best_agent, env, n_eval_episodes=num_episodes)
print(f"Test Mean Reward: {test_mean_reward}")