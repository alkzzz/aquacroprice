import os
import numpy as np
import matplotlib.pyplot as plt
from comet_ml import OfflineExperiment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from aquacroprice.envs.rice import Rice

# Custom callback for logging and plotting rewards
class RewardLoggingCallback(BaseCallback):
    def __init__(self, experiment, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.experiment = experiment
        self.episode_rewards = []
        self.episode_timesteps = []
        self.current_rewards = 0
        self.current_timesteps = 0

    def _on_step(self) -> bool:
        # Update cumulative reward and timesteps
        self.current_rewards += np.mean(self.locals['rewards'])  # Mean reward across environments
        self.current_timesteps += 1
        
        # Check if any environment is done
        if any(self.locals['dones']):
            self.episode_rewards.append(self.current_rewards)
            self.episode_timesteps.append(self.num_timesteps)
            self.experiment.log_metric("reward", self.current_rewards, step=self.num_timesteps)
            self.current_rewards = 0
            self.current_timesteps = 0
        
        return True

    def _on_training_end(self):
        self.plot_rewards()

    def plot_rewards(self):
        # Plot the rewards
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_timesteps, self.episode_rewards)
        plt.xlabel('Timesteps')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        # Save the plot as an image file
        plt.savefig('reward_plot.png')
        print("Reward plot saved as reward_plot.png")

# Initialize Comet.ml experiment in offline mode
experiment = OfflineExperiment(
    project_name="aqua-gym-rice",
    workspace="alkzzz",
    offline_directory="/comet_logs"
)

# Create the environment
env = DummyVecEnv([lambda: Rice()])

# Custom reward logging callback
reward_logging_callback = RewardLoggingCallback(experiment)

# Define PPO hyperparameters
ppo_params = {
    "learning_rate": 1e-3,  # Increased learning rate for potential better learning
    "n_steps": 4096,         # Increased steps per update
    "batch_size": 128,       # Larger batch size
    "n_epochs": 20,          # More epochs per update
    "gamma": 0.98,           # Slightly adjusted discount factor
    "gae_lambda": 0.9,       # Slightly adjusted GAE lambda
    "clip_range": 0.3,       # Adjusted clip range
}

# Create the PPO model
model = PPO(
    "MlpPolicy", env, verbose=1,
    **ppo_params
)

# Log hyperparameters to Comet.ml
experiment.log_parameters(ppo_params)

# Train the PPO model
model.learn(total_timesteps=100000, callback=reward_logging_callback)

# Evaluate the PPO agent using Stable Baselines' `evaluate_policy`
ppo_mean_reward, ppo_std_reward = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=False)

# Log PPO evaluation results to Comet.ml
experiment.log_metric("ppo_agent_mean_reward", ppo_mean_reward)
experiment.log_metric("ppo_agent_std_reward", ppo_std_reward)

# Implement the random agent
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()

# Evaluate the random agent
random_agent = RandomAgent(env.action_space)

# Function to evaluate the random agent
def evaluate_random_agent(env, agent, n_eval_episodes=10):
    episode_rewards = []
    for episode in range(n_eval_episodes):
        obs = env.reset()  # Only get the observation from reset
        done = False
        total_reward = 0
        while not done:
            action = agent.act()
            # Wrap the action in a list to ensure compatibility with DummyVecEnv
            obs, reward, done, _ = env.step([action])  # Action wrapped in a list
            total_reward += reward
        episode_rewards.append(total_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward



# Run the random agent evaluation
random_mean_reward, random_std_reward = evaluate_random_agent(env, random_agent, n_eval_episodes=10)

# Log random agent evaluation results to Comet.ml
experiment.log_metric("random_agent_mean_reward", random_mean_reward)
experiment.log_metric("random_agent_std_reward", random_std_reward)

# End Comet.ml experiment
experiment.end()

# Print final evaluation results
print(f"PPO Agent - Mean Reward: {ppo_mean_reward}, Std Dev: {ppo_std_reward}")
print(f"Random Agent - Mean Reward: {random_mean_reward}, Std Dev: {random_std_reward}")

# Plot comparison of Mean Rewards
agents = ['PPO Agent', 'Random Agent']
mean_rewards = [ppo_mean_reward, random_mean_reward]
std_rewards = [ppo_std_reward, random_std_reward]

plt.figure(figsize=(10, 6))
plt.bar(agents, mean_rewards, yerr=std_rewards, capsize=10)
plt.ylabel('Mean Reward')
plt.title('Comparison of Mean Reward: PPO Agent vs Random Agent')
plt.grid(True)
plt.savefig('agent_comparison.png')
print("Comparison plot saved as agent_comparison.png")
plt.show()
