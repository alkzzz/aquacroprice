import numpy as np
import matplotlib.pyplot as plt
from comet_ml import OfflineExperiment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import ARS, RecurrentPPO
from aquacroprice.envs.rice import Rice

import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)

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
        if 'rewards' in self.locals:
            self.current_rewards += np.mean(self.locals['rewards'])  # Mean reward across environments
            self.current_timesteps += 1

        if 'dones' in self.locals:
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
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_timesteps, self.episode_rewards)
        plt.xlabel('Timesteps')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.savefig('reward_plot.png')
        print("Reward plot saved as reward_plot.png")

# Initialize Comet.ml experiment in offline mode
experiment = OfflineExperiment(
    project_name="aqua-gym-rice",
    workspace="alkzzz",
    offline_directory="/home/alkaff/phd/aquacroprice/comet_logs"
)

# Create the environment for training (years 1678-2159)
train_env = DummyVecEnv([lambda: Rice(mode='train', year1=1678, year2=2159)])

# Custom reward logging callback
reward_logging_callback = RewardLoggingCallback(experiment)

# Training parameters (shared among algorithms)
train_timesteps = 20000

# Define algorithms and hyperparameters
algorithms = {
    "PPO": PPO("MlpPolicy", train_env, verbose=1, learning_rate=1e-3, n_steps=2048, batch_size=64, n_epochs=10),
    "ARS": ARS("MlpPolicy", train_env, verbose=1, n_delta=32, n_top=16),
    "RecurrentPPO": RecurrentPPO("MlpLstmPolicy", train_env, verbose=1, learning_rate=1e-3, n_steps=2048, batch_size=64, n_epochs=10),
}

# Define the Random Agent
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        action = self.action_space.sample()
        return [action], state  # Return the action as a list

# Initialize the random agent
random_agent = RandomAgent(train_env.action_space)

# Train and evaluate each algorithm
mean_rewards = []
std_rewards = []
agents = []

for name, model in algorithms.items():
    print(f"Training {name}...")
    model.learn(total_timesteps=train_timesteps, callback=reward_logging_callback)
    
    # Create the environment for evaluation (years 2160-2260)
    eval_env = DummyVecEnv([lambda: Rice(mode='eval', year1=2160, year2=2260)])
    
    print(f"Evaluating {name}...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000, return_episode_rewards=False)
    
    # Log evaluation results to Comet.ml
    experiment.log_metric(f"{name}_mean_reward", mean_reward)
    experiment.log_metric(f"{name}_std_reward", std_reward)
    
    # Store results for comparison
    mean_rewards.append(mean_reward)
    std_rewards.append(std_reward)
    agents.append(name)

# Evaluate the Random Agent
print("Evaluating Random Agent...")
mean_reward, std_reward = evaluate_policy(random_agent, eval_env, n_eval_episodes=1000, return_episode_rewards=False)

# Log evaluation results to Comet.ml
experiment.log_metric(f"RandomAgent_mean_reward", mean_reward)
experiment.log_metric(f"RandomAgent_std_reward", std_reward)

# Store results for comparison
mean_rewards.append(mean_reward)
std_rewards.append(std_reward)
agents.append("RandomAgent")

# End Comet.ml experiment
experiment.end()

# Print final evaluation results
for i, agent in enumerate(agents):
    print(f"{agent} - Mean Reward: {mean_rewards[i]}, Std Dev: {std_rewards[i]}")

# Plot comparison of Mean Rewards
plt.figure(figsize=(12, 7))
plt.bar(agents, mean_rewards, yerr=std_rewards, capsize=10)
plt.ylabel('Mean Reward')
plt.title('Comparison of Mean Reward across Algorithms')
plt.grid(True)
plt.savefig('algorithm_comparison.png')
print("Comparison plot saved as algorithm_comparison.png")
plt.show()
