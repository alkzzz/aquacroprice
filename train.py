import numpy as np
import matplotlib.pyplot as plt
from comet_ml import OfflineExperiment
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import ARS
from aquacroprice.envs.rice import Rice

import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)

# Custom callback for logging and plotting rewards and irrigation
class RewardLoggingCallback(BaseCallback):
    def __init__(self, experiment, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.experiment = experiment
        self.episode_rewards = []
        self.episode_schedules = []
        self.current_rewards = 0
        self.highest_reward = -np.inf
        self.best_schedule = []

    def _on_step(self) -> bool:
        if 'rewards' in self.locals:
            mean_reward = np.mean(self.locals['rewards'])
            self.current_rewards += mean_reward

        if 'dones' in self.locals and any(self.locals['dones']):
            env = self.locals['env'].envs[0]
            total_irrigation = env.unwrapped.model._outputs.final_stats['Seasonal irrigation (mm)'].mean()

            self.episode_rewards.append(self.current_rewards)
            self.episode_schedules.append(env.unwrapped.irrigation_schedule)

            if self.current_rewards > self.highest_reward:
                self.highest_reward = self.current_rewards
                self.best_schedule = env.unwrapped.irrigation_schedule.copy()

            self.experiment.log_metric("reward", self.current_rewards, step=len(self.episode_rewards))

            self.current_rewards = 0
            env.unwrapped.irrigation_schedule = []

        return True

    def _on_training_end(self):
        self.plot_rewards()
        self.plot_irrigation_schedule()

    def plot_rewards(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.savefig('reward_plot.png')
        print("Reward plot saved as reward_plot.png")

    def plot_irrigation_schedule(self):
        if len(self.best_schedule) > 0:
            timesteps, actions = zip(*self.best_schedule)
            plt.figure(figsize=(10, 5))
            plt.plot(timesteps, actions, label='Best Episode Irrigation Schedule')
            plt.xlabel('Timestep (Day)')
            plt.ylabel('Irrigation Action (0 or 25)')
            plt.title('Irrigation Schedule for Highest Reward Episode')
            plt.legend()
            plt.savefig('best_irrigation_schedule_plot.png')
            print("Irrigation schedule plot saved as best_irrigation_schedule_plot.png")
        else:
            print("No irrigation schedule found for any episode.")


# Initialize Comet.ml experiment in offline mode
experiment = OfflineExperiment(
    project_name="aqua-gym-rice",
    workspace="alkzzz",
    offline_directory="/home/alkaff/phd/aquacroprice/comet_logs"
)

# Create the environment for training
train_env = DummyVecEnv([lambda: Monitor(Rice(mode='train', year1=1982, year2=2002))])

# Custom reward logging callback
reward_logging_callback = RewardLoggingCallback(experiment)

# Training parameters (shared among algorithms)
train_timesteps = 1000

# Define algorithms and hyperparameters with exploration encouragement
algorithms = {
    "PPO": PPO(
        "MlpPolicy", train_env, verbose=1,
        learning_rate=5e-4,
        n_steps=4096,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01
    ),
    "DQN": DQN(
        "MlpPolicy", train_env, verbose=1,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=64,
        target_update_interval=1000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        exploration_fraction=0.2
    ),
    "ARS": ARS(
        "MlpPolicy", train_env, verbose=1,
        n_delta=64,
        n_top=16,
        delta_std=0.05
    ),
}

# Define the Random Agent
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        action = self.action_space.sample()
        return [action], state

# Initialize the random agent
random_agent = RandomAgent(train_env.action_space)

# Train and evaluate PPO, DQN, and ARS
mean_rewards = []
std_rewards = []
agents = []

for name, model in algorithms.items():
    print(f"Training {name} with enhanced exploration...")
    model.learn(total_timesteps=train_timesteps, callback=reward_logging_callback)
    
    eval_env = DummyVecEnv([lambda: Monitor(Rice(mode='eval', year1=2003, year2=2018))])
    
    print(f"Evaluating {name}...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000, return_episode_rewards=False, deterministic=False)
    
    experiment.log_metric(f"{name}_mean_reward", mean_reward)
    experiment.log_metric(f"{name}_std_reward", std_reward)
    
    mean_rewards.append(mean_reward)
    std_rewards.append(std_reward)
    agents.append(name)

# Evaluate the Random Agent
print("Evaluating Random Agent...")
mean_reward, std_reward = evaluate_policy(random_agent, eval_env, n_eval_episodes=1000, return_episode_rewards=False)

experiment.log_metric(f"RandomAgent_mean_reward", mean_reward)
experiment.log_metric(f"RandomAgent_std_reward", std_reward)

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
