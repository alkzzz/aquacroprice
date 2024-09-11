import numpy as np
import matplotlib.pyplot as plt
from comet_ml import OfflineExperiment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from aquacroprice.envs.maize import Maize

import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)

# Custom callback for logging and plotting rewards, yields, and irrigation
class RewardLoggingCallback(BaseCallback):
    def __init__(self, experiment, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.experiment = experiment
        self.episode_rewards = []
        self.episode_schedules = []
        self.current_episode_rewards = []  # To store rewards per episode
        self.highest_reward = -np.inf
        self.best_schedule = []
        self.yields = []  # To store yield values for each episode
        self.irrigations = []  # To store irrigation values for each episode

    def _on_step(self) -> bool:
        # Track individual rewards
        if 'rewards' in self.locals:
            reward = self.locals['rewards'][0]
            self.current_episode_rewards.append(reward)

        # Check if the episode is done
        if 'dones' in self.locals and any(self.locals['dones']):
            env = self.locals['env'].envs[0]
            info = self.locals['infos'][0]

            dry_yield = info.get('dry_yield', float('nan'))
            total_irrigation = info.get('total_irrigation', float('nan'))

            if not np.isnan(dry_yield) and not np.isnan(total_irrigation):
                self.yields.append(dry_yield)
                self.irrigations.append(total_irrigation)

            # Calculate mean reward for the current episode
            mean_reward = np.mean(self.current_episode_rewards)
            self.episode_rewards.append(mean_reward)

            # Log the mean reward to Comet.ml
            self.experiment.log_metric("PPO_mean_reward", mean_reward, step=len(self.episode_rewards))

            # Check if this is the best episode
            if mean_reward > self.highest_reward:
                self.highest_reward = mean_reward
                self.best_schedule = env.unwrapped.irrigation_schedule.copy()

            # Reset for the next episode
            self.current_episode_rewards = []
            env.unwrapped.irrigation_schedule = []

        return True

    def _on_training_end(self):
        self.plot_rewards()
        self.plot_irrigation_schedule()

        if self.yields and self.irrigations:
            mean_yield = np.mean(self.yields)
            mean_irrigation = np.mean(self.irrigations)
        else:
            mean_yield = float('nan')
            mean_irrigation = float('nan')

        # Log additional metrics to Comet.ml
        self.experiment.log_metric("PPO_mean_yield", mean_yield)
        self.experiment.log_metric("PPO_mean_irrigation", mean_irrigation)

    def plot_rewards(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Mean Reward')
        plt.title('Mean Reward per Episode (PPO)')
        plt.savefig('ppo_reward_plot.png')

    def plot_irrigation_schedule(self):
        if len(self.best_schedule) > 0:
            timesteps, actions = zip(*self.best_schedule)
            plt.figure(figsize=(10, 5))
            plt.plot(timesteps, actions, label='Best Episode Irrigation Schedule (PPO)')
            plt.xlabel('Timestep (Day)')
            plt.ylabel('Irrigation Action (0 or 25)')
            plt.title('Irrigation Schedule for Highest Reward Episode (PPO)')
            plt.legend()
            plt.savefig('ppo_best_irrigation_schedule_plot.png')

# Initialize Comet.ml experiment in offline mode
experiment = OfflineExperiment(
    project_name="aqua-gym-rice",
    workspace="alkzzz",
    offline_directory="/home/alkaff/phd/aquacroprice/comet_logs"
)

# Create the environment for training
train_env = DummyVecEnv([lambda: Monitor(Maize(mode='train', year1=1982, year2=2002))])

# Custom reward logging callback
reward_logging_callback = RewardLoggingCallback(experiment)

# Training parameters
train_timesteps = 10

# Define PPO algorithm with hyperparameters
ppo_model = PPO(
    "MlpPolicy", train_env, verbose=1,
    learning_rate=5e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=20,
    ent_coef=0.01
)

# Define the Random Agent
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        action = self.action_space.sample()
        return [action], state

# Function to evaluate agents with debugging
def evaluate_agent(agent, env, n_eval_episodes=10, agent_name="RandomAgent"):
    total_rewards = []
    yields = []
    irrigations = []

    with open(f"{agent_name}_evaluation_debug.txt", "w") as log_file:
        for episode in range(n_eval_episodes):
            obs, done = env.reset(), False
            episode_rewards = []  # Track rewards per step
            total_irrigation = 0

            log_file.write(f"\nEpisode {episode+1}:\n")
            step_count = 0

            while not done:
                # Get the action from the agent
                action, _states = agent.predict(obs)

                # Take a step in the environment
                obs, reward, done, info = env.step(action)

                # Accumulate rewards and irrigation
                episode_rewards.append(reward)
                total_irrigation += info[0].get('total_irrigation', 0)
                dry_yield = info[0].get('dry_yield', 0)

                # Log action, irrigation, yield, and cumulative reward after each step
                log_file.write(f"Step {step_count}: Action: {action}, Reward: {reward}, "
                               f"Dry Yield: {dry_yield}, Total Irrigation: {total_irrigation}\n")

                step_count += 1

            # Store the mean reward, yield, and irrigation for the episode
            total_rewards.append(np.mean(episode_rewards))
            yields.append(dry_yield)
            irrigations.append(total_irrigation)

            # Log final values for the episode
            log_file.write(f"End of Episode {episode+1}: Mean Reward: {np.mean(episode_rewards)}, "
                           f"Final Yield: {dry_yield}, Total Irrigation: {total_irrigation}\n")

    # Compute and return the mean and standard deviation of the rewards
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_yield = np.mean(yields)
    mean_irrigation = np.mean(irrigations)

    return mean_reward, std_reward, mean_yield, mean_irrigation

# Train PPO Model
print("Training PPO Model...")
ppo_model.learn(total_timesteps=train_timesteps, callback=reward_logging_callback)

# Initialize the random agent
random_agent = RandomAgent(train_env.action_space)

# Evaluate the PPO Model
print("Evaluating PPO Model with Debugging...")
ppo_mean_reward, ppo_std_reward, ppo_mean_yield, ppo_mean_irrigation = evaluate_agent(ppo_model, train_env, n_eval_episodes=100, agent_name="PPO")

# Evaluate the Random Agent
print("Evaluating Random Agent with Debugging...")
random_mean_reward, random_std_reward, random_mean_yield, random_mean_irrigation = evaluate_agent(random_agent, train_env, n_eval_episodes=100, agent_name="RandomAgent")

# Log the results to Comet.ml
experiment.log_metric(f"PPO_mean_reward", ppo_mean_reward)
experiment.log_metric(f"PPO_std_reward", ppo_std_reward)
experiment.log_metric(f"PPO_mean_yield", ppo_mean_yield)
experiment.log_metric(f"PPO_mean_irrigation", ppo_mean_irrigation)

experiment.log_metric(f"RandomAgent_mean_reward", random_mean_reward)
experiment.log_metric(f"RandomAgent_std_reward", random_std_reward)
experiment.log_metric(f"RandomAgent_mean_yield", random_mean_yield)
experiment.log_metric(f"RandomAgent_mean_irrigation", random_mean_irrigation)

# Print the results
print(f"PPO - Mean Reward: {ppo_mean_reward}, Std Dev: {ppo_std_reward}, Mean Yield: {ppo_mean_yield}, Mean Irrigation: {ppo_mean_irrigation}")
print(f"RandomAgent - Mean Reward: {random_mean_reward}, Std Dev: {random_std_reward}, Mean Yield: {random_mean_yield}, Mean Irrigation: {random_mean_irrigation}")

# End Comet.ml experiment
experiment.end()

# Plot comparison of Mean Rewards
plt.figure(figsize=(12, 7))
agents = ['PPO', 'RandomAgent']
mean_rewards = [ppo_mean_reward, random_mean_reward]
std_rewards = [ppo_std_reward, random_std_reward]

plt.bar(agents, mean_rewards, yerr=std_rewards, capsize=10, color=['green', 'blue'])
plt.ylabel('Mean Reward')
plt.title('Comparison of Mean Rewards (PPO vs Random Agent)')
plt.grid(True)
plt.savefig('ppo_vs_random_reward.png')
plt.show()

# Plot comparison of Mean Yields
plt.figure(figsize=(12, 7))
mean_yields = [ppo_mean_yield, random_mean_yield]

plt.bar(agents, mean_yields, capsize=10, color=['green', 'blue'])
plt.ylabel('Mean Yield (tonne/ha)')
plt.title('Comparison of Mean Yields (PPO vs Random Agent)')
plt.grid(True)
plt.savefig('ppo_vs_random_yield.png')
plt.show()

# Plot comparison of Mean Irrigation
plt.figure(figsize=(12, 7))
mean_irrigations = [ppo_mean_irrigation, random_mean_irrigation]

plt.bar(agents, mean_irrigations, capsize=10, color=['green', 'blue'])
plt.ylabel('Mean Irrigation (mm)')
plt.title('Comparison of Mean Irrigation (PPO vs Random Agent)')
plt.grid(True)
plt.savefig('ppo_vs_random_irrigation.png')
plt.show()

