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

# Custom callback for logging and plotting rewards, yields, and irrigation
class RewardLoggingCallback(BaseCallback):
    def __init__(self, experiment, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.experiment = experiment
        self.episode_rewards = []
        self.episode_schedules = []
        self.current_rewards = 0
        self.highest_reward = -np.inf
        self.best_schedule = []
        self.yields = []  # To store yield values for each episode
        self.irrigations = []  # To store irrigation values for each episode

    def _on_step(self) -> bool:
        if 'rewards' in self.locals:
            mean_reward = np.mean(self.locals['rewards'])
            self.current_rewards += mean_reward

        if 'dones' in self.locals and any(self.locals['dones']):
            env = self.locals['env'].envs[0]
            info = self.locals['infos'][0]  # Access the info dictionary from the environment

            dry_yield = info.get('dry_yield', float('nan'))
            total_irrigation = info.get('total_irrigation', float('nan'))

            if not np.isnan(dry_yield) and not np.isnan(total_irrigation):
                print(f"Episode ended. Dry yield: {dry_yield}, Total irrigation: {total_irrigation}")

                self.yields.append(dry_yield)
                self.irrigations.append(total_irrigation)
            else:
                print("Final stats not available or contain NaN values.")

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

        # Calculate and log mean yield and irrigation at the end of training
        if self.yields and self.irrigations:
            mean_yield = np.mean(self.yields)
            mean_irrigation = np.mean(self.irrigations)
        else:
            mean_yield = float('nan')
            mean_irrigation = float('nan')

        self.experiment.log_metric("mean_yield", mean_yield)
        self.experiment.log_metric("mean_irrigation", mean_irrigation)
        print(f"Mean Yield: {mean_yield}, Mean Irrigation: {mean_irrigation}")

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

mean_rewards = []
std_rewards = []
mean_yields = []
mean_irrigations = []
agents = []

for name, model in algorithms.items():
    print(f"Training {name} with enhanced exploration...")
    model.learn(total_timesteps=train_timesteps, callback=reward_logging_callback)
    
    eval_env = DummyVecEnv([lambda: Monitor(Rice(mode='eval', year1=2003, year2=2018))])
    
    print(f"Evaluating {name}...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000, return_episode_rewards=False, deterministic=False)
    
    # Calculate mean yield and irrigation over evaluation episodes
    yields = []
    irrigations = []
    
    for _ in range(10):  # Assuming you want to evaluate over 10 episodes
        obs, done = eval_env.reset(), False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            if done:
                dry_yield = info[0].get('dry_yield', float('nan'))
                total_irrigation = info[0].get('total_irrigation', float('nan'))
                yields.append(dry_yield)
                irrigations.append(total_irrigation)
                print(f"Evaluation Episode Ended. Dry yield: {dry_yield}, Total irrigation: {total_irrigation}")

    mean_yield = np.mean(yields) if yields else float('nan')
    mean_irrigation = np.mean(irrigations) if irrigations else float('nan')

    experiment.log_metric(f"{name}_mean_reward", mean_reward)
    experiment.log_metric(f"{name}_std_reward", std_reward)
    experiment.log_metric(f"{name}_mean_yield", mean_yield)
    experiment.log_metric(f"{name}_mean_irrigation", mean_irrigation)
    
    mean_rewards.append(mean_reward)
    std_rewards.append(std_reward)
    mean_yields.append(mean_yield)
    mean_irrigations.append(mean_irrigation)
    agents.append(name)

# Evaluate the Random Agent
print("Evaluating Random Agent...")
mean_reward, std_reward = evaluate_policy(random_agent, eval_env, n_eval_episodes=1000, return_episode_rewards=False)

# Repeating the yield and irrigation logging for the Random Agent
yields = []
irrigations = []

for _ in range(10):  # Assuming you want to evaluate over 10 episodes
    obs, done = eval_env.reset(), False
    while not done:
        action, _states = random_agent.predict(obs)
        obs, reward, done, info = eval_env.step(action)
        if done:
            dry_yield = info[0].get('dry_yield', float('nan'))
            total_irrigation = info[0].get('total_irrigation', float('nan'))
            yields.append(dry_yield)
            irrigations.append(total_irrigation)
            print(f"Evaluation Episode Ended. Dry yield: {dry_yield}, Total irrigation: {total_irrigation}")

mean_yield = np.mean(yields) if yields else float('nan')
mean_irrigation = np.mean(irrigations) if irrigations else float('nan')

experiment.log_metric(f"RandomAgent_mean_reward", mean_reward)
experiment.log_metric(f"RandomAgent_std_reward", std_reward)
experiment.log_metric(f"RandomAgent_mean_yield", mean_yield)
experiment.log_metric(f"RandomAgent_mean_irrigation", mean_irrigation)

mean_rewards.append(mean_reward)
std_rewards.append(std_reward)
mean_yields.append(mean_yield)
mean_irrigations.append(mean_irrigation)
agents.append("RandomAgent")

# End Comet.ml experiment
experiment.end()

# Print final evaluation results
for i, agent in enumerate(agents):
    print(f"{agent} - Mean Reward: {mean_rewards[i]}, Std Dev: {std_rewards[i]}, Mean Yield: {mean_yields[i]}, Mean Irrigation: {mean_irrigations[i]}")

# Plot comparison of Mean Rewards
plt.figure(figsize=(12, 7))
plt.bar(agents, mean_rewards, yerr=std_rewards, capsize=10, color='blue')
plt.ylabel('Mean Reward')
plt.title('Comparison of Mean Reward across Algorithms')
plt.grid(True)
plt.savefig('algorithm_comparison_reward.png')
print("Reward comparison plot saved as algorithm_comparison_reward.png")
plt.show()

# Plot comparison of Mean Yields
plt.figure(figsize=(12, 7))
plt.bar(agents, mean_yields, yerr=np.std(mean_yields), capsize=10, color='green')
plt.ylabel('Mean Yield (tonne/ha)')
plt.title('Comparison of Mean Yield across Algorithms')
plt.grid(True)
plt.savefig('algorithm_comparison_yield.png')
print("Yield comparison plot saved as algorithm_comparison_yield.png")
plt.show()

# Plot comparison of Mean Irrigation
plt.figure(figsize=(12, 7))
plt.bar(agents, mean_irrigations, yerr=np.std(mean_irrigations), capsize=10, color='orange')
plt.ylabel('Mean Irrigation (mm)')
plt.title('Comparison of Mean Irrigation across Algorithms')
plt.grid(True)
plt.savefig('algorithm_comparison_irrigation.png')
print("Irrigation comparison plot saved as algorithm_comparison_irrigation.png")
plt.show()
