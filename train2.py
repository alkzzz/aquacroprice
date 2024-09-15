import numpy as np
import matplotlib.pyplot as plt
from comet_ml import OfflineExperiment
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from aquacroprice.envs.maize import Maize
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)

# Custom callback for logging final rewards, yields, and irrigation with debug output
class RewardLoggingCallback(BaseCallback):
    def __init__(self, experiment, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.experiment = experiment
        self.episode_rewards = []
        self.current_episode_rewards = []
        self.total_steps = 0  # Keep track of the total steps

    def _on_step(self) -> bool:
        # Capture reward for each step
        reward = self.locals['rewards'][0]
        self.current_episode_rewards.append(reward)
        self.total_steps += 1

        # Check if the episode has ended
        if 'dones' in self.locals and any(self.locals['dones']):
            # Calculate total reward for the episode
            total_reward = np.sum(self.current_episode_rewards)
            self.episode_rewards.append(total_reward)
            
            # Log mean reward for the episode in Comet.ml
            self.experiment.log_metric("episode_reward", total_reward)

            # Reset for the next episode
            self.current_episode_rewards = []

        return True

    def _on_training_end(self):
        # Calculate and log the final mean and std of rewards at the end of training
        if self.episode_rewards:
            final_mean_reward = np.mean(self.episode_rewards)
            final_std_reward = np.std(self.episode_rewards)

            # Log final values to Comet.ml
            self.experiment.log_metric("final_mean_reward", final_mean_reward)
            self.experiment.log_metric("final_std_reward", final_std_reward)

        # Plot rewards
        self.plot_rewards()

    def plot_rewards(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.savefig('reward_plot.png')

# Initialize Comet.ml experiment in offline mode
experiment = OfflineExperiment(
    project_name="aqua-gym-rice",
    workspace="alkzzz",
    offline_directory="/home/alkaff/phd/aquacroprice/comet_logs"
)

# Create the environment for training
train_env = VecNormalize(DummyVecEnv([lambda: Monitor(Maize(mode='train', year1=1982, year2=2002))]), norm_obs=True, norm_reward=True)

# Custom reward logging callback
reward_logging_callback = RewardLoggingCallback(experiment)

# Training parameters
train_timesteps = 200000  # Adjust the number of timesteps as needed

# Define PPO algorithm with hyperparameters
ppo_model = PPO(
    "MlpPolicy", train_env, verbose=1,
    learning_rate=5e-4,
    n_steps=4096,
    batch_size=64,
    n_epochs=20,
    ent_coef=0.01,
    tensorboard_log="./tensorboard/"
)

# Define DQN algorithm with standard hyperparameters
dqn_model = DQN(
    "MlpPolicy", train_env, verbose=1,
    tensorboard_log="./tensorboard/"
)

# Evaluation function
def evaluate_agent(agent, env, n_eval_episodes=100, agent_name="RandomAgent"):
    total_rewards = []
    yields = []
    irrigations = []

    with open(f"{agent_name}_evaluation_debug.txt", "w") as log_file:
        for episode in range(n_eval_episodes):
            obs, done = env.reset(), False
            episode_rewards = []
            total_irrigation = 0

            step_count = 0

            while not done:
                action, _states = agent.predict(obs)
                obs, reward, done, info = env.step(action)

                episode_rewards.append(reward)
                total_irrigation += info[0].get('total_irrigation', 0)
                dry_yield = info[0].get('dry_yield', 0)

                step_count += 1

            total_rewards.append(np.sum(episode_rewards))
            yields.append(dry_yield)
            irrigations.append(total_irrigation)

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_yield = np.mean(yields)
    mean_irrigation = np.mean(irrigations)

    return mean_reward, std_reward, mean_yield, mean_irrigation

# Train PPO Model
print("Training PPO Model...")
ppo_model.learn(total_timesteps=train_timesteps, callback=reward_logging_callback)

# Train DQN Model
print("Training DQN Model...")
dqn_model.learn(total_timesteps=train_timesteps, callback=reward_logging_callback)

# Initialize the random agent
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        action = self.action_space.sample()
        return [action], state

random_agent = RandomAgent(train_env.action_space)

# Evaluate the PPO Model
print("Evaluating PPO Model with Debugging...")
ppo_mean_reward, ppo_std_reward, ppo_mean_yield, ppo_mean_irrigation = evaluate_agent(ppo_model, train_env, n_eval_episodes=100, agent_name="PPO")

# Evaluate the DQN Model
print("Evaluating DQN Model with Debugging...")
dqn_mean_reward, dqn_std_reward, dqn_mean_yield, dqn_mean_irrigation = evaluate_agent(dqn_model, train_env, n_eval_episodes=100, agent_name="DQN")

# Evaluate the Random Agent
print("Evaluating Random Agent with Debugging...")
random_mean_reward, random_std_reward, random_mean_yield, random_mean_irrigation = evaluate_agent(random_agent, train_env, n_eval_episodes=100, agent_name="RandomAgent")

# Log the final values for PPO, DQN, and RandomAgent
experiment.log_metric(f"PPO_final_mean_reward", ppo_mean_reward)
experiment.log_metric(f"PPO_final_std_reward", ppo_std_reward)
experiment.log_metric(f"PPO_final_mean_yield", ppo_mean_yield)
experiment.log_metric(f"PPO_final_mean_irrigation", ppo_mean_irrigation)

experiment.log_metric(f"DQN_final_mean_reward", dqn_mean_reward)
experiment.log_metric(f"DQN_final_std_reward", dqn_std_reward)
experiment.log_metric(f"DQN_final_mean_yield", dqn_mean_yield)
experiment.log_metric(f"DQN_final_mean_irrigation", dqn_mean_irrigation)

experiment.log_metric(f"RandomAgent_final_mean_reward", random_mean_reward)
experiment.log_metric(f"RandomAgent_final_std_reward", random_std_reward)
experiment.log_metric(f"RandomAgent_final_mean_yield", random_mean_yield)
experiment.log_metric(f"RandomAgent_final_mean_irrigation", random_mean_irrigation)

# End Comet.ml experiment
experiment.end()

# Plot comparison of Mean Rewards
plt.figure(figsize=(12, 7))
agents = ['PPO', 'DQN', 'RandomAgent']
mean_rewards = [ppo_mean_reward, dqn_mean_reward, random_mean_reward]
std_rewards = [ppo_std_reward, dqn_std_reward, random_std_reward]

plt.bar(agents, mean_rewards, yerr=std_rewards, capsize=10, color=['green', 'orange', 'blue'])
plt.ylabel('Mean Reward')
plt.title('Comparison of Total Rewards (PPO, DQN, Random Agent)')
plt.grid(True)
plt.savefig('ppo_vs_dqn_vs_random_reward.png')
plt.show()

# Plot comparison of Mean Yields
plt.figure(figsize=(12, 7))
mean_yields = [ppo_mean_yield, dqn_mean_yield, random_mean_yield]

plt.bar(agents, mean_yields, capsize=10, color=['green', 'orange', 'blue'])
plt.ylabel('Mean Yield (tonne/ha)')
plt.title('Comparison of Mean Yields (PPO, DQN, Random Agent)')
plt.grid(True)
plt.savefig('ppo_vs_dqn_vs_random_yield.png')
plt.show()

# Plot comparison of Mean Irrigation
plt.figure(figsize=(12, 7))
mean_irrigations = [ppo_mean_irrigation, dqn_mean_irrigation, random_mean_irrigation]

plt.bar(agents, mean_irrigations, capsize=10, color=['green', 'orange', 'blue'])
plt.ylabel('Mean Irrigation (mm)')
plt.title('Comparison of Mean Irrigation (PPO, DQN, Random Agent)')
plt.grid(True)
plt.savefig('ppo_vs_dqn_vs_random_irrigation.png')
plt.show()
