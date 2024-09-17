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
            total_reward = np.sum(self.current_episode_rewards)
            self.episode_rewards.append(total_reward)
            self.experiment.log_metric("episode_reward", total_reward)
            self.current_episode_rewards = []  # Reset for the next episode
        return True

    def _on_training_end(self):
        if self.episode_rewards:
            final_mean_reward = np.mean(self.episode_rewards)
            final_std_reward = np.std(self.episode_rewards)
            self.experiment.log_metric("final_mean_reward", final_mean_reward)
            self.experiment.log_metric("final_std_reward", final_std_reward)
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
    project_name="aqua-gym-rice-training",
    workspace="alkzzz",
    offline_directory="/home/alkaff/phd/aquacroprice/comet_logs"
)

### Training Phase ###
# Create the environment for training
train_env = DummyVecEnv([lambda: Monitor(Maize(mode='train', year1=1982, year2=2002))])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

# Custom reward logging callback
reward_logging_callback = RewardLoggingCallback(experiment)

# Training parameters
train_timesteps = 50000  # Adjust the number of timesteps as needed

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

# Define DQN algorithm with hyperparameters
dqn_model = DQN(
    "MlpPolicy", train_env, verbose=1,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    target_update_interval=500,
    train_freq=7,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    tensorboard_log="./tensorboard/"
)

# Train PPO Model
print("Training PPO Model...")
ppo_model.learn(total_timesteps=train_timesteps, callback=reward_logging_callback)

# Save the trained PPO model and the normalization stats
ppo_model.save("ppo_model")
train_env.save("ppo_vecnormalize.pkl")

# Train DQN Model
print("Training DQN Model...")
dqn_model.learn(total_timesteps=train_timesteps, callback=reward_logging_callback)

# Save the trained DQN model and the normalization stats
dqn_model.save("dqn_model")
train_env.save("dqn_vecnormalize.pkl")

# End the training Comet.ml experiment
experiment.end()
