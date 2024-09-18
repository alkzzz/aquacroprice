import numpy as np
import matplotlib.pyplot as plt
import csv
from comet_ml import OfflineExperiment
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from aquacroprice.envs.maize import Maize
import warnings
import logging
import random
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)

### Evaluation Phase ###
print("Starting evaluation phase...")

# Function to set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

# Function to create the evaluation environment
def make_eval_env(seed):
    def _init():
        env = Monitor(Maize(mode='train', year1=2003, year2=2018))
        # Instead of env.seed(seed), we will seed the random number generators manually
        set_seed(seed)
        return env
    eval_env = DummyVecEnv([_init])
    return eval_env

# Load the saved normalization statistics and set training=False, with fixed seed
def load_env_and_model(model_name, vecnormalize_filename, seed):
    # Set seed
    set_seed(seed)
    # Create a new evaluation environment
    eval_env = make_eval_env(seed)
    # Load the saved VecNormalize statistics
    eval_env = VecNormalize.load(vecnormalize_filename, eval_env)
    # Do not update them at test time
    eval_env.training = False
    eval_env.norm_reward = False
    # Load the saved model and set the environment
    if model_name == "ppo_model":
        model = PPO.load(model_name, env=eval_env)
    elif model_name == "dqn_model":
        model = DQN.load(model_name, env=eval_env)
    else:
        raise ValueError("Unknown model name")
    return model, eval_env

# Evaluation function
def evaluate_agent(agent, env, n_eval_episodes=100, agent_name="Agent"):
    total_rewards = []
    yields = []
    irrigations = []

    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_rewards = []
        total_irrigation = 0

        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            total_irrigation += info[0].get('total_irrigation', 0)
            dry_yield = info[0].get('dry_yield', 0)

        total_rewards.append(np.sum(episode_rewards))
        yields.append(dry_yield)
        irrigations.append(total_irrigation)

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_yield = np.mean(yields)
    std_yield = np.std(yields)
    mean_irrigation = np.mean(irrigations)
    std_irrigation = np.std(irrigations)

    print(f"{agent_name} Mean Reward: {mean_reward}, Std: {std_reward}")
    print(f"{agent_name} Mean Yield: {mean_yield}, Std: {std_yield}")
    print(f"{agent_name} Mean Irrigation: {mean_irrigation}, Std: {std_irrigation}")

    # Return the results as a dictionary
    return {
        'agent_name': agent_name,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_yield': mean_yield,
        'std_yield': std_yield,
        'mean_irrigation': mean_irrigation,
        'std_irrigation': std_irrigation
    }

# Initialize a new Comet.ml experiment for evaluation
experiment = OfflineExperiment(
    project_name="aqua-gym-rice-evaluation",
    workspace="alkzzz",
    offline_directory="/home/alkaff/phd/aquacroprice/comet_logs"
)

# List to store evaluation results
evaluation_results = []

# Set the seed values
seeds = [1, 2, 3, 4, 5]

# Loop through different seeds and evaluate for PPO
for seed in seeds:
    print(f"Evaluating PPO Model with seed {seed}...")
    ppo_model, ppo_eval_env = load_env_and_model("ppo_model", "ppo_vecnormalize.pkl", seed)
    ppo_results = evaluate_agent(ppo_model, ppo_eval_env, n_eval_episodes=100, agent_name=f"PPO_seed_{seed}")
    evaluation_results.append(ppo_results)

    # Log PPO metrics to Comet.ml
    experiment.log_metric(f"PPO_seed_{seed}_mean_reward", ppo_results['mean_reward'])
    experiment.log_metric(f"PPO_seed_{seed}_std_reward", ppo_results['std_reward'])
    experiment.log_metric(f"PPO_seed_{seed}_mean_yield", ppo_results['mean_yield'])
    experiment.log_metric(f"PPO_seed_{seed}_std_yield", ppo_results['std_yield'])
    experiment.log_metric(f"PPO_seed_{seed}_mean_irrigation", ppo_results['mean_irrigation'])
    experiment.log_metric(f"PPO_seed_{seed}_std_irrigation", ppo_results['std_irrigation'])

# Loop through different seeds and evaluate for DQN
for seed in seeds:
    print(f"Evaluating DQN Model with seed {seed}...")
    dqn_model, dqn_eval_env = load_env_and_model("dqn_model", "dqn_vecnormalize.pkl", seed)
    dqn_results = evaluate_agent(dqn_model, dqn_eval_env, n_eval_episodes=100, agent_name=f"DQN_seed_{seed}")
    evaluation_results.append(dqn_results)

    # Log DQN metrics to Comet.ml
    experiment.log_metric(f"DQN_seed_{seed}_mean_reward", dqn_results['mean_reward'])
    experiment.log_metric(f"DQN_seed_{seed}_std_reward", dqn_results['std_reward'])
    experiment.log_metric(f"DQN_seed_{seed}_mean_yield", dqn_results['mean_yield'])
    experiment.log_metric(f"DQN_seed_{seed}_std_yield", dqn_results['std_yield'])
    experiment.log_metric(f"DQN_seed_{seed}_mean_irrigation", dqn_results['mean_irrigation'])
    experiment.log_metric(f"DQN_seed_{seed}_std_irrigation", dqn_results['std_irrigation'])

# Loop through different seeds and evaluate Random Agent
def make_random_env(seed):
    def _init():
        env = Monitor(Maize(mode='train', year1=2003, year2=2018))
        set_seed(seed)
        return env
    random_env = DummyVecEnv([_init])
    return random_env

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        action = self.action_space.sample()
        return [action], state

for seed in seeds:
    print(f"Evaluating Random Agent with seed {seed}...")
    random_env = make_random_env(seed)
    random_agent = RandomAgent(random_env.action_space)
    random_results = evaluate_agent(random_agent, random_env, n_eval_episodes=100, agent_name=f"RandomAgent_seed_{seed}")
    evaluation_results.append(random_results)

    # Log Random Agent metrics to Comet.ml
    experiment.log_metric(f"RandomAgent_seed_{seed}_mean_reward", random_results['mean_reward'])
    experiment.log_metric(f"RandomAgent_seed_{seed}_std_reward", random_results['std_reward'])
    experiment.log_metric(f"RandomAgent_seed_{seed}_mean_yield", random_results['mean_yield'])
    experiment.log_metric(f"RandomAgent_seed_{seed}_std_yield", random_results['std_yield'])
    experiment.log_metric(f"RandomAgent_seed_{seed}_mean_irrigation", random_results['mean_irrigation'])
    experiment.log_metric(f"RandomAgent_seed_{seed}_std_irrigation", random_results['std_irrigation'])

# Save evaluation results to a CSV file
csv_filename = 'evaluation_results.csv'
csv_columns = ['agent_name', 'mean_reward', 'std_reward', 'mean_yield', 'std_yield', 'mean_irrigation', 'std_irrigation']

try:
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in evaluation_results:
            writer.writerow(data)
    print(f"Evaluation results saved to {csv_filename}")
except IOError:
    print("I/O error while writing evaluation results to CSV")

# End the evaluation Comet.ml experiment
experiment.end()

# Plot comparison of Mean Rewards
plt.figure(figsize=(12, 7))
agents = [result['agent_name'] for result in evaluation_results]
mean_rewards = [result['mean_reward'] for result in evaluation_results]
std_rewards = [result['std_reward'] for result in evaluation_results]

plt.bar(agents, mean_rewards, yerr=std_rewards, capsize=10, color=['green', 'orange', 'blue'])
plt.ylabel('Mean Reward')
plt.title('Comparison of Total Rewards (PPO, DQN, Random Agent)')
plt.grid(True)
plt.savefig('ppo_vs_dqn_vs_random_reward.png')
plt.show()

# Plot comparison of Mean Yields with Standard Deviations
plt.figure(figsize=(12, 7))
mean_yields = [result['mean_yield'] for result in evaluation_results]
std_yields = [result['std_yield'] for result in evaluation_results]

plt.bar(agents, mean_yields, yerr=std_yields, capsize=10, color=['green', 'orange', 'blue'])
plt.ylabel('Mean Yield (tonne/ha)')
plt.title('Comparison of Mean Yields (PPO, DQN, Random Agent)')
plt.grid(True)
plt.savefig('ppo_vs_dqn_vs_random_yield.png')
plt.show()

# Plot comparison of Mean Irrigation with Standard Deviations
plt.figure(figsize=(12, 7))
mean_irrigations = [result['mean_irrigation'] for result in evaluation_results]
std_irrigations = [result['std_irrigation'] for result in evaluation_results]

plt.bar(agents, mean_irrigations, yerr=std_irrigations, capsize=10, color=['green', 'orange', 'blue'])
plt.ylabel('Mean Irrigation (mm)')
plt.title('Comparison of Mean Irrigation (PPO, DQN, Random Agent)')
plt.grid(True)
plt.savefig('ppo_vs_dqn_vs_random_irrigation.png')
plt.show()
