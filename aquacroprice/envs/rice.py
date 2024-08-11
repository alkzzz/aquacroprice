import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from aquacrop.entities.crop import Crop
from aquacrop.entities.soil import Soil
from aquacrop.entities.inititalWaterContent import InitialWaterContent
from aquacrop.entities.irrigationManagement import IrrigationManagement
from aquacrop.core import AquaCropModel
from aquacrop.utils import prepare_weather, get_filepath

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="The 'delim_whitespace' keyword in pd.read_csv is deprecated and will be removed in a future version.")
warnings.filterwarnings("ignore", category=FutureWarning, message="DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.")

# Configuration dictionary for the environment
config = dict(
    name='generated_weather_data',
    year1=2000,
    year2=2050,
    crop='PaddyRice',
    soil='Paddy',
    init_wc=InitialWaterContent(depth_layer=[1, 2], value=['FC', 'FC']),
    planting_date='08/01',
    days_to_irr=1,
    max_irr=25,
    action_set='depth',
)

# Define the Rice environment class
class Rice(gym.Env):
    def __init__(self, render_mode=None):
        super(Rice, self).__init__()
        print("Initializing Rice environment...")
        self.render_mode = render_mode
        self.planting_date = config['planting_date']
        self.days_to_irr = config["days_to_irr"]
        self.year1 = config["year1"]
        self.year2 = config["year2"]
        self.max_irr = config['max_irr']
        self.init_wc = config["init_wc"]
        self.action_set = config["action_set"]
        
        soil = config['soil']
        if isinstance(soil, str):
            self.soil = Soil(soil)
        else:
            assert isinstance(soil, Soil), "soil needs to be 'str' or 'Soil'"
            self.soil = soil

        # Define observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # Define discrete action space for irrigation depth
        self.action_space = spaces.Discrete(self.max_irr + 1)  # Discrete space from 0 to 25


    def reset(self, seed=None, options=None):
        print("Resetting environment...")
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Corrected line
        sim_year = np.random.randint(self.year1, self.year2 + 1)
        self.simcalyear = sim_year
        print(f"Chosen Year: {self.simcalyear}")

        # Initialize the Crop object with the selected planting date
        crop = config['crop']
        if isinstance(crop, str):
            self.crop = Crop(crop, self.planting_date)
        else:
            assert isinstance(crop, Crop), "crop needs to be 'str' or 'Crop'"
            self.crop = crop

        print(f"Crop Planting Date: {self.crop.planting_date}")

        # Update weather data for the selected year
        self.wdf = prepare_weather('generated_weather_data.txt')
        self.wdf['Year'] = self.simcalyear

        self.irr_sched = []

        # Initialize the AquaCrop model for the new simulation year
        self.model = AquaCropModel(f'{self.simcalyear}/{self.planting_date}', 
                                f'{self.simcalyear}/12/31', self.wdf, self.soil, self.crop,
                                irrigation_management=IrrigationManagement(irrigation_method=5),
                                initial_water_content=self.init_wc)
        self.model.run_model()
        
        self.cumulative_reward = 0.0
        
        obs = self._get_obs()
        info = dict()

        return obs, info

    def _get_obs(self):
        cond = self.model._init_cond
        total_precipitation_last_7_days = self._get_total_precipitation_last_7_days()
        obs = np.array([
            cond.canopy_cover,
            cond.biomass,
            cond.harvest_index,
            cond.DryYield,
            cond.z_root,
            total_precipitation_last_7_days
        ], dtype=np.float32)

        print(f'Obs: ', obs)

        return obs

    def _get_total_precipitation_last_7_days(self):
        current_day = self.model._clock_struct.time_step_counter
        last_7_days = self.wdf.iloc[max(0, current_day - 7):current_day]
        total_precipitation = last_7_days['Precipitation'].sum()
        return total_precipitation

    def step(self, action):
        depth = np.clip(action, 0, self.max_irr)  # Clip the action to ensure it's within the valid range
        self.model.irrigation_management.depth = depth
        print(f"Applied irrigation depth: {depth}")
        
        self.model.run_model(initialize_model=False)  # Run the model for the current timestep
        print(f'Timestep: ', self.model._clock_struct.time_step_counter)
        
        terminated = self.model._clock_struct.model_is_finished
        truncated = False
        next_obs = self._get_obs()

        # Define the weights for rewards
        intermediate_reward_weight = 0.05  # Lower weight for intermediate rewards
        final_reward_weight = 20.0  # Higher weight for the final reward based on yield

        # Extract relevant observations
        biomass = next_obs[1]
        harvest_index = next_obs[2]
        canopy_cover = next_obs[0]

        # Calculate intermediate reward based on the current growth stage
        if self.model._clock_struct.time_step_counter < 50:  # Early to mid-season
            reward = intermediate_reward_weight * (
                (biomass * 0.5) + 
                (canopy_cover * 0.3) + 
                (harvest_index * 0.2)
            )
        else:  # Late season
            reward = intermediate_reward_weight * (
                (biomass * 0.7) + 
                (harvest_index * 0.3)
            )
        
        # Accumulate the intermediate reward
        self.cumulative_reward += reward

        # Handle episode termination
        if terminated:
            dry_yield = self.model._outputs.final_stats['Dry yield (tonne/ha)'].mean()
            final_reward = dry_yield * final_reward_weight  # Emphasize the final yield reward

            # Add a penalty if the dry yield is below a certain threshold (optional)
            penalty_threshold = 6.5  # Example threshold for low yield
            if dry_yield < penalty_threshold:
                penalty = -500.0  # Penalty for low yield
                self.cumulative_reward += penalty
                print(f'Applied Penalty: {penalty} due to low yield: {dry_yield}')

            # Add the final reward to the cumulative reward
            self.cumulative_reward += final_reward
            reward = self.cumulative_reward  # The final reward is the cumulative reward

            print(f'Final Cumulative Reward: {reward} (Dry Yield: {dry_yield})')
        
        # Info dictionary for additional details (optional, can be used for debugging)
        info = dict()

        # Return the next observation, the reward, whether the episode is terminated, and additional info
        return next_obs, reward, terminated, truncated, info




