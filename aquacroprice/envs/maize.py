import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from aquacrop.core import AquaCropModel
from aquacrop.entities.crop import Crop
from aquacrop.entities.inititalWaterContent import InitialWaterContent
from aquacrop.entities.irrigationManagement import IrrigationManagement
from aquacrop.entities.soil import Soil
from aquacrop.utils import prepare_weather, get_filepath

# Configuration dictionary for the environment
config = dict(
    climate='champion_climate.txt',
    year1=1982,
    year2=2018,
    crop='Maize',
    soil='SandyLoam',
    init_wc=InitialWaterContent(value=['FC']),
    days_to_irr=7,
)

class Maize(gym.Env):
    def __init__(self, render_mode=None, mode='train', year1=None, year2=None):
        super(Maize, self).__init__()
        print("Initializing Maize environment...")
        self.render_mode = render_mode
        self.days_to_irr = config["days_to_irr"]
        self.day_counter = 0  # Counter to track days since the last action
        self.consecutive_zero_irrigation_episodes = 0  # Counter for consecutive zero irrigation episodes

        # If year1 and year2 are provided, override the default config
        self.year1 = year1 if year1 is not None else config["year1"]
        self.year2 = year2 if year2 is not None else config["year2"]

        self.init_wc = config["init_wc"]
        self.climate = config['climate']
        self.irrigation_schedule = []  # Store Irrigation Schedule
        self.mode = mode  # 'train' or 'eval'
        
        soil = config['soil']
        if isinstance(soil, str):
            self.soil = Soil(soil)
        else:
            assert isinstance(soil, Soil), "soil needs to be 'str' or 'Soil'"
            self.soil = soil

        # Define observation space: Includes weather-related observations
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        self.action_depths = [0, 25]
        self.action_space = spaces.Discrete(len(self.action_depths))  # Discrete action space with 6 actions

    def reset(self, seed=None, options=None):
        print("Resetting environment...")
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        sim_year = np.random.randint(self.year1, self.year2 + 1)
        self.simcalyear = sim_year
        print(f"Chosen Year: {self.simcalyear}")

        crop = config['crop']
        self.planting_date = '05/01'

        if isinstance(crop, str):
            self.crop = Crop(crop, self.planting_date)
        else:
            assert isinstance(crop, Crop), "crop needs to be 'str' or 'Crop'"
            self.crop = crop

        print(f"Crop Planting Date: {self.crop.planting_date}")

        self.wdf = prepare_weather(get_filepath(self.climate))
        self.wdf['Year'] = self.simcalyear

        self.irr_sched = []
        self.day_counter = 0  # Reset day counter

        # Initialize the AquaCrop model
        self.model = AquaCropModel(
            f'{self.simcalyear}/{self.planting_date}', 
            f'{self.simcalyear}/12/31', 
            self.wdf, 
            self.soil, 
            self.crop,
            irrigation_management=IrrigationManagement(irrigation_method=5),  # SMT method
            initial_water_content=self.init_wc
        )

        self.model.run_model()

        self.cumulative_reward = 0.0
        
        obs = self._get_obs()
        info = dict()

        return obs, info

    def _get_obs(self):
        cond = self.model._init_cond

        total_precipitation_last_7_days = self._get_total_precipitation_last_7_days()
        cum_min_temp_last_7_days = self._get_cumulative_temp_last_7_days("MinTemp")
        cum_max_temp_last_7_days = self._get_cumulative_temp_last_7_days("MaxTemp")
        prev_day_min_temp = self._get_previous_day_value("MinTemp")
        prev_day_max_temp = self._get_previous_day_value("MaxTemp")
        prev_day_precipitation = self._get_previous_day_value("Precipitation")

        obs = np.array([
            cond.age_days,
            cond.canopy_cover,
            cond.biomass,
            cond.z_root,
            cond.depletion,
            cond.taw,
            total_precipitation_last_7_days,
            cum_min_temp_last_7_days,
            cum_max_temp_last_7_days,
            prev_day_min_temp,
            prev_day_max_temp,
            prev_day_precipitation,
        ], dtype=np.float32)

        return obs

    def _get_total_precipitation_last_7_days(self):
        current_day = self.model._clock_struct.time_step_counter
        last_7_days = self.wdf.iloc[max(0, current_day - 7):current_day]
        total_precipitation = last_7_days['Precipitation'].sum()
        return total_precipitation

    def _get_cumulative_temp_last_7_days(self, temp_col):
        current_day = self.model._clock_struct.time_step_counter
        last_7_days = self.wdf.iloc[max(0, current_day - 7):current_day]
        cumulative_temp = last_7_days[temp_col].sum()
        return cumulative_temp

    def _get_previous_day_value(self, col):
        current_day = self.model._clock_struct.time_step_counter
        if current_day > 0:
            prev_day_value = self.wdf.iloc[current_day - 1][col]
        else:
            prev_day_value = 0.0  # If it's the first day, there's no previous day data
        return prev_day_value

    def step(self, action):
        # Increment the day counter
        self.day_counter += 1

        # Check if 7 days have passed since the last action
        if self.day_counter >= self.days_to_irr:
            self.day_counter = 0
            depth = self.action_depths[int(action)]  # Get irrigation depth based on the agent's action
        else:
            depth = 0

        # Apply the depth to the model
        self.model._param_struct.IrrMngt.depth = depth
        self.model.run_model(initialize_model=False)
        
        truncated = False
        next_obs = self._get_obs()
        
        # Retrieve biomass and biomass non-stress
        biomass = self.model._init_cond.biomass
        biomass_ns = self.model._init_cond.biomass_ns
        
        # Calculate the biomass difference for this step
        biomass_diff = abs(biomass_ns - biomass)
        
        # Accumulate total biomass difference over the episode
        if not hasattr(self, 'total_biomass_diff'):
            self.total_biomass_diff = 0  # Initialize if not already present
        self.total_biomass_diff += biomass_diff  # Accumulate the biomass difference

        # First irrigation reward logic
        if depth > 0:
            if not getattr(self, 'irrigated_once', False):
                # Reward heavily for the first irrigation
                first_irrigation_reward = 50.0  # Large reward for first irrigation
                self.irrigated_once = True  # Mark that the agent has irrigated at least once
                print("First irrigation: Large reward given!")
            else:
                # Penalize for any subsequent irrigation
                first_irrigation_reward = -5.0  # Fixed penalty for subsequent irrigations
                print("Subsequent irrigation: Small penalty applied.")
        else:
            first_irrigation_reward = 0  # No special reward if no irrigation

        # Calculate the step reward based on biomass difference and first irrigation reward
        reward = (1 / (1 + biomass_diff)) * 5 + first_irrigation_reward  # Biomass delta weight increased
        
        terminated = self.model._clock_struct.model_is_finished
        
        current_timestep = self.model._clock_struct.time_step_counter
        self.irrigation_schedule.append((current_timestep, depth))  # Log the applied irrigation depth
        
        info = {
            'dry_yield': 0.0,  # Placeholder until episode ends
            'total_irrigation': sum([item[1] for item in self.irrigation_schedule])  # Cumulative irrigation
        }

        # If the season is finished, apply final biomass difference reward
        if terminated:
            # Get dry yield and total irrigation from the model
            dry_yield = self.model._outputs.final_stats['Dry yield (tonne/ha)'].mean()
            total_irrigation = self.model._outputs.final_stats['Seasonal irrigation (mm)'].mean()
            
            # Calculate final cumulative biomass reward based on total biomass difference
            final_biomass_diff_reward = (1 / (1 + self.total_biomass_diff)) * 5  # Biomass difference reward for the episode
            
            # Print final statistics
            print(f"Dry Yield: {dry_yield}")
            print(f"Total Irrigation: {total_irrigation}")
            print(f"Final Biomass Difference Reward: {final_biomass_diff_reward}")
            
            # Add final stats to the info dictionary
            info['dry_yield'] = dry_yield
            info['total_irrigation'] = total_irrigation

            # Reset cumulative values to 0 for the next episode
            self.total_biomass_diff = 0
            self.irrigated_once = False  # Reset irrigation flag for the next episode

            # Apply the final biomass difference reward to the total reward
            reward += final_biomass_diff_reward

        return next_obs, reward, terminated, truncated, info











