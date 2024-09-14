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
    days_to_irr=3,
)

class Maize(gym.Env):
    def __init__(self, render_mode=None, mode='train', year1=None, year2=None):
        super(Maize, self).__init__()
        print("Initializing Maize environment...")
        self.render_mode = render_mode
        self.days_to_irr = config["days_to_irr"]
        self.day_counter = 0

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)

        self.action_depths = [0, 25]
        self.action_space = spaces.Discrete(len(self.action_depths))  # Discrete action space

        # Open a log file for debugging output
        self.log_file = open("debug.txt", "a")

    def log(self, message):
        """Helper method to log a message to the text file."""
        self.log_file.write(message + "\n")
        self.log_file.flush()  # Ensure the message is written immediately

    def reset(self, seed=None, options=None):
        self.log("Resetting environment...")
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        sim_year = np.random.randint(self.year1, self.year2 + 1)
        self.simcalyear = sim_year
        self.log(f"Chosen Year: {self.simcalyear}")

        crop = config['crop']
        self.planting_date = '05/01'

        if isinstance(crop, str):
            self.crop = Crop(crop, self.planting_date)
        else:
            assert isinstance(crop, Crop), "crop needs to be 'str' or 'Crop'"
            self.crop = crop

        self.log(f"Crop Planting Date: {self.crop.planting_date}")

        self.wdf = prepare_weather(get_filepath(self.climate))
        self.wdf['Year'] = self.simcalyear

        self.irr_sched = []
        self.total_irrigation_applied = 0
        self.day_counter = 0  # Reset day counter
        self.total_irrigation_applied = 0
        self.cumulative_reward = 0

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

        obs = self._get_obs()
        info = dict()

        self.log("Environment reset complete.")
        return obs, info

    def _get_obs(self):
        cond = self.model._init_cond

        # Get daily values for the last 7 days for each variable
        precip_last_7_days = self._get_last_7_days_values('Precipitation')
        min_temp_last_7_days = self._get_last_7_days_values('MinTemp')
        max_temp_last_7_days = self._get_last_7_days_values('MaxTemp')

        # Combine the weather data into one array
        weather_obs = np.concatenate([precip_last_7_days, min_temp_last_7_days, max_temp_last_7_days])

        # Construct the observation by combining crop conditions and weather data
        obs = np.array([
            cond.age_days,
            cond.canopy_cover,
            cond.biomass,
            cond.z_root,
            cond.depletion,
            cond.taw
        ], dtype=np.float32)

        # Add the weather data to the observation
        obs = np.concatenate([obs, weather_obs])

        return obs

    def _get_last_7_days_values(self, column):
        """Helper method to get the last 7 days of data for a specific column."""
        current_day = self.model._clock_struct.time_step_counter
        last_7_days = self.wdf.iloc[max(0, current_day - 7):current_day][column]
        
        # If fewer than 7 days of data exist, pad the start with zeros
        if len(last_7_days) < 7:
            padding = np.zeros(7 - len(last_7_days))
            last_7_days = np.concatenate([padding, last_7_days])
        
        return last_7_days


    # Example code for step function with penalty system and random agent info
    def step(self, action):
        # Apply irrigation based on action (either 25 mm or 0 mm)
        depth = self.action_depths[int(action)]
        self.model._param_struct.IrrMngt.depth = depth
        self.model.run_model(initialize_model=False)

        next_obs = self._get_obs()
        terminated = self.model._clock_struct.model_is_finished

        # Add the current irrigation depth to the irrigation schedule
        current_timestep = self.model._clock_struct.time_step_counter
        self.irrigation_schedule.append((current_timestep, depth))

        # Update the total irrigation applied so far
        self.total_irrigation_applied += depth
        current_total_irrigation = self.total_irrigation_applied
        # print(f"Total Irrigation Applied: {current_total_irrigation}")

        # Initialize reward for this step
        step_reward = 0  

        # Biomass reward
        biomass_ns = self.model._init_cond.biomass_ns
        biomass = self.model._init_cond.biomass
        if biomass_ns > 0:
            biomass_reward = 1 / (1 + (biomass_ns - biomass))
            step_reward += biomass_reward
            # print(f"Biomass NS: {biomass_ns}, Biomass: {biomass}, Biomass Reward: {biomass_reward}")

        # Penalty for excessive irrigation
        # print(f"Current Irrigation: {current_total_irrigation}")
        if current_total_irrigation >= 200 and depth > 0:
            penalty = current_total_irrigation / 10
            step_reward -= penalty
            # print(f"Penalty for Irrigation (Total Irrigation {current_total_irrigation} mm): -{penalty}, Current Step Reward: {step_reward}")

        # Accumulate the reward for the final step
        if not hasattr(self, 'cumulative_reward'):
            self.cumulative_reward = 0  # Initialize cumulative reward

        self.cumulative_reward += step_reward  # Add current step reward to cumulative reward

        # If the season is finished, calculate the final reward
        if terminated:
            dry_yield = self.model._outputs.final_stats['Dry yield (tonne/ha)'].mean()
            total_irrigation = self.model._outputs.final_stats['Seasonal irrigation (mm)'].mean()
            
            print(f"Current Cumulative Reward: {self.cumulative_reward}")
            # Add yield-based reward separately from penalties
            yield_reward = (2 * (dry_yield ** 3)) - (total_irrigation * 10)
            self.cumulative_reward += yield_reward  # Add final yield reward to cumulative reward

            print(f"Dry Yield: {dry_yield}, Total Irrigation: {total_irrigation}")
            print(f"Final Cumulative Reward: {self.cumulative_reward}")

            info = {'dry_yield': dry_yield, 'total_irrigation': total_irrigation}

            # Reset cumulative reward for the next episode
            total_reward = self.cumulative_reward
            self.cumulative_reward = 0  # Reset for the next episode
        else:
            info = {'dry_yield': 0.0, 'total_irrigation': 0}
            total_reward = step_reward

        # Return the next observation, the step reward (not cumulative), and whether the episode is done
        return next_obs, total_reward, terminated, False, info

    def close(self):
        # Close the log file when the environment is closed
        self.log_file.close()
