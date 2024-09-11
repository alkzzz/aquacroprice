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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

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

        self.total_irrigation_applied = 0
        self.cumulative_reward = 0

        obs = self._get_obs()
        info = dict()

        self.log("Environment reset complete.")
        return obs, info

    def _get_obs(self):
        cond = self.model._init_cond

        total_precipitation_last_7_days = self._get_total_precipitation_last_7_days()
        cum_min_temp_last_7_days = self._get_cumulative_temp_last_7_days("MinTemp")
        cum_max_temp_last_7_days = self._get_cumulative_temp_last_7_days("MaxTemp")

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
        previous_total_irrigation = self.total_irrigation_applied
        self.total_irrigation_applied += depth
        self.log(f"Action taken: {action}, Depth: {depth}, Total Irrigation: {self.total_irrigation_applied}")

        # Apply penalty for irrigation once total irrigation exceeds 300 mm
        if previous_total_irrigation >= 400 and depth > 0:
            # Penalize for irrigation after exceeding 300 mm
            self.cumulative_reward -= depth
            self.log(f"Penalty applied for exceeding 300 mm: -{depth}. Cumulative reward: {self.cumulative_reward}")

        # If the season is finished, calculate the final reward
        if terminated:
            dry_yield = self.model._outputs.final_stats['Dry yield (tonne/ha)'].mean()
            total_irrigation = self.model._outputs.final_stats['Seasonal irrigation (mm)'].mean()

            # Add yield-based reward separately from penalties
            yield_reward = dry_yield ** 2
            
            if total_irrigation < 400:
                self.cumulative_reward += (yield_reward - (400 - total_irrigation))
            else:
                self.cumulative_reward += yield_reward
                
            self.log(f"Dry Yield: {dry_yield}, Total Irrigation: {total_irrigation}")
            self.log(f"Yield Reward: {yield_reward}. Final cumulative reward: {self.cumulative_reward}")

            info = {'dry_yield': dry_yield, 'total_irrigation': total_irrigation}
        else:
            info = {'dry_yield': 0.0, 'total_irrigation': 0.0}

        # Return the next observation, the cumulative reward, and whether the episode is done
        return next_obs, self.cumulative_reward, terminated, False, info

    def close(self):
        # Close the log file when the environment is closed
        self.log_file.close()
