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

# Define the Rice environment class
class Rice(gym.Env):
    def __init__(self, render_mode=None, mode='train', year1=None, year2=None):
        super(Rice, self).__init__()
        print("Initializing Rice environment...")
        self.render_mode = render_mode
        self.days_to_irr = config["days_to_irr"]

        # If year1 and year2 are provided, override the default config
        self.year1 = year1 if year1 is not None else config["year1"]
        self.year2 = year2 if year2 is not None else config["year2"]

        self.init_wc = config["init_wc"]
        self.climate = config['climate']
        self.irrigation_schedule = [] # Store Irrigation Schedule
        self.mode = mode  # 'train' or 'eval'
        
        soil = config['soil']
        if isinstance(soil, str):
            self.soil = Soil(soil)
        else:
            assert isinstance(soil, Soil), "soil needs to be 'str' or 'Soil'"
            self.soil = soil

        # Define observation space: Includes weather-related observations
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        
        # self.action_depths = list(range(0, 26))  # This creates a list from 0 to 25
        self.action_depths = [0, 15, 30]
        self.action_space = spaces.Discrete(len(self.action_depths))  # Discrete action space with 25 actions


    def reset(self, seed=None, options=None):
        print("Resetting environment...")
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        sim_year = np.random.randint(self.year1, self.year2 + 1)
        self.simcalyear = sim_year
        print(f"Chosen Year: {self.simcalyear}")

        crop = config['crop']
        # self.planting_date = self._get_random_planting_date()
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

        # Initialize the AquaCrop model with irrigation method 1 (for SMT)
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

    def _get_random_planting_date(self):
        # Generate a random planting date between January 1 and August 1
        start_date = datetime.strptime("01/01", "%m/%d")
        end_date = datetime.strptime("08/01", "%m/%d")

        delta_days = (end_date - start_date).days
        random_days = np.random.randint(0, delta_days + 1)
        random_date = start_date + timedelta(days=random_days)

        return random_date.strftime("%m/%d")

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
            cond.harvest_index,
            cond.DryYield,
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
        # Scale the action values to the range [0, 100] for SMTs
        depth = self.action_depths[int(action)]
        self.model._param_struct.IrrMngt.depth = depth
        self.model.run_model(initialize_model=False)
        
        truncated = False
        next_obs = self._get_obs()
        
        reward = 0
        
        terminated = self.model._clock_struct.model_is_finished
        
        current_timestep = self.model._clock_struct.time_step_counter
        self.irrigation_schedule.append((current_timestep, action))
        
        if terminated:
            dry_yield = self.model._outputs.final_stats['Dry yield (tonne/ha)'].mean()
            total_irrigation = self.model._outputs.final_stats['Seasonal irrigation (mm)'].mean()
            
            if total_irrigation == 0:
                # Assign a fixed reward if total irrigation is 0
                reward = -10  # Fixed penalty reward
            else:
                # Calculate the normal reward
                reward = (dry_yield ** 3) - ((total_irrigation + 1) * 15)

            # Logging to help debug and understand the reward structure
            print(f"Dry Yield: {dry_yield}")
            print(f"Total Irrigation: {total_irrigation}")
            print(f"Final Reward: {reward}")
        
        info = dict()

        return next_obs, reward, terminated, truncated, info






