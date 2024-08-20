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
from aquacrop.utils import prepare_weather

# Configuration dictionary for the environment
config = dict(
    name='generated_weather_data',
    year1=1678,
    year2=2261,
    crop='PaddyRice',
    soil='Paddy',
    init_wc=InitialWaterContent(depth_layer=[1, 2], value=['FC', 'FC']),
    days_to_irr=1,
    max_irr=25,
    action_set='smt4',  # Action set to 'smt4' for SMT optimization
)

# Define the Rice environment class
class Rice(gym.Env):
    def __init__(self, render_mode=None, mode='train'):
        super(Rice, self).__init__()
        print("Initializing Rice environment...")
        self.render_mode = render_mode
        self.days_to_irr = config["days_to_irr"]
        self.year1 = config["year1"]
        self.year2 = config["year2"]
        self.max_irr = config['max_irr']
        self.init_wc = config["init_wc"]
        self.mode = mode  # 'train' or 'eval'
        
        soil = config['soil']
        if isinstance(soil, str):
            self.soil = Soil(soil)
        else:
            assert isinstance(soil, Soil), "soil needs to be 'str' or 'Soil'"
            self.soil = soil

        # Define observation space: Includes weather-related observations
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        
        # Define action space for SMTs (4 continuous values between -1 and 1)
        self.action_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        print("Resetting environment...")
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        sim_year = np.random.randint(self.year1, self.year2 + 1)
        self.simcalyear = sim_year
        print(f"Chosen Year: {self.simcalyear}")

        crop = config['crop']
        self.planting_date = self._get_random_planting_date()

        if isinstance(crop, str):
            self.crop = Crop(crop, self.planting_date)
        else:
            assert isinstance(crop, Crop), "crop needs to be 'str' or 'Crop'"
            self.crop = crop

        print(f"Crop Planting Date: {self.crop.planting_date}")

        self.wdf = prepare_weather('generated_weather_data.txt')
        self.wdf['Year'] = self.simcalyear

        self.irr_sched = []

        # Initialize the AquaCrop model with irrigation method 1 (for SMT)
        self.model = AquaCropModel(
            f'{self.simcalyear}/{self.planting_date}', 
            f'{self.simcalyear}/12/31', 
            self.wdf, 
            self.soil, 
            self.crop,
            irrigation_management=IrrigationManagement(irrigation_method=1),  # SMT method
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
        smts = np.clip((action + 1) * 50, 0, 100)
        # print(f'SMT: ', smts)

        # Iterate over days until the next irrigation decision
        for _ in range(self.days_to_irr):
            # Calculate relative depletion
            if self.model._init_cond.taw > 0:
                dep = self.model._init_cond.depletion / self.model._init_cond.taw
            else:
                dep = 0

            # Determine growth stage
            gs = int(self.model._init_cond.growth_stage) - 1

            if 0 <= gs <= 3:
                if (1 - dep) < (smts[gs] / 100):
                    depth = np.clip(self.model._init_cond.depletion, 0, self.max_irr)
                else:
                    depth = 0
            else:
                depth = 0

            self.model.irrigation_management.depth = depth
            self.irr_sched.append(depth)

            # Simulate one day in the AquaCrop model
            self.model.run_model(initialize_model=False)

            # Termination conditions
            if self.model._clock_struct.model_is_finished:
                break

        terminated = self.model._clock_struct.model_is_finished
        truncated = False
        next_obs = self._get_obs()
        
        reward = 0
        
        # Check if the episode has terminated
        if terminated:
            reward = self.model._outputs.final_stats['Dry yield (tonne/ha)'].mean()
            if self.mode == 'train':
                reward *= 1000  # Scale reward by 1000 during training
            print(f'Final Reward: {reward}')
        
        info = dict()

        return next_obs, reward, terminated, truncated, info
