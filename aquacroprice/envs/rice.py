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
    year1=1678,
    year2=2261,
    crop='PaddyRice',
    soil='Paddy',
    init_wc=InitialWaterContent(depth_layer=[1, 2], value=['FC', 'FC']),
    planting_date='08/01',
    days_to_irr=1,
    max_irr=25,
    action_set='binary',
)

# Define the Rice environment class
class Rice(gym.Env):
    def __init__(self, render_mode=None, mode='train'):
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
        self.mode = mode  # 'train' or 'eval'
        
        soil = config['soil']
        if isinstance(soil, str):
            self.soil = Soil(soil)
        else:
            assert isinstance(soil, Soil), "soil needs to be 'str' or 'Soil'"
            self.soil = soil

        # Define observation space: Updated to include additional weather-related observations
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        # Define binary action space: 0 for no irrigation, 1 for maximum irrigation depth
        self.action_space = spaces.Discrete(2)  # 0 or 1

    def reset(self, seed=None, options=None):
        print("Resetting environment...")
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        sim_year = np.random.randint(self.year1, self.year2 + 1)
        self.simcalyear = sim_year
        print(f"Chosen Year: {self.simcalyear}")

        crop = config['crop']
        if isinstance(crop, str):
            self.crop = Crop(crop, self.planting_date)
        else:
            assert isinstance(crop, Crop), "crop needs to be 'str' or 'Crop'"
            self.crop = crop

        print(f"Crop Planting Date: {self.crop.planting_date}")

        self.wdf = prepare_weather('generated_weather_data.txt')
        self.wdf['Year'] = self.simcalyear

        self.irr_sched = []

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
        cum_min_temp_last_7_days = self._get_cumulative_temp_last_7_days("MinTemp")
        cum_max_temp_last_7_days = self._get_cumulative_temp_last_7_days("MaxTemp")
        prev_day_min_temp = self._get_previous_day_value("MinTemp")
        prev_day_max_temp = self._get_previous_day_value("MaxTemp")
        prev_day_precipitation = self._get_previous_day_value("Precipitation")

        obs = np.array([
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

        # print(f'Obs: {obs}')

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
        # Map the binary action to irrigation depth
        depth = 0 if action == 0 else self.max_irr
        self.model.irrigation_management.depth = depth
        # print(f"Applied irrigation depth: {depth}")
        
        self.model.run_model(initialize_model=False)
        # print(f'Timestep: {self.model._clock_struct.time_step_counter}')
        
        terminated = self.model._clock_struct.model_is_finished
        truncated = False
        next_obs = self._get_obs()
        
        reward = 0
        
        if terminated:
            fresh_yield = self.model._outputs.final_stats['Fresh yield (tonne/ha)'].mean()
            avg_yield = 7.0
            if self.mode == 'train':
                if fresh_yield < avg_yield:
                    reward = fresh_yield - 3.0
                    print(f'Final Reward with Penalty: {reward} (Fresh Yield: {fresh_yield})')
                else:
                    reward = fresh_yield
                    print(f'Final Reward with No Penalty: {reward} (Fresh Yield: {fresh_yield})')
            else:
                reward = fresh_yield
                print(f'Final Reward: {reward} (Fresh Yield: {fresh_yield})')
        
        info = dict()

        return next_obs, reward, terminated, truncated, info
