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
    days_to_irr=3,
    max_irr=25,
    action_set='depth',
    normalize_obs=True,
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
        self.normalize_obs = config["normalize_obs"]
        
        soil = config['soil']
        if isinstance(soil, str):
            self.soil = Soil(soil)
        else:
            assert isinstance(soil, Soil), "soil needs to be 'str' or 'Soil'"
            self.soil = soil

        # Define observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)
        
        # Define discrete action space for irrigation depth
        self.action_space = spaces.Discrete(self.max_irr + 1)  # Discrete space from 0 to 25
        
        self.mean = 0
        self.std = 1

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
        obs = self._get_obs()
        info = dict()

        return obs, info

    def _get_obs(self):
        cond = self.model._init_cond
        total_precipitation_last_7_days = self._get_total_precipitation_last_7_days()
        obs = np.array([
            cond.canopy_cover,
            cond.canopy_cover_adj,
            cond.canopy_cover_ns,
            cond.canopy_cover_adj_ns,
            cond.biomass,
            cond.biomass_ns,
            cond.YieldPot,
            cond.harvest_index,
            cond.harvest_index_adj,
            cond.ccx_act,
            cond.ccx_act_ns,
            cond.ccx_w,
            cond.ccx_w_ns,
            cond.ccx_early_sen,
            cond.cc_prev,
            cond.DryYield,
            cond.FreshYield,
            cond.z_root,
            total_precipitation_last_7_days
        ], dtype=np.float32)

        if self.normalize_obs:
            return (obs - self.mean) / self.std
        else:
            return obs

    def _get_total_precipitation_last_7_days(self):
        current_day = self.model._clock_struct.time_step_counter
        last_7_days = self.wdf.iloc[max(0, current_day - 7):current_day]
        total_precipitation = last_7_days['Precipitation'].sum()
        return total_precipitation

    def step(self, action):
        depth = np.clip(action, 0, self.max_irr)  # Action is already an integer within this range
        self.model.irrigation_management.depth = depth
        print(f"Applied irrigation depth: {depth}")
        
        self.model.run_model(initialize_model=False)
        print(f'Timestep: ', self.model._clock_struct.time_step_counter)
        
        terminated = self.model._clock_struct.model_is_finished
        truncated = False
        next_obs = self._get_obs()

        if terminated:
            dry_yield = self.model._outputs.final_stats['Dry yield (tonne/ha)'].mean()
            reward = dry_yield
            print(f'Chosen Year: {self.simcalyear}')
            print(f'Final Reward: {reward} (Dry Yield: {dry_yield})')
        else:
            reward = 0.0

        info = dict()

        return next_obs, reward, terminated, truncated, info

    def get_mean_std(self, num_reps):
        self.mean = 0
        self.std = 1
        obs = []
        for _ in range(num_reps):
            self.reset(seed=0)

            done = False
            while not done:
                observation, reward, done, _, _ = self.step(np.random.randint(0, 2))  # Randomly choosing 0 or 1 action
                obs.append(observation)

        obs = np.vstack(obs)
        self.mean = obs.mean(axis=0)
        self.std = obs.std(axis=0)
        self.std[self.std == 0] = 1
