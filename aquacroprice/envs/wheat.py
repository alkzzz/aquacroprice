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
    soil='ClayLoam',
    init_wc=InitialWaterContent(value=['FC']),
    days_to_irr=7,
)

class Wheat(gym.Env):
    def __init__(self, render_mode=None, mode='train', year1=None, year2=None):
        super(Wheat, self).__init__()
        print("Initializing Wheat environment...")
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

        # Soil initialization
        soil = config['soil']
        if isinstance(soil, str):
            self.soil = Soil(soil)
        else:
            assert isinstance(soil, Soil), "soil needs to be 'str' or 'Soil'"
            self.soil = soil

        # Define observation space and action space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        # self.action_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=np.float32)

        self.action_space = spaces.MultiDiscrete([101, 101, 101, 101])
        self.max_irr_season = 250  # Max irrigation for the entire season (300 mm)

        # Store the SMT value to be used during one episode
        self.current_smt = None  # This will hold the SMT during the episode

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

        # If no SMT is set, we assign default values (initial episode)
        if self.current_smt is None:
            self.current_smt = np.ones(4) * 50  # Default SMT of 50 for all thresholds

        # Initialize the AquaCrop model with SMT irrigation method
        self.model = AquaCropModel(
            f'{self.simcalyear}/{self.planting_date}', 
            f'{self.simcalyear}/12/31', 
            self.wdf, 
            self.soil, 
            self.crop,
            irrigation_management=IrrigationManagement(irrigation_method=2, SMT=self.current_smt, MaxIrrSeason=self.max_irr_season),  # SMT method
            initial_water_content=self.init_wc
        )

        # Run the model for the first step
        self.model.run_model()

        # Set SMT values for the entire episode
        # self.model._param_struct.IrrMngt.SMT = self.current_smt
        print(f"Set initial SMT for the episode: {self.current_smt}")

        self.cumulative_reward = 0.0
        
        obs = self._get_obs()
        info = dict()

        return obs, info

    def _get_obs(self):
        cond = self.model._init_cond

        week_min_temp = self._get_cumulative_temp_last_7_days("MinTemp")
        week_max_temp = self._get_cumulative_temp_last_7_days("MaxTemp")
        week_precipitation = self._get_total_precipitation_last_7_days()

        obs = np.array([
            cond.age_days,
            cond.canopy_cover,
            cond.biomass,
            cond.z_root,
            cond.depletion,
            cond.taw,
            week_min_temp,
            week_max_temp,
            week_precipitation
            # prev_day_min_temp,
            # prev_day_max_temp,
            # prev_day_precipitation,
        ], dtype=np.float32)

        return obs

    def _get_previous_day_value(self, col):
        current_day = self.model._clock_struct.time_step_counter
        if current_day > 0:
            prev_day_value = self.wdf.iloc[current_day - 1][col]
        else:
            prev_day_value = 0.0  # If it's the first day, there's no previous day data
        return prev_day_value
    
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

    def step(self, action):
        # Do not update SMT during the episode, we keep using the same SMT values
        # print(f"Using constant SMT for the episode: {self.current_smt}")

        # Run the AquaCrop model for the next step in the growing season
        self.model.run_model(initialize_model=False)

        next_obs = self._get_obs()
        truncated = False

        # Check if the model is finished (season ends)
        terminated = self.model._clock_struct.model_is_finished

        # Log the irrigation and calculate rewards at the end of the season
        info = {'dry_yield': 0.0, 'total_irrigation': 0.0}
        reward = 0

        if terminated:
            dry_yield = self.model._outputs.final_stats['Dry yield (tonne/ha)'].mean()
            total_irrigation = self.model._outputs.final_stats['Seasonal irrigation (mm)'].mean()

            reward = dry_yield ** 2

            print(f"Dry Yield: {dry_yield}")
            print(f"Total Irrigation: {total_irrigation}")
            print(f"Reward: {reward}")
            
            info['dry_yield'] = dry_yield
            info['total_irrigation'] = total_irrigation

            # Update the SMT values for the next episode based on the action
            self.current_smt = np.array(action)
            print(f"New SMT for the next episode: {self.current_smt}")

        return next_obs, reward, terminated, truncated, info
