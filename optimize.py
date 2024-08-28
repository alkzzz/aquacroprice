import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # For exporting to CSV
from aquacrop import (AquaCropModel, Crop, InitialWaterContent,
                      IrrigationManagement, Soil)
from aquacrop.utils import get_filepath, prepare_weather
from scipy.optimize import fmin

path = get_filepath('champion_climate.txt')
wdf = prepare_weather(path)

def run_model(smts, max_irr_season, year1, year2):
    """
    Function to run the model and return results for a given set of soil moisture targets
    """
    maize = Crop('Maize', planting_date='05/01')  # Define crop
    loam = Soil('ClayLoam')  # Define soil
    init_wc = InitialWaterContent(wc_type='Pct', value=[70])  # Define initial soil water conditions

    irrmngt = IrrigationManagement(irrigation_method=1, SMT=smts, MaxIrrSeason=max_irr_season)  # Define irrigation management

    # Create and run the model
    model = AquaCropModel(f'{year1}/05/01', f'{year2}/10/31', wdf, loam, maize,
                          irrigation_management=irrmngt, initial_water_content=init_wc)
    
    model.run_model(till_termination=True)
    return model.get_simulation_results()

def evaluate(smts, max_irr_season, test=False):
    """
    Function to run the model and calculate reward (yield) for a given set of soil moisture targets
    """
    # Run the model
    out = run_model(smts, max_irr_season, year1=2016, year2=2018)
    print(out)

    # Get yields and total irrigation
    yld = out['Dry yield (tonne/ha)'].mean()
    tirr = out['Seasonal irrigation (mm)'].mean()

    reward = yld

    # Export results to CSV
    out.to_csv('simulation_results.csv', index=False)
    print("Simulation results saved to 'simulation_results.csv'")

    # Return either the negative reward (for the optimization)
    # or the yield and total irrigation (for analysis)
    if test:
        return yld, tirr, reward
    else:
        return -reward

def get_starting_point(num_smts, max_irr_season, num_searches):
    """
    Find good starting threshold(s) for optimization
    """
    # Get random SMT's
    x0list = np.random.rand(num_searches, num_smts) * 100
    rlist = []
    # Evaluate random SMT's
    for xtest in x0list:
        r = evaluate(xtest, max_irr_season)
        rlist.append(r)

    # Save best SMT
    x0 = x0list[np.argmin(rlist)]
    
    return x0

def optimize(num_smts, max_irr_season, num_searches=100):
    """ 
    Optimize thresholds to be profit-maximizing
    """
    # Get starting optimization strategy
    x0 = get_starting_point(num_smts, max_irr_season, num_searches)
    # Run optimization
    res = fmin(evaluate, x0, disp=0, args=(max_irr_season,))
    # Reshape array
    smts = res.squeeze()
    # Evaluate optimal strategy
    return smts

smts = optimize(4, 300)

evaluate(smts, 300, True)

from tqdm.autonotebook import tqdm  # Progress bar

opt_smts = []
yld_list = []
tirr_list = []
for max_irr in tqdm(range(0, 500, 50)):
    # Find optimal thresholds and save to list
    smts = optimize(4, max_irr)
    opt_smts.append(smts)

    # Save the optimal yield and total irrigation
    yld, tirr, _ = evaluate(smts, max_irr, True)
    yld_list.append(yld)
    tirr_list.append(tirr)

# Create plot
fig, ax = plt.subplots(1, 1, figsize=(13, 8))

# Plot results
ax.scatter(tirr_list, yld_list)
ax.plot(tirr_list, yld_list)

# Labels
ax.set_xlabel('Total Irrigation (ha-mm)', fontsize=18)
ax.set_ylabel('Dry Yield (tonne/ha)', fontsize=18)
ax.set_xlim([-20, 600])
ax.set_ylim([2, 15.5])

# Annotate with optimal thresholds
bbox = dict(boxstyle="round", fc="1")
offset = [15, 15, 15, 15, 15, -125, -100, -5, 10, 10]
yoffset = [0, -5, -10, -15, -15, 0, 10, 15, -20, 10]
for i, smt in enumerate(opt_smts):
    smt = smt.clip(0, 100)
    ax.annotate('(%.0f, %.0f, %.0f, %.0f)' % (smt[0], smt[1], smt[2], smt[3]),
                (tirr_list[i], yld_list[i]), xytext=(offset[i], yoffset[i]), textcoords='offset points',
                bbox=bbox, fontsize=12)
    
fig.savefig('optimized_irrigation_yield.png')
