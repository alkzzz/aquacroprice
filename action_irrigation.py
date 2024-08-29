from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load weather data
path = get_filepath('hyderabad_climate.txt')
wdf = prepare_weather(path)

sim_start = '2008/08/01'
sim_end = '2010/12/31'

# Define soil, crop, and initial water content
soil = Soil('Paddy')
crop = Crop('localpaddy', planting_date='08/01')
initWC = InitialWaterContent(value=['FC'])

# Define irrigation actions with a constant depth every 3 days
def get_depth_action1(model):
    if model._clock_struct.time_step_counter % 3 == 0:
        return 5  # Apply 5 mm every 3 days
    else:
        return 0

def get_depth_action2(model):
    if model._clock_struct.time_step_counter % 3 == 0:
        return 10  # Apply 10 mm every 3 days
    else:
        return 0

def get_depth_action3(model):
    if model._clock_struct.time_step_counter % 3 == 0:
        return 20  # Apply 20 mm every 3 days
    else:
        return 0

# List of actions for comparison
actions = [get_depth_action1, get_depth_action2, get_depth_action3]
labels = ['action1', 'action2', 'action3']

outputs = []
for i, action in enumerate(actions):
    crop.Name = labels[i]  # Add helpful label
    model = AquaCropModel(sim_start, sim_end, wdf, soil, crop, initial_water_content=initWC,
                          irrigation_management=IrrigationManagement(irrigation_method=5))
    model._initialize()

    while model._clock_struct.model_is_finished is False:
        depth = action(model)  # Get depth based on action
        model._param_struct.IrrMngt.depth = depth
        model.run_model(initialize_model=False)

    outputs.append(model._outputs.final_stats)  # Save results

# Combine results into a DataFrame for analysis
dflist = outputs
outlist = []
for i in range(len(dflist)):
    temp = pd.DataFrame(dflist[i][['Dry yield (tonne/ha)', 'Seasonal irrigation (mm)']])
    temp['label'] = labels[i]
    outlist.append(temp)

results = pd.concat(outlist)

# Save results to a CSV file
results.to_csv("irrigation_action_comparison_new.csv")

# Create figure with 2 plots for yield and irrigation comparison
fig, ax = plt.subplots(2, 1, figsize=(10, 14))

# Box plots for yield and irrigation
sns.boxplot(data=results, x='label', y='Dry yield (tonne/ha)', ax=ax[0])
sns.boxplot(data=results, x='label', y='Seasonal irrigation (mm)', ax=ax[1])

# Set labels and font sizes
ax[0].tick_params(labelsize=15)
ax[0].set_xlabel(' ')
ax[0].set_ylabel('Yield (t/ha)', fontsize=18)

ax[1].tick_params(labelsize=15)
ax[1].set_xlabel(' ')
ax[1].set_ylabel('Total Irrigation (ha-mm)', fontsize=18)

plt.legend(fontsize=18)

# Save the figure
fig.savefig('irrigation_action_comparison_new.png')

plt.show()
