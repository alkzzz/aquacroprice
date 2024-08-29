from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



path = get_filepath('hyderabad_climate.txt')
wdf = prepare_weather(path)
wdf



sim_start = '2008/08/01'
sim_end = '2010/12/31'



soil= Soil('SandyLoam')

crop = Crop('Maize',planting_date='05/01')

initWC = InitialWaterContent(value=['FC'])



# define labels to help after
labels=[]

outputs=[]
for smt in range(0,110,20):
    crop.Name = str(smt) # add helpfull label
    labels.append(str(smt))
    irr_mngt = IrrigationManagement(irrigation_method=1,SMT=[smt]*4) # specify irrigation management
    model = AquaCropModel(sim_start,
                        sim_end,
                        wdf,
                        soil,
                        crop,
                        initial_water_content=initWC,
                        irrigation_management=irr_mngt) # create model
    model.run_model(till_termination=True) # run model till the end
    outputs.append(model._outputs.final_stats) # save results



import pandas as pd

dflist=outputs
labels[0]='Rainfed'
outlist=[]
for i in range(len(dflist)):
    temp = pd.DataFrame(dflist[i][['Dry yield (tonne/ha)',
                                   'Seasonal irrigation (mm)']])
    temp['label']=labels[i]
    outlist.append(temp)

all_outputs = pd.concat(outlist,axis=0)



# combine all results
results=pd.concat(outlist)



# import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# create figure consisting of 2 plots
fig,ax=plt.subplots(2,1,figsize=(10,14))

# create two box plots
sns.boxplot(data=results,x='label',y='Dry yield (tonne/ha)',ax=ax[0])
sns.boxplot(data=results,x='label',y='Seasonal irrigation (mm)',ax=ax[1])

# labels and font sizes
ax[0].tick_params(labelsize=15)
ax[0].set_xlabel('Soil-moisture threshold (%TAW)',fontsize=18)
ax[0].set_ylabel('Dry yield (t/ha)',fontsize=18)

ax[1].tick_params(labelsize=15)
ax[1].set_xlabel('Soil-moisture threshold (%TAW)',fontsize=18)
ax[1].set_ylabel('Total Irrigation (ha-mm)',fontsize=18)

plt.legend(fontsize=18)



# define irrigation management
rainfed = IrrigationManagement(irrigation_method=0)



# irrigate according to 4 different soil-moisture thresholds
threshold4_irrigate = IrrigationManagement(irrigation_method=1,SMT=[40,60,70,30]*4)



# irrigate every 7 days
interval_7 = IrrigationManagement(irrigation_method=2,IrrInterval=7)



import pandas as pd # import pandas library

all_days = pd.date_range(sim_start,sim_end) # list of all dates in simulation period

new_month=True
dates=[]
# iterate through all simulation days
for date in all_days:
    #check if new month
    if date.is_month_start:
        new_month=True

    if new_month:
        # check if tuesday (dayofweek=1)
        if date.dayofweek==1:
            #save date
            dates.append(date)
            new_month=False



depths = [25]*len(dates) # depth of irrigation applied
schedule=pd.DataFrame([dates,depths]).T # create pandas DataFrame
schedule.columns=['Date','Depth'] # name columns

schedule



irrigate_schedule = IrrigationManagement(irrigation_method=3,schedule=schedule)



net_irrigation = IrrigationManagement(irrigation_method=4,NetIrrSMT=70)



# define labels to help after
labels=['rainfed','four thresholds','interval','schedule','net']
strategies = [rainfed,threshold4_irrigate,interval_7,irrigate_schedule,net_irrigation]

outputs=[]
for i,irr_mngt in enumerate(strategies): # for both irrigation strategies...
    crop.Name = labels[i] # add helpfull label
    model = AquaCropModel(sim_start,
                        sim_end,
                        wdf,
                        soil,
                        crop,
                        initial_water_content=initWC,
                        irrigation_management=irr_mngt) # create model
    model.run_model(till_termination=True) # run model till the end
    outputs.append(model._outputs.final_stats) # save results



# function to return the irrigation depth to apply on next day
def get_depth(model):    
    t = model._clock_struct.time_step_counter # current timestep
    # get weather data for next 7 days
    weather10 = model._weather[t+1:min(t+10+1,len(model._weather))]
    # if it will rain in next 7 days
    if sum(weather10[:,2])>0:
        # check if soil is over 70% depleted
        if t>0 and model._init_cond.depletion/model._init_cond.taw > 0.7:
            depth=10
        else:
            depth=0
    else:
        # no rain for next 10 days
        depth=10


    return depth



model._clock_struct.time_step_counter



# create model with IrrMethod= Constant depth
crop.Name = 'weather' # add helpfull label

model = AquaCropModel(sim_start,sim_end,wdf,soil,crop,initial_water_content=initWC,
                      irrigation_management=IrrigationManagement(irrigation_method=5,)) 

model._initialize()

while model._clock_struct.model_is_finished is False:    
    # get depth to apply
    depth=get_depth(model)
    
    model._param_struct.IrrMngt.depth=depth

    model.run_model(initialize_model=False)



outputs.append(model._outputs.final_stats) # save results
labels.append('weather')



dflist=outputs
outlist=[]
for i in range(len(dflist)):
    temp = pd.DataFrame(dflist[i][['Dry yield (tonne/ha)','Seasonal irrigation (mm)']])
    temp['label']=labels[i]
    outlist.append(temp)

all_outputs = pd.concat(outlist,axis=0)



# combine all results
results=pd.concat(outlist)

results.to_csv("irrigation_comparison.csv")



# import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# create figure consisting of 2 plots
fig,ax=plt.subplots(2,1,figsize=(10,14))

# create two box plots
sns.boxplot(data=results,x='label',y='Dry yield (tonne/ha)',ax=ax[0])
sns.boxplot(data=results,x='label',y='Seasonal irrigation (mm)',ax=ax[1])

# labels and font sizes
ax[0].tick_params(labelsize=15)
ax[0].set_xlabel(' ')
ax[0].set_ylabel('Yield (t/ha)',fontsize=18)

ax[1].tick_params(labelsize=15)
ax[1].set_xlabel(' ')
ax[1].set_ylabel('Total Irrigation (ha-mm)',fontsize=18)

plt.legend(fontsize=18)

fig.savefig('irrigation_comparison.png')