import numpy as np
import pandas as pd
from tqdm import tqdm

# Define parameters for the weather data generation
start_year = 1678
end_year = 2261  # Maximum allowed end year for Pandas datetime64

min_temp_range = (5, 20)  # Minimum temperature range in degrees Celsius
max_temp_range = (25, 40)  # Maximum temperature range in degrees Celsius
precipitation_range = (0, 10)  # Precipitation range in mm
et_range = (3, 6)  # Reference Evapotranspiration range in mm/day

# Initialize an empty list to store the weather data
weather_data = []

# Generate weather data for each day from start_year to end_year
for year in tqdm(range(start_year, end_year + 1)):
    for month in range(1, 13):
        # Determine the number of days in the current month
        if month in [1, 3, 5, 7, 8, 10, 12]:
            days_in_month = 31
        elif month in [4, 6, 9, 11]:
            days_in_month = 30
        elif month == 2:
            # Check for leap year
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                days_in_month = 29
            else:
                days_in_month = 28

        for day in range(1, days_in_month + 1):
            # Ensure we stop before exceeding the maximum valid date
            if year == 2262 and month == 4 and day > 11:
                break

            # Ensure we start after the minimum valid date
            if year == 1677 and month == 9 and day < 21:
                continue

            # Generate random weather data for the day
            min_temp = round(np.random.uniform(*min_temp_range), 1)
            max_temp = round(np.random.uniform(min_temp + 5, max_temp_range[1]), 1)  # Ensure max_temp > min_temp
            precipitation = round(np.random.uniform(*precipitation_range), 1)
            reference_et = round(np.random.uniform(*et_range), 1)

            # Append the generated data to the list
            weather_data.append([day, month, year, min_temp, max_temp, precipitation, reference_et])

# Convert the list to a DataFrame
weather_df = pd.DataFrame(weather_data, columns=["Day", "Month", "Year", "MinTemp", "MaxTemp", "Precipitation", "ReferenceET"])

# Save the DataFrame to a text file
weather_df.to_csv("generated_weather_data.txt", sep="\t", index=False)

print("Weather data generation completed. The file 'generated_weather_data.txt' has been created.")
