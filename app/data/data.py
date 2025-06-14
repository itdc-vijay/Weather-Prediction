from meteostat import Point, Hourly
from datetime import datetime
import pandas as pd
import os

cities = {
    "ahmedabad": {"coords": Point(23.0225, 72.5714, 53), "timezone": "Asia/Kolkata"},
    "mumbai": {"coords": Point(19.0760, 72.8777, 14), "timezone": "Asia/Kolkata"},
    "delhi": {"coords": Point(28.7041, 77.1025, 216), "timezone": "Asia/Kolkata"},
    "bengaluru": {"coords": Point(12.9716, 77.5946, 920), "timezone": "Asia/Kolkata"}
}

start = datetime(2020, 1, 1)
end = datetime.now()

rename_map = {
    "time": "Timestamp",
    "temp": "Temperature (°C)",
    "rhum": "Humidity (%)",
    "wspd": "Wind Speed (km/h)",
    "wdir": "Wind Direction (°)"
}

script_dir = os.path.dirname(os.path.abspath(__file__))

def process_weather_data(city_name, city_info):
    # Fetch hourly weather data for the city
    data = Hourly(city_info['coords'], start, end, timezone=city_info['timezone']).fetch()

    # If no data is returned, exit the function
    if data.empty:
        return

    # Reset index and fill missing values
    data.reset_index(inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    # Rename columns and format the Timestamp column
    data.rename(columns={col: rename_map.get(col, col) for col in data.columns}, inplace=True)
    data = data[list(rename_map.values())]
    data['Timestamp'] = pd.to_datetime(data['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Save the processed data to a CSV file
    filename = os.path.join(script_dir, f"{city_name}.csv")
    data.to_csv(filename, index=False)

# Process weather data for each city in the dictionary
for city_name, city_info in cities.items():
    process_weather_data(city_name, city_info)

print("Data processing completed successfully.")
