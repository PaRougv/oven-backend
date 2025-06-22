import joblib
import requests
import pandas as pd
from datetime import datetime, timedelta

# Configuration
WEATHER_API_KEY = "18a9e977d32e4a7a8e961308252106"
LOCATION = "Pune"
MODEL_PATH = "D:/IndustrialOvenHeatUpPrediction/oven_time_predictor.pkl"
TARGET_TIME = input("Enter the Time of Start in HH:MM , 3 am in the morning is 03:00 : ")  # Target completion time

# Sensor targets
SENSOR_TARGETS = {
    'WU311': 160,
    'WU312': 190,
    'WU314': 190,
    'WU321': 190,
    'WU322': 190,
    'WU323': 190
}

def get_weather():
    """Fetch current weather data and calculate oven temperature"""
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={LOCATION}"
    response = requests.get(url)
    data = response.json()
    
    ambient_temp = data["current"]["temp_c"]
    oven_temp = ambient_temp + 5  # Add 5°C to atmospheric temperature
    
    return {
        "ambient_temp": ambient_temp,
        "oven_temp": oven_temp,
        "humidity": data["current"]["humidity"]
    }

def calculate_start_times():
    """Calculate start times for all sensors to reach target by 6:30 AM"""
    try:
        # Load model and feature names
        model, feature_names = joblib.load(MODEL_PATH)
        
        # Get weather and oven temperature
        weather_data = get_weather()
        current_temp = weather_data["oven_temp"]
        
        # Calculate for all sensors
        start_times = {}
        target_datetime = datetime.strptime(TARGET_TIME, "%H:%M")
        
        for sensor in SENSOR_TARGETS:
            # Prepare input data
            input_data = pd.DataFrame({
                'start_temp': [current_temp],
                'ambient_temp': [weather_data["ambient_temp"]],
                'humidity': [weather_data["humidity"]],
                'target_temp': [SENSOR_TARGETS[sensor]],
                'sensor_WU311': [1 if sensor == 'WU311' else 0],
                'sensor_WU312': [1 if sensor == 'WU312' else 0],
                'sensor_WU314': [1 if sensor == 'WU314' else 0],
                'sensor_WU321': [1 if sensor == 'WU321' else 0],
                'sensor_WU322': [1 if sensor == 'WU322' else 0],
                'sensor_WU323': [1 if sensor == 'WU323' else 0]
            })[feature_names]

            # Predict heating time (with 10 minute buffer)
            heating_time = model.predict(input_data)[0] + 10
            
            # Calculate start time
            start_time = target_datetime - timedelta(minutes=heating_time)
            start_times[sensor] = {
                'heating_time': heating_time,
                'start_time': start_time.strftime("%H:%M"),
                'target_temp': SENSOR_TARGETS[sensor]
            }
        
        # Find the earliest required start time
        latest_start = min(start_times.values(), key=lambda x: x['start_time'])
        
        # Print results
        print("\nCurrent Conditions:")
        print(f"Atmospheric Temperature: {weather_data['ambient_temp']}°C")
        print(f"Calculated Oven Temperature: {current_temp}°C")
        print(f"Humidity: {weather_data['humidity']}%\n")
        
        print("Individual Sensor Requirements:")
        for sensor, data in start_times.items():
            print(f"{sensor} (Target: {data['target_temp']}°C):")
            print(f"→ Heating Time: {data['heating_time']:.1f} minutes")
            print(f"→ Start By: {data['start_time']}\n")
        
        print(f"\nOPERATIONAL DECISION:")
        print(f"To reach all targets by {TARGET_TIME}, start ALL burners by:")
        print(f"→ {latest_start['start_time']} (based on {latest_start['heating_time']:.1f} min heating time)")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    calculate_start_times()