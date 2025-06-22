from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import requests
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

WEATHER_API_KEY = "18a9e977d32e4a7a8e961308252106"
LOCATION = "Pune"
MODEL_PATH = "oven_time_predictor.pkl"


SENSOR_TARGETS = {
    'WU311': 160,
    'WU312': 190,
    'WU314': 190,
    'WU321': 190,
    'WU322': 190,
    'WU323': 190
}

def get_weather():
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={LOCATION}"
    response = requests.get(url)
    data = response.json()

    ambient_temp = data["current"]["temp_c"]
    oven_temp = ambient_temp + 5

    return {
        "ambient_temp": ambient_temp,
        "oven_temp": oven_temp,
        "humidity": data["current"]["humidity"]
    }

@app.route("/predict", methods=["POST"])
def predict():
    try:
        target_time = request.json.get("time")
        target_datetime = datetime.strptime(target_time, "%H:%M")

        model, feature_names = joblib.load(MODEL_PATH)
        weather_data = get_weather()
        current_temp = weather_data["oven_temp"]

        start_times = {}

        for sensor in SENSOR_TARGETS:
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

            heating_time = model.predict(input_data)[0] + 10
            start_time = target_datetime - timedelta(minutes=heating_time)

            start_times[sensor] = {
                'heating_time': round(heating_time, 1),
                'start_time': start_time.strftime("%H:%M"),
                'target_temp': SENSOR_TARGETS[sensor]
            }

        earliest = min(start_times.values(), key=lambda x: x['start_time'])

        return jsonify({
            'ambient_temp': weather_data['ambient_temp'],
            'oven_temp': current_temp,
            'humidity': weather_data['humidity'],
            'sensors': start_times,
            'earliest_start': earliest['start_time']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT env variable
    app.run(debug=True, host="0.0.0.0", port=port)

