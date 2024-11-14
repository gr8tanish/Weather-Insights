from flask import Flask, render_template
from dotenv import load_dotenv
import pickle
import pandas as pd
import requests
import os

app = Flask(__name__)

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv("API_KEY")
imputer_link = os.getenv("IMPUTER_LINK")
scaler_link = os.getenv("SCALER_LINK")
encoder_link = os.getenv("ENCODER_LINK")
model_link = os.getenv("MODEL_LINK")

# Dropbox links (with dl=1 for direct download)
DROPBOX_URLS = {
    "imputer": imputer_link,
    "scaler": scaler_link,
    "encoder": encoder_link,
    "model": model_link
}

# Path to save the downloaded pickle files
FILE_PATHS = {
    "imputer": "imputer.pkl",
    "scaler": "scaler.pkl",
    "encoder": "encoder.pkl",
    "model": "model.pkl"
}

# Function to download the pickle files from Dropbox
def download_pickle_from_dropbox(url, file_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"{file_path} has been downloaded successfully!")
        else:
            print(f"Failed to download {file_path}")
    except Exception as e:
        print(f"Error downloading {file_path}: {e}")

# Download files if they are not already downloaded
for key, url in DROPBOX_URLS.items():
    if not os.path.exists(FILE_PATHS[key]):
        download_pickle_from_dropbox(url, FILE_PATHS[key])

# Load pre-trained components (imputer, scaler, encoder, model)
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Load models and preprocessing files
imputer = load_pickle(FILE_PATHS["imputer"])
scaler = load_pickle(FILE_PATHS["scaler"])
encoder = load_pickle(FILE_PATHS["encoder"])
model = load_pickle(FILE_PATHS["model"])

# Define column configurations
NUMERIC_COLUMNS = [
    'MinTemp', 'MaxTemp', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
    'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
    'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
]
CATEGORICAL_COLUMNS = ['Location']
ENCODED_COLUMNS = encoder.get_feature_names_out(CATEGORICAL_COLUMNS)

# Sending Get Request to the API
# Define the API URL
url = f"https://api.openweathermap.org/data/2.5/forecast?lat=25&lon=70&appid={api_key}&units=metric"

# Send GET request
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the response (assuming it's JSON)
    data = response.json()
else:
    print(f"Request failed with status code: {response.status_code}")

# Manually provided input data
input_data_rain_prediction = {
    'MinTemp': data['list'][0]['main']['temp_min'],
    'MaxTemp': data['list'][0]['main']['temp_max'],
    'WindGustSpeed': data['list'][0]['wind']['speed'],
    'WindSpeed9am': data['list'][0]['wind']['speed'],
    'WindSpeed3pm': data['list'][2]['wind']['speed'],
    'Humidity9am': data['list'][0]['main']['humidity'],
    'Humidity3pm': data['list'][2]['main']['humidity'],
    'Pressure9am': data['list'][0]['main']['pressure'],
    'Pressure3pm': data['list'][2]['main']['pressure'],
    'Cloud9am': data['list'][0]['clouds']['all'],
    'Cloud3pm': data['list'][0]['clouds']['all'],
    'Temp9am': data['list'][0]['main']['temp'],
    'Temp3pm': data['list'][0]['main']['temp'],
    'Location': 'Greater Noida'
}

@app.route('/')
def home():
    """Perform prediction on page load and render the result on the home page."""
    try:
        # Convert manual input data to DataFrame
        input_data = pd.DataFrame([input_data_rain_prediction])

        # Preprocess input data
        input_data[NUMERIC_COLUMNS] = imputer.transform(input_data[NUMERIC_COLUMNS])
        input_data[NUMERIC_COLUMNS] = scaler.transform(input_data[NUMERIC_COLUMNS])
        
        # Encode categorical columns and create final input DataFrame
        encoded_input_data = encoder.transform(input_data[CATEGORICAL_COLUMNS])
        encoded_input_df = pd.DataFrame(encoded_input_data, columns=ENCODED_COLUMNS)
        
        # Combine numeric and encoded categorical data
        final_input_data = pd.concat(
            [input_data[NUMERIC_COLUMNS].reset_index(drop=True), 
             encoded_input_df.reset_index(drop=True)], axis=1
        ).reindex(columns=model.feature_names_in_, fill_value=0)

        # Predict
        prediction = model.predict(final_input_data)
        rain_prediction = True if prediction[0] == 'Yes' else False
        
        # Render template with prediction result
        return render_template('index.html', rain_prediction=rain_prediction)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
