# c:\Users\Tanzil Sayed\Documents\Projs\InjuryPred Merged\injury_prediction_portal\backend\app.py
import os
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import traceback

# MODIFICATION 2: template_folder and static_folder are now defined
app = Flask(__name__,
            template_folder='../frontend',
            static_folder='../frontend')
CORS(app)


# MODIFICATION 3: THIS IS THE MISSING HOMEPAGE ROUTE TO SHOW YOUR index.html
@app.route('/')
def index():
    return render_template('index.html')


# ==== Load Models (This section is for a different, unused part of the code) ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEVERity_MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'severity_model.pkl')
LOCATION_MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'location_model.pkl')
LOCATION_ENCODER_PATH = os.path.join(BASE_DIR, '..', 'model', 'location_encoder.pkl')

severity_model = joblib.load(SEVERity_MODEL_PATH)
location_model = joblib.load(LOCATION_MODEL_PATH)
location_encoder = joblib.load(LOCATION_ENCODER_PATH)


# Get the encoding mapping from the location encoder
encoding_mapping = location_encoder

# Function to decode the predicted location category
def decode_location(predicted_location_encoded):
    # Get the category name for the predicted value
    category_name = [category for category, index in encoding_mapping.items() if index == predicted_location_encoded]
    
    # Return the category name
    return category_name[0]

# Specify the feature names
feature_names = [
  'Injury Duration (weeks)',
  'Injury Occurred (weeks ago)',
  'Trunk Flexion (cm)',
  'BMI',
  'Weekly Training Hours',
  'Stick Test (cm)',
  'Current discomfort / Injury',
  'Shoulder Flexion (deg)',
  'Weight (kg)'
]

# Create a pandas DataFrame with the feature names
def create_dataframe(data):
  df = pd.DataFrame([data], columns=feature_names)
  return df

# Create a prediction function
def predict(data):
  df = create_dataframe(data)
  predicted_severity = severity_model.predict(df)[0]
  predicted_location_encoded = location_model.predict(df)[0]
  predicted_body_part = decode_location(predicted_location_encoded)
  return predicted_severity, predicted_body_part

# Train the model with the feature names
model = RandomForestClassifier()
# model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # --- This is the data sent from your webpage form ---
        form_data = request.get_json()

        # Load trained models
        severity_model_pred = joblib.load("../results/models/xgboost_severity_model.pkl")
        location_model_pred = joblib.load("../results/models/random_forest_location_model.pkl")

        # Load encoders
        severity_encoder = joblib.load("../results/models/severity_encoder.pkl")
        location_encoder_pred = joblib.load("../results/models/location_encoder.pkl")

        # --- This is the "template" of all features the model expects ---
        all_features_df = pd.read_excel("../injurypred/data/balanced_sheet1.xlsx")
        full_feature_list = list(all_features_df.drop(columns=["Injury Severity", "Injury Location"]).columns)

        # --- NEW ROBUST LOGIC STARTS HERE ---
        # 1. Create a dictionary to hold the data for the model, starting with all expected features set to a default of 0.
        model_input = {feature: 0 for feature in full_feature_list}

        # 2. Update the dictionary with the actual data received from the form.
        # This ensures we use all the user's data, and only missing features remain 0.
        for feature in full_feature_list:
            if feature in form_data:
                model_input[feature] = form_data[feature]
        
        # 3. Convert the clean, ordered dictionary into a DataFrame.
        input_df = pd.DataFrame([model_input])
        # --- NEW ROBUST LOGIC ENDS HERE ---


        # Encode known categorical fields (e.g. Coach)
        if 'Coach' in input_df.columns:
            coach_encoder_path = os.path.join("..", "results", "models", "coach_encoder.pkl")
            coach_encoder = joblib.load(coach_encoder_path)

            try:
                # The input might be a number from the form, so convert to string for the encoder
                input_df['Coach'] = coach_encoder.transform(input_df[['Coach']].astype(str))
            except ValueError:
                # If the coach is unknown, use the most common coach from training data
                most_common_label_encoded = np.argmax(np.bincount(coach_encoder.transform(coach_encoder.classes_)))
                input_df['Coach'] = most_common_label_encoded
        
        # Ensure correct numeric types for all other columns
        for col in input_df.columns:
            if col != 'Coach': # The 'Coach' column is already handled
                # Convert all other inputs to numbers (float). If it fails, it will become NaN (Not a Number)
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        
        # Fill any NaNs that might have resulted from bad conversion with 0
        input_df.fillna(0, inplace=True)


        # Make predictions
        severity_pred_encoded = severity_model_pred.predict(input_df)[0]
        location_pred_encoded = location_model_pred.predict(input_df)[0]

        # Decode predictions
        severity_pred = severity_encoder.inverse_transform([severity_pred_encoded])[0]
        location_pred = location_encoder_pred.inverse_transform([location_pred_encoded])[0]

        return jsonify({
            "Injury Severity": str(severity_pred),
            "Injury Location": str(location_pred)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/submit', methods=['POST'])
def submit_data():
    try:
        data = request.get_json()

        # Where to save the data
        save_path = os.path.join(BASE_DIR, '..', 'data', 'user_submitted_data.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # If file doesn't exist yet → write header
        file_exists = os.path.isfile(save_path)

        with open(save_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(data.keys()) + ['timestamp'])

            if not file_exists:
                writer.writeheader()

            # Add timestamp to the data
            data['timestamp'] = datetime.now().isoformat()

            writer.writerow(data)

        return jsonify({"message": "Data submitted successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)