from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import subprocess
import json

import sys
print(sys.executable)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
with open('./model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the prediction endpoint
@app.route('/predict', methods=['GET'])
def predict():
    try:

        with open('./data/schedule.json', 'r') as f:
            games = json.load(f)

        if not games:
            return jsonify({"message": "There are no games today"}), 204
        
        required_features = [
            'team_code', 'opp_code',
            'rest_days_home', 'rest_days_away', 'home_AVG_gf/gp', 'away_AVG_gf/gp',
            'home_AVG_ga/gp', 'away_AVG_ga/gp', 'home_AVG_pp%', 'away_AVG_pp%',
            'home_AVG_fow%', 'away_AVG_fow%'
        ]
        
        predictions = []
        for game in games:
            prediction_data = {feature: game.get(feature, None) for feature in required_features}
            input_df = pd.DataFrame([prediction_data])
            
            if input_df.isnull().values.any():
                continue  # Skip games with missing features

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0].tolist()
            winner = game['home'] if prediction == 1 else game['away']
            print(winner)

            predictions.append({
                "home": game['home'],
                "away": game['away'],
                "prediction": winner,
                "probabilities": probability
            })

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
# Define the prediction endpoint
@app.route('/games', methods=['GET'])
def get_games():
    try:

        venv_python = r"C:\Users\riger\Desktop\NHL_predictor\main\nhl_venv\Scripts\python.exe"

        result = subprocess.run(['python', './scripts/get_games.py'], check=True, capture_output=True, text=True)
        print(result.stdout) 
        
        with open('./data/schedule.json', 'r') as f:
            games_data = json.load(f)
            
        return jsonify({"games": games_data})
        
    except subprocess.CalledProcessError as e:
        print(f"Error running subprocess: {e.stderr}")  # Log any error
        return jsonify({"error": "Error executing get_games.py"}), 500
    except FileNotFoundError:
        return jsonify({"error": "schedule.json file not found"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# Run the app/start server
if __name__ == '__main__':
    app.run(debug=True)