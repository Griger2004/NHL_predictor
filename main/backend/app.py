from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import subprocess
import os
import requests
from datetime import datetime

BACKEND_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.abspath(os.path.join(BACKEND_DIR, '..'))

VENV_PYTHON     = os.path.join(PROJECT_ROOT, 'nhl_venv', 'Scripts', 'python.exe')
ML_SCRIPTS_DIR  = os.path.join(PROJECT_ROOT, 'ml_dev', 'scripts')
PREDICT_SCRIPT  = os.path.join(ML_SCRIPTS_DIR, 'predict_games.py')
PREDICTIONS_DIR      = os.path.join(ML_SCRIPTS_DIR, 'predictions')
HISTORICAL_DATA_FILE = os.path.join(ML_SCRIPTS_DIR, 'generated', 'data', 'nhl_data.csv')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/api/health', methods=['GET'])
def health():
    missing = [p for p in [VENV_PYTHON, PREDICT_SCRIPT, HISTORICAL_DATA_FILE] if not os.path.exists(p)]
    if missing:
        return jsonify({"status": "degraded", "missing": missing})
    return jsonify({"status": "ok"})

@app.route('/api/predict', methods=['GET'])
def predict():
    try:
        date_str = datetime.now().strftime("%Y-%m-%d")
        csv_path = os.path.join(PREDICTIONS_DIR, f"predictions_{date_str}.csv")

        result = subprocess.run(
            [VENV_PYTHON, PREDICT_SCRIPT],
            capture_output=True, text=True,
            cwd=ML_SCRIPTS_DIR,
            timeout=120,
        )

        if result.returncode != 0:
            return jsonify({"error": result.stderr}), 500

        df = pd.read_csv(csv_path)
        return jsonify({"predictions": df.to_dict(orient='records')})

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Prediction script timed out after 120 seconds"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/games', methods=['GET'])
def get_games():
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        response = requests.get(
            f'https://api-web.nhle.com/v1/schedule/{today_str}',
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        today_games = next(
            (day["games"] for day in data.get("gameWeek", []) if day.get("date") == today_str),
            [],
        )

        games = [
            {
                "game_id": game.get("id"),
                "away_team": game.get("awayTeam", {}).get("abbrev"),
                "home_team": game.get("homeTeam", {}).get("abbrev"),
                "away_team_name": game.get("awayTeam", {}).get("placeName", {}).get("default", ""),
                "home_team_name": game.get("homeTeam", {}).get("placeName", {}).get("default", ""),
                "game_time": game.get("startTimeUTC"),
            }
            for game in today_games
        ]

        return jsonify({"games": games})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# Run the app/start server
if __name__ == '__main__':
    app.run(debug=True)