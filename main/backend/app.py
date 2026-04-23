from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import subprocess
import os
import requests
from datetime import datetime, timedelta

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
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        today_str = today.strftime("%Y-%m-%d")
        yesterday_str = yesterday.strftime("%Y-%m-%d")

        today_csv     = os.path.join(PREDICTIONS_DIR, f"predictions_{today_str}.csv")
        yesterday_csv = os.path.join(PREDICTIONS_DIR, f"predictions_{yesterday_str}.csv")

        result = subprocess.run(
            [VENV_PYTHON, PREDICT_SCRIPT],
            capture_output=True, text=True,
            cwd=ML_SCRIPTS_DIR,
            timeout=120,
        )

        if result.returncode != 0:
            return jsonify({"error": result.stderr}), 500

        all_predictions = []

        df_today = pd.read_csv(today_csv)
        df_today['date'] = today_str
        all_predictions.extend(df_today.to_dict(orient='records'))

        if os.path.exists(yesterday_csv):
            df_yesterday = pd.read_csv(yesterday_csv)
            df_yesterday['date'] = yesterday_str
            all_predictions.extend(df_yesterday.to_dict(orient='records'))

        return jsonify({"predictions": all_predictions})

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Prediction script timed out after 120 seconds"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/games', methods=['GET'])
def get_games():
    try:
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        today_str = today.strftime("%Y-%m-%d")
        yesterday_str = yesterday.strftime("%Y-%m-%d")

        # Requesting yesterday's schedule returns a gameWeek spanning yesterday→today+
        response = requests.get(
            f'https://api-web.nhle.com/v1/schedule/{yesterday_str}',
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        target_dates = {yesterday_str, today_str}
        games = []
        for day in data.get("gameWeek", []):
            date = day.get("date")
            if date not in target_dates:
                continue
            for game in day.get("games", []):
                games.append({
                    "game_id": game.get("id"),
                    "date": date,
                    "away_team": game.get("awayTeam", {}).get("abbrev"),
                    "home_team": game.get("homeTeam", {}).get("abbrev"),
                    "away_team_name": game.get("awayTeam", {}).get("placeName", {}).get("default", ""),
                    "home_team_name": game.get("homeTeam", {}).get("placeName", {}).get("default", ""),
                    "away_team_full_name": (game.get("awayTeam", {}).get("placeName", {}).get("default", "") + " " + game.get("awayTeam", {}).get("commonName", {}).get("default", "")).strip(),
                    "home_team_full_name": (game.get("homeTeam", {}).get("placeName", {}).get("default", "") + " " + game.get("homeTeam", {}).get("commonName", {}).get("default", "")).strip(),
                    "game_time": game.get("startTimeUTC"),
                    "game_state": game.get("gameState"),
                    "home_score": game.get("homeTeam", {}).get("score", 0),
                    "away_score": game.get("awayTeam", {}).get("score", 0),
                })

        return jsonify({"games": games})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# Run the app/start server
if __name__ == '__main__':
    app.run(debug=True)