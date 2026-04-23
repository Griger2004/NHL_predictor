from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import os
import json
import requests
from datetime import datetime

from db import (
    init_db, upsert_game, upsert_final_result,
    get_predictions_for_date,
    get_prediction_history_for_date, get_accuracy_stats,
)

BACKEND_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.abspath(os.path.join(BACKEND_DIR, '..'))

VENV_PYTHON     = os.path.join(PROJECT_ROOT, 'nhl_venv', 'Scripts', 'python.exe')
ML_SCRIPTS_DIR  = os.path.join(PROJECT_ROOT, 'ml_dev', 'scripts')
PREDICT_SCRIPT  = os.path.join(ML_SCRIPTS_DIR, 'predict_games.py')
HISTORICAL_DATA_FILE = os.path.join(ML_SCRIPTS_DIR, 'generated', 'data', 'nhl_data.csv')

app = Flask(__name__)
CORS(app)

# Create DB tables on startup (no-op if they already exist)
init_db()


@app.route('/api/health', methods=['GET'])
def health():
    missing = [p for p in [VENV_PYTHON, PREDICT_SCRIPT, HISTORICAL_DATA_FILE] if not os.path.exists(p)]
    if missing:
        return jsonify({"status": "degraded", "missing": missing})
    return jsonify({"status": "ok"})


@app.route('/api/predict', methods=['GET'])
def predict():
    try:
        date_str = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))

        # 1. Fetch today's schedule to get current game states
        resp = requests.get(
            f'https://api-web.nhle.com/v1/schedule/{date_str}', timeout=10
        )
        resp.raise_for_status()
        today_games = next(
            (day["games"] for day in resp.json().get("gameWeek", [])
             if day.get("date") == date_str),
            []
        )

        games_needing_prediction = []  # non-FINAL game_ids to pass to the script

        for game in today_games:
            gid = game["id"]
            state = game.get("gameState", "")
            home_score = game.get("homeTeam", {}).get("score") or 0
            away_score = game.get("awayTeam", {}).get("score") or 0

            home_place  = game.get("homeTeam", {}).get("placeName", {}).get("default", "")
            home_common = game.get("homeTeam", {}).get("commonName", {}).get("default", "")
            away_place  = game.get("awayTeam", {}).get("placeName", {}).get("default", "")
            away_common = game.get("awayTeam", {}).get("commonName", {}).get("default", "")

            # Persist latest game state to DB
            upsert_game(
                game_id=gid,
                game_date=date_str,
                season=game.get("season"),
                home_team=game.get("homeTeam", {}).get("abbrev", ""),
                away_team=game.get("awayTeam", {}).get("abbrev", ""),
                home_team_name=f"{home_place} {home_common}".strip(),
                away_team_name=f"{away_place} {away_common}".strip(),
                game_time_utc=game.get("startTimeUTC"),
                game_state=state,
                home_score=home_score,
                away_score=away_score,
            )

            if state in ("FINAL", "OFF"):
                upsert_final_result(gid, home_score, away_score)
            else:
                # Let predict_games.py decide internally if goalie changed
                games_needing_prediction.append(gid)

        # 2. Run subprocess only for non-FINAL games
        if games_needing_prediction:
            result = subprocess.run(
                [VENV_PYTHON, PREDICT_SCRIPT, json.dumps(games_needing_prediction)],
                capture_output=True, text=True,
                cwd=ML_SCRIPTS_DIR,
                timeout=180,
            )
            if result.returncode != 0:
                return jsonify({"error": result.stderr}), 500

        # 3. Load all of today's latest predictions from DB and return them
        rows = get_predictions_for_date(date_str)
        predictions = [
            {
                "game_id":        r["game_id"],
                "date":           r["game_date"],
                "time":           r["game_time_utc"],
                "away_team":      r["away_team"],
                "home_team":      r["home_team"],
                "away_team_name": r.get("away_team_name", ""),
                "home_team_name": r.get("home_team_name", ""),
                "away_goalie":    r["away_goalie"],
                "home_goalie":    r["home_goalie"],
                "pred_home_win":  r["pred_home_win"],
                "prob_home_win":  float(r["prob_home_win"]),
                "prob_away_win":  float(r["prob_away_win"]),
                "confidence":     float(r["confidence"]),
            }
            for r in rows
        ]

        return jsonify({"predictions": predictions})

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Prediction script timed out after 180 seconds"}), 500
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
                "away_team_full_name": (game.get("awayTeam", {}).get("placeName", {}).get("default", "") + " " + game.get("awayTeam", {}).get("commonName", {}).get("default", "")).strip(),
                "home_team_full_name": (game.get("homeTeam", {}).get("placeName", {}).get("default", "") + " " + game.get("homeTeam", {}).get("commonName", {}).get("default", "")).strip(),
                "game_time": game.get("startTimeUTC"),
                "game_state": game.get("gameState"),
                "home_score": game.get("homeTeam", {}).get("score", 0),
                "away_score": game.get("awayTeam", {}).get("score", 0),
            }
            for game in today_games
        ]

        return jsonify({"games": games})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predictions/history', methods=['GET'])
def prediction_history():
    """All prediction attempts for a given date (shows repredictions on goalie changes)."""
    date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    try:
        rows = get_prediction_history_for_date(date_str)
        return jsonify({"history": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predictions/accuracy', methods=['GET'])
def prediction_accuracy():
    """Overall model accuracy across all finalized games stored in DB."""
    try:
        stats = get_accuracy_stats()
        return jsonify({"accuracy": stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
