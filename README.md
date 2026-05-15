# NHL Win Predictor

A machine learning system that predicts NHL game outcomes using exponentially weighted team statistics, goalie performance metrics, season standings, and historical matchup data. For every game on today's schedule, the model outputs a win probability and prediction for the home team.

---

## What It Does

The predictor ingests raw game data from the official NHL API, engineers over 70 features from it, and runs them through a trained Random Forest classifier. Every day, a prediction script assembles a live feature vector for each scheduled matchup — pulling real-time standings, identifying probable starting goalies, and computing up-to-date rolling stats — then produces a CSV of predictions with confidence levels.

**Current model accuracy: 67.18%** on a fully held-out test season (Season 3), compared to a 54.19% naive home-win baseline.

---

## How It Works

The system operates as a sequential four-stage pipeline:

```
NHL API  ──►  Feature Engineering  ──►  Model Training  ──►  Daily Predictions
             (main.py)                   (ts_predict.ipynb)   (predict_games.py)
```

**1. Data Collection**
Historical games are fetched asynchronously from the NHL API across multiple seasons. Each completed game yields raw box score data: goals, shots, faceoff percentages, power play stats, hits, blocked shots, and goalie breakdowns.

**2. Feature Engineering**
Raw stats are transformed into predictive signals. Team performance is tracked with exponentially weighted moving averages (EWM) that respect season boundaries. Goalie EWMs are maintained per-starter. Season standings, rest days, and head-to-head matchup history round out the feature set.

**3. Model Training**
A Random Forest classifier is trained on seasons of historical data using a season-level expanding window cross-validation strategy — the temporal equivalent of walk-forward testing. Hyperparameters are tuned via grid search constrained to training folds only.

**4. Daily Prediction**
On prediction day, the script fetches today's schedule, identifies starting goalies using a three-tier detection strategy (live boxscore → pre-game lineup → historical starter frequency), builds feature vectors for each matchup, and runs inference. Output is a CSV with per-game win probabilities.

---

## Project Structure

```
NHL_predictor/
├── docker-compose.yml           # Runs backend + frontend together
└── main/
    ├── backend/                 # Flask API + SQLite database layer
    │   ├── app.py               # API server (6 endpoints)
    │   ├── db.py                # SQLAlchemy schema and query functions
    │   ├── migrate_csv_to_db.py # One-time: load historical CSV into DB
    │   ├── wsgi.py              # WSGI entry point for production
    │   ├── Dockerfile
    │   └── requirements.txt
    ├── frontend/                # React + Vite UI
    │   ├── src/
    │   │   ├── components/GameCard/
    │   │   │   ├── GameCard.jsx         # Prediction card with goalie change diffs
    │   │   │   └── GameResultCard.jsx   # Finished-game card when no prediction exists
    │   │   ├── constants/gameStates.js  # GAME_STATE enum, FINISHED_STATES, LIVE_STATES
    │   │   ├── hooks/useNHLData.js      # All data fetching and app state
    │   │   ├── utils/
    │   │   │   ├── api.js               # BASE_URL (dev vs. prod)
    │   │   │   ├── dates.js             # todayStr, yesterdayStr, dateLabel
    │   │   │   └── gameUtils.js         # buildHistoryMap, getGoalieChanges
    │   │   ├── App.jsx                  # Root component: layout and derived state
    │   │   └── App.css
    │   ├── index.html
    │   ├── Dockerfile
    │   └── package.json
    └── ml_dev/                  # ML pipeline: data collection, feature engineering, training
        ├── scripts/
        │   ├── main.py              # Full data pipeline (fetch → feature engineering → CSV)
        │   ├── predict_games.py     # Daily prediction engine (called by Flask as subprocess)
        │   ├── config.py            # Seasons, API base URL, feature column lists
        │   ├── api/
        │   │   └── client.py        # Async HTTP client with retry logic
        │   ├── utils/
        │   │   └── helpers.py       # Stat extraction and calculation utilities
        │   └── generated/data/
        │       └── nhl_data.csv     # Engineered dataset output (~3,877 games)
        ├── notebooks/
        │   ├── ts_predict.ipynb     # Model training, validation, hyperparameter tuning
        │   ├── verify_calculations.ipynb  # Feature correctness spot-checks
        │   └── models/
        │       ├── nhl_rf_model.pkl     # Trained Random Forest (300 trees)
        │       └── feature_names.pkl    # Ordered feature name list (71 features)
        └── tests/
            └── test_main.py         # Unit tests for the data pipeline
```

---

## Tech Stack

| Category | Library / Tool |
|---|---|
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn |
| Async HTTP | aiohttp |
| Model persistence | joblib |
| Visualization | matplotlib, seaborn |
| Testing | pytest, pytest-mock |
| Web API | Flask, flask-cors |
| Database | SQLite (dev), PostgreSQL (prod) via SQLAlchemy |
| Frontend | React, Vite |
| Containerization | Docker, docker-compose |

---

## Performance

| Metric | Value |
|---|---|
| Test accuracy (Season 3 holdout) | **67.18%** |
| Naive baseline (always predict home win) | 54.19% |
| Precision — home win | 65% |
| Recall — home win | 80% |
| F1-score — home win | 0.72 |

The model is trained on Seasons 1 and 2 (2,585 games) and evaluated on Season 3 (1,292 games) without any look-ahead. Cross-validation across season folds yields a mean baseline of 67.92%, and the final held-out test lands at 67.18% — confirming that the model generalizes across seasons rather than just memorizing historical patterns.

---

## How to Use

### Option 1 — Docker (recommended)

Requires [Docker](https://docs.docker.com/get-docker/) and [docker-compose](https://docs.docker.com/compose/).

```bash
docker-compose up --build
```

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:5000`

The database is persisted in a named Docker volume (`nhl_db`). On first run the database will be empty — click **Generate** in the UI to run the first prediction and populate it.

> **Note:** The historical feature data (`nhl_data.csv`) and trained model (`nhl_rf_model.pkl`) are bundled into the backend image at build time. If you regenerate the CSV or retrain the model, rebuild the image with `docker-compose up --build`.

---

### Option 2 — Manual setup

#### Prerequisites

- Python 3.12 with the project virtualenv (`main/nhl_venv`)
- Node.js 22+

#### 1. Start the backend

```bash
# Windows
main\nhl_venv\Scripts\activate
# macOS/Linux
source main/nhl_venv/bin/activate

cd main/backend
python app.py
```

The API starts on `http://localhost:5000`. On first run it creates `nhl_predictions.db` automatically.

#### 2. Start the frontend

In a separate terminal:

```bash
cd main/frontend
npm install      # first time only
npm run dev
```

The UI is available at `http://localhost:5173`.

#### 3. (First time only) Load historical data into the database

The ML pipeline needs historical game data to build feature vectors. After generating `nhl_data.csv` with `main.py` (or using the pre-existing one), run:

```bash
cd main/backend
python migrate_csv_to_db.py
```

This loads the CSV into the `nhl_game_data` table. The prediction engine falls back to the CSV directly if this step is skipped, but the database path is faster.

#### 4. (Optional) Re-run the full data pipeline

To regenerate `nhl_data.csv` from scratch using the live NHL API:

```bash
cd main/ml_dev/scripts
python main.py
```

This fetches all historical games and re-engineers the 71-feature dataset. Then re-run `migrate_csv_to_db.py` to reload the database.

---

### Using the UI

1. Open `http://localhost:5173`. The page loads today's (and yesterday's) schedule automatically.
2. Click **Generate** to run predictions for today's games.
3. The **Predictions** section shows each matchup's predicted winner with win probability and detected starting goalies.
4. The **Results** section shows finished games with actual outcomes compared to predictions.
5. A "Goalies updated" badge appears on any game where the detected starter changed between prediction runs.

---

## Detailed Documentation

- [backend/README.md](main/backend/README.md) — API endpoints, database schema, and write-path summary.
- [frontend/README.md](main/frontend/README.md) — Component structure, data flow, and state management.
- [ml_dev/README.md](main/ml_dev/README.md) — Full ML pipeline documentation: feature engineering methods, EWM design, season boundary handling, cold start logic, and all 71 features explained.
- [ml_dev/notebooks/README.md](main/ml_dev/notebooks/README.md) — Model training documentation: dataset construction, cross-validation strategy, hyperparameter tuning, evaluation metrics, and feature importances.
