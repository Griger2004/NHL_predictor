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
└── main/
    └── ml_dev/
        ├── scripts/
        │   ├── main.py              # Full data pipeline (fetch → feature engineering → CSV)
        │   ├── predict_games.py     # Daily prediction engine
        │   ├── config.py            # Seasons, API base URL, feature column lists
        │   └── api/
        │       └── client.py        # Async HTTP client with retry logic
        ├── notebooks/
        │   ├── ts_predict.ipynb     # Model training, validation, hyperparameter tuning
        │   ├── verify_calculations.ipynb  # Feature correctness verification
        │   └── models/
        │       ├── nhl_rf_model.pkl     # Trained Random Forest (300 trees)
        │       └── feature_names.pkl    # Ordered feature name list (71 features)
        └── scripts/generated/
            └── data/
                └── nhl_data.csv     # Engineered dataset output (~3,877 games)
```

---

## Tech Stack

| Category | Library |
|---|---|
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn |
| Async HTTP | aiohttp |
| Model persistence | joblib |
| Visualization | matplotlib, seaborn |
| Testing | pytest, pytest-mock |

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

## Detailed Documentation

- [ml_dev/README.md](main/ml_dev/README.md) — Full ML pipeline documentation: feature engineering methods, EWM design, season boundary handling, cold start logic, and all 71 features explained.
- [ml_dev/notebooks/README.md](main/ml_dev/notebooks/README.md) — Model training documentation: dataset construction, cross-validation strategy, hyperparameter tuning, evaluation metrics, and feature importances.
