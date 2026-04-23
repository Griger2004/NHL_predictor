# Backend — Flask API & Database Layer

This directory contains the Flask API server and the SQLAlchemy database layer that connects the React frontend to the ML prediction pipeline.

---

## Directory Structure

```
backend/
├── app.py                 Flask API server (5 endpoints)
├── db.py                  SQLAlchemy engine, schema, and query functions
├── migrate_csv_to_db.py   One-time utility: loads historical CSV into nhl_game_data table
├── wsgi.py                WSGI entry point for production deployments
└── nhl_predictions.db     SQLite database file (not committed)
```

---

## Database

### Overview

The database is SQLite in development (default) and PostgreSQL in production, selectable via the `DATABASE_URL` environment variable. The engine is created in `db.py` with `pool_pre_ping=True` for connection health checks, and automatically rewrites the deprecated `postgres://` scheme to `postgresql://` for Heroku/Render compatibility.

```
DATABASE_URL=sqlite:///./nhl_predictions.db   # default
DATABASE_URL=postgresql://user:pass@host/db   # production
```

Tables are created on Flask startup via `init_db()`, which wraps every `CREATE TABLE IF NOT EXISTS` in a single transaction. Re-running is a no-op.

---

### Schema

#### `games`

Stores live game metadata and real-time state. Written on every `/api/predict` call.

| Column | Type | Notes |
|---|---|---|
| `game_id` | INTEGER PK | NHL API game identifier |
| `game_date` | TEXT NOT NULL | `YYYY-MM-DD` |
| `season` | INTEGER | NHL season year (e.g. `20242025`) |
| `home_team` / `away_team` | TEXT | 3-letter abbreviation (e.g. `BOS`) |
| `home_team_name` / `away_team_name` | TEXT | Full name (e.g. `Boston Bruins`) |
| `game_time_utc` | TEXT | ISO 8601 timestamp |
| `game_state` | TEXT | `FUT`, `PRE`, `LIVE`, `CRIT`, `FINAL`, `OFF` |
| `home_score` / `away_score` | INTEGER | Current or final score |
| `created_at` / `updated_at` | TEXT | Auto-set via `datetime('now')` |

`upsert_game()` uses `INSERT ... ON CONFLICT(game_id) DO UPDATE` to keep the row's state, score, and `updated_at` current across repeated calls without duplicating rows.

#### `goalie_states`

Tracks which goalie was identified for each side and how confidently. The prediction engine reads this table to detect goalie changes and decide whether to repridict.

| Column | Type | Notes |
|---|---|---|
| `game_id` | INTEGER FK → `games` | |
| `side` | TEXT | `'home'` or `'away'` (CHECK constraint enforced) |
| `goalie_name` | TEXT | Starter name or `"Unknown"` |
| `detection_tier` | TEXT | `LIVE`, `PRE`, or `FUT` (see below) |
| `recorded_at` | TEXT | When this state was written |

**Primary key is `(game_id, side)`** — one row per team per game, overwritten on each prediction run.

Detection tiers reflect confidence:
- `FUT` — most-started goalie this season (probabilistic guess, lineup not yet announced)
- `PRE` — goalie flagged `startingLineup=True` in the NHL API (1–2 hours before puck drop)
- `LIVE` — goalie with highest time-on-ice from the boxscore (most accurate, during/after game)

A reprediction is triggered if the stored name differs from the current name, or if the tier upgraded (e.g. `FUT` → `PRE`). This is evaluated entirely inside `predict_games.py`.

#### `predictions`

Every prediction attempt, including repredictions. Multiple rows per game are expected and intentional — the full history is preserved.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | Unique prediction row ID |
| `game_id` | INTEGER FK → `games` | |
| `predicted_at` | TEXT | Timestamp of this prediction |
| `home_goalie` / `away_goalie` | TEXT | Starter names at prediction time |
| `home_goalie_tier` / `away_goalie_tier` | TEXT | Detection tiers at prediction time |
| `pred_home_win` | INTEGER | `1` = home wins, `0` = away wins |
| `prob_home_win` / `prob_away_win` | REAL | Model probabilities, sum to 1.0 |
| `confidence` | REAL | `max(prob_home_win, prob_away_win)` |
| `feature_snapshot` | TEXT | Optional JSON of the 71-feature vector |
| `model_version` | TEXT | Which `.pkl` file generated this |
| `invalidation_reason` | TEXT | Populated if this row was superseded |

`get_predictions_for_date()` retrieves only the **latest** prediction per game using:
```sql
WHERE p.id IN (SELECT MAX(id) FROM predictions GROUP BY game_id)
```

`get_prediction_history_for_date()` returns **all** rows for a date, ordered by `(game_id, predicted_at)`, which the frontend uses to detect goalie changes and show the "Goalie updated" badge.

#### `game_results`

Final outcomes, written once per game when state becomes `FINAL` or `OFF`.

| Column | Type | Notes |
|---|---|---|
| `game_id` | INTEGER PK FK → `games` | |
| `home_score` / `away_score` | INTEGER | Final score |
| `actual_home_win` | INTEGER | `1` if home won, `0` otherwise |
| `finalized_at` | TEXT | Auto-set |

Uses `ON CONFLICT(game_id) DO NOTHING` — the first result written is permanent. This prevents a live game score update from overwriting a finalized result.

#### `nhl_game_data`

Created by `migrate_csv_to_db.py`. Contains 3,877+ historical games from seasons 2022–2025, each with all 71 engineered features. `predict_games.py` queries this table first (falls back to the CSV if unavailable) to compute rolling stats, goalie EWMs, head-to-head records, and rest days for today's feature vectors.

---

### Write Path Summary

```
/api/predict called
    ├── upsert_game()            → games table (every game, every call)
    ├── upsert_final_result()    → game_results (only FINAL/OFF games)
    └── subprocess: predict_games.py
            ├── INSERT INTO predictions   (one row per game needing prediction)
            └── UPSERT goalie_states      (one row per side per game)
```

---

## API Endpoints

### `GET /api/health`

Checks whether the ML dependencies are in place.

**Response:**
```json
{ "status": "ok" }
// or
{ "status": "degraded", "missing": ["/path/to/nhl_venv/...", ...] }
```

---

### `GET /api/predict?date=YYYY-MM-DD`

The primary endpoint. Orchestrates the full prediction pipeline:

1. Fetches today's schedule from `https://api-web.nhle.com/v1/schedule/{date}`
2. Upserts every game's current state and score into `games`
3. Writes final results for completed games into `game_results`
4. Runs `predict_games.py` as a subprocess, passing a JSON list of non-final game IDs — the script writes to `predictions` and `goalie_states`
5. Reads latest predictions from the DB and returns them as JSON

The subprocess has a 180-second timeout. If it exceeds that, a `500` is returned.

**Response:**
```json
{
  "predictions": [
    {
      "game_id": 2025030143,
      "date": "2025-04-23",
      "time": "2025-04-23T18:00:00Z",
      "away_team": "NYR",
      "home_team": "OTT",
      "away_team_name": "New York Rangers",
      "home_team_name": "Ottawa Senators",
      "away_goalie": "Shesterkin",
      "home_goalie": "Ullmark",
      "pred_home_win": 1,
      "prob_home_win": 0.63,
      "prob_away_win": 0.37,
      "confidence": 0.63
    }
  ]
}
```

---

### `GET /api/games`

Returns today's schedule directly from the NHL API without hitting the database. Used by the frontend on initial page load before predictions are generated.

---

### `GET /api/predictions/history?date=YYYY-MM-DD`

Returns all prediction rows for a date, including repredictions. The frontend uses this to:
- Detect if `history.length > 1` (goalie was updated mid-day)
- Show which goalie changed between prediction runs

---

### `GET /api/predictions/accuracy`

Computes live accuracy across all finalized games in the database by joining `predictions` (latest per game) with `game_results`.

**Response:**
```json
{
  "accuracy": {
    "total_games": 1234,
    "correct": 829,
    "accuracy_pct": 67.2
  }
}
```

---

## Running the Backend

```bash
cd main
source nhl_venv/Scripts/activate   # Windows: nhl_venv\Scripts\activate
cd backend
python app.py
```

The server starts on `http://localhost:5000`. `init_db()` runs on startup and creates all tables if they don't exist — safe to run on an existing database.

For production, use `wsgi.py` with Gunicorn:
```bash
gunicorn wsgi:app
```

Set `DATABASE_URL` in your environment (or a `.env` file) to point at a PostgreSQL instance.
