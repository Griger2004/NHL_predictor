# Frontend — React UI

This directory contains the React application that renders today's NHL game predictions, live scores, and model accuracy.

Built with Vite + React. Talks exclusively to the Flask backend at `http://localhost:5000`.

---

## Directory Structure

```
frontend/
├── src/
│   ├── App.jsx        Root component: data fetching, state management, layout
│   ├── GameCard.jsx   Individual game prediction card
│   ├── App.css        Styles for layout, accuracy bar, prediction notes
│   └── main.jsx       React entry point
├── index.html
└── package.json
```

---

## Data Flow

The frontend is read-only — it never writes to the database directly. All mutations happen server-side when the user clicks "Generate Predictions".

```
User clicks "Generate"
    │
    ▼
fetchPredictions()
    ├── POST /api/predict
    │       └── Flask upserts games, runs predict_games.py, returns latest predictions
    │
    └── GET /api/predictions/history?date=...
            └── Flask reads all prediction rows for today (used to detect goalie changes)

fetchAccuracy()
    └── GET /api/predictions/accuracy
            └── Flask joins predictions × game_results, returns accuracy stats
```

On page load, `fetchGames()` calls `GET /api/games` to show today's schedule before any predictions are generated.

---

## State

`App.jsx` manages four pieces of state:

| State | Type | Source |
|---|---|---|
| `games` | Array | `GET /api/games` — NHL schedule (no DB) |
| `predictions` | Array | `GET /api/predict` — latest prediction per game |
| `predictionHistory` | Map `game_id → []` | `GET /api/predictions/history` — all attempts |
| `accuracy` | Object | `GET /api/predictions/accuracy` — running accuracy |

`predictionHistory` is a `Map` keyed by `game_id`. When `GameCard` receives `history` for a game, it checks `history.length > 1` to determine whether to show the "Goalie updated" badge and render a diff of goalie changes across prediction runs.

---

## GameCard

`GameCard.jsx` receives three props:

| Prop | Description |
|---|---|
| `prediction` | Latest prediction object from the DB |
| `gameStatus` | Live game state from the NHL schedule API (`FUT`, `PRE`, `LIVE`, `FINAL`, etc.) |
| `history` | Array of all prediction rows for this game (from `predictionHistory`) |

What each card renders:

- **Matchup header** — Away @ Home abbreviations and full names
- **Status badge** — game state with a pulsing dot for live games
- **Scoreboard** — only shown when `gameStatus` is `FINAL` or `OFF`
- **Goalie line** — starters identified at prediction time, with detection tier (FUT/PRE/LIVE)
- **"Goalie updated" badge** — shown when `history.length > 1` (a reprediction occurred)
- **Goalie change diff** — lists each goalie substitution across prediction runs (e.g. `NYR: Shesterkin → Drury`)
- **Prediction winner** — team predicted to win with confidence percentage
- **Actual winner** — shown only after game completes (from DB `game_results`)
- **Probability bar** — stacked horizontal bar with home/away win percentages

Cards receive a `.finished` CSS class when the game is done (gray background) and a `.correct` class when the prediction matched the actual outcome.

---

## How Reprediction History Is Used

Every time the backend detects a goalie change (different name or tier upgrade), `predict_games.py` inserts a new row into the `predictions` table rather than overwriting the old one. The history endpoint returns all rows for a date in `(game_id, predicted_at)` order.

The frontend groups rows by `game_id` into the `predictionHistory` map. `GameCard` walks through the history array and compares consecutive `home_goalie` / `away_goalie` values to build the list of changes displayed to the user. This lets users see exactly when lineups were confirmed and how that updated the model's outlook.

---

## Development

```bash
cd main/frontend
npm install
npm run dev
```

The dev server runs on `http://localhost:5173` and proxies API calls to the Flask backend. Make sure the backend is running on port 5000 first.

```bash
npm run build   # production build → dist/
```
