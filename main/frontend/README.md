# Frontend — React UI

This directory contains the React application that renders today's NHL game predictions, live scores, and results.

Built with Vite + React. Talks exclusively to the Flask backend at `http://localhost:5000`.

---

## Directory Structure

```
frontend/
├── src/
│   ├── components/
│   │   └── GameCard/
│   │       ├── GameCard.jsx        Prediction card with goalie change diffs
│   │       └── GameResultCard.jsx  Finished-game card for games with no prediction
│   ├── constants/
│   │   └── gameStates.js      GAME_STATE enum, FINISHED_STATES, LIVE_STATES, STATUS_LABELS
│   ├── hooks/
│   │   └── useNHLData.js      All data fetching and app state
│   ├── utils/
│   │   ├── api.js             BASE_URL (dev vs. prod)
│   │   ├── dates.js           todayStr, yesterdayStr, dateLabel
│   │   └── gameUtils.js       buildHistoryMap, getGoalieChanges
│   ├── App.jsx                Root component: layout and derived state
│   ├── App.css                Styles, CSS custom properties, animations
│   ├── index.css              Global base styles
│   └── main.jsx               React entry point
├── public/
├── index.html
├── vite.config.js
└── package.json
```

---

## Data Flow

The frontend is read-only — it never writes to the database directly. All mutations happen server-side when the user clicks "Generate Predictions".

```
Page load
    ├── GET /api/games
    │       └── Flask fetches yesterday + today NHL schedules, returns game list
    │
    └── GET /api/predictions/today?date=YYYY-MM-DD    (called twice: today + yesterday)
            └── Flask reads latest prediction per game for the given date
                If predictions exist, also fetches prediction history

User clicks "Generate"
    │
    ▼
GET /api/predict
    └── Flask runs predict_games.py, upserts predictions, returns latest set
        Then fetches /api/predictions/history for today + yesterday

After predictions load (both page-load and generate paths):
    └── GET /api/predictions/history?date=YYYY-MM-DD  (today + yesterday)
            └── Flask returns all prediction rows in (game_id, predicted_at) order
                Used to detect goalie changes across reprediction runs
```

---

## State

All state lives in the `useNHLData` hook (`src/hooks/useNHLData.js`). `App.jsx` is render-only.

| State | Type | Source |
|---|---|---|
| `games` | `Array` | `GET /api/games` — live NHL schedule |
| `predictions` | `Array` | Today's latest prediction per game |
| `yesterdayPredictions` | `Array` | Yesterday's latest prediction per game |
| `predictionHistory` | `Object` (`game_id → Array`) | All prediction rows, grouped by game |
| `hasGenerated` | `boolean` | Whether the user has clicked Generate this session |
| `resultsTab` | `'today' \| 'yesterday' \| null` | Active tab in the Results section |
| `loadingGames` | `boolean` | Schedule fetch in flight |
| `loadingPredictions` | `boolean` | Predict fetch in flight |
| `gamesError` | `string \| null` | Schedule fetch error message |
| `predictionsError` | `string \| null` | Predict fetch error message |

`predictionHistory` is a plain object keyed by `game_id`. When `GameCard` receives `history` for a game, it passes it to `getGoalieChanges()` which walks consecutive prediction rows to build the list of goalie substitutions.

---

## Components

### GameCard

`src/components/GameCard/GameCard.jsx` receives three props:

| Prop | Description |
|---|---|
| `prediction` | Latest prediction object from the DB |
| `gameStatus` | Matching game object from the NHL schedule (`game_state`, scores, etc.) |
| `history` | Array of all prediction rows for this game (from `predictionHistory`) |

What each card renders:

- **Matchup header** — Away @ Home abbreviations with a status badge (`Upcoming`, `Pre-Game`, `Live`, `Final`)
- **Live indicator** — pulsing red dot when `game_state` is `LIVE` or `CRIT`
- **Scoreboard** — absolute-positioned top-right, only shown when `isFinished`
- **Goalie line** — starters at prediction time; labeled "unconfirmed" before game start
- **"Goalies updated" badge** — shown when `history.length > 1` (a reprediction occurred)
- **Goalie change diff** — lists each substitution across runs (e.g. `NYR: Shesterkin → Drury`)
- **Prediction winner** — team predicted to win with confidence percentage
- **Actual winner** — shown only after game completes
- **Probability bar** — stacked horizontal bar, away (purple) left, home (blue) right

Cards receive a `.finished` CSS class when the game is done and a `.correct` class (green glow) when the prediction matched the actual outcome.

Game state constants (`FINISHED_STATES`, `LIVE_STATES`, `STATUS_LABELS`) are defined in `src/constants/gameStates.js` and shared between `App.jsx` and `GameCard`.

### GameResultCard

`src/components/GameCard/GameResultCard.jsx` renders finished games for which no prediction was ever generated (e.g., games that completed before "Generate" was clicked). It receives one prop:

| Prop | Description |
|---|---|
| `game` | Game object from the NHL schedule (`game_state`, scores, team names) |

Renders the final score, matchup header with a status badge, the winner's name, and a "No prediction made" label. Uses the same `.game-card.finished` CSS class as `GameCard`.

---

## How Reprediction History Is Used

Every time the backend detects a goalie change, `predict_games.py` inserts a new row into the `predictions` table rather than overwriting the old one. The history endpoint returns all rows for a date in `(game_id, predicted_at)` order.

`buildHistoryMap()` in `src/utils/gameUtils.js` groups these rows by `game_id` into the `predictionHistory` object. `getGoalieChanges()` then walks the array, comparing consecutive `home_goalie` / `away_goalie` values to build the diff shown to the user.

---

## Development

### With Docker (recommended)

From the repo root:
```bash
docker-compose up --build
```

### Without Docker

```bash
cd main/frontend
npm install
npm run dev     # dev server at http://localhost:5173
```

Make sure the Flask backend is running on port 5000 before starting the dev server. The API base URL is set in `.env`:

```
VITE_API_URL="http://127.0.0.1:5000/api"       # used in dev
VITE_API_URL_PROD="/api"                         # used in production build
```

```bash
npm run build   # production build → dist/
npm run lint    # ESLint
```
