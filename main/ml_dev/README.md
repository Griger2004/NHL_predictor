# ML Pipeline — Feature Engineering & Data Architecture

This document covers the full data pipeline: how raw NHL API data is fetched, what transformations turn it into model-ready features, why each design choice was made, and how edge cases like season boundaries and missing goalies are handled.

---

## Pipeline Overview

The pipeline is implemented in [scripts/main.py](scripts/main.py) as a class with six sequential steps. Each step builds on the previous, appending new feature columns to a growing DataFrame. The final output is `scripts/generated/data/nhl_data.csv` — a flat table of games with all engineered features.

```
Step 1: fetch_games()               → raw box scores from NHL API
Step 2: add_team_rolling_features() → EWM stats + L5 rolling windows
Step 3: add_goalie_features()       → per-goalie EWM + team-level aggregates
Step 4: add_standings_features()    → season standings (win %, point %)
Step 5: add_rest_days()             → team rest days + goalie rest days
Step 6: add_head_to_head()          → cumulative season matchup stats
```

---

## Step 1 — Game Fetching

**Source:** NHL public API (`https://api-web.nhle.com`)

**Endpoints used:**
- `/v1/wsc/game-story/{game_id}` — final game stats and outcomes
- `/v1/gamecenter/{game_id}/boxscore` — per-goalie stats

**Seasons fetched:** Configurable in [scripts/config.py](scripts/config.py). Currently spans 2022–2025.

**What gets extracted per game:**
- Game ID, date, season identifier
- Home and away team names and three-letter abbreviations
- Final score (goals for / goals against per side)
- Shots on goal
- Faceoff win percentage
- Power play opportunities and percentage
- Penalty kill percentage
- Penalty minutes (PIM)
- Hits, blocked shots, takeaways, giveaways
- Starting goalie name (per side)

**Filtering:** Only games with state `FINAL` are processed. Games in `FUT`, `PRE`, or `LIVE` states are skipped entirely — this ensures the dataset contains only complete, verified outcomes.

**Team normalization:** The Arizona Coyotes relocated to Utah as the Utah Hockey Club (UTA) mid-history. The pipeline normalizes all Arizona/Utah references to `UTA` to maintain continuity in rolling stats across the team's history.

---

## Step 2 — Team Rolling Features

This is the most important step. It converts raw per-game statistics into forward-looking estimates of each team's current form using two methods: exponential weighted moving averages (EWM) and fixed-length rolling windows.

### Exponential Weighted Moving Average (EWM)

**Why EWM instead of a simple rolling average?**

A simple 10-game rolling average treats a game from 10 games ago identically to yesterday's game. EWM assigns exponentially decreasing weight to older observations, meaning recent performance matters more without completely discarding history. This better reflects how a team's current capability actually works.

**Alpha = 0.3** — for each new game observation `x`, the updated EWM is:

```
new_ewm = 0.3 * x + 0.7 * previous_ewm
```

This means: weight the current game at 30%, carry forward 70% of the prior estimate. An alpha of 0.3 is intentionally conservative — it captures trends without overreacting to a single hot or cold game.

**Stats tracked with EWM (per team, home and away):**

| Feature | Description |
|---|---|
| `gf_ewm` | Goals for per game |
| `ga_ewm` | Goals against per game |
| `sog_ewm` | Shots on goal per game |
| `powerplay_pct_ewm` | Power play conversion percentage |
| `pk_pct_ewm` | Penalty kill percentage |
| `faceoffwin_pct_ewm` | Faceoff win percentage |
| `pims_ewm` | Penalty minutes per game |
| `hits_ewm` | Hits per game |
| `blockedshots_ewm` | Blocked shots per game |
| `giveaways_ewm` | Giveaways per game |
| `takeaways_ewm` | Takeaways per game |

All EWM features are computed **prior to the current game** — the update from game `i` produces the feature value used for game `i+1`. This is critical for preventing data leakage.

### Season Boundary Regression-to-Mean

Between seasons, roster changes are significant: players get traded, goalies retire, coaching staffs turn over. Carrying last season's EWM forward unchanged would overweight stale information.

**The approach:** When a team's first game of a new season is encountered, the EWM seed is computed as a weighted blend:

```
season_start_seed = 0.7 * (last_season_final_ewm) + 0.3 * (prior_season_league_average)
```

This anchors the new season's estimate 70% in the team's actual prior performance and pulls it 30% toward the league mean. The result is a soft reset — teams that were outliers move slightly toward average, but genuine performance differences are preserved.

**Why 70/30?** This split was chosen empirically to balance two competing needs: honoring continuity for established teams while injecting enough uncertainty to account for roster changes. A 50/50 split over-regresses; a 90/10 split barely adjusts at all.

### Last-5 Rolling Windows

EWMs are good at long-run form but can miss sharp short-term momentum shifts. Four last-5 features complement the EWMs:

| Feature | Description |
|---|---|
| `wins_l5` | Number of wins in last 5 games |
| `win_pct_l5` | Win percentage across last 5 games |
| `powerplays_l5` | Power play opportunities in last 5 games |
| `penalty_kills_l5` | Penalty kill attempts in last 5 games |

**Season-bounded:** The L5 window resets at season boundaries. A team's April wins do not count toward their October momentum. This prevents stale cross-season leakage in the most recent-form features.

**Cold start:** The first 1–4 games of a season produce NaN values for L5 features (there aren't 5 prior games yet). In training, these rows are handled via forward-fill or exclusion. In prediction, if fewer than 5 games exist in the current season, the available games are used.

### Differential Features

Three computed differentials capture relative team strength directly:

| Feature | Calculation |
|---|---|
| `home_goal_diff_ewm` | `home_gf_ewm - away_gf_ewm` |
| `home_ga_diff_ewm` | `home_ga_ewm - away_ga_ewm` |
| `home_shot_diff_ewm` | `home_sog_ewm - away_sog_ewm` |

These encode the relative advantage directly as single features, which can help tree-based models learn matchup dynamics without needing to compare the component features implicitly.

**Total features from this step: 30**

---

## Step 3 — Goalie Features

Goalie performance is one of the highest-variance factors in any given NHL game. A hot goalie can steal a game for an outmatched team; a struggling one can sink a favorite. The pipeline tracks goalie performance at two levels: per-individual-starter and per-team-aggregate.

### Per-Goalie EWM Stats

Each named goalie has their own EWM state, updated only when they start a game. This avoids polluting a starter's EWM with backup appearances (which are usually under different circumstances).

**Stats tracked per goalie (alpha = 0.3, same as team stats):**

| Feature | Description |
|---|---|
| `save_pct_ewm` | Save percentage (saves / shots faced) |
| `ga_ewm` | Goals allowed per start |
| `saves_ewm` | Saves per start |
| `ev_sa_ewm` | Even-strength shots against per start |
| `pp_sa_ewm` | Power play shots against per start |
| `sh_sa_ewm` | Shorthanded shots against per start |
| `ev_ga_ewm` | Even-strength goals against per start |
| `pp_ga_ewm` | Power play goals against per start |

These features are prefixed `home_goalie_` and `away_goalie_` in the dataset.

**Season Boundary — Goalie EWM:**
The same 70/30 regression-to-mean logic applies at season boundaries. A goalie's last-season EWM is blended with the prior-season league average for that stat (e.g., ~0.915 for save percentage).

### Rookie and Debut Handling

When a goalie has no prior starts in the dataset (new to the league, called up, trade deadline arrival), there is no historical EWM to carry forward. Using zero would create a severe negative bias.

**Solution:** Use the season-average EWM across all goalies as a neutral prior. For save percentage this is approximately 0.915. For goals against, approximately 2.8. This means a first-time starter is treated as a league-average goalie rather than an unknown or a poor one — an appropriate epistemic prior given no evidence.

### Team-Level Goalie Aggregates

Individual starter features assume the model knows who is starting. The team-level aggregates provide a coarser but more robust signal — the overall quality of a team's goaltending across all their goalies this season:

| Feature | Description |
|---|---|
| `home_team_save_pct_ewm` | Weighted EWM of save % across all goalies who played for the home team this season |
| `away_team_save_pct_ewm` | Same for away team |

These are useful when the starter is uncertain (Tier 3 predictions) and provide a fallback signal that captures depth chart quality.

**Total features from this step: 18** (8 per-goalie × 2 sides + 2 team aggregates)

---

## Step 4 — Season Standings Features

Real-time standings capture information that rolling stats can miss: overall season trajectory, home/road splits, and current streak.

**Source:** NHL standings API (`/v1/standings/{date}`), called at the date of each historical game (and fresh on prediction day).

**Features extracted:**

| Feature | Description |
|---|---|
| `home_win_pct_season` | (homeWins + roadWins) / gamesPlayed |
| `away_win_pct_season` | Same for away team |
| `home_home_win_pct` | homeWins / homeGamesPlayed |
| `away_away_win_pct` | roadWins / roadGamesPlayed |
| `home_gf_per_game_season` | Goals for per game from standings |
| `away_gf_per_game_season` | Same for away team |
| `home_pointPctg_season` | Point percentage (pts / max possible pts) |
| `away_pointPctg_season` | Same for away team |
| `pointPctg_diff` | `home_pointPctg_season - away_pointPctg_season` |

**Home/road win splits** are particularly valuable because NHL home advantage is real and measurable. A team that is 20-4 at home but 8-18 on the road has a very different home-game profile than a team with a balanced 14-14 / 14-14 split.

**Streak features:**

| Feature | Description |
|---|---|
| `home_win_streak` | Signed streak going into the game (positive = win streak, negative = losing streak) |
| `away_win_streak` | Same for away team |

**Leakage prevention:** Streaks use the prior game's streak state, not the current game's outcome. The current game's result contributes to the streak seen in the *next* game's features.

**Total features from this step: 9** (plus 2 streak features = **11 from standings**)

---

## Step 5 — Rest Days

Fatigue and scheduling context matter in the NHL, especially on back-to-back games or for goalies playing their third start in four nights.

### Team Rest Days

```
team_rest_days = (current_game_date - last_game_date).days - 1
```

A game the day after the previous one gives 0 rest days (back-to-back). A game two days later gives 1 rest day. Default: 3 days when no prior game exists in the dataset (assumes reasonable rest for season openers).

The calculation is **season-bounded** — it does not count across seasons. A team's last game in April does not subtract from their October rest days.

### Goalie Rest Days

Tracked independently per starter:

```
goalie_rest_days = (current_game_date - last_start_date_for_this_goalie).days - 1
```

Default: 7 days when the goalie has no recorded prior start. This default is higher than the team default because an unknown goalie is more likely to be a fresh recall than a regular starter on short rest.

**Why track goalie rest separately from team rest?** Teams often use a backup goalie midway through a back-to-back, so the starter may have had 2–3 days of rest even when the team had 0. Team rest and goalie rest carry different predictive signals.

| Feature | Description |
|---|---|
| `home_rest_days` | Days since home team's last game |
| `away_rest_days` | Days since away team's last game |
| `home_goalie_rest_days` | Days since home starter's last start |
| `away_goalie_rest_days` | Days since away starter's last start |

**Total features from this step: 4**

---

## Step 6 — Head-to-Head Matchup Stats

Within a season, teams play each other multiple times. Some matchups develop a one-sided character — a team may consistently struggle against a particular opponent's defensive style or goaltender. H2H features capture this.

**Season-bounded:** H2H stats reset each season. Cross-season matchup history is not used because roster and coaching changes make it less reliable.

**Leakage prevention:** All H2H features use `.shift(1)` — they reflect results *before* the current game. The first matchup of the season between two teams has H2H features of zero (no prior history).

| Feature | Calculation |
|---|---|
| `home_h2h_wins` | Cumulative wins by home team against this opponent this season (lagged) |
| `home_h2h_gf` | Expanding mean goals scored by home team against this opponent this season (lagged) |
| `away_h2h_wins` | Same for away team |
| `away_h2h_gf` | Same for away team |
| `home_h2h_wins_diff` | `home_h2h_wins - away_h2h_wins` |

**Total features from this step: 5**

---

## Complete Feature Set (71 Features)

| Category | Features | Count |
|---|---|---|
| Home team EWM stats | gf, ga, sog, powerplay_pct, pk_pct, faceoffwin_pct, pims, hits, blockedshots, giveaways, takeaways | 11 |
| Away team EWM stats | (same as home) | 11 |
| Home team L5 rolling | wins_l5, win_pct_l5, powerplays_l5, penalty_kills_l5 | 4 |
| Away team L5 rolling | (same as home) | 4 |
| EWM differentials | home_goal_diff_ewm, home_ga_diff_ewm, home_shot_diff_ewm | 3 |
| Home goalie EWM | save_pct, ga, saves, ev_sa, pp_sa, sh_sa, ev_ga, pp_ga | 8 |
| Away goalie EWM | (same as home) | 8 |
| Team goalie aggregates | home_team_save_pct_ewm, away_team_save_pct_ewm | 2 |
| Season standings | win_pct_season, home_win_pct, away_win_pct, gf_per_game_season, pointPctg_season (× 2 sides), pointPctg_diff | 9 |
| Streak features | home_win_streak, away_win_streak | 2 |
| Rest days | home_rest_days, away_rest_days, home_goalie_rest_days, away_goalie_rest_days | 4 |
| Head-to-head | home_h2h_wins, home_h2h_gf, away_h2h_wins, away_h2h_gf, home_h2h_wins_diff | 5 |
| **Total** | | **71** |

### What Is Explicitly Excluded (and Why)

The following raw game stats are present in the CSV but excluded from model training:

| Excluded Column | Reason |
|---|---|
| `home_gf`, `away_gf` | Direct outcome — would be data leakage |
| `home_sog`, `away_sog` | Game-time stat, not available pre-game |
| `home_goalie_save_pct` (raw) | Game-time stat; EWM version is used instead |
| `faceoffwin_pct` (raw) | Game-time stat; EWM version is used instead |
| All other per-game raw stats | Same reasoning — only EWM transformations are features |

### Data Leakage Prevention

All temporal features are designed so that `feature[i]` uses only information from games `[0..i-1]`:

- **EWM:** `_ewm_with_season_regression` stores `ewm_vals[i] = cur` *before* updating `cur` from `values[i]`.
- **L5 rolling:** groupby transform applies `.shift(1)` before the rolling window.
- **H2H:** `cumsum().shift(1)` and `expanding().mean().shift(1)` in step 6.
- **Standings risk:** `/v1/standings/{game_date}` is called per game date. If the NHL API returns end-of-day standings (after that day's games complete), `win_pct_season` and `pointPctg` features would carry a small leakage. To eliminate this risk, change `add_standings_features()` to fetch with `date - 1`.

---

## Cold Start & Missing Data Handling

| Scenario | What Happens |
|---|---|
| Team's first 1–4 games of season | L5 features are NaN; training drops these rows |
| First game between two teams this season | H2H features are 0 (no prior history) |
| Rookie or unknown goalie | Goalie EWM seeded with season-average EWM (neutral prior) |
| No prior team game in dataset (first game ever) | Team EWM seeded with league average |
| Missing standings data | Features derived from last available standings date |
| Unknown goalie rest | Defaults to 7 days |
| Unknown team rest | Defaults to 3 days |

---

## Prediction Day Feature Assembly

When [predict_games.py](scripts/predict_games.py) runs, it must construct a feature vector for each of today's games without any in-game data. The process mirrors training exactly:

1. **Team EWM:** Load each team's most recent EWM values from the historical CSV, then apply one more EWM update step using their last game's raw stats.
2. **Goalie EWM:** Find the identified starter's last recorded start, apply one EWM update.
3. **Season standings:** Fetch live from the API using today's date.
4. **Rest days:** Compute from today's date minus the last game date in the historical data.
5. **L5 rolling stats:** Pull the last 5 games from the historical data within the current season.
6. **H2H:** Aggregate all prior games this season between these two teams.

### Starter Goalie Detection (3-Tier Strategy)

Lineup availability changes depending on how far in advance you're predicting.

**Tier 1 — Live game:** Parse the boxscore API. Use the goalie with highest time on ice. Most accurate.

**Tier 2 — Pre-game (lineup submitted):** Read `rosterSpots` from the API. Find the goalie flagged with `startingLineup=True`. Available 1–2 hours before puck drop.

**Tier 3 — Future game (lineup not announced):** Fall back to the most-started goalie for that team in the current season prior to today. This is a probabilistic best guess — the most-used starter is most likely to start again. Unavoidable uncertainty at this tier.

---

## Design Decisions Summary

| Decision | Choice | Why |
|---|---|---|
| Temporal smoothing method | EWM (alpha=0.3) | Captures trend decay; recent games matter more than old ones without complete forgetfulness |
| Season boundary handling | 70/30 regression-to-mean | Preserves performance continuity while accounting for roster changes |
| L5 window size | 5 games | Captures recent momentum without being too noisy (3 games) or too slow (10 games) |
| Goalie unknown prior | Season-average EWM | Neutral, evidence-based prior — avoids penalizing unknowns |
| H2H scope | Current season only | Cross-season rosters differ too much; stale matchup history adds noise |
| Target variable | Binary home win | Interpretable, directly actionable; regression (goal differential) adds complexity for marginal gain |
| Data leakage prevention | `.shift(1)` on all cumulative stats; EWM updated after, not during, each game | Ensures features represent only information available before the game |

---

## Database Integration

### How the ML Pipeline Reads and Writes the Database

The ML scripts interact with the same SQLite database (`main/nhl_predictions.db`) used by the Flask backend. There are two distinct phases of database interaction: the one-time historical data load and the daily prediction run.

---

### Phase 1 — Historical Data Load (`migrate_csv_to_db.py`)

`main.py` generates a flat CSV (`scripts/generated/data/nhl_data.csv`) containing every historical game with all 71 engineered features. This is a one-time operation that does not touch the database.

After the CSV is generated, `main/backend/migrate_csv_to_db.py` loads it into the `nhl_game_data` table:

```
main.py → nhl_data.csv → migrate_csv_to_db.py → nhl_game_data table
```

The `nhl_game_data` table has one row per historical game and one column per engineered feature. It is the source `predict_games.py` queries to compute rolling stats, goalie EWMs, rest days, and head-to-head records for today's feature vectors.

**Why a database instead of reading the CSV directly?**

- Faster startup for the prediction subprocess (indexed SQL queries vs. full CSV parse)
- Enables future querying across games without loading everything into memory
- `predict_games.py` tries the DB first and falls back to the CSV if the table is unavailable

---

### Phase 2 — Daily Predictions (`predict_games.py`)

`predict_games.py` is invoked as a subprocess by the Flask backend on every `/api/predict` call. It receives a JSON list of game IDs to predict and writes three things to the database:

#### Reads

```sql
-- Load historical feature data
SELECT * FROM nhl_game_data
WHERE season = :season
ORDER BY game_date
```

This dataset drives all feature computation: EWM lookups, L5 rolling windows, goalie rest days, head-to-head records.

```sql
-- Check if goalie has changed since last prediction
SELECT goalie_name, detection_tier
FROM goalie_states
WHERE game_id = :game_id AND side = :side
```

If the stored goalie name differs from the currently identified starter, or the detection tier has upgraded (e.g. `FUT` → `PRE`), the game is marked for reprediction. Games where the goalie is unchanged are skipped entirely.

#### Writes

**`predictions` table** — one INSERT per game that needed a prediction:

```python
INSERT INTO predictions (
    game_id, predicted_at,
    home_goalie, away_goalie, home_goalie_tier, away_goalie_tier,
    pred_home_win, prob_home_win, prob_away_win, confidence,
    model_version
) VALUES (...)
```

Multiple rows per game are expected. The Flask backend always queries for `MAX(id)` per `game_id` to get the latest prediction. The history of all rows is preserved and exposed via the `/api/predictions/history` endpoint so the frontend can show goalie change diffs.

**`goalie_states` table** — one UPSERT per side per game:

```python
INSERT INTO goalie_states (game_id, side, goalie_name, detection_tier, recorded_at)
VALUES (...)
ON CONFLICT(game_id, side) DO UPDATE SET
    goalie_name    = excluded.goalie_name,
    detection_tier = excluded.detection_tier,
    recorded_at    = excluded.recorded_at
```

This is what the next run will compare against to detect changes. The record is always overwritten with the current run's goalie name and tier.

**`games` table** — the Flask backend writes to this before invoking the subprocess (via `upsert_game()`). `predict_games.py` may also insert a games row if one doesn't exist yet for a given game ID, but this is a safety fallback.

---

### Goalie Change Detection — Full Logic

```
For each game_id in the input list:
    1. Identify current starter (3-tier: LIVE > PRE > FUT)
    2. SELECT stored goalie from goalie_states WHERE game_id AND side
    3. Compare:
         changed = (no stored record)
                OR (stored_name != current_name)
                OR (stored_tier == 'FUT' AND current_tier IN ('PRE', 'LIVE'))
    4. If changed → build feature vector → run inference → INSERT predictions
       If unchanged → skip (no new prediction row)
    5. UPSERT goalie_states with current name + tier
```

The tier upgrade condition (`FUT` → `PRE`) is intentional: even if the same goalie is confirmed by lineup submission, the prediction is re-run because the model may produce a slightly different confidence with a confirmed starter vs. a probabilistic guess.

---

### Full Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ ONE-TIME: Historical Data Collection                        │
├─────────────────────────────────────────────────────────────┤
│ ml_dev/scripts/main.py                                      │
│   └── Fetch 3,877+ games from NHL API (Seasons 2022–2025)  │
│   └── Engineer 71 features per game                        │
│   └── Write: scripts/generated/data/nhl_data.csv           │
│                                                              │
│ main/backend/migrate_csv_to_db.py                           │
│   └── Read: nhl_data.csv                                   │
│   └── Write: nhl_game_data table                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ ONE-TIME: Model Training                                    │
├─────────────────────────────────────────────────────────────┤
│ ml_dev/notebooks/ts_predict.ipynb                           │
│   └── Load: nhl_data.csv                                   │
│   └── Train: Random Forest (300 trees, 71 features)        │
│   └── Validate: season-expanding cross-validation          │
│   └── Save: notebooks/models/nhl_rf_model.pkl              │
│   └── Save: notebooks/models/feature_names.pkl             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ DAILY: Prediction Pipeline                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  React Frontend                                             │
│    └── User clicks "Generate Predictions"                   │
│                                                              │
│  Flask Backend (app.py: GET /api/predict)                   │
│    ├── Fetch: NHL API schedule                              │
│    ├── Write: games table (upsert all games)                │
│    ├── Write: game_results table (for FINAL/OFF games)      │
│    └── Subprocess: predict_games.py [game_ids]             │
│                                                              │
│  predict_games.py                                           │
│    ├── Load: nhl_rf_model.pkl + feature_names.pkl          │
│    ├── Read: nhl_game_data table (historical stats)        │
│    ├── Fetch: NHL API (schedule, boxscores, standings)      │
│    ├── Identify goalies (LIVE > PRE > FUT tier)            │
│    ├── Read: goalie_states table (detect changes)          │
│    ├── For changed games:                                   │
│    │     Build 71-feature vector                           │
│    │     Run model.predict_proba(X)                        │
│    │     Write: predictions table (INSERT)                  │
│    └── Write: goalie_states table (UPSERT)                 │
│                                                              │
│  Flask Backend (continued)                                  │
│    ├── Read: predictions table (MAX(id) per game_id)       │
│    └── Return JSON to frontend                              │
│                                                              │
│  React Frontend                                             │
│    ├── Fetch: /api/predictions/history (all rows)          │
│    ├── Fetch: /api/predictions/accuracy                    │
│    └── Render: GameCard per game                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Database tables written per day:
  games          ← upserted every /api/predict call
  game_results   ← inserted once per completed game
  predictions    ← inserted once per prediction run per changed game
  goalie_states  ← upserted once per prediction run per game side
```
