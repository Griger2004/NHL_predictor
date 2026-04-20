"""
NHL Game Predictor - Today's Games
Fetches today's NHL games and predicts outcomes using trained Random Forest model.
"""

import os
import pickle
import pandas as pd
import numpy as np
import asyncio
import warnings
warnings.filterwarnings('ignore')

from api.client import ApiClient
from config import API_BASE_URL, TIMEOUT, MAX_CONCURRENT_REQUESTS, RETRIES
from utils.helpers import extract_name

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_FILE = "../notebooks/models/nhl_rf_model.pkl"
FEATURES_FILE = "../notebooks/models/feature_names.pkl"
HISTORICAL_DATA_FILE = "generated/data/nhl_data.csv"

EWM_ALPHA = 0.3

# Feature columns (must match training exactly)
HOME_TEAM_L5_COLS = [
    'home_gf_ewm', 'home_ga_ewm', 'home_sog_ewm',
    'home_wins_l5', 'home_win_pct_l5', 'home_powerplay_pct_ewm',
    'home_penalty_kill_pct_ewm', 'home_powerplays_l5', 'home_penalty_kills_l5',
    'home_faceoffwin_pct_ewm', 'home_pims_ewm', 'home_hits_ewm',
    'home_blockedshots_ewm', 'home_giveaways_ewm', 'home_takeaways_ewm',
]

AWAY_TEAM_L5_COLS = [
    'away_gf_ewm', 'away_ga_ewm', 'away_sog_ewm',
    'away_wins_l5', 'away_win_pct_l5', 'away_powerplay_pct_ewm',
    'away_penalty_kill_pct_ewm', 'away_powerplays_l5', 'away_penalty_kills_l5',
    'away_faceoffwin_pct_ewm', 'away_pims_ewm', 'away_hits_ewm',
    'away_blockedshots_ewm', 'away_giveaways_ewm', 'away_takeaways_ewm',
]

GOALIE_L5_COLS = [
    'home_goalie_save_pct_ewm', 'home_goalie_ga_ewm', 'home_goalie_saves_ewm',
    'home_goalie_ev_sa_ewm', 'home_goalie_pp_sa_ewm', 'home_goalie_sh_sa_ewm',
    'home_goalie_ev_ga_ewm', 'home_goalie_pp_ga_ewm',
    'away_goalie_save_pct_ewm', 'away_goalie_ga_ewm', 'away_goalie_saves_ewm',
    'away_goalie_ev_sa_ewm', 'away_goalie_pp_sa_ewm', 'away_goalie_sh_sa_ewm',
    'away_goalie_ev_ga_ewm', 'away_goalie_pp_ga_ewm',
]

TEAM_GOALIE_COLS = ['home_team_save_pct_ewm', 'away_team_save_pct_ewm']

SEASON_COLS = [
    'home_win_pct_season', 'away_win_pct_season',
    'home_home_win_pct', 'away_away_win_pct',
    'home_gf_per_game_season', 'away_gf_per_game_season',
    'home_pointPctg_season', 'away_pointPctg_season', 'pointPctg_diff',
]

DIFF_COLS = ['home_goal_diff_ewm', 'home_ga_diff_ewm', 'home_shot_diff_ewm']

STREAKS_AND_REST = [
    'home_win_streak', 'away_win_streak',
    'home_rest_days', 'away_rest_days',
    'home_goalie_rest_days', 'away_goalie_rest_days',
]

HEAD_TO_HEAD = [
    'home_h2h_wins', 'home_h2h_gf', 'away_h2h_wins',
    'away_h2h_gf', 'home_h2h_wins_diff',
]

# =============================================================================
# GOALIE SELECTION
# =============================================================================

def _goalie_by_toi(goalies):
    """Pick goalie with highest TOI — works post-game and when lineups are locked."""
    if not goalies:
        return None

    def toi_secs(toi):
        try:
            m, s = (toi or "0:00").split(":")
            return int(m) * 60 + int(s)
        except ValueError:
            return 0

    best = max(goalies, key=lambda g: toi_secs(g.get("toi", "0:00")))
    name = extract_name(best.get("name", {}))
    return name or None


def _goalie_by_games_played(players):
    """
    Pick goalie with most games played this season.
    Used when landing-page `players` list is available.
    The workhorse starter almost always has the most GP.
    """
    if not players:
        return None
    best = max(players, key=lambda p: p.get("gamesPlayed", 0))
    name = extract_name(best.get("name", {}))
    return name or None


def get_season_starter(df, team_abbrev, season, current_date):
    """
    Fall-back: count starts per goalie this season in historical data
    and return the one who has started the most games.
    """
    mask = (
        ((df['home_team_abbrev'] == team_abbrev) | (df['away_team_abbrev'] == team_abbrev)) &
        (df['season'] == season) &
        (df['date'] < pd.Timestamp(current_date))
    )
    team_games = df[mask]

    if team_games.empty:
        return None

    from collections import Counter
    starters = []
    for _, g in team_games.iterrows():
        is_home = g['home_team_abbrev'] == team_abbrev
        name = g['home_goalie_starter'] if is_home else g['away_goalie_starter']
        if name and str(name) not in ("", "nan"):
            starters.append(name)

    if not starters:
        return None

    return Counter(starters).most_common(1)[0][0]


async def get_todays_goalies(games, df=None):
    """
    Fetch starting goalies for today's games using three-tier logic:
    1. Boxscore — works once lineups are locked (~90 min pre-game) or post-game
    2. Landing `players` sorted by gamesPlayed — workhorse goalie, better than 'leaders'
    3. Historical starts count — most-used goalie this season from nhl_data.csv
    """
    print("\n" + "=" * 60)
    print("FETCHING TODAY'S STARTING GOALIES")
    print("=" * 60)

    goalies_dict = {}

    async with ApiClient(API_BASE_URL, TIMEOUT, MAX_CONCURRENT_REQUESTS, RETRIES) as client:
        boxscore_results = await asyncio.gather(*[
            client.get_json(f"/v1/gamecenter/{g['game_id']}/boxscore") for g in games
        ])
        landing_results = await asyncio.gather(*[
            client.get_json(f"/v1/gamecenter/{g['game_id']}/landing") for g in games
        ])

    for i, game in enumerate(games):
        gid = game['game_id']
        home_goalie = away_goalie = None
        home_source = away_source = "unknown"

        # --- Tier 1: Boxscore (TOI-based, works when lineups submitted) ---
        boxscore = boxscore_results[i]
        if boxscore:
            ps = boxscore.get("playerByGameStats", {})
            home_g = ps.get("homeTeam", {}).get("goalies", [])
            away_g = ps.get("awayTeam", {}).get("goalies", [])
            if home_g:
                home_goalie = _goalie_by_toi(home_g)
                home_source = "boxscore"
            if away_g:
                away_goalie = _goalie_by_toi(away_g)
                away_source = "boxscore"

        # --- Tier 2: Landing players sorted by gamesPlayed (NOT leaders) ---
        if not home_goalie or not away_goalie:
            landing = landing_results[i]
            if landing:
                gc = landing.get("matchup", {}).get("goalieComparison", {})
                if not home_goalie:
                    players = gc.get("homeTeam", {}).get("players", [])
                    home_goalie = _goalie_by_games_played(players)
                    if home_goalie:
                        home_source = "landing (GP)"
                if not away_goalie:
                    players = gc.get("awayTeam", {}).get("players", [])
                    away_goalie = _goalie_by_games_played(players)
                    if away_goalie:
                        away_source = "landing (GP)"

        # --- Tier 3: Historical starts count ---
        if df is not None and (not home_goalie or not away_goalie):
            season = game['season']
            current_date = game['date']
            if not home_goalie:
                home_goalie = get_season_starter(df, game['home_team'], season, current_date)
                if home_goalie:
                    home_source = "historical starts"
            if not away_goalie:
                away_goalie = get_season_starter(df, game['away_team'], season, current_date)
                if away_goalie:
                    away_source = "historical starts"

        goalies_dict[gid] = {
            "home_goalie": home_goalie or "Unknown",
            "away_goalie": away_goalie or "Unknown",
        }

        print(f"  {game['away_team']} @ {game['home_team']}")
        print(f"    Away: {goalies_dict[gid]['away_goalie']} [{away_source}]"
              f"  |  Home: {goalies_dict[gid]['home_goalie']} [{home_source}]")

    return goalies_dict


# =============================================================================
# FEATURE COMPUTATION — EWM UPDATE APPROACH
# Training EWM: stored[N] = EWM(games 0..N-1).
# For today's game N+1 we need EWM(0..N):
#   new_ewm = alpha * actual[N] + (1-alpha) * stored[N]
# =============================================================================

_EWM_RAW_COLS = [
    'gf', 'ga', 'sog', 'powerplay_pct', 'pk_pct',
    'faceoffwin_pct', 'pims', 'hits', 'blockedshots', 'giveaways', 'takeaways',
]

_GOALIE_STAT_MAP = {
    'save_pct': 'goalie_save_pct',
    'ga':       'goalie_ga',
    'saves':    'goalie_saves',
    'ev_sa':    'goalie_evenStrengthShotsAgainst',
    'pp_sa':    'goalie_powerPlayShotsAgainst',
    'sh_sa':    'goalie_shorthandedShotsAgainst',
    'ev_ga':    'goalie_evenStrengthGoalsAgainst',
    'pp_ga':    'goalie_powerPlayGoalsAgainst',
}


def _ewm_step(actual, stored, alpha=EWM_ALPHA):
    return round(alpha * float(actual or 0) + (1 - alpha) * float(stored or 0), 3)


def get_team_ewm_stats(df, team_abbrev, current_date):
    """
    Returns dict with _ewm and _l5 keys for this team prior to current_date.
    Uses one-step EWM update from last historical game.
    """
    mask = (
        ((df['home_team_abbrev'] == team_abbrev) | (df['away_team_abbrev'] == team_abbrev)) &
        (df['date'] < pd.Timestamp(current_date))
    )
    team_games = df[mask].sort_values('date', ascending=False)

    if team_games.empty:
        return {}

    last = team_games.iloc[0]
    pfx = 'home' if last['home_team_abbrev'] == team_abbrev else 'away'

    result = {}

    # One-step EWM update for each stat
    for col in _EWM_RAW_COLS:
        result[f'{col}_ewm'] = _ewm_step(
            last.get(f'{pfx}_{col}'),
            last.get(f'{pfx}_{col}_ewm'),
        )

    # Rolling L5 counts (simple sum/mean, matches training)
    last_5 = team_games.head(5)
    wins, pps, pks = [], [], []
    for _, g in last_5.iterrows():
        h = g['home_team_abbrev'] == team_abbrev
        p = 'home' if h else 'away'
        wins.append(int(g['home_win'] if h else 1 - g['home_win']))
        pps.append(float(g.get(f'{p}_powerplays') or 0))
        pks.append(float(g.get(f'{p}_pk') or 0))

    result['wins_l5'] = sum(wins)
    result['win_pct_l5'] = round(np.mean(wins), 3) if wins else 0.0
    result['powerplays_l5'] = round(np.mean(pps), 3) if pps else 0.0
    result['penalty_kills_l5'] = round(np.mean(pks), 3) if pks else 0.0

    return result


def get_goalie_ewm_stats(df, goalie_name, current_date):
    """
    Returns dict with goalie EWM stats prior to current_date.
    Groups by goalie only (matches training groupby logic).
    """
    if not goalie_name or goalie_name == "Unknown":
        return {}

    mask = (
        ((df['home_goalie_starter'] == goalie_name) | (df['away_goalie_starter'] == goalie_name)) &
        (df['date'] < pd.Timestamp(current_date))
    )
    goalie_games = df[mask].sort_values('date', ascending=False)

    if goalie_games.empty:
        return {}

    last = goalie_games.iloc[0]
    pfx = 'home' if last['home_goalie_starter'] == goalie_name else 'away'

    result = {}
    for stat_key, raw_col in _GOALIE_STAT_MAP.items():
        result[f'{stat_key}_ewm'] = _ewm_step(
            last.get(f'{pfx}_{raw_col}'),
            last.get(f'{pfx}_goalie_{stat_key}_ewm'),
        )

    return result


def get_team_save_pct_ewm(df, team_abbrev, current_date):
    """Returns updated team save pct EWM for today."""
    mask = (
        ((df['home_team_abbrev'] == team_abbrev) | (df['away_team_abbrev'] == team_abbrev)) &
        (df['date'] < pd.Timestamp(current_date))
    )
    team_games = df[mask].sort_values('date', ascending=False)

    if team_games.empty:
        return 0.0

    last = team_games.iloc[0]
    pfx = 'home' if last['home_team_abbrev'] == team_abbrev else 'away'
    return _ewm_step(last.get(f'{pfx}_save_pct'), last.get(f'{pfx}_team_save_pct_ewm'))


def get_rest_days(df, team_abbrev, current_date):
    """Days since last game (no season filter — matches training)."""
    mask = (
        ((df['home_team_abbrev'] == team_abbrev) | (df['away_team_abbrev'] == team_abbrev)) &
        (df['date'] < pd.Timestamp(current_date))
    )
    team_games = df[mask].sort_values('date', ascending=False)

    if team_games.empty:
        return 7

    return max((pd.Timestamp(current_date) - team_games.iloc[0]['date']).days - 1, 0)


def get_goalie_rest_days(df, goalie_name, current_date):
    """Days since goalie's last start (no season filter)."""
    if not goalie_name or goalie_name == "Unknown":
        return 7

    mask = (
        ((df['home_goalie_starter'] == goalie_name) | (df['away_goalie_starter'] == goalie_name)) &
        (df['date'] < pd.Timestamp(current_date))
    )
    goalie_games = df[mask].sort_values('date', ascending=False)

    if goalie_games.empty:
        return 7

    return max((pd.Timestamp(current_date) - goalie_games.iloc[0]['date']).days - 1, 0)


def get_season_stats(standings_data, team_abbrev):
    """Extract season-to-date stats from standings data."""
    if not standings_data:
        return {}

    for team in standings_data.get("standings", []):
        if team.get("teamAbbrev", {}).get("default") == team_abbrev:
            total = max(team.get("gamesPlayed", 1), 1)
            home_gp = max(team.get("homeGamesPlayed", 1), 1)
            road_gp = max(team.get("roadGamesPlayed", 1), 1)
            return {
                'win_pct_season':    (team.get("homeWins", 0) + team.get("roadWins", 0)) / total,
                'home_win_pct':      team.get("homeWins", 0) / home_gp,
                'away_win_pct':      team.get("roadWins", 0) / road_gp,
                'gf_per_game_season': team.get("goalsForPctg", 0),
                'pointPctg_season':  team.get("pointPctg", 0),
                'win_streak':        team.get("streakCount", 0) if team.get("streakCode") == "W" else 0,
            }

    return {}


def get_h2h_stats(df, home_team, away_team, current_date, season):
    """Head-to-head stats within the same season (matches training groupby)."""
    mask = (
        (((df['home_team_abbrev'] == home_team) & (df['away_team_abbrev'] == away_team)) |
         ((df['home_team_abbrev'] == away_team) & (df['away_team_abbrev'] == home_team))) &
        (df['date'] < pd.Timestamp(current_date)) &
        (df['season'] == season)
    )
    h2h_games = df[mask]

    if h2h_games.empty:
        return {'home_h2h_wins': 0, 'home_h2h_gf': 0,
                'away_h2h_wins': 0, 'away_h2h_gf': 0, 'home_h2h_wins_diff': 0}

    home_wins, away_wins = 0, 0
    home_gf_list, away_gf_list = [], []

    for _, g in h2h_games.iterrows():
        if g['home_team_abbrev'] == home_team:
            if g['home_win'] == 1:
                home_wins += 1
            else:
                away_wins += 1
            home_gf_list.append(g['home_gf'])
            away_gf_list.append(g['away_gf'])
        else:
            if g['home_win'] == 1:
                away_wins += 1
            else:
                home_wins += 1
            home_gf_list.append(g['away_gf'])
            away_gf_list.append(g['home_gf'])

    return {
        'home_h2h_wins':      home_wins,
        'home_h2h_gf':        round(np.mean(home_gf_list), 3) if home_gf_list else 0,
        'away_h2h_wins':      away_wins,
        'away_h2h_gf':        round(np.mean(away_gf_list), 3) if away_gf_list else 0,
        'home_h2h_wins_diff': home_wins - away_wins,
    }


# =============================================================================
# LOAD MODEL AND HISTORICAL DATA
# =============================================================================

def load_model_and_features():
    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    for path in (MODEL_FILE, FEATURES_FILE):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_FILE, 'rb') as f:
        feature_names = pickle.load(f)

    print(f"Model: {type(model).__name__}  |  Features: {len(feature_names)}")
    return model, feature_names


def load_historical_data():
    print("\n" + "=" * 60)
    print("LOADING HISTORICAL DATA")
    print("=" * 60)

    if not os.path.exists(HISTORICAL_DATA_FILE):
        raise FileNotFoundError(
            f"Historical data not found: {HISTORICAL_DATA_FILE}\n"
            "Run main.py first to generate it."
        )

    df = pd.read_csv(HISTORICAL_DATA_FILE, parse_dates=["date"])
    print(f"Loaded {len(df)} games  ({df['date'].min().date()} – {df['date'].max().date()})")
    return df


# =============================================================================
# FETCH TODAY'S GAMES
# =============================================================================

async def get_todays_games(date_str=None):
    print("\n" + "=" * 60)
    print("FETCHING TODAY'S GAMES")
    print("=" * 60)

    if date_str is None:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"Date: {date_str}")

    async with ApiClient(API_BASE_URL, TIMEOUT, MAX_CONCURRENT_REQUESTS, RETRIES) as client:
        schedule_data = await client.get_json(f"/v1/schedule/{date_str}")
        standings_data = await client.get_json(f"/v1/standings/{date_str}")

    if not schedule_data or "gameWeek" not in schedule_data:
        print("No games found for today")
        return None, None

    today_games = next(
        (day["games"] for day in schedule_data.get("gameWeek", []) if day.get("date") == date_str),
        []
    )

    games = []
    for game in today_games:
        games.append({
            "game_id":        game.get("id"),
            "date":           date_str,
            "season":         game.get("season"),
            "home_team":      game.get("homeTeam", {}).get("abbrev"),
            "away_team":      game.get("awayTeam", {}).get("abbrev"),
            "home_team_name": game.get("homeTeam", {}).get("placeName", {}).get("default", ""),
            "away_team_name": game.get("awayTeam", {}).get("placeName", {}).get("default", ""),
            "game_time":      game.get("startTimeUTC"),
        })

    print(f"Found {len(games)} games")
    for i, g in enumerate(games, 1):
        print(f"  {i}. {g['away_team_name']} @ {g['home_team_name']}  {g['game_time']}")

    return games, standings_data


# =============================================================================
# BUILD FEATURE ROW
# =============================================================================

def build_feature_row(game, df, standings_data, goalies_dict):
    """Assemble the 71-feature vector that matches the trained model."""
    current_date = pd.Timestamp(game['date'])
    season = game['season']
    home_team = game['home_team']
    away_team = game['away_team']
    gid = game['game_id']

    print(f"\n  Building features: {away_team} @ {home_team}")

    goalie_info = goalies_dict.get(gid, {})
    home_goalie = goalie_info.get('home_goalie', 'Unknown')
    away_goalie = goalie_info.get('away_goalie', 'Unknown')

    home_ewm = get_team_ewm_stats(df, home_team, current_date)
    away_ewm = get_team_ewm_stats(df, away_team, current_date)

    home_season = get_season_stats(standings_data, home_team)
    away_season = get_season_stats(standings_data, away_team)

    h2h = get_h2h_stats(df, home_team, away_team, current_date, season)

    home_goalie_ewm = get_goalie_ewm_stats(df, home_goalie, current_date)
    away_goalie_ewm = get_goalie_ewm_stats(df, away_goalie, current_date)

    features = {}

    # --- Team EWM stats (11 cols × 2 teams) ---
    for col in _EWM_RAW_COLS:
        features[f'home_{col}_ewm'] = home_ewm.get(f'{col}_ewm', 0)
        features[f'away_{col}_ewm'] = away_ewm.get(f'{col}_ewm', 0)

    # --- L5 rolling stats (4 cols × 2 teams) ---
    features['home_wins_l5']         = home_ewm.get('wins_l5', 0)
    features['away_wins_l5']         = away_ewm.get('wins_l5', 0)
    features['home_win_pct_l5']      = home_ewm.get('win_pct_l5', 0)
    features['away_win_pct_l5']      = away_ewm.get('win_pct_l5', 0)
    features['home_powerplays_l5']   = home_ewm.get('powerplays_l5', 0)
    features['away_powerplays_l5']   = away_ewm.get('powerplays_l5', 0)
    features['home_penalty_kills_l5'] = home_ewm.get('penalty_kills_l5', 0)
    features['away_penalty_kills_l5'] = away_ewm.get('penalty_kills_l5', 0)

    # --- Differentials (derived from EWM, not L5 mean) ---
    features['home_goal_diff_ewm'] = round(features['home_gf_ewm'] - features['away_gf_ewm'], 3)
    features['home_ga_diff_ewm']   = round(features['home_ga_ewm'] - features['away_ga_ewm'], 3)
    features['home_shot_diff_ewm'] = round(features['home_sog_ewm'] - features['away_sog_ewm'], 3)

    # --- Season stats ---
    features['home_win_pct_season']    = home_season.get('win_pct_season', 0)
    features['away_win_pct_season']    = away_season.get('win_pct_season', 0)
    features['home_home_win_pct']      = home_season.get('home_win_pct', 0)
    features['away_away_win_pct']      = away_season.get('away_win_pct', 0)
    features['home_gf_per_game_season'] = home_season.get('gf_per_game_season', 0)
    features['away_gf_per_game_season'] = away_season.get('gf_per_game_season', 0)
    features['home_pointPctg_season']  = home_season.get('pointPctg_season', 0)
    features['away_pointPctg_season']  = away_season.get('pointPctg_season', 0)
    features['pointPctg_diff']         = (
        home_season.get('pointPctg_season', 0) - away_season.get('pointPctg_season', 0)
    )

    # --- Streaks & rest ---
    features['home_win_streak']      = home_season.get('win_streak', 0)
    features['away_win_streak']      = away_season.get('win_streak', 0)
    features['home_rest_days']       = get_rest_days(df, home_team, current_date)
    features['away_rest_days']       = get_rest_days(df, away_team, current_date)
    features['home_goalie_rest_days'] = get_goalie_rest_days(df, home_goalie, current_date)
    features['away_goalie_rest_days'] = get_goalie_rest_days(df, away_goalie, current_date)

    # --- Head-to-head ---
    features.update(h2h)

    # --- Goalie EWM stats (8 × 2) ---
    for stat in _GOALIE_STAT_MAP:
        features[f'home_goalie_{stat}_ewm'] = home_goalie_ewm.get(f'{stat}_ewm', 0)
        features[f'away_goalie_{stat}_ewm'] = away_goalie_ewm.get(f'{stat}_ewm', 0)

    # --- Team save pct EWM ---
    features['home_team_save_pct_ewm'] = get_team_save_pct_ewm(df, home_team, current_date)
    features['away_team_save_pct_ewm'] = get_team_save_pct_ewm(df, away_team, current_date)

    return features


# =============================================================================
# PREDICTIONS
# =============================================================================

def make_predictions(model, feature_names, games, df, standings_data, goalies_dict, threshold=0.5):
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)

    predictions = []

    for game in games:
        features = build_feature_row(game, df, standings_data, goalies_dict)

        X = np.array([features.get(f, 0) for f in feature_names]).reshape(1, -1)

        prob_home_win = model.predict_proba(X)[0][1]
        pred_home_win = int(prob_home_win >= threshold)

        predictions.append({
            'game_id':        game['game_id'],
            'date':           game['date'],
            'time':           game['game_time'],
            'away_team':      game['away_team'],
            'home_team':      game['home_team'],
            'away_team_name': game['away_team_name'],
            'home_team_name': game['home_team_name'],
            'away_goalie':    goalies_dict[game['game_id']]['away_goalie'],
            'home_goalie':    goalies_dict[game['game_id']]['home_goalie'],
            'pred_home_win':  pred_home_win,
            'prob_home_win':  prob_home_win,
            'prob_away_win':  1 - prob_home_win,
            'confidence':     max(prob_home_win, 1 - prob_home_win),
        })

    return predictions


def display_predictions(predictions, threshold=0.5):
    print("\n" + "=" * 60)
    print("TODAY'S NHL PREDICTIONS")
    print("=" * 60)
    print(f"Threshold: {threshold:.2f}  |  Games: {len(predictions)}\n")

    for i, pred in enumerate(predictions, 1):
        winner = pred['home_team_name'] if pred['pred_home_win'] == 1 else pred['away_team_name']
        prob   = pred['prob_home_win']  if pred['pred_home_win'] == 1 else pred['prob_away_win']

        print(f"Game {i}: {pred['away_team']} ({pred['away_goalie']}) @ "
              f"{pred['home_team']} ({pred['home_goalie']})")
        print(f"  Predicted Winner: {winner}  ({prob:.1%} confidence)")
        print(f"  Home: {pred['prob_home_win']:.1%}  |  Away: {pred['prob_away_win']:.1%}")
        print()

    df_pred = pd.DataFrame(predictions)
    output_file = f"predictions_{predictions[0]['date']}.csv"
    df_pred.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

async def main(date_str=None, threshold=0.5):
    print("\n" + "=" * 60)
    print("NHL GAME PREDICTOR")
    print("=" * 60)

    model, feature_names = load_model_and_features()
    df = load_historical_data()

    games, standings_data = await get_todays_games(date_str)
    if not games:
        print("\nNo games to predict today!")
        return

    goalies_dict = await get_todays_goalies(games, df=df)
    predictions = make_predictions(model, feature_names, games, df, standings_data, goalies_dict, threshold)
    display_predictions(predictions, threshold)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main(date_str=None, threshold=0.5))
