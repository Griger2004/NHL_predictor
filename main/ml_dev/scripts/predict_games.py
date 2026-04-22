"""
NHL Game Predictor - Today's Games
Fetches today's NHL games and predicts outcomes using trained Random Forest model
"""

import os
import pickle
import pandas as pd
import numpy as np
import requests
import asyncio
from aiohttp import ClientTimeout
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from api.client import ApiClient

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = "https://api-web.nhle.com"
TIMEOUT = ClientTimeout(total=15)
MAX_CONCURRENT_REQUESTS = 9
RETRIES = 3
ROLLING_N = 5
EWM_ALPHA = 0.3
EWM_CARRY_WEIGHT = 0.7

# Maps EWM feature suffix → raw stat column suffix in nhl_data.csv
TEAM_EWM_MAP = {
    'gf_ewm': 'gf',
    'ga_ewm': 'ga',
    'sog_ewm': 'sog',
    'powerplay_pct_ewm': 'powerplay_pct',
    'pk_pct_ewm': 'pk_pct',
    'faceoffwin_pct_ewm': 'faceoffwin_pct',
    'pims_ewm': 'pims',
    'hits_ewm': 'hits',
    'blockedshots_ewm': 'blockedshots',
    'giveaways_ewm': 'giveaways',
    'takeaways_ewm': 'takeaways',
}

# Maps goalie EWM suffix → raw stat column suffix in nhl_data.csv
GOALIE_EWM_TO_RAW = {
    'save_pct_ewm': 'save_pct',
    'ga_ewm': 'ga',
    'saves_ewm': 'saves',
    'ev_sa_ewm': 'evenStrengthShotsAgainst',
    'pp_sa_ewm': 'powerPlayShotsAgainst',
    'sh_sa_ewm': 'shorthandedShotsAgainst',
    'ev_ga_ewm': 'evenStrengthGoalsAgainst',
    'pp_ga_ewm': 'powerPlayGoalsAgainst',
}

# Model files
MODEL_FILE = "../notebooks/models/nhl_rf_model.pkl"
FEATURES_FILE = "../notebooks/models/feature_names.pkl"

# Historical data file (needed for computing rolling stats)
HISTORICAL_DATA_FILE = "generated/data/nhl_data.csv"

# Feature definitions (must match training)
HOME_TEAM_L5_COLS = [
    'home_gf_ewm', 'home_ga_ewm', 'home_sog_ewm',
    'home_wins_l5', 'home_win_pct_l5', 'home_powerplay_pct_ewm',
    'home_pk_pct_ewm', 'home_powerplays_l5', 'home_penalty_kills_l5',
    'home_faceoffwin_pct_ewm', 'home_pims_ewm', 'home_hits_ewm',
    'home_blockedshots_ewm', 'home_giveaways_ewm', 'home_takeaways_ewm',
]

AWAY_TEAM_L5_COLS = [
    'away_gf_ewm', 'away_ga_ewm', 'away_sog_ewm',
    'away_wins_l5', 'away_win_pct_l5', 'away_powerplay_pct_ewm',
    'away_pk_pct_ewm', 'away_powerplays_l5', 'away_penalty_kills_l5',
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

TEAM_GOALIE_PERFORMANCE = [
    'home_team_save_pct_ewm', 'away_team_save_pct_ewm',
]

SEASON_COLS = [
    'home_win_pct_season', 'away_win_pct_season',
    'home_home_win_pct', 'away_away_win_pct',
    'home_gf_per_game_season', 'away_gf_per_game_season',
    'home_pointPctg_season', 'away_pointPctg_season', 'pointPctg_diff',
]

DIFF_COLS = [
    'home_goal_diff_ewm', 'home_ga_diff_ewm', 'home_shot_diff_ewm',
]

STREAKS_AND_REST = [
    'home_win_streak', 'away_win_streak',
    'home_rest_days', 'away_rest_days',
    'home_goalie_rest_days', 'away_goalie_rest_days',
]

HEAD_TO_HEAD = [
    'home_h2h_wins', 'home_h2h_gf', 'away_h2h_wins', 
    'away_h2h_gf', 'home_h2h_wins_diff',
]

# --------------------------
# TODO: save hisorical data to a database 
# --------------------------

# =============================================================================
# GOALIE DATA HELPERS
# =============================================================================

def extract_name(obj):
    """Extract name from nested object."""
    if not obj:
        return ""
    if isinstance(obj, dict):
        return obj.get("default", "") if "default" in obj else ""
    return str(obj)

def get_starter_goalie(goalies):
    """Get the starting goalie based on highest TOI (for live games)."""
    if not goalies:
        return ""

    def toi_to_seconds(toi):
        if not toi:
            return 0
        parts = toi.split(":")
        return int(parts[0]) * 60 + int(parts[1])

    starter = max(goalies, key=lambda g: toi_to_seconds(g.get("toi", "00:00")))
    return extract_name(starter.get("name", {}))


def _goalie_name_from_roster_spot(spot):
    """Build full name from a rosterSpots entry (first + last)."""
    first = spot.get("firstName", {})
    last = spot.get("lastName", {})
    first_str = first.get("default", "") if isinstance(first, dict) else str(first)
    last_str = last.get("default", "") if isinstance(last, dict) else str(last)
    return f"{first_str} {last_str}".strip()


def _starter_from_roster_spots(spots):
    """Find starting goalie from rosterSpots (available once lineup is submitted pre-game).
    Trusts startingLineup flag; falls back to the sole dressed goalie if unambiguous."""
    goalies = [s for s in spots if s.get("positionCode") == "G"]
    for g in goalies:
        if g.get("startingLineup"):
            return _goalie_name_from_roster_spot(g)
    if len(goalies) == 1:
        return _goalie_name_from_roster_spot(goalies[0])
    return None


def get_most_starts_goalie(df, team_abbrev, current_date, season):
    """Return the goalie with the most starts for a team this season before current_date.
    Best available proxy when the starting lineup hasn't been announced yet."""
    season_games = df[
        ((df['home_team_abbrev'] == team_abbrev) | (df['away_team_abbrev'] == team_abbrev)) &
        (df['date'] < current_date) &
        (df['season'] == season)
    ]
    home_g = season_games[season_games['home_team_abbrev'] == team_abbrev]['home_goalie_starter']
    away_g = season_games[season_games['away_team_abbrev'] == team_abbrev]['away_goalie_starter']
    all_starts = pd.concat([home_g, away_g]).dropna()
    if all_starts.empty:
        return None
    return all_starts.value_counts().index[0]


async def get_todays_goalies(games, df):
    """Fetch today's starting goalies using a 3-tier approach:
    1. LIVE: highest TOI from playerByGameStats
    2. PRE (lineup submitted): startingLineup flag in rosterSpots
    3. FUT (lineup not announced): most starts this season from historical data
    """
    print("\n" + "="*60)
    print("FETCHING TODAY'S STARTING GOALIES")
    print("="*60)

    goalies_dict = {}

    async with ApiClient(API_BASE_URL, TIMEOUT, MAX_CONCURRENT_REQUESTS, RETRIES) as client:
        boxscore_results = await asyncio.gather(*[
            client.get_json(f"/v1/gamecenter/{game['game_id']}/boxscore")
            for game in games
        ])

    for i, game in enumerate(games):
        gid = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']
        current_date = pd.to_datetime(game['date'])
        season = game['season']

        home_goalie = None
        away_goalie = None

        boxscore = boxscore_results[i]
        if boxscore:
            # Tier 1: LIVE — use highest TOI from in-game stats
            home_stats = boxscore.get("playerByGameStats", {}).get("homeTeam", {}).get("goalies", [])
            away_stats = boxscore.get("playerByGameStats", {}).get("awayTeam", {}).get("goalies", [])
            if home_stats:
                home_goalie = get_starter_goalie(home_stats)
            if away_stats:
                away_goalie = get_starter_goalie(away_stats)

            # Tier 2: PRE — lineup submitted, read rosterSpots with startingLineup flag
            if not home_goalie:
                home_goalie = _starter_from_roster_spots(
                    boxscore.get("homeTeam", {}).get("rosterSpots", [])
                )
            if not away_goalie:
                away_goalie = _starter_from_roster_spots(
                    boxscore.get("awayTeam", {}).get("rosterSpots", [])
                )

        # Tier 3: FUT — lineup not announced, use most-started goalie this season
        if not home_goalie:
            home_goalie = get_most_starts_goalie(df, home_team, current_date, season)
        if not away_goalie:
            away_goalie = get_most_starts_goalie(df, away_team, current_date, season)

        goalies_dict[gid] = {
            "game_id": gid,
            "home_team": home_team,
            "away_team": away_team,
            "home_goalie": home_goalie or "Unknown",
            "away_goalie": away_goalie or "Unknown",
        }

        print(f"  {away_team} @ {home_team}")
        print(f"    Away Goalie: {goalies_dict[gid]['away_goalie']}")
        print(f"    Home Goalie: {goalies_dict[gid]['home_goalie']}")

    return goalies_dict


def get_goalie_ewm_stats(df, goalie_name, current_date, season, team_abbrev):
    """
    Compute EWM stats for a goalie going into today's game.

    Looks up the goalie's last start in historical data, reads the stored EWM,
    and updates it with that game's actual stat. Applies regression-to-mean at
    season boundaries (70% last-season final EWM, 30% prior-season league avg).
    """
    if not goalie_name or goalie_name == "Unknown":
        return {}

    current_date = pd.to_datetime(current_date)

    goalie_games = df[
        (((df['home_team_abbrev'] == team_abbrev) & (df['home_goalie_starter'] == goalie_name)) |
         ((df['away_team_abbrev'] == team_abbrev) & (df['away_goalie_starter'] == goalie_name))) &
        (df['date'] < current_date)
    ].sort_values('date')

    if len(goalie_games) == 0:
        return {}

    last_game = goalie_games.iloc[-1]
    is_home = last_game['home_goalie_starter'] == goalie_name
    prefix = 'home' if is_home else 'away'
    last_game_season = last_game['season']

    result = {}

    for ewm_suffix, raw_suffix in GOALIE_EWM_TO_RAW.items():
        ewm_col = f'{prefix}_goalie_{ewm_suffix}'
        raw_col = f'{prefix}_goalie_{raw_suffix}'

        stored_ewm = last_game.get(ewm_col, np.nan)
        last_stat = last_game.get(raw_col, np.nan)

        if pd.isna(stored_ewm):
            current_ewm = float(last_stat) if not pd.isna(last_stat) else np.nan
        else:
            current_ewm = EWM_ALPHA * float(last_stat) + (1 - EWM_ALPHA) * float(stored_ewm)

        # Season boundary: regression-to-mean
        if last_game_season != season and not pd.isna(current_ewm):
            prev = df[df['season'] == last_game_season]
            league_avg = pd.concat([
                prev[f'home_goalie_{raw_suffix}'].dropna() if f'home_goalie_{raw_suffix}' in prev.columns else pd.Series(dtype=float),
                prev[f'away_goalie_{raw_suffix}'].dropna() if f'away_goalie_{raw_suffix}' in prev.columns else pd.Series(dtype=float),
            ]).mean()
            if not pd.isna(league_avg):
                current_ewm = EWM_CARRY_WEIGHT * current_ewm + (1 - EWM_CARRY_WEIGHT) * float(league_avg)

        result[ewm_suffix] = round(float(current_ewm), 3) if not pd.isna(current_ewm) else 0

    return result


def get_goalie_rest_days(df, goalie_name, current_date, season, team_abbrev):
    """Calculate rest days for a goalie."""
    if not goalie_name or goalie_name == "Unknown":
        return 7
    
    goalie_games = df[
        (((df['home_team_abbrev'] == team_abbrev) & (df['home_goalie_starter'] == goalie_name)) |
         ((df['away_team_abbrev'] == team_abbrev) & (df['away_goalie_starter'] == goalie_name))) &
        (df['date'] < current_date) &
        (df['season'] == season)
    ].sort_values('date', ascending=False)
    
    if len(goalie_games) == 0:
        return 7  # Default if no recent games
    
    last_game_date = goalie_games.iloc[0]['date']
    rest_days = (pd.to_datetime(current_date) - last_game_date).days - 1
    
    return max(rest_days, 0)


def get_team_save_pct_ewm(df, team_abbrev, current_date, season):
    """Compute EWM team save percentage for today's game."""
    current_date = pd.to_datetime(current_date)

    all_team_games = df[
        ((df['home_team_abbrev'] == team_abbrev) | (df['away_team_abbrev'] == team_abbrev)) &
        (df['date'] < current_date)
    ].sort_values('date')

    if len(all_team_games) == 0:
        return 0

    last_game = all_team_games.iloc[-1]
    is_home = last_game['home_team_abbrev'] == team_abbrev
    prefix = 'home' if is_home else 'away'
    last_game_season = last_game['season']

    stored_ewm = last_game.get(f'{prefix}_team_save_pct_ewm', np.nan)
    last_stat = last_game.get(f'{prefix}_save_pct', np.nan)

    if pd.isna(stored_ewm):
        current_ewm = float(last_stat) if not pd.isna(last_stat) else np.nan
    else:
        current_ewm = EWM_ALPHA * float(last_stat) + (1 - EWM_ALPHA) * float(stored_ewm)

    # Season boundary: regression-to-mean
    if last_game_season != season and not pd.isna(current_ewm):
        prev = df[df['season'] == last_game_season]
        league_avg = pd.concat([
            prev['home_save_pct'].dropna() if 'home_save_pct' in prev.columns else pd.Series(dtype=float),
            prev['away_save_pct'].dropna() if 'away_save_pct' in prev.columns else pd.Series(dtype=float),
        ]).mean()
        if not pd.isna(league_avg):
            current_ewm = EWM_CARRY_WEIGHT * current_ewm + (1 - EWM_CARRY_WEIGHT) * float(league_avg)

    return round(float(current_ewm), 3) if not pd.isna(current_ewm) else 0


# =============================================================================
# LOAD MODEL AND HISTORICAL DATA
# =============================================================================

def load_model_and_features():
    """Load trained model and feature names."""
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")
    
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    
    with open(FEATURES_FILE, 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"Model loaded: {type(model).__name__}")
    print(f"Features loaded: {len(feature_names)} features")
    
    return model, feature_names


def load_historical_data():
    """Load historical data for computing rolling stats."""
    print("\n" + "="*60)
    print("LOADING HISTORICAL DATA")
    print("="*60)
    
    if not os.path.exists(HISTORICAL_DATA_FILE):
        raise FileNotFoundError(
            f"Historical data file not found: {HISTORICAL_DATA_FILE}\n"
            "Please run your data scraper first to generate this file."
        )
    
    df = pd.read_csv(HISTORICAL_DATA_FILE, parse_dates=["date"])
    print(f"Loaded {len(df)} historical games")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


# =============================================================================
# FETCH TODAY'S GAMES
# =============================================================================

async def get_todays_games(date_str=None):
    """Fetch all games scheduled for today."""
    print("\n" + "="*60)
    print("FETCHING TODAY'S GAMES")
    print("="*60)
    
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Date: {date_str}")
    
    async with ApiClient(
        API_BASE_URL,
        TIMEOUT,
        MAX_CONCURRENT_REQUESTS,
        RETRIES,
    ) as client:
        schedule_data = await client.get_json(f"/v1/schedule/{date_str}")
        standings_data = await client.get_json(f"/v1/standings/{date_str}")
    
    if not schedule_data or "gameWeek" not in schedule_data:
        print("No games found for today")
        return None, None
    
    games = []

    from datetime import date

    # Extract today's games only out of an entire week schedule
    today_games = next(
        (
            day["games"]
            for day in schedule_data.get("gameWeek", [])
            if day.get("date") == date_str
        ),
        []
    )


    for game in today_games:
        # if game.get("gameState") not in ["FUT", "PRE"]:
        #     continue  # Skip games that already started
        
        games.append({
            "game_id": game.get("id"),
            "date": date_str,
            "season": game.get("season"),
            "home_team": game.get("homeTeam", {}).get("abbrev"),
            "away_team": game.get("awayTeam", {}).get("abbrev"),
            "home_team_name": game.get("homeTeam", {}).get("placeName", {}).get("default", ""),
            "away_team_name": game.get("awayTeam", {}).get("placeName", {}).get("default", ""),
            "game_time": game.get("startTimeUTC"),
        })
    
    print(f"Found {len(games)} games scheduled for today")
    
    for i, game in enumerate(games, 1):
        print(f"  {i}. {game['away_team_name']} @ {game['home_team_name']} - {game['game_time']}")
    
    return games, standings_data

# =============================================================================
# COMPUTE FEATURES FOR TODAY'S GAMES
# =============================================================================

def get_team_ewm_stats(df, team_abbrev, current_date, season):
    """
    Compute EWM and rolling features for a team going into today's game.

    EWM stats are looked up from the team's last historical game and updated
    with that game's actual result. At a season boundary, regression-to-mean
    is applied: 70% last-season final EWM + 30% prior-season league average.

    Rolling L5 features (wins, powerplays, penalty_kills) remain season-bounded
    simple windows to match training.
    """
    current_date = pd.to_datetime(current_date)

    all_team_games = df[
        ((df['home_team_abbrev'] == team_abbrev) | (df['away_team_abbrev'] == team_abbrev)) &
        (df['date'] < current_date)
    ].sort_values('date')

    if len(all_team_games) == 0:
        return {}

    last_game = all_team_games.iloc[-1]
    is_home = last_game['home_team_abbrev'] == team_abbrev
    prefix = 'home' if is_home else 'away'
    last_game_season = last_game['season']

    result = {}

    # EWM features: look up stored EWM, apply one update step
    for ewm_key, raw_key in TEAM_EWM_MAP.items():
        stored_ewm = last_game.get(f'{prefix}_{ewm_key}', np.nan)
        last_stat = last_game.get(f'{prefix}_{raw_key}', np.nan)

        if pd.isna(stored_ewm):
            current_ewm = float(last_stat) if not pd.isna(last_stat) else np.nan
        else:
            current_ewm = EWM_ALPHA * float(last_stat) + (1 - EWM_ALPHA) * float(stored_ewm)

        # Season boundary: regression-to-mean
        if last_game_season != season and not pd.isna(current_ewm):
            prev = df[df['season'] == last_game_season]
            league_avg = pd.concat([
                prev[f'home_{raw_key}'].dropna(),
                prev[f'away_{raw_key}'].dropna()
            ]).mean() if f'home_{raw_key}' in prev.columns else np.nan
            if not pd.isna(league_avg):
                current_ewm = EWM_CARRY_WEIGHT * current_ewm + (1 - EWM_CARRY_WEIGHT) * float(league_avg)

        result[ewm_key] = round(float(current_ewm), 3) if not pd.isna(current_ewm) else 0

    # Rolling L5 features (season-bounded)
    season_games = df[
        ((df['home_team_abbrev'] == team_abbrev) | (df['away_team_abbrev'] == team_abbrev)) &
        (df['date'] < current_date) &
        (df['season'] == season)
    ].sort_values('date', ascending=False).head(5)

    wins, powerplays, penalty_kills = [], [], []
    for _, game in season_games.iterrows():
        is_h = game['home_team_abbrev'] == team_abbrev
        p = 'home' if is_h else 'away'
        wins.append(float(game['home_win']) if is_h else 1 - float(game['home_win']))
        powerplays.append(game.get(f'{p}_powerplays', 0))
        penalty_kills.append(game.get(f'{p}_pk', 0))

    result['wins_l5'] = sum(wins)
    result['win_pct_l5'] = np.mean(wins) if wins else 0
    result['powerplays_l5'] = np.mean(powerplays) if powerplays else 0
    result['penalty_kills_l5'] = np.mean(penalty_kills) if penalty_kills else 0

    return result


def get_season_stats(standings_data, team_abbrev):
    """Extract season stats from standings data."""
    if not standings_data:
        return {}
    
    for team in standings_data.get("standings", []):
        if team.get("teamAbbrev", {}).get("default") == team_abbrev:
            home_games = team.get("homeGamesPlayed", 1)
            road_games = team.get("roadGamesPlayed", 1)
            total_games = team.get("gamesPlayed", 1)
            
            return {
                'win_pct_season': (team.get("homeWins", 0) + team.get("roadWins", 0)) / max(total_games, 1),
                'home_win_pct': team.get("homeWins", 0) / max(home_games, 1),
                'away_win_pct': team.get("roadWins", 0) / max(road_games, 1),
                'gf_per_game_season': team.get("goalsForPctg", 0),
                'pointPctg_season': team.get("pointPctg", 0),
                'win_streak': team.get("streakCount", 0) if team.get("streakCode") == "W" else 0,
            }
    
    return {}


def get_rest_days(df, team_abbrev, current_date, season):
    """Calculate rest days for a team."""
    team_games = df[
        ((df['home_team_abbrev'] == team_abbrev) | (df['away_team_abbrev'] == team_abbrev)) &
        (df['date'] < current_date) &
        (df['season'] == season)
    ].sort_values('date', ascending=False)
    
    if len(team_games) == 0:
        return 7
    
    last_game_date = team_games.iloc[0]['date']
    rest_days = (pd.to_datetime(current_date) - last_game_date).days - 1
    
    return max(rest_days, 0)


def get_h2h_stats(df, home_team, away_team, current_date, season):
    """Get head-to-head stats between two teams."""
    h2h_games = df[
        (((df['home_team_abbrev'] == home_team) & (df['away_team_abbrev'] == away_team)) |
         ((df['home_team_abbrev'] == away_team) & (df['away_team_abbrev'] == home_team))) &
        (df['date'] < current_date) &
        (df['season'] == season)
    ]
    
    if len(h2h_games) == 0:
        return {
            'home_h2h_wins': 0, 'home_h2h_gf': 0,
            'away_h2h_wins': 0, 'away_h2h_gf': 0,
            'home_h2h_wins_diff': 0
        }
    
    home_wins = 0
    away_wins = 0
    home_gf = []
    away_gf = []
    
    for _, game in h2h_games.iterrows():
        if game['home_team_abbrev'] == home_team:
            if game['home_win'] == 1:
                home_wins += 1
            else:
                away_wins += 1
            home_gf.append(game['home_gf'])
            away_gf.append(game['away_gf'])
        else:
            if game['home_win'] == 1:
                away_wins += 1
            else:
                home_wins += 1
            away_gf.append(game['home_gf'])
            home_gf.append(game['away_gf'])
    
    return {
        'home_h2h_wins': home_wins,
        'home_h2h_gf': np.mean(home_gf) if home_gf else 0,
        'away_h2h_wins': away_wins,
        'away_h2h_gf': np.mean(away_gf) if away_gf else 0,
        'home_h2h_wins_diff': home_wins - away_wins
    }


def get_goalie_league_avg_ewm(df, season):
    """Return season-average EWM values for all goalie EWM columns.
    Used as a neutral prior when a goalie has no prior history (debut)."""
    cols = GOALIE_L5_COLS + TEAM_GOALIE_PERFORMANCE
    season_df = df[df["season"] == season]
    return {col: season_df[col].mean() for col in cols if col in season_df.columns}


def build_feature_row(game, df, standings_data, goalies_dict):
    """Build complete feature row for a single game."""
    current_date = pd.to_datetime(game['date'])
    season = game['season']
    home_team = game['home_team']
    away_team = game['away_team']
    gid = game['game_id']
    
    print(f"\n  Building features for {away_team} @ {home_team}...")
    
    # Get goalie names
    goalie_info = goalies_dict.get(gid, {})
    home_goalie = goalie_info.get('home_goalie')
    away_goalie = goalie_info.get('away_goalie')
    
    print(f"    Home Goalie: {home_goalie}")
    print(f"    Away Goalie: {away_goalie}")
    
    # EWM + rolling stats
    home_ewm = get_team_ewm_stats(df, home_team, current_date, season)
    away_ewm = get_team_ewm_stats(df, away_team, current_date, season)

    # Season stats
    home_season = get_season_stats(standings_data, home_team)
    away_season = get_season_stats(standings_data, away_team)

    # Rest days
    home_rest = get_rest_days(df, home_team, current_date, season)
    away_rest = get_rest_days(df, away_team, current_date, season)

    # H2H stats
    h2h = get_h2h_stats(df, home_team, away_team, current_date, season)

    # Goalie stats
    home_goalie_ewm = get_goalie_ewm_stats(df, home_goalie, current_date, season, home_team)
    away_goalie_ewm = get_goalie_ewm_stats(df, away_goalie, current_date, season, away_team)
    home_goalie_rest = get_goalie_rest_days(df, home_goalie, current_date, season, home_team)
    away_goalie_rest = get_goalie_rest_days(df, away_goalie, current_date, season, away_team)

    # Team save pct EWM
    home_team_save_pct = get_team_save_pct_ewm(df, home_team, current_date, season)
    away_team_save_pct = get_team_save_pct_ewm(df, away_team, current_date, season)

    # Build feature dictionary
    features = {}

    # Team EWM + rolling features (keys already match model: gf_ewm, wins_l5, etc.)
    for key, val in home_ewm.items():
        features[f'home_{key}'] = val
    for key, val in away_ewm.items():
        features[f'away_{key}'] = val

    # Season stats
    for key, val in home_season.items():
        features[f'home_{key}'] = val
    for key, val in away_season.items():
        features[f'away_{key}'] = val

    # Point percentage difference
    features['pointPctg_diff'] = home_season.get('pointPctg_season', 0) - away_season.get('pointPctg_season', 0)

    # Differentials (EWM-based to match training features)
    features['home_goal_diff_ewm'] = home_ewm.get('gf_ewm', 0) - away_ewm.get('gf_ewm', 0)
    features['home_ga_diff_ewm'] = home_ewm.get('ga_ewm', 0) - away_ewm.get('ga_ewm', 0)
    features['home_shot_diff_ewm'] = home_ewm.get('sog_ewm', 0) - away_ewm.get('sog_ewm', 0)

    # Rest days
    features['home_rest_days'] = home_rest
    features['away_rest_days'] = away_rest

    # H2H
    features.update(h2h)

    # Goalie EWM stats (keys: save_pct_ewm, ga_ewm, … → home_goalie_save_pct_ewm, …)
    for key, val in home_goalie_ewm.items():
        features[f'home_goalie_{key}'] = val
    for key, val in away_goalie_ewm.items():
        features[f'away_goalie_{key}'] = val

    features['home_goalie_rest_days'] = home_goalie_rest
    features['away_goalie_rest_days'] = away_goalie_rest

    # Team save pct EWM
    features['home_team_save_pct_ewm'] = home_team_save_pct
    features['away_team_save_pct_ewm'] = away_team_save_pct

    # Fill any missing goalie EWM features with the season mean as a neutral prior
    league_avg = get_goalie_league_avg_ewm(df, season)
    for feat in GOALIE_L5_COLS + TEAM_GOALIE_PERFORMANCE + ['home_goalie_rest_days', 'away_goalie_rest_days']:
        if feat not in features:
            features[feat] = league_avg.get(feat, 3 if 'rest_days' in feat else 0)

    return features


# =============================================================================
# MAKE PREDICTIONS
# =============================================================================

def make_predictions(model, feature_names, games, df, standings_data, goalies_dict, threshold=0.52):
    """Generate predictions for all games."""
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    predictions = []
    
    for game in games:
        # Build features
        features = build_feature_row(game, df, standings_data, goalies_dict)

        # print(f"    Features for {game['away_team']} @ {game['home_team']}:")
        # for key in feature_names:
        #     print(f"      {key}: {features.get(key, 0)}")
        
        # Create feature vector in correct order
        X = []
        for feat in feature_names:
            X.append(features.get(feat, 0))
        
        X = np.array(X).reshape(1, -1)
        
        # Make prediction
        prob_home_win = model.predict_proba(X)[0][1]
        pred_home_win = int(prob_home_win >= threshold)
        
        predictions.append({
            'game_id': game['game_id'],
            'date': game['date'],
            'time': game['game_time'],
            'away_team': game['away_team'],
            'home_team': game['home_team'],
            'away_team_name': game['away_team_name'],
            'home_team_name': game['home_team_name'],
            'away_goalie': goalies_dict[game['game_id']]['away_goalie'],
            'home_goalie': goalies_dict[game['game_id']]['home_goalie'],
            'pred_home_win': pred_home_win,
            'prob_home_win': prob_home_win,
            'prob_away_win': 1 - prob_home_win,
            'confidence': max(prob_home_win, 1 - prob_home_win)
        })
    
    return predictions


def display_predictions(predictions, threshold=0.5):
    """Display predictions in a nice format."""
    print("\n" + "="*60)
    print("TODAY'S NHL PREDICTIONS")
    print("="*60)
    print(f"Prediction Threshold: {threshold:.2f}")
    print(f"Total Games: {len(predictions)}\n")
    
    for i, pred in enumerate(predictions, 1):
        winner = pred['home_team_name'] if pred['pred_home_win'] == 1 else pred['away_team_name']
        prob = pred['prob_home_win'] if pred['pred_home_win'] == 1 else pred['prob_away_win']
        
        print(f"Game {i}:")
        print(f"  {pred['away_team']} ({pred['away_goalie']}) @ {pred['home_team']} ({pred['home_goalie']})")
        print(f"  Predicted Winner: {winner}")
        print(f"  Confidence: {prob:.1%}")
        print(f"  Home Win Probability: {pred['prob_home_win']:.1%}")
        print(f"  Away Win Probability: {pred['prob_away_win']:.1%}")
        print()
    
    # Save to CSV
    df_pred = pd.DataFrame(predictions)
    output_file = f"predictions_{predictions[0]['date']}.csv"
    df_pred.to_csv(output_file, index=False)
    print(f"✓ Predictions saved to: {output_file}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

async def main(date_str=None, threshold=0.5):
    """Main prediction workflow."""
    print("\n" + "="*60)
    print("NHL GAME PREDICTOR")
    print("="*60)
    
    try:
        # Load model and data
        model, feature_names = load_model_and_features()
        df = load_historical_data()
        
        # Fetch today's games
        games, standings_data = await get_todays_games(date_str)
        
        if not games:
            print("\nNo games to predict today!")
            return
        
        # Fetch goalie information
        goalies_dict = await get_todays_goalies(games, df)
        
        # Make predictions
        predictions = make_predictions(model, feature_names, games, df, standings_data, goalies_dict, threshold)
        
        # Display results
        display_predictions(predictions, threshold)
        
        print("\n" + "="*60)
        print("PREDICTION COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # threshold = 0.60  # Higher threshold = more conservative predictions    
    asyncio.run(main(date_str=None, threshold=0.52))