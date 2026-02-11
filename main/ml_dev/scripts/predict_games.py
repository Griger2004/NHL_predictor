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
import aiohttp
from aiohttp import ClientTimeout
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = "https://api-web.nhle.com"
TIMEOUT = ClientTimeout(total=15)
MAX_CONCURRENT_REQUESTS = 9
RETRIES = 3
ROLLING_N = 5

# Model files
MODEL_FILE = "../notebooks/models/nhl_rf_model.pkl"
FEATURES_FILE = "../notebooks/models/feature_names.pkl"

# Historical data file (needed for computing rolling stats)
HISTORICAL_DATA_FILE = "generated/data/nhl_data.csv"

# Feature definitions (must match training)
HOME_TEAM_L5_COLS = [
    'home_gf_l5', 'home_ga_l5', 'home_sog_l5',
    'home_wins_l5', 'home_win_pct_l5', 'home_powerplay_pct_l5',
    'home_penalty_kill_pct_l5', 'home_powerplays_l5', 'home_penalty_kills_l5',
    'home_faceoffwin_pct_l5', 'home_pims_l5', 'home_hits_l5',
    'home_blockedshots_l5', 'home_giveaways_l5', 'home_takeaways_l5',
]

AWAY_TEAM_L5_COLS = [
    'away_gf_l5', 'away_ga_l5', 'away_sog_l5',
    'away_wins_l5', 'away_win_pct_l5', 'away_powerplay_pct_l5',
    'away_penalty_kill_pct_l5', 'away_powerplays_l5', 'away_penalty_kills_l5',
    'away_faceoffwin_pct_l5', 'away_pims_l5', 'away_hits_l5',
    'away_blockedshots_l5', 'away_giveaways_l5', 'away_takeaways_l5',
]

GOALIE_L5_COLS = [
    'home_goalie_save_pct_l5', 'home_goalie_ga_l5', 'home_goalie_saves_l5',
    'home_goalie_ev_sa_l5', 'home_goalie_pp_sa_l5', 'home_goalie_sh_sa_l5',
    'home_goalie_ev_ga_l5', 'home_goalie_pp_ga_l5', 
    'away_goalie_save_pct_l5', 'away_goalie_ga_l5', 'away_goalie_saves_l5',
    'away_goalie_ev_sa_l5', 'away_goalie_pp_sa_l5', 'away_goalie_sh_sa_l5',
    'away_goalie_ev_ga_l5', 'away_goalie_pp_ga_l5',
]

TEAM_GOALIE_PERFORMANCE = [
    'home_team_save_pct_l5', 'away_team_save_pct_l5',
]

SEASON_COLS = [
    'home_win_pct_season', 'away_win_pct_season',
    'home_home_win_pct', 'away_away_win_pct',
    'home_gf_per_game_season', 'away_gf_per_game_season',
    'home_pointPctg_season', 'away_pointPctg_season', 'pointPctg_diff',
]

DIFF_COLS = [
    'home_goal_diff_l5', 'home_ga_diff_l5', 'home_shot_diff_l5',
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
# ASYNC API HELPERS
# =============================================================================

async def fetch_json(session, url, semaphore):
    """Fetch JSON data from URL with retry logic."""
    async with semaphore:
        for attempt in range(RETRIES):
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status in (429, 500, 502, 503, 504):
                        await asyncio.sleep(2 ** attempt)
                    else:
                        return None
            except Exception:
                await asyncio.sleep(2 ** attempt)
    return None


async def fetch_schedule(session, date_str, semaphore):
    """Fetch schedule for a specific date."""
    url = f"{API_BASE_URL}/v1/schedule/{date_str}"
    return await fetch_json(session, url, semaphore)

async def fetch_standings(session, date_str, semaphore):
    """Fetch standings for a specific date."""
    url = f"{API_BASE_URL}/v1/standings/{date_str}"
    return await fetch_json(session, url, semaphore)

async def fetch_boxscores(session, gid, semaphore):
    """Fetch boxscore for a game."""
    url = f"{API_BASE_URL}/v1/gamecenter/{gid}/boxscore"
    return await fetch_json(session, url, semaphore)

async def fetch_landings(session, gid, semaphore):
    """Fetch landing page for a game (has expected starters for future games)."""
    url = f"{API_BASE_URL}/v1/gamecenter/{gid}/landing"
    return await fetch_json(session, url, semaphore)

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

# Unfortunately, it seems that the 'starter' value only appears post-game.

# def get_starter_goalie(goalies):
#     """Get the starting goalie from a list of goalies."""
#     if not goalies:
#         return ""
#     starter = next((g for g in goalies if g.get("starter")), goalies[0] if goalies else None)
#     if not starter:
#         return ""
#     return extract_name(starter.get("name", {}))

def get_starter_goalie(goalies):
    """Get the starting goalie based on highest TOI."""
    if not goalies:
        return ""

    def toi_to_seconds(toi):
        if not toi:
            return 0
        minutes, seconds = toi.split(":")
        return int(minutes) * 60 + int(seconds)

    starter = max(
        goalies,
        key=lambda g: toi_to_seconds(g.get("toi", "00:00"))
    )

    return extract_name(starter.get("name", {}))


async def get_todays_goalies(games):
    """Fetch today's starting goalies - tries boxscore first, then landing."""
    print("\n" + "="*60)
    print("FETCHING TODAY'S STARTING GOALIES")
    print("="*60)
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    goalies_dict = {}
    
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        # Try boxscore first (for live/completed games)
        boxscore_tasks = [fetch_boxscores(session, game['game_id'], semaphore) for game in games]
        boxscore_results = await asyncio.gather(*boxscore_tasks)
        
        # Try landing for games without boxscore data (future games)
        landing_tasks = [fetch_landings(session, game['game_id'], semaphore) for game in games]
        landing_results = await asyncio.gather(*landing_tasks)
    
    for i, game in enumerate(games):
        gid = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']
        
        home_goalie = None
        away_goalie = None
        
        # Try to get from boxscore first
        boxscore = boxscore_results[i]
        if boxscore:
            home_players = boxscore.get("playerByGameStats", {}).get("homeTeam", {}).get("goalies", [])
            away_players = boxscore.get("playerByGameStats", {}).get("awayTeam", {}).get("goalies", [])
            
            if home_players:
                home_goalie = get_starter_goalie(home_players)
            if away_players:
                away_goalie = get_starter_goalie(away_players)
        
        # If boxscore didn't have goalies, try landing
        if not home_goalie or not away_goalie:
            landing = landing_results[i]
            if landing:

                home_team_data = landing.get("matchup", {}).get("goalieComparison", {}).get("homeTeam", {})
                away_team_data = landing.get("matchup", {}).get("goalieComparison", {}).get("awayTeam", {})
                
                # Get BEST performing goalies from landing
                if not home_goalie:
                    home_goalie_obj = home_team_data.get("leaders", [{}])[0] if home_team_data.get("leaders") else {}
                    home_goalie = extract_name(home_goalie_obj.get("name", {}))
                
                if not away_goalie:
                    away_goalie_obj = away_team_data.get("leaders", [{}])[0] if away_team_data.get("leaders") else {}
                    away_goalie = extract_name(away_goalie_obj.get("name", {}))
        
        goalies_dict[gid] = {
            "game_id": gid,
            "home_team": home_team,
            "away_team": away_team,
            "home_goalie": home_goalie if home_goalie else "Unknown",
            "away_goalie": away_goalie if away_goalie else "Unknown",
        }
        
        print(f"  {away_team} @ {home_team}")
        print(f"    Away Goalie: {goalies_dict[gid]['away_goalie']}")
        print(f"    Home Goalie: {goalies_dict[gid]['home_goalie']}")
    
    return goalies_dict


def get_goalie_last_5_stats(df, goalie_name, current_date, season, team_abbrev):
    """Get last 5 games stats for a goalie."""
    if not goalie_name or goalie_name == "Unknown":
        return {}
    
    # Find games where this goalie was the starter for this team
    goalie_games = df[
        (((df['home_team_abbrev'] == team_abbrev) & (df['home_goalie_starter'] == goalie_name)) |
         ((df['away_team_abbrev'] == team_abbrev) & (df['away_goalie_starter'] == goalie_name))) &
        (df['date'] < current_date) &
        (df['season'] == season)
    ].sort_values('date', ascending=False).head(5)
    
    if len(goalie_games) == 0:
        return {}
    
    # Aggregate stats from last 5 games
    stats = {
        'save_pct': [],
        'ga': [],
        'saves': [],
        'ev_sa': [],
        'pp_sa': [],
        'sh_sa': [],
        'ev_ga': [],
        'pp_ga': []
    }
    
    for _, game in goalie_games.iterrows():
        is_home = game['home_goalie_starter'] == goalie_name
        prefix = 'home' if is_home else 'away'
        
        # Collect goalie stats
        stats['save_pct'].append(game.get(f'{prefix}_goalie_save_pct', 0))
        stats['ga'].append(game.get(f'{prefix}_goalie_ga', 0))
        stats['saves'].append(game.get(f'{prefix}_goalie_saves', 0))
        stats['ev_sa'].append(game.get(f'{prefix}_goalie_evenStrengthShotsAgainst', 0))
        stats['pp_sa'].append(game.get(f'{prefix}_goalie_powerPlayShotsAgainst', 0))
        stats['sh_sa'].append(game.get(f'{prefix}_goalie_shorthandedShotsAgainst', 0))
        stats['ev_ga'].append(game.get(f'{prefix}_goalie_evenStrengthGoalsAgainst', 0))
        stats['pp_ga'].append(game.get(f'{prefix}_goalie_powerPlayGoalsAgainst', 0))
    
    # Calculate averages
    result = {}
    for key in stats.keys():
        result[f'{key}_l5'] = np.mean(stats[key]) if stats[key] else 0
    
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


def get_team_save_pct_l5(df, team_abbrev, current_date, season):
    """Get team's last 5 games save percentage."""
    team_games = df[
        ((df['home_team_abbrev'] == team_abbrev) | (df['away_team_abbrev'] == team_abbrev)) &
        (df['date'] < current_date) &
        (df['season'] == season)
    ].sort_values('date', ascending=False).head(5)
    
    if len(team_games) == 0:
        return 0
    
    save_pcts = []
    for _, game in team_games.iterrows():
        is_home = game['home_team_abbrev'] == team_abbrev
        prefix = 'home' if is_home else 'away'
        save_pcts.append(game.get(f'{prefix}_save_pct', 0))
    
    return np.mean(save_pcts) if save_pcts else 0


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
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        schedule_data = await fetch_schedule(session, date_str, semaphore)
        standings_data = await fetch_standings(session, date_str, semaphore)
    
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

def get_team_last_5_stats(df, team_abbrev, current_date, season):
    """Get last 5 games stats for a team."""
    team_games = df[
        ((df['home_team_abbrev'] == team_abbrev) | (df['away_team_abbrev'] == team_abbrev)) &
        (df['date'] < current_date) &
        (df['season'] == season)
    ].sort_values('date', ascending=False).head(5)
    
    if len(team_games) == 0:
        return {}
    
    stats = {}
    
    for _, game in team_games.iterrows():
        is_home = game['home_team_abbrev'] == team_abbrev
        prefix = 'home' if is_home else 'away'
        
        for stat in ['gf', 'ga', 'sog', 'powerplay_pct', 'pk_pct', 
                     'faceoffwin_pct', 'pims', 'hits', 'blockedshots', 
                     'giveaways', 'takeaways']:
            col = f'{prefix}_{stat}'
            if col in game:
                stats.setdefault(stat, []).append(game[col])
        
        if is_home:
            stats.setdefault('wins', []).append(game['home_win'])
        else:
            stats.setdefault('wins', []).append(1 - game['home_win'])
        
        stats.setdefault('powerplays', []).append(game.get(f'{prefix}_powerplays', 0))
        stats.setdefault('pk', []).append(game.get(f'{prefix}_pk', 0))
    
    result = {}
    result['gf_per_game_l5'] = np.mean(stats.get('gf', [0]))
    result['ga_per_game_l5'] = np.mean(stats.get('ga', [0]))
    result['sog_per_game_l5'] = np.mean(stats.get('sog', [0]))
    result['wins_l5'] = np.sum(stats.get('wins', [0]))
    result['win_pct_l5'] = np.mean(stats.get('wins', [0]))
    result['powerplay_pct_l5'] = np.mean(stats.get('powerplay_pct', [0]))
    result['penalty_kill_pct_l5'] = np.mean(stats.get('pk_pct', [0]))
    result['powerplay_opps_l5'] = np.sum(stats.get('powerplays', [0]))
    result['pk_opps_l5'] = np.sum(stats.get('pk', [0]))
    result['faceoffwin_pct_l5'] = np.mean(stats.get('faceoffwin_pct', [0]))
    result['pims_l5'] = np.mean(stats.get('pims', [0]))
    result['hits_l5'] = np.mean(stats.get('hits', [0]))
    result['blockedshots_l5'] = np.mean(stats.get('blockedshots', [0]))
    result['giveaways_l5'] = np.mean(stats.get('giveaways', [0]))
    result['takeaways_l5'] = np.mean(stats.get('takeaways', [0]))
    
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
    
    # Get last 5 games stats
    home_l5 = get_team_last_5_stats(df, home_team, current_date, season)
    away_l5 = get_team_last_5_stats(df, away_team, current_date, season)
    
    # Get season stats
    home_season = get_season_stats(standings_data, home_team)
    away_season = get_season_stats(standings_data, away_team)
    
    # Get rest days
    home_rest = get_rest_days(df, home_team, current_date, season)
    away_rest = get_rest_days(df, away_team, current_date, season)
    
    # Get H2H stats
    h2h = get_h2h_stats(df, home_team, away_team, current_date, season)
    
    # Get goalie stats
    home_goalie_l5 = get_goalie_last_5_stats(df, home_goalie, current_date, season, home_team)
    away_goalie_l5 = get_goalie_last_5_stats(df, away_goalie, current_date, season, away_team)
    home_goalie_rest = get_goalie_rest_days(df, home_goalie, current_date, season, home_team)
    away_goalie_rest = get_goalie_rest_days(df, away_goalie, current_date, season, away_team)
    
    # Get team save pct
    home_team_save_pct = get_team_save_pct_l5(df, home_team, current_date, season)
    away_team_save_pct = get_team_save_pct_l5(df, away_team, current_date, season)
    
    # Build feature dictionary
    features = {}
    
    # Last 5 games stats
    for key, val in home_l5.items():
        features[f'home_{key}'] = val
    for key, val in away_l5.items():
        features[f'away_{key}'] = val
    
    # Season stats
    for key, val in home_season.items():
        features[f'home_{key}'] = val
    for key, val in away_season.items():
        features[f'away_{key}'] = val
    
    # Point percentage difference
    features['pointPctg_diff'] = home_season.get('pointPctg_season', 0) - away_season.get('pointPctg_season', 0)
    
    # Differentials
    features['home_goal_diff_l5'] = home_l5.get('gf_per_game_l5', 0) - away_l5.get('gf_per_game_l5', 0)
    features['home_ga_diff_l5'] = home_l5.get('ga_per_game_l5', 0) - away_l5.get('ga_per_game_l5', 0)
    features['home_shot_diff_l5'] = home_l5.get('sog_per_game_l5', 0) - away_l5.get('sog_per_game_l5', 0)
    
    # Rest days
    features['home_rest_days'] = home_rest
    features['away_rest_days'] = away_rest
    
    # H2H
    features.update(h2h)
    
    # Goalie stats
    for key, val in home_goalie_l5.items():
        features[f'home_goalie_{key}'] = val
    for key, val in away_goalie_l5.items():
        features[f'away_goalie_{key}'] = val
    
    features['home_goalie_rest_days'] = home_goalie_rest
    features['away_goalie_rest_days'] = away_goalie_rest
    
    # Team save pct
    features['home_team_save_pct_l5'] = home_team_save_pct
    features['away_team_save_pct_l5'] = away_team_save_pct
    
    # Fill in any missing goalie features with 0
    for feat in GOALIE_L5_COLS + TEAM_GOALIE_PERFORMANCE + ['home_goalie_rest_days', 'away_goalie_rest_days']:
        if feat not in features:
            features[feat] = 0
    
    return features


# =============================================================================
# MAKE PREDICTIONS
# =============================================================================

def make_predictions(model, feature_names, games, df, standings_data, goalies_dict, threshold=0.5):
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
        goalies_dict = await get_todays_goalies(games)
        
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
    asyncio.run(main(date_str=None, threshold=0.5))