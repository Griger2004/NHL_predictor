import os
import time
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import csv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import asyncio
import aiohttp
from aiohttp import ClientTimeout

# ===== CONSTANTS =====
API_BASE_URL = "https://api-web.nhle.com"
OUTPUT_DIR = "generated/data/"
CSV_FILE = f"{OUTPUT_DIR}/nhl_data.csv"

TIMEOUT = ClientTimeout(total=15)
MAX_CONCURRENT_REQUESTS = 9
RETRIES = 3

MAX_GAMES = 1312
SLEEP_SEC = 0.1

SEASONS = [2023, 2024, 2025]

FIELDNAMES = [
    "game_id", "date", "season", "home_team", "away_team",
    "home_team_abbrev", "away_team_abbrev", "home_win",
    "home_gf", "away_gf", "home_ga", "away_ga", "home_sog", "away_sog",
    "home_faceoffwin_pct", "away_faceoffwin_pct", "home_powerplays", "away_powerplays",
    "home_powerplay_pct", "away_powerplay_pct", "home_pk", "away_pk", "home_pk_pct", "away_pk_pct",
    "home_pims", "away_pims", "home_hits", "away_hits", "home_blockedshots", "away_blockedshots",
    "home_takeaways", "away_takeaways", "home_giveaways", "away_giveaways"
]

ROLLING_N = 5

STANDINGS_FIELDS = [
    "pointPctg", "gamesPlayed", "goalsForPctg", "homeGamesPlayed", "homeWins",
    "homeLosses", "roadGamesPlayed", "roadWins", "roadLosses", "streakCode", "streakCount",
]

HOME_RENAME = {
    "home_goalie_starter": "goalie",
    "home_goalie_save_pct": "save_pct",
    "home_goalie_ga": "ga",
    "home_goalie_saves": "saves",
    "home_goalie_evenStrengthShotsAgainst": "ev_sa",
    "home_goalie_powerPlayShotsAgainst": "pp_sa",
    "home_goalie_shorthandedShotsAgainst": "sh_sa",
    "home_goalie_evenStrengthGoalsAgainst": "ev_ga",
    "home_goalie_powerPlayGoalsAgainst": "pp_ga",
}

AWAY_RENAME = {
    "away_goalie_starter": "goalie",
    "away_goalie_save_pct": "save_pct",
    "away_goalie_ga": "ga",
    "away_goalie_saves": "saves",
    "away_goalie_evenStrengthShotsAgainst": "ev_sa",
    "away_goalie_powerPlayShotsAgainst": "pp_sa",
    "away_goalie_shorthandedShotsAgainst": "sh_sa",
    "away_goalie_evenStrengthGoalsAgainst": "ev_ga",
    "away_goalie_powerPlayGoalsAgainst": "pp_ga",
}

GOALIE_STATS = ["save_pct", "ga", "saves", "ev_sa", "pp_sa", "sh_sa", "ev_ga", "pp_ga"]

GOALIE_MERGE_COLS = [
    "game_id", "home_goalie_starter", "away_goalie_starter", "home_save_pct", "away_save_pct",
    "home_goalie_save_pct", "away_goalie_save_pct", "home_goalie_ga", "away_goalie_ga",
    "home_goalie_saves", "away_goalie_saves", "home_goalie_evenStrengthShotsAgainst",
    "away_goalie_evenStrengthShotsAgainst", "home_goalie_powerPlayShotsAgainst",
    "away_goalie_powerPlayShotsAgainst", "home_goalie_shorthandedShotsAgainst",
    "away_goalie_shorthandedShotsAgainst", "home_goalie_evenStrengthGoalsAgainst",
    "away_goalie_evenStrengthGoalsAgainst", "home_goalie_powerPlayGoalsAgainst",
    "away_goalie_powerPlayGoalsAgainst", "home_goalie_save_pct_l5", "home_goalie_ga_l5",
    "home_goalie_saves_l5", "home_goalie_ev_sa_l5", "home_goalie_pp_sa_l5", "home_goalie_sh_sa_l5",
    "home_goalie_ev_ga_l5", "home_goalie_pp_ga_l5", "away_goalie_save_pct_l5", "away_goalie_ga_l5",
    "away_goalie_saves_l5", "away_goalie_ev_sa_l5", "away_goalie_pp_sa_l5", "away_goalie_sh_sa_l5",
    "away_goalie_ev_ga_l5", "away_goalie_pp_ga_l5", "home_team_save_pct_l5", "away_team_save_pct_l5",
]

SEASON_STATS = [
    "home_win_pct_season", "away_win_pct_season", "home_home_win_pct", "away_away_win_pct",
    "home_gf_per_game_season", "away_gf_per_game_season", "home_pointPctg_season",
    "away_pointPctg_season", "pointPctg_diff", "home_win_streak", "away_win_streak",
]

# Global semaphore for async requests
# semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# ===== ASYNC HELPER FUNCTIONS =====
async def fetch_json(session, url, semaphore):
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



async def fetch_game_story_async(session, game_id, semaphore):
    url = f"{API_BASE_URL}/v1/wsc/game-story/{game_id}"
    return await fetch_json(session, url, semaphore)


async def fetch_boxscores(session, gid, semaphore):
    url = f"{API_BASE_URL}/v1/gamecenter/{gid}/boxscore"
    return await fetch_json(session, url, semaphore)


async def fetch_standings_info(session, date, semaphore):
    url = f"{API_BASE_URL}/v1/standings/{date}"
    return await fetch_json(session, url, semaphore)



# ===== STAT EXTRACTION HELPER FUNCTIONS =====
def get_num_powerplays(powerplay_str):
    """Extract power play goals and opportunities from string."""
    if not powerplay_str:
        return 0, 0
    try:
        goals, opps = map(int, powerplay_str.split("/"))
        return goals, opps
    except ValueError:
        return 0, 0


def calc_num_penalty_kills(powerplays_against, ppg_against):
    """Calculate penalty kills from power play stats."""
    return powerplays_against - ppg_against


def calc_penalty_kill_pct(powerplays, ppg_against):
    """Calculate penalty kill percentage."""
    if powerplays == 0:
        return 0.0
    pk_successes = powerplays - ppg_against
    return round((pk_successes / powerplays) * 100, 2)


def extract_name(obj):
    """Extract name from nested object."""
    if not obj:
        return ""
    if isinstance(obj, dict):
        return obj.get("default", "") if "default" in obj else ""
    return str(obj)


def extract_category_stat(stats, category):
    """Extract category stats for home and away teams."""
    for stat in stats:
        if stat.get("category") == category:
            return stat.get("homeValue", 0), stat.get("awayValue", 0)
    return 0, 0


def calc_team_save_pct(saves, shots_against):
    """Calculate team save percentage."""
    if shots_against == 0:
        return 0.0
    return round((saves / shots_against), 3)


def extract_fractional_stat(stat_str):
    """Extract fractional stat (e.g., '5/10' -> 0.5)."""
    if not stat_str:
        return 0.0
    try:
        numerator, denominator = map(int, stat_str.split("/"))
        if denominator == 0:
            return 0.0
        return round(numerator / denominator, 3)
    except ValueError:
        return 0.0


def get_starter_goalie(goalies):
    """Get the starting goalie from a list of goalies."""
    if not goalies:
        return "", None
    starter = next((g for g in goalies if g.get("starter")), goalies[0])
    return extract_name(starter.get("name", {})), starter


# ===== MAIN STAT EXTRACTION FUNCTIONS =====
def extract_all_basic_stats(game_data):
    """Extract all game statistics from game data."""
    is_future_or_live_game = game_data.get("gameState") in ["FUT", "LIVE", "PRE"]
    if is_future_or_live_game:
        return None

    date_str = game_data.get("gameDate", "")
    season_year = game_data.get("season", 0)

    home = game_data.get("homeTeam", {})
    away = game_data.get("awayTeam", {})
    team_stats = game_data.get("summary", {}).get("teamGameStats", [])

    home_place = extract_name(home.get("placeName"))
    home_name = extract_name(home.get("name"))
    home_abbrev = home.get("abbrev", "")
    away_place = extract_name(away.get("placeName"))
    away_name = extract_name(away.get("name"))
    away_abbrev = away.get("abbrev", "")

    full_home_name = f"{home_place} {home_name}".strip()
    full_away_name = f"{away_place} {away_name}".strip()

    if full_home_name in ("Arizona Coyotes", "Utah Utah Hockey Club"):
        home_place = "Utah"
        home_name = "Mammoth"
        home_abbrev = "UTA"

    if full_away_name in ("Arizona Coyotes", "Utah Utah Hockey Club"):
        away_place = "Utah"
        away_name = "Mammoth"
        away_abbrev = "UTA"

    home_faceoffwin_pct, away_faceoffwin_pct = extract_category_stat(team_stats, "faceoffWinningPctg")
    home_powerplays_str, away_powerplays_str = extract_category_stat(team_stats, "powerPlay")

    home_pp_goals, home_pp_opps = get_num_powerplays(home_powerplays_str)
    away_pp_goals, away_pp_opps = get_num_powerplays(away_powerplays_str)

    home_penaltykills = calc_num_penalty_kills(away_pp_opps, away_pp_goals)
    away_penaltykills = calc_num_penalty_kills(home_pp_opps, home_pp_goals)
    home_pk_pct = calc_penalty_kill_pct(away_pp_opps, away_pp_goals)
    away_pk_pct = calc_penalty_kill_pct(home_pp_opps, home_pp_goals)
    home_powerplay_pct, away_powerplay_pct = extract_category_stat(team_stats, "powerPlayPctg")
    home_pims, away_pims = extract_category_stat(team_stats, "pim")
    home_hits, away_hits = extract_category_stat(team_stats, "hits")
    home_blockedshots, away_blockedshots = extract_category_stat(team_stats, "blockedShots")
    home_takeaways, away_takeaways = extract_category_stat(team_stats, "takeaways")
    home_giveaways, away_giveaways = extract_category_stat(team_stats, "giveaways")

    row = {
        "game_id": game_data.get("id"),
        "date": date_str,
        "season": season_year,
        "home_team": f"{home_place} {home_name}".strip(),
        "away_team": f"{away_place} {away_name}".strip(),
        "home_team_abbrev": home_abbrev,
        "away_team_abbrev": away_abbrev,
        "home_win": 1 if home.get("score", 0) > away.get("score", 0) else 0,
        "home_gf": home.get("score", 0),
        "away_gf": away.get("score", 0),
        "home_ga": away.get("score", 0),
        "away_ga": home.get("score", 0),
        "home_sog": home.get("sog", 0),
        "away_sog": away.get("sog", 0),
        "home_faceoffwin_pct": home_faceoffwin_pct,
        "away_faceoffwin_pct": away_faceoffwin_pct,
        "home_powerplays": home_pp_opps,
        "away_powerplays": away_pp_opps,
        "home_powerplay_pct": home_powerplay_pct,
        "away_powerplay_pct": away_powerplay_pct,
        "home_pk": home_penaltykills,
        "away_pk": away_penaltykills,
        "home_pk_pct": home_pk_pct,
        "away_pk_pct": away_pk_pct,
        "home_pims": home_pims,
        "away_pims": away_pims,
        "home_hits": home_hits,
        "away_hits": away_hits,
        "home_blockedshots": home_blockedshots,
        "away_blockedshots": away_blockedshots,
        "home_takeaways": home_takeaways,
        "away_takeaways": away_takeaways,
        "home_giveaways": home_giveaways,
        "away_giveaways": away_giveaways
    }
    return row


def extract_all_goalie_basic_stats(boxscore_data):
    """Extract all goalie statistics from boxscore data."""
    date_str = boxscore_data.get("gameDate", "")
    season_year = boxscore_data.get("season", 0)

    home = boxscore_data.get("homeTeam", {})
    away = boxscore_data.get("awayTeam", {})
    home_players = boxscore_data.get("playerByGameStats", {}).get("homeTeam", {}).get("goalies", [])
    away_players = boxscore_data.get("playerByGameStats", {}).get("awayTeam", {}).get("goalies", [])

    home_starter_goalie, home_starter_obj = get_starter_goalie(home_players)
    away_starter_goalie, away_starter_obj = get_starter_goalie(away_players)

    # Calculate team save percentages
    if len(home_players) < 2:
        home_save_pct = home_players[0].get("savePctg", 0.0) if home_players else 0.0
    else:
        home_save_pct = calc_team_save_pct(
            home_players[0].get("saves", 0) + home_players[1].get("saves", 0),
            home_players[0].get("shotsAgainst", 0) + home_players[1].get("shotsAgainst", 0)
        )

    if len(away_players) < 2:
        away_save_pct = away_players[0].get("savePctg", 0.0) if away_players else 0.0
    else:
        away_save_pct = calc_team_save_pct(
            away_players[0].get("saves", 0) + away_players[1].get("saves", 0),
            away_players[0].get("shotsAgainst", 0) + away_players[1].get("shotsAgainst", 0)
        )

    home_starter_obj = home_players[0] if home_players and home_players[0].get("starter") else (home_players[1] if len(home_players) > 1 else {})
    away_starter_obj = away_players[0] if away_players and away_players[0].get("starter") else (away_players[1] if len(away_players) > 1 else {})

    home_abbrev = home.get("abbrev", "")
    away_abbrev = away.get("abbrev", "")

    if home_abbrev == "ARI":
        home_abbrev = "UTA"
    if away_abbrev == "ARI":
        away_abbrev = "UTA"

    row = {
        "game_id": boxscore_data.get("id"),
        "date": date_str,
        "season": season_year,
        "home_team_abbrev": home_abbrev,
        "away_team_abbrev": away_abbrev,
        "home_goalie_starter": home_starter_goalie,
        "away_goalie_starter": away_starter_goalie,
        "home_save_pct": home_save_pct,
        "away_save_pct": away_save_pct,
        "home_goalie_save_pct": home_starter_obj.get("savePctg", 0.0),
        "away_goalie_save_pct": away_starter_obj.get("savePctg", 0.0),
        "home_goalie_ga": home_starter_obj.get("goalsAgainst", 0),
        "away_goalie_ga": away_starter_obj.get("goalsAgainst", 0),
        "home_goalie_saves": home_starter_obj.get("saves", 0),
        "away_goalie_saves": away_starter_obj.get("saves", 0),
        "home_goalie_evenStrengthShotsAgainst": extract_fractional_stat(home_starter_obj.get("evenStrengthShotsAgainst", "0/0")),
        "away_goalie_evenStrengthShotsAgainst": extract_fractional_stat(away_starter_obj.get("evenStrengthShotsAgainst", "0/0")),
        "home_goalie_powerPlayShotsAgainst": extract_fractional_stat(home_starter_obj.get("powerPlayShotsAgainst", "0/0")),
        "away_goalie_powerPlayShotsAgainst": extract_fractional_stat(away_starter_obj.get("powerPlayShotsAgainst", "0/0")),
        "home_goalie_shorthandedShotsAgainst": extract_fractional_stat(home_starter_obj.get("shorthandedShotsAgainst", "0/0")),
        "away_goalie_shorthandedShotsAgainst": extract_fractional_stat(away_starter_obj.get("shorthandedShotsAgainst", "0/0")),
        "home_goalie_evenStrengthGoalsAgainst": home_starter_obj.get("evenStrengthGoalsAgainst", 0),
        "away_goalie_evenStrengthGoalsAgainst": away_starter_obj.get("evenStrengthGoalsAgainst", 0),
        "home_goalie_powerPlayGoalsAgainst": home_starter_obj.get("powerPlayGoalsAgainst", 0),
        "away_goalie_powerPlayGoalsAgainst": away_starter_obj.get("powerPlayGoalsAgainst", 0),
    }
    return row


def extract_all_teams_playing_on_date(game_date):
    """Extract all teams playing on a specific date."""
    df = pd.read_csv(CSV_FILE, parse_dates=["date"])
    teams_home = df[df["date"] == game_date]["home_team_abbrev"].unique().tolist()
    teams_away = df[df["date"] == game_date]["away_team_abbrev"].unique().tolist()
    all_teams = set(teams_home + teams_away)
    return all_teams


def extract_all_standing_stats(standings_data, playing_teams):
    """Extract standing stats for teams playing on a date."""
    rows = {}

    for team in standings_data.get("standings", []):
        abbrev = team.get("teamAbbrev", {}).get("default", "")

        if abbrev == "ARI":
            abbrev = "UTA"

        if abbrev not in playing_teams:
            continue

        rows[abbrev] = {field: team.get(field) for field in STANDINGS_FIELDS}

    return rows


# ===== STEP 1: FETCH BASIC GAME INFO =====
async def fetch_season_games(season):
    """Fetch all games for a season."""
    game_ids = [f"{season}02{str(game_num).zfill(4)}" for game_num in range(1, MAX_GAMES + 1)]
    rows = []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        for i in range(0, len(game_ids), MAX_CONCURRENT_REQUESTS):
            batch = game_ids[i:i + MAX_CONCURRENT_REQUESTS]
            tasks = [fetch_game_story_async(session, gid, semaphore) 
                     for gid in batch
                    ]
            results = await asyncio.gather(*tasks)

            for game_data in results:
                if not game_data:
                    continue
                row = extract_all_basic_stats(game_data)
                if row is None:
                    break
                else:
                    rows.append(row)

            await asyncio.sleep(0.1)

    return rows


def step1_fetch_basic_game_info():
    """Step 1: Fetch basic game information for all seasons."""
    print("\n=== STEP 1: Fetch Basic Game Info ===")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)

    all_rows = []
    for season in SEASONS:
        print(f"📅 Fetching season {season}...")
        season_rows = asyncio.run(fetch_season_games(season))
        all_rows.extend(season_rows)

    df = pd.DataFrame(all_rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(CSV_FILE, index=False)
    print(f"✅ Saved {len(df)} games to {CSV_FILE}")


# ===== STEP 2: COMPUTE ROLLING AVERAGES STATISTICS AND DIFFERENCIALS =====
def step2_compute_rolling_averages():
    """Step 2: Compute rolling average statistics for last 5 games."""
    print("\n=== STEP 2: Compute Rolling Averages ===")

    df = pd.read_csv(CSV_FILE, parse_dates=["date"])

    # Prepare combined dataframe
    home_stats = df[[
        "date", "season", "home_team_abbrev", "home_gf", "home_ga", "home_sog", "home_win",
        "home_powerplay_pct", "home_pk_pct", "home_powerplays", "home_pk", "home_faceoffwin_pct",
        "home_pims", "home_hits", "home_blockedshots", "home_giveaways", "home_takeaways"
    ]].rename(columns={
        "home_team_abbrev": "team_abbrev", "home_gf": "goals_for", "home_ga": "goals_against",
        "home_sog": "shots_on_goal", "home_win": "win", "home_powerplay_pct": "powerplay_pct",
        "home_pk_pct": "penalty_kill_pct", "home_powerplays": "powerplays", "home_pk": "penalty_kills",
        "home_faceoffwin_pct": "faceoffwin_pct", "home_pims": "pims", "home_hits": "hits",
        "home_blockedshots": "blockedshots", "home_giveaways": "giveaways", "home_takeaways": "takeaways"
    })

    away_stats = df[[
        "date", "season", "away_team_abbrev", "away_gf", "away_ga", "away_sog", "home_win",
        "away_powerplay_pct", "away_pk_pct", "away_powerplays", "away_pk", "away_faceoffwin_pct",
        "away_pims", "away_hits", "away_blockedshots", "away_giveaways", "away_takeaways"
    ]].rename(columns={
        "away_team_abbrev": "team_abbrev", "away_gf": "goals_for", "away_ga": "goals_against",
        "away_sog": "shots_on_goal", "away_powerplay_pct": "powerplay_pct",
        "away_pk_pct": "penalty_kill_pct", "away_powerplays": "powerplays", "away_pk": "penalty_kills",
        "away_faceoffwin_pct": "faceoffwin_pct", "away_pims": "pims", "away_hits": "hits",
        "away_blockedshots": "blockedshots", "away_giveaways": "giveaways", "away_takeaways": "takeaways"
    })

    away_stats["win"] = 1 - away_stats["home_win"]
    away_stats = away_stats.drop(columns=["home_win"])

    combined_stats = pd.concat([home_stats, away_stats], ignore_index=True)
    combined_stats = combined_stats.sort_values(by=["team_abbrev", "season", "date"]).reset_index(drop=True)

    # Compute rolling stats
    combined_stats["gf_per_game_l5"] = combined_stats.groupby(["team_abbrev", "season"])["goals_for"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    combined_stats["ga_per_game_l5"] = combined_stats.groupby(["team_abbrev", "season"])["goals_against"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    combined_stats["sog_per_game_l5"] = combined_stats.groupby(["team_abbrev", "season"])["shots_on_goal"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    combined_stats["wins_l5"] = combined_stats.groupby(["team_abbrev", "season"])["win"].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum().shift(1))
    combined_stats["powerplay_pct_l5"] = combined_stats.groupby(["team_abbrev", "season"])["powerplay_pct"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    combined_stats["penalty_kill_pct_l5"] = combined_stats.groupby(["team_abbrev", "season"])["penalty_kill_pct"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    combined_stats["powerplays_l5"] = combined_stats.groupby(["team_abbrev", "season"])["powerplays"].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum().shift(1))
    combined_stats["penalty_kills_l5"] = combined_stats.groupby(["team_abbrev", "season"])["penalty_kills"].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum().shift(1))
    combined_stats["faceoffwin_pct_l5"] = combined_stats.groupby(["team_abbrev", "season"])["faceoffwin_pct"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    combined_stats["pims_l5"] = combined_stats.groupby(["team_abbrev", "season"])["pims"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    combined_stats["hits_l5"] = combined_stats.groupby(["team_abbrev", "season"])["hits"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    combined_stats["blockedshots_l5"] = combined_stats.groupby(["team_abbrev", "season"])["blockedshots"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    combined_stats["giveaways_l5"] = combined_stats.groupby(["team_abbrev", "season"])["giveaways"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    combined_stats["takeaways_l5"] = combined_stats.groupby(["team_abbrev", "season"])["takeaways"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))

    combined_stats["games_l5"] = combined_stats.groupby(["team_abbrev", "season"])["win"].transform(
        lambda x: x.rolling(5, min_periods=1).count().shift(1))
    combined_stats["win_pct_l5"] = combined_stats["wins_l5"] / combined_stats["games_l5"]

    # Merge rolling stats back to original dataframe
    df = df.merge(
        combined_stats[[
            "date", "season", "team_abbrev", "gf_per_game_l5", "ga_per_game_l5", "sog_per_game_l5",
            "wins_l5", "win_pct_l5", "powerplay_pct_l5", "penalty_kill_pct_l5", "powerplays_l5",
            "penalty_kills_l5", "faceoffwin_pct_l5", "pims_l5", "hits_l5", "blockedshots_l5",
            "giveaways_l5", "takeaways_l5"
        ]],
        left_on=["home_team_abbrev", "date", "season"],
        right_on=["team_abbrev", "date", "season"],
        how="left"
    ).rename(columns={
        "gf_per_game_l5": "home_gf_per_game_l5", "ga_per_game_l5": "home_ga_per_game_l5",
        "sog_per_game_l5": "home_sog_per_game_l5", "wins_l5": "home_wins_l5", "win_pct_l5": "home_win_pct_l5",
        "powerplay_pct_l5": "home_powerplay_pct_l5", "penalty_kill_pct_l5": "home_penalty_kill_pct_l5",
        "powerplays_l5": "home_powerplay_opps_l5", "penalty_kills_l5": "home_pk_opps_l5",
        "faceoffwin_pct_l5": "home_faceoffwin_pct_l5", "pims_l5": "home_pims_l5", "hits_l5": "home_hits_l5",
        "blockedshots_l5": "home_blockedshots_l5", "giveaways_l5": "home_giveaways_l5", "takeaways_l5": "home_takeaways_l5"
    }).drop(columns=["team_abbrev"])

    df = df.merge(
        combined_stats[[
            "date", "season", "team_abbrev", "gf_per_game_l5", "ga_per_game_l5", "sog_per_game_l5",
            "wins_l5", "win_pct_l5", "powerplay_pct_l5", "penalty_kill_pct_l5", "powerplays_l5",
            "penalty_kills_l5", "faceoffwin_pct_l5", "pims_l5", "hits_l5", "blockedshots_l5",
            "giveaways_l5", "takeaways_l5"
        ]],
        left_on=["away_team_abbrev", "date", "season"],
        right_on=["team_abbrev", "date", "season"],
        how="left"
    ).rename(columns={
        "gf_per_game_l5": "away_gf_per_game_l5", "ga_per_game_l5": "away_ga_per_game_l5",
        "sog_per_game_l5": "away_sog_per_game_l5", "wins_l5": "away_wins_l5", "win_pct_l5": "away_win_pct_l5",
        "powerplay_pct_l5": "away_powerplay_pct_l5", "penalty_kill_pct_l5": "away_penalty_kill_pct_l5",
        "powerplays_l5": "away_powerplay_opps_l5", "penalty_kills_l5": "away_pk_opps_l5",
        "faceoffwin_pct_l5": "away_faceoffwin_pct_l5", "pims_l5": "away_pims_l5", "hits_l5": "away_hits_l5",
        "blockedshots_l5": "away_blockedshots_l5", "giveaways_l5": "away_giveaways_l5", "takeaways_l5": "away_takeaways_l5"
    }).drop(columns=["team_abbrev"])

    cols_to_round = [c for c in df.columns if c.endswith(("_l5"))]
    df[cols_to_round] = df[cols_to_round].round(3)
    df.to_csv(CSV_FILE, index=False)
    print("✅ Rolling averages computed and saved")


def step2_compute_goal_diffs():
    """Step 2B: Compute goal and shot differences."""
    print("=== STEP 2B: Compute Goal/Shot Differences ===")

    df = pd.read_csv(CSV_FILE, parse_dates=["date"])

    df["home_goal_diff_l5"] = df["home_gf_per_game_l5"] - df["away_gf_per_game_l5"]
    df["home_ga_diff_l5"] = df["home_ga_per_game_l5"] - df["away_ga_per_game_l5"]
    df["home_shot_diff_l5"] = df["home_sog_per_game_l5"] - df["away_sog_per_game_l5"]

    cols_to_round = [
        "home_gf_per_game_l5", "away_gf_per_game_l5", "home_ga_per_game_l5", "away_ga_per_game_l5",
        "home_sog_per_game_l5", "away_sog_per_game_l5", "home_goal_diff_l5", "home_ga_diff_l5", "home_shot_diff_l5",
    ]

    df[cols_to_round] = df[cols_to_round].round(3)
    df.to_csv(CSV_FILE, index=False)
    print("✅ Goal/shot differences computed and saved")


# ===== STEP 3: FETCH GOALIE DATA =====
async def fetch_all_goalie_data(game_ids):
    """Fetch goalie data for all games (optimized)."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:

        tasks = [
            fetch_boxscores(session, game_id, semaphore)
            for game_id in game_ids
        ]

        boxscores = await asyncio.gather(*tasks)

    # Extract stats (CPU-bound, keep it outside async context)
    results = [
        extract_all_goalie_basic_stats(boxscore)
        for boxscore in boxscores
        if boxscore is not None
    ]

    return results



def step3_fetch_goalie_data():
    """Step 3: Fetch goalie performance data."""
    print("\n=== STEP 3: Fetch Goalie Performance Data ===")

    df = pd.read_csv(CSV_FILE, parse_dates=["date"])
    game_ids = df["game_id"].tolist()

    print(f"Fetching goalie data for {len(game_ids)} games...")
    goalie_data_rows = asyncio.run(fetch_all_goalie_data(game_ids))
    goalie_df = pd.DataFrame(goalie_data_rows)
    print(f"✅ Fetched {len(goalie_data_rows)} goalie records")

    # Prepare goalie long format
    goalie_df["date"] = pd.to_datetime(goalie_df["date"])
    goalie_df = goalie_df.sort_values("date").reset_index(drop=True)

    home_goalies = goalie_df[["game_id", "date", "season", *HOME_RENAME.keys()]].rename(columns=HOME_RENAME)
    away_goalies = goalie_df[["game_id", "date", "season", *AWAY_RENAME.keys()]].rename(columns=AWAY_RENAME)

    goalie_long = (
        pd.concat([home_goalies, away_goalies], ignore_index=True)
        .sort_values(["goalie", "season", "date"])
        .reset_index(drop=True)
    )

    # Compute rolling stats for goalies
    for stat in GOALIE_STATS:
        goalie_long[f"{stat}_l5"] = (
            goalie_long.groupby(["goalie", "season"])[stat]
            .transform(lambda x: x.rolling(ROLLING_N, min_periods=1).mean().shift(1))
        )

    goalie_l5 = goalie_long[["game_id", "goalie"] + [f"{s}_l5" for s in GOALIE_STATS]]

    # Merge home goalie stats
    goalie_df = goalie_df.merge(
        goalie_l5,
        left_on=["game_id", "home_goalie_starter"],
        right_on=["game_id", "goalie"],
        how="left"
    ).rename(columns={f"{s}_l5": f"home_goalie_{s}_l5" for s in GOALIE_STATS}).drop(columns=["goalie"])

    # Merge away goalie stats
    goalie_df = goalie_df.merge(
        goalie_l5,
        left_on=["game_id", "away_goalie_starter"],
        right_on=["game_id", "goalie"],
        how="left"
    ).rename(columns={f"{s}_l5": f"away_goalie_{s}_l5" for s in GOALIE_STATS}).drop(columns=["goalie"])

    # Team save pct L5
    team_long = pd.concat(
        [
            goalie_df[["game_id", "date", "season", "home_team_abbrev", "home_save_pct"]]
            .rename(columns={"home_team_abbrev": "team", "home_save_pct": "save_pct"}),
            goalie_df[["game_id", "date", "season", "away_team_abbrev", "away_save_pct"]]
            .rename(columns={"away_team_abbrev": "team", "away_save_pct": "save_pct"}),
        ],
        ignore_index=True
    ).sort_values(["team", "season", "date"]).reset_index(drop=True)

    team_long["team_save_pct_l5"] = (
        team_long.groupby(["team", "season"])["save_pct"]
        .transform(lambda x: x.rolling(ROLLING_N, min_periods=1).mean().shift(1))
    )

    # Merge team stats back
    goalie_df = goalie_df.merge(
        team_long[["game_id", "team", "team_save_pct_l5"]],
        left_on=["game_id", "home_team_abbrev"],
        right_on=["game_id", "team"],
        how="left"
    ).rename(columns={"team_save_pct_l5": "home_team_save_pct_l5"}).drop(columns=["team"])

    goalie_df = goalie_df.merge(
        team_long[["game_id", "team", "team_save_pct_l5"]],
        left_on=["game_id", "away_team_abbrev"],
        right_on=["game_id", "team"],
        how="left"
    ).rename(columns={"team_save_pct_l5": "away_team_save_pct_l5"}).drop(columns=["team"])

    # Merge into main dataframe

    goalie_df = (
        goalie_df
        .sort_values("date")
        .groupby("game_id", as_index=False)
        .first()
    )

    main_df = pd.read_csv(CSV_FILE, parse_dates=["date"])
    main_df = main_df.merge(
        goalie_df[GOALIE_MERGE_COLS],
        on="game_id",
        how="left",
        validate="one_to_one"
    )
    main_df.to_csv(CSV_FILE, index=False)
    print("✅ Goalie data merged into main dataframe")


# ===== STEP 4: FETCH SEASON STANDINGS DATA =====
async def build_season_stats_dataframe():
    """Fetch standings data for all dates in the dataset (optimized)."""

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    df = pd.read_csv(CSV_FILE, parse_dates=["date"])

    # Group games by date once (HUGE speedup)
    games_by_date = dict(tuple(df.groupby("date")))
    unique_dates = list(games_by_date.keys())

    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:

        # Create tasks for all dates
        tasks = [
            fetch_standings_info(session, date_val.strftime("%Y-%m-%d"), semaphore)
            for date_val in unique_dates
        ]

        # Run requests concurrently (semaphore controls rate)
        standings_results = await asyncio.gather(*tasks)

    standing_rows = []

    for i, (date_val, standings_data) in enumerate(
        zip(unique_dates, standings_results)
    ):
        if not standings_data:
            continue

        date_str = date_val.strftime("%Y-%m-%d")

        playing_teams = extract_all_teams_playing_on_date(date_val)
        standings_by_team = extract_all_standing_stats(
            standings_data, playing_teams
        )

        # Iterate only games on this date
        for _, row in games_by_date[date_val].iterrows():
            out_row = {
                "game_id": row["game_id"],
                "date": date_str,
            }

            home_stats = standings_by_team.get(row["home_team_abbrev"], {})
            away_stats = standings_by_team.get(row["away_team_abbrev"], {})

            for field in STANDINGS_FIELDS:
                out_row[f"home_{field}"] = home_stats.get(field)
                out_row[f"away_{field}"] = away_stats.get(field)

            standing_rows.append(out_row)

        if (i + 1) % 100 == 0:
            print(f"✅ Processed {i + 1}/{len(unique_dates)} dates")

    return pd.DataFrame(standing_rows)


def step4_fetch_season_stats():
    """Step 4: Fetch and merge season standings data."""
    print("\n=== STEP 4: Fetch Season Standings Data ===")

    standing_df = asyncio.run(build_season_stats_dataframe())

    standing_df["home_gamesPlayed"] = standing_df["home_gamesPlayed"].astype(int)
    standing_df["away_gamesPlayed"] = standing_df["away_gamesPlayed"].astype(int)

    standing_df["home_win_pct_season"] = (
        (standing_df["home_homeWins"] + standing_df["home_roadWins"]) / standing_df["home_gamesPlayed"]
    )

    standing_df["away_win_pct_season"] = (
        (standing_df["away_homeWins"] + standing_df["away_roadWins"]) / standing_df["away_gamesPlayed"]
    )

    standing_df["home_home_win_pct"] = standing_df["home_homeWins"] / standing_df["home_homeGamesPlayed"]
    standing_df["away_away_win_pct"] = standing_df["away_roadWins"] / standing_df["away_roadGamesPlayed"]

    standing_df["home_gf_per_game_season"] = standing_df["home_goalsForPctg"]
    standing_df["away_gf_per_game_season"] = standing_df["away_goalsForPctg"]

    standing_df["home_pointPctg_season"] = standing_df["home_pointPctg"]
    standing_df["away_pointPctg_season"] = standing_df["away_pointPctg"]

    standing_df["pointPctg_diff"] = (
        standing_df["home_pointPctg_season"] - standing_df["away_pointPctg_season"]
    )

    standing_df["home_win_streak"] = standing_df.apply(
        lambda r: (r["home_streakCount"] - 1) if r["home_streakCode"] == "W" else 0, axis=1
    )

    standing_df["away_win_streak"] = standing_df.apply(
        lambda r: (r["away_streakCount"] - 1) if r["away_streakCode"] == "W" else 0, axis=1
    )

    standing_df = standing_df.fillna(0)

    df = pd.read_csv(CSV_FILE, parse_dates=["date"])
    df = df.merge(
        standing_df[["game_id"] + SEASON_STATS],
        on="game_id",
        how="left",
        validate="one_to_one"
    )
    df.to_csv(CSV_FILE, index=False)
    print("✅ Season stats data merged and saved")

# ===== STEP 5: ROUND NUMERIC COLUMNS =====
def step5_round_numeric_columns():
    """Step 5: Round all numeric columns to 3 decimal places."""
    print("\n=== STEP 5: Round Numeric Columns ===")

    df = pd.read_csv(CSV_FILE, parse_dates=["date"])
    df = df.round(3)
    df.to_csv(CSV_FILE, index=False)
    print("✅ All numeric columns rounded to 3 decimals")


# ===== STEP 6: COMPUTE REST DAYS =====
def step6_compute_rest_days():
    """Step 6: Compute team and goalie rest days."""
    print("\n=== STEP 6: Compute Rest Days ===")

    df = pd.read_csv(CSV_FILE, parse_dates=["date"])

    # Team rest days
    home_games = df[["date", "season", "home_team_abbrev"]].rename(columns={"home_team_abbrev": "team"})
    away_games = df[["date", "season", "away_team_abbrev"]].rename(columns={"away_team_abbrev": "team"})
    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.sort_values(["team", "season", "date"])

    team_games["team_rest_days"] = (
        team_games.groupby(["team", "season"])["date"].diff().dt.days - 1
    )

    df = df.merge(
        team_games[["team", "season", "date", "team_rest_days"]],
        left_on=["home_team_abbrev", "season", "date"],
        right_on=["team", "season", "date"],
        how="left"
    ).rename(columns={"team_rest_days": "home_rest_days"}).drop(columns=["team"])

    df = df.merge(
        team_games[["team", "season", "date", "team_rest_days"]],
        left_on=["away_team_abbrev", "season", "date"],
        right_on=["team", "season", "date"],
        how="left"
    ).rename(columns={"team_rest_days": "away_rest_days"}).drop(columns=["team"])

    # Goalie rest days
    home_goalie_games = df[[
        "date", "season", "home_team_abbrev", "home_goalie_starter"
    ]].rename(columns={"home_team_abbrev": "team", "home_goalie_starter": "goalie"})

    away_goalie_games = df[[
        "date", "season", "away_team_abbrev", "away_goalie_starter"
    ]].rename(columns={"away_team_abbrev": "team", "away_goalie_starter": "goalie"})

    goalie_games = pd.concat([home_goalie_games, away_goalie_games], ignore_index=True)
    goalie_games = goalie_games.sort_values(["goalie", "team", "season", "date"])

    goalie_games["goalie_rest_days"] = (
        goalie_games.groupby(["goalie", "team", "season"])["date"].diff().dt.days - 1
    )

    df = df.merge(
        goalie_games[["goalie", "team", "season", "date", "goalie_rest_days"]],
        left_on=["home_goalie_starter", "home_team_abbrev", "season", "date"],
        right_on=["goalie", "team", "season", "date"],
        how="left"
    ).rename(columns={"goalie_rest_days": "home_goalie_rest_days"}).drop(columns=["goalie", "team"])

    df = df.merge(
        goalie_games[["goalie", "team", "season", "date", "goalie_rest_days"]],
        left_on=["away_goalie_starter", "away_team_abbrev", "season", "date"],
        right_on=["goalie", "team", "season", "date"],
        how="left"
    ).rename(columns={"goalie_rest_days": "away_goalie_rest_days"}).drop(columns=["goalie", "team"])

    df.to_csv(CSV_FILE, index=False)
    print("✅ Rest days computed and saved")


# ===== STEP 7: HEAD-TO-HEAD DATA =====
def step7_compute_head_to_head():
    """Step 7: Compute head-to-head statistics."""
    print("\n=== STEP 7: Compute Head-to-Head Data ===")

    df = pd.read_csv(CSV_FILE, parse_dates=["date"])

    df["matchup"] = df.apply(
        lambda r: "_".join(sorted([r["home_team_abbrev"], r["away_team_abbrev"]])), axis=1
    )

    df = df.sort_values(["season", "date"]).reset_index(drop=True)

    # Build head-to-head long format
    h2h_long = pd.concat([
        df[[
            "game_id", "date", "season", "matchup", "home_team_abbrev", "away_team_abbrev", "home_gf", "home_win"
        ]]
        .rename(columns={
            "home_team_abbrev": "team", "away_team_abbrev": "opponent", "home_gf": "gf", "home_win": "win"
        }),

        df[[
            "game_id", "date", "season", "matchup", "away_team_abbrev", "home_team_abbrev", "away_gf", "home_win"
        ]]
        .assign(win=lambda x: 1 - x["home_win"])
        .rename(columns={"away_team_abbrev": "team", "home_team_abbrev": "opponent", "away_gf": "gf"})
        .drop(columns="home_win")
    ], ignore_index=True)

    h2h_long = h2h_long.sort_values(["season", "matchup", "date"])

    h2h_long["h2h_wins"] = (
        h2h_long.groupby(["season", "matchup", "team"])["win"]
        .transform(lambda s: s.cumsum().shift(1))
        .fillna(0)
    )

    h2h_long["h2h_gf"] = (
        h2h_long.groupby(["season", "matchup", "team"])["gf"]
        .transform(lambda s: s.expanding().mean().shift(1))
        .fillna(0)
    ).round(3)

    home_stats = h2h_long.rename(columns={
        "team": "home_team_abbrev", "h2h_wins": "home_h2h_wins", "h2h_gf": "home_h2h_gf"
    })[["game_id", "home_team_abbrev", "home_h2h_wins", "home_h2h_gf"]]

    away_stats = h2h_long.rename(columns={
        "team": "away_team_abbrev", "h2h_wins": "away_h2h_wins", "h2h_gf": "away_h2h_gf"
    })[["game_id", "away_team_abbrev", "away_h2h_wins", "away_h2h_gf"]]

    df = df.merge(home_stats, on=["game_id", "home_team_abbrev"], how="left")
    df = df.merge(away_stats, on=["game_id", "away_team_abbrev"], how="left")

    df["home_h2h_wins_diff"] = df["home_h2h_wins"] - df["away_h2h_wins"]

    df.to_csv(CSV_FILE, index=False)
    print("✅ Head-to-head data computed and saved")


# ===== MAIN FUNCTION =====
def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("NHL PREDICTOR - DATA SCRAPER")
    print("=" * 60)

    try:
        step1_fetch_basic_game_info()
        step2_compute_rolling_averages()
        step2_compute_goal_diffs()
        step3_fetch_goalie_data()
        step4_fetch_season_stats()
        step5_round_numeric_columns()
        step6_compute_rest_days()
        step7_compute_head_to_head()

        print("\n" + "=" * 60)
        print("ALL STEPS COMPLETED SUCCESSFULLY!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
