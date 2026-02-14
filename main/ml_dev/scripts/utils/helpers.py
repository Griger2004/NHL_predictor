import pandas as pd
from config import CSV_FILE, STANDINGS_FIELDS


# ==========================================================
# BASIC STAT HELPERS
# ==========================================================

def get_num_powerplays(powerplay_str):
    """Extract power play goals and opportunities from string (e.g., '2/5')."""
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
    """Extract name from nested object structure."""
    if not obj:
        return ""
    if isinstance(obj, dict):
        return obj.get("default", "")
    return str(obj)


def extract_category_stat(stats, category):
    """
    Extract category stats for home and away teams.
    Returns (home_value, away_value).
    """
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
    """Convert fractional stat (e.g., '5/10') to float (0.5)."""
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
    """
    Get starting goalie from list.
    Returns (goalie_name, goalie_object).
    """
    if not goalies:
        return "", None

    starter = next((g for g in goalies if g.get("starter")), goalies[0])
    return extract_name(starter.get("name", {})), starter


# ==========================================================
# TEAM / GOALIE EXTRACTION HELPERS
# ==========================================================

def extract_all_teams_playing_on_date(game_date, df=None):
    """
    Extract all team abbreviations playing on a specific date.
    """
    if df is None:
        df = pd.read_csv(CSV_FILE, parse_dates=["date"])

    teams_home = df[df["date"] == game_date]["home_team_abbrev"].unique().tolist()
    teams_away = df[df["date"] == game_date]["away_team_abbrev"].unique().tolist()

    return set(teams_home + teams_away)


def extract_all_standings_stats(standings_data, playing_teams):
    """
    Extract standings stats only for teams playing on a specific date.
    Returns dict keyed by team abbreviation.
    """
    rows = {}

    for team in standings_data.get("standings", []):
        abbrev = team.get("teamAbbrev", {}).get("default", "")

        # Handle ARI → UTA rename
        if abbrev == "ARI":
            abbrev = "UTA"

        if abbrev not in playing_teams:
            continue

        rows[abbrev] = {field: team.get(field) for field in STANDINGS_FIELDS}

    return rows
