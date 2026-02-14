from aiohttp import ClientTimeout

API_BASE_URL = "https://api-web.nhle.com"
OUTPUT_DIR = "generated/data/"
CSV_FILE = f"{OUTPUT_DIR}/nhl_data.csv"

TIMEOUT = ClientTimeout(total=15)
MAX_CONCURRENT_REQUESTS = 9
RETRIES = 3

MAX_GAMES = 1312  # there are 1312 games in a full NHL regular season (32 * 82 / 2)
SLEEP_SEC = 0.1

SEASONS = [2022, 2023, 2024, 2025]

FIELDNAMES = [
    "game_id", "date", "season", "home_team", "away_team",
    "home_team_abbrev", "away_team_abbrev", "home_win",
    "home_gf", "away_gf", "home_ga", "away_ga", "home_sog", "away_sog",
    "home_faceoffwin_pct", "away_faceoffwin_pct", "home_powerplays", "away_powerplays",
    "home_powerplay_pct", "away_powerplay_pct", "home_penalty_kill_pct", "away_penalty_kill_pct",
    "home_pims", "away_pims", "home_hits", "away_hits", "home_blockedshots", "away_blockedshots",
    "home_takeaways", "away_takeaways", "home_giveaways", "away_giveaways",
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
    "away_goalie_powerPlayGoalsAgainst", "home_goalie_save_pct_ewm", "home_goalie_ga_ewm",
    "home_goalie_saves_ewm", "home_goalie_ev_sa_ewm", "home_goalie_pp_sa_ewm", "home_goalie_sh_sa_ewm",
    "home_goalie_ev_ga_ewm", "home_goalie_pp_ga_ewm", "away_goalie_save_pct_ewm", "away_goalie_ga_ewm",
    "away_goalie_saves_ewm", "away_goalie_ev_sa_ewm", "away_goalie_pp_sa_ewm", "away_goalie_sh_sa_ewm",
    "away_goalie_ev_ga_ewm", "away_goalie_pp_ga_ewm", "home_team_save_pct_ewm", "away_team_save_pct_ewm",
]

SEASON_STATS = [
    "home_win_pct_season", "away_win_pct_season", "home_home_win_pct", "away_away_win_pct",
    "home_gf_per_game_season", "away_gf_per_game_season", "home_pointPctg_season",
    "away_pointPctg_season", "pointPctg_diff", "home_win_streak", "away_win_streak",
]

MAIN_STATS_TO_BE_BLEND = [
    "gf", "ga", "sog", "faceoffwin_pct", "powerplays", "powerplay_pct",
    "pk", "pk_pct", "pims", "hits", "blockedshots", "takeaways", "giveaways",
]

HOME_TEAM_STATS_COLS = [
    "date",
    "season",
    "home_team_abbrev",
    "home_gf",
    "home_ga",
    "home_sog",
    "home_win",
    "home_powerplay_pct",
    "home_pk_pct",
    "home_powerplays",
    "home_pk",
    "home_faceoffwin_pct",
    "home_pims",
    "home_hits",
    "home_blockedshots",
    "home_giveaways",
    "home_takeaways",
]

AWAY_TEAM_STATS_COLS = [
    "date",
    "season",
    "away_team_abbrev",
    "away_gf",
    "away_ga",
    "away_sog",
    "home_win",
    "away_powerplay_pct",
    "away_pk_pct",
    "away_powerplays",
    "away_pk",
    "away_faceoffwin_pct",
    "away_pims",
    "away_hits",
    "away_blockedshots",
    "away_giveaways",
    "away_takeaways",
]
