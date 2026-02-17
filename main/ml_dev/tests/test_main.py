import os
import sys

import pandas as pd
import pytest

TESTS_DIR = os.path.dirname(__file__)
SCRIPTS_DIR = os.path.abspath(os.path.join(TESTS_DIR, "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import main as pipeline_main


# Ensures future/pre-game entries are skipped in team stat extraction.
def test_extract_all_basic_team_stats_future_game():
    game_data = {"gameState": "FUT"}
    result = pipeline_main.extract_all_basic_team_stats(game_data)
    assert result is None


# Validates basic team stat extraction with power-play math and ARI -> UTA rename.
def test_extract_all_basic_team_stats_basic():
    game_data = {
        "id": "2024020001",
        "gameState": "FINAL",
        "gameDate": "2024-10-01",
        "season": 2024,
        "homeTeam": {
            "placeName": {"default": "Arizona"},
            "name": {"default": "Coyotes"},
            "abbrev": "ARI",
            "score": 4,
            "sog": 32,
        },
        "awayTeam": {
            "placeName": {"default": "Boston"},
            "name": {"default": "Bruins"},
            "abbrev": "BOS",
            "score": 2,
            "sog": 28,
        },
        "summary": {
            "teamGameStats": [
                {
                    "category": "faceoffWinningPctg",
                    "homeValue": 51.2,
                    "awayValue": 48.8,
                },
                {"category": "powerPlay", "homeValue": "2/5", "awayValue": "1/4"},
                {"category": "powerPlayPctg", "homeValue": 40.0, "awayValue": 25.0},
                {"category": "pim", "homeValue": 10, "awayValue": 8},
                {"category": "hits", "homeValue": 20, "awayValue": 15},
                {"category": "blockedShots", "homeValue": 12, "awayValue": 9},
                {"category": "takeaways", "homeValue": 7, "awayValue": 5},
                {"category": "giveaways", "homeValue": 6, "awayValue": 4},
            ]
        },
    }

    row = pipeline_main.extract_all_basic_team_stats(game_data)

    assert row["home_team"] == "Utah Mammoth"
    assert row["home_team_abbrev"] == "UTA"
    assert row["away_team"] == "Boston Bruins"
    assert row["home_win"] == 1
    assert row["home_powerplays"] == 5
    assert row["away_powerplays"] == 4
    assert row["home_pk"] == 3
    assert row["away_pk"] == 3
    assert row["home_pk_pct"] == 75.0
    assert row["away_pk_pct"] == 60.0


# Validates goalie stat extraction including starter selection and save pct aggregation.
def test_extract_all_basic_goalie_stats():
    boxscore_data = {
        "id": "2024020001",
        "gameDate": "2024-10-01",
        "season": 2024,
        "homeTeam": {"abbrev": "ARI"},
        "awayTeam": {"abbrev": "NYR"},
        "playerByGameStats": {
            "homeTeam": {
                "goalies": [
                    {
                        "starter": True,
                        "name": {"default": "Goalie A"},
                        "savePctg": 0.938,
                        "goalsAgainst": 2,
                        "saves": 30,
                        "shotsAgainst": 32,
                        "evenStrengthShotsAgainst": "20/22",
                        "powerPlayShotsAgainst": "8/10",
                        "shorthandedShotsAgainst": "2/2",
                        "evenStrengthGoalsAgainst": 1,
                        "powerPlayGoalsAgainst": 1,
                    },
                    {
                        "starter": False,
                        "name": {"default": "Goalie A2"},
                        "savePctg": 0.833,
                        "goalsAgainst": 2,
                        "saves": 10,
                        "shotsAgainst": 12,
                    },
                ]
            },
            "awayTeam": {
                "goalies": [
                    {
                        "starter": True,
                        "name": {"default": "Goalie B"},
                        "savePctg": 0.933,
                        "goalsAgainst": 2,
                        "saves": 28,
                        "shotsAgainst": 30,
                        "evenStrengthShotsAgainst": "18/20",
                        "powerPlayShotsAgainst": "10/10",
                        "shorthandedShotsAgainst": "0/0",
                        "evenStrengthGoalsAgainst": 2,
                        "powerPlayGoalsAgainst": 0,
                    }
                ]
            },
        },
    }

    row = pipeline_main.extract_all_basic_goalie_stats(boxscore_data)

    assert row["home_team_abbrev"] == "UTA"
    assert row["away_team_abbrev"] == "NYR"
    assert row["home_goalie_starter"] == "Goalie A"
    assert row["away_goalie_starter"] == "Goalie B"
    assert row["home_save_pct"] == pytest.approx(0.909, abs=1e-3)
    assert row["home_goalie_evenStrengthShotsAgainst"] == pytest.approx(0.909, abs=1e-3)


# Checks head-to-head rolling stats and win diff logic.
def test_add_head_to_head():
    pipeline = pipeline_main.NHLPipeline()
    pipeline.df = pd.DataFrame(
        [
            {
                "game_id": "2024020001",
                "date": pd.to_datetime("2024-10-01"),
                "season": 2024,
                "home_team_abbrev": "AAA",
                "away_team_abbrev": "BBB",
                "home_gf": 3,
                "away_gf": 2,
                "home_win": 1,
            },
            {
                "game_id": "2024020002",
                "date": pd.to_datetime("2024-10-05"),
                "season": 2024,
                "home_team_abbrev": "BBB",
                "away_team_abbrev": "AAA",
                "home_gf": 1,
                "away_gf": 4,
                "home_win": 0,
            },
        ]
    )

    pipeline.add_head_to_head()
    df = pipeline.df.set_index("game_id")

    assert df.loc["2024020002", "home_h2h_wins"] == 0
    assert df.loc["2024020002", "away_h2h_wins"] == 1
    assert df.loc["2024020002", "home_h2h_gf"] == 2.0
    assert df.loc["2024020002", "away_h2h_gf"] == 3.0
    assert df.loc["2024020002", "home_h2h_wins_diff"] == -1


# Confirms fetch_games creates a dataframe and parses dates.
def test_fetch_games_sets_dataframe(tmp_path, mocker):
    async def fake_fetch_season_games(season):
        return [
            {
                "game_id": "2024020001",
                "date": "2024-10-01",
                "season": season,
                "home_team_abbrev": "AAA",
                "away_team_abbrev": "BBB",
            }
        ]

    mocker.patch.object(pipeline_main, "SEASONS", [2024])
    mocker.patch.object(pipeline_main, "OUTPUT_DIR", str(tmp_path / "out"))
    mocker.patch.object(
        pipeline_main,
        "CSV_FILE",
        str(tmp_path / "out" / "nhl_data.csv"),
    )
    mocker.patch.object(pipeline_main, "fetch_season_games", new=fake_fetch_season_games)

    pipeline = pipeline_main.NHLPipeline()
    pipeline.fetch_games()

    assert pipeline.df is not None
    assert len(pipeline.df) == 1
    assert pd.api.types.is_datetime64_any_dtype(pipeline.df["date"])


# Verifies rolling EWM and diff columns are computed for team stats.
def test_add_team_rolling_features():
    pipeline = pipeline_main.NHLPipeline()
    pipeline.df = pd.DataFrame(
        [
            {
                "game_id": "2024020001",
                "date": pd.to_datetime("2024-10-01"),
                "season": 2024,
                "home_team_abbrev": "AAA",
                "away_team_abbrev": "BBB",
                "home_gf": 3,
                "home_ga": 2,
                "home_sog": 30,
                "home_win": 1,
                "home_powerplay_pct": 30.0,
                "home_pk_pct": 80.0,
                "home_powerplays": 4,
                "home_pk": 3,
                "home_faceoffwin_pct": 52.0,
                "home_pims": 8,
                "home_hits": 15,
                "home_blockedshots": 10,
                "home_giveaways": 5,
                "home_takeaways": 6,
                "away_gf": 2,
                "away_ga": 3,
                "away_sog": 28,
                "away_powerplay_pct": 25.0,
                "away_pk_pct": 75.0,
                "away_powerplays": 4,
                "away_pk": 3,
                "away_faceoffwin_pct": 48.0,
                "away_pims": 10,
                "away_hits": 12,
                "away_blockedshots": 9,
                "away_giveaways": 6,
                "away_takeaways": 4,
            },
            {
                "game_id": "2024020002",
                "date": pd.to_datetime("2024-10-03"),
                "season": 2024,
                "home_team_abbrev": "BBB",
                "away_team_abbrev": "AAA",
                "home_gf": 4,
                "home_ga": 1,
                "home_sog": 35,
                "home_win": 1,
                "home_powerplay_pct": 40.0,
                "home_pk_pct": 90.0,
                "home_powerplays": 5,
                "home_pk": 4,
                "home_faceoffwin_pct": 53.0,
                "home_pims": 6,
                "home_hits": 18,
                "home_blockedshots": 11,
                "home_giveaways": 4,
                "home_takeaways": 7,
                "away_gf": 1,
                "away_ga": 4,
                "away_sog": 26,
                "away_powerplay_pct": 20.0,
                "away_pk_pct": 70.0,
                "away_powerplays": 3,
                "away_pk": 2,
                "away_faceoffwin_pct": 47.0,
                "away_pims": 12,
                "away_hits": 14,
                "away_blockedshots": 8,
                "away_giveaways": 7,
                "away_takeaways": 5,
            },
        ]
    )

    pipeline.add_team_rolling_features()
    df = pipeline.df.set_index("game_id")

    assert df.loc["2024020002", "home_gf_ewm"] == pytest.approx(2.0, abs=1e-3)
    assert df.loc["2024020002", "away_gf_ewm"] == pytest.approx(3.0, abs=1e-3)
    assert df.loc["2024020002", "home_goal_diff_ewm"] == pytest.approx(-1.0, abs=1e-3)


# Verifies goalie EWM features and team save pct EWMs are merged into main data.
def test_add_goalie_features(mocker):
    async def fake_fetch_all_goalie_data(game_ids):
        return [
            {
                "game_id": "2024020001",
                "date": "2024-10-01",
                "season": 2024,
                "home_team_abbrev": "AAA",
                "away_team_abbrev": "BBB",
                "home_goalie_starter": "Goalie A",
                "away_goalie_starter": "Goalie B",
                "home_save_pct": 0.91,
                "away_save_pct": 0.92,
                "home_goalie_save_pct": 0.91,
                "away_goalie_save_pct": 0.92,
                "home_goalie_ga": 2,
                "away_goalie_ga": 3,
                "home_goalie_saves": 30,
                "away_goalie_saves": 28,
                "home_goalie_evenStrengthShotsAgainst": 0.9,
                "away_goalie_evenStrengthShotsAgainst": 0.9,
                "home_goalie_powerPlayShotsAgainst": 0.8,
                "away_goalie_powerPlayShotsAgainst": 0.85,
                "home_goalie_shorthandedShotsAgainst": 1.0,
                "away_goalie_shorthandedShotsAgainst": 1.0,
                "home_goalie_evenStrengthGoalsAgainst": 1,
                "away_goalie_evenStrengthGoalsAgainst": 2,
                "home_goalie_powerPlayGoalsAgainst": 1,
                "away_goalie_powerPlayGoalsAgainst": 1,
            },
            {
                "game_id": "2024020002",
                "date": "2024-10-03",
                "season": 2024,
                "home_team_abbrev": "BBB",
                "away_team_abbrev": "AAA",
                "home_goalie_starter": "Goalie B",
                "away_goalie_starter": "Goalie A",
                "home_save_pct": 0.93,
                "away_save_pct": 0.9,
                "home_goalie_save_pct": 0.93,
                "away_goalie_save_pct": 0.9,
                "home_goalie_ga": 1,
                "away_goalie_ga": 4,
                "home_goalie_saves": 32,
                "away_goalie_saves": 25,
                "home_goalie_evenStrengthShotsAgainst": 0.95,
                "away_goalie_evenStrengthShotsAgainst": 0.85,
                "home_goalie_powerPlayShotsAgainst": 0.9,
                "away_goalie_powerPlayShotsAgainst": 0.75,
                "home_goalie_shorthandedShotsAgainst": 1.0,
                "away_goalie_shorthandedShotsAgainst": 1.0,
                "home_goalie_evenStrengthGoalsAgainst": 1,
                "away_goalie_evenStrengthGoalsAgainst": 3,
                "home_goalie_powerPlayGoalsAgainst": 0,
                "away_goalie_powerPlayGoalsAgainst": 1,
            },
        ]

    mocker.patch.object(
        pipeline_main,
        "fetch_all_goalie_data",
        new=fake_fetch_all_goalie_data,
    )

    pipeline = pipeline_main.NHLPipeline()
    pipeline.df = pd.DataFrame(
        [
            {
                "game_id": "2024020001",
                "date": pd.to_datetime("2024-10-01"),
                "season": 2024,
                "home_team_abbrev": "AAA",
                "away_team_abbrev": "BBB",
            },
            {
                "game_id": "2024020002",
                "date": pd.to_datetime("2024-10-03"),
                "season": 2024,
                "home_team_abbrev": "BBB",
                "away_team_abbrev": "AAA",
            },
        ]
    )

    pipeline.add_goalie_features()
    df = pipeline.df.set_index("game_id")

    assert df.loc["2024020002", "home_goalie_save_pct_ewm"] == pytest.approx(0.92, abs=1e-3)
    assert df.loc["2024020002", "away_goalie_save_pct_ewm"] == pytest.approx(0.91, abs=1e-3)
    assert df.loc["2024020002", "home_team_save_pct_ewm"] == pytest.approx(0.92, abs=1e-3)


# Ensures standings-derived season stats and streak logic are merged correctly.
def test_add_standings_features(mocker):
    async def fake_build_season_stats_dataframe(df):
        return pd.DataFrame(
            [
                {
                    "game_id": "2024020001",
                    "date": pd.to_datetime("2024-10-01"),
                    "home_teamAbbrev": "AAA",
                    "away_teamAbbrev": "BBB",
                    "home_gamesPlayed": 2,
                    "away_gamesPlayed": 2,
                    "home_homeWins": 1,
                    "home_roadWins": 1,
                    "away_homeWins": 1,
                    "away_roadWins": 0,
                    "home_homeGamesPlayed": 1,
                    "away_roadGamesPlayed": 1,
                    "home_goalsForPctg": 3.0,
                    "away_goalsForPctg": 2.5,
                    "home_pointPctg": 0.75,
                    "away_pointPctg": 0.5,
                    "home_streakCount_CURR": 2,
                    "home_streakCode_CURR": "W",
                    "away_streakCount_CURR": 1,
                    "away_streakCode_CURR": "L",
                },
                {
                    "game_id": "2024020002",
                    "date": pd.to_datetime("2024-10-03"),
                    "home_teamAbbrev": "BBB",
                    "away_teamAbbrev": "AAA",
                    "home_gamesPlayed": 3,
                    "away_gamesPlayed": 3,
                    "home_homeWins": 2,
                    "home_roadWins": 0,
                    "away_homeWins": 1,
                    "away_roadWins": 2,
                    "home_homeGamesPlayed": 2,
                    "away_roadGamesPlayed": 2,
                    "home_goalsForPctg": 3.2,
                    "away_goalsForPctg": 2.8,
                    "home_pointPctg": 0.67,
                    "away_pointPctg": 0.75,
                    "home_streakCount_CURR": 1,
                    "home_streakCode_CURR": "W",
                    "away_streakCount_CURR": 3,
                    "away_streakCode_CURR": "W",
                },
            ]
        )

    mocker.patch.object(
        pipeline_main,
        "build_season_stats_dataframe",
        new=fake_build_season_stats_dataframe,
    )

    pipeline = pipeline_main.NHLPipeline()
    pipeline.df = pd.DataFrame(
        [
            {
                "game_id": "2024020001",
                "date": pd.to_datetime("2024-10-01"),
                "season": 2024,
                "home_team_abbrev": "AAA",
                "away_team_abbrev": "BBB",
            },
            {
                "game_id": "2024020002",
                "date": pd.to_datetime("2024-10-03"),
                "season": 2024,
                "home_team_abbrev": "BBB",
                "away_team_abbrev": "AAA",
            },
        ]
    )

    pipeline.add_standings_features()
    df = pipeline.df.set_index("game_id")

    assert df.loc["2024020002", "home_win_streak"] == 0
    assert df.loc["2024020002", "away_win_streak"] == 2
    assert df.loc["2024020002", "pointPctg_diff"] == pytest.approx(-0.08, abs=1e-3)


# Ensures team and goalie rest days are computed from prior game dates.
def test_add_rest_days():
    pipeline = pipeline_main.NHLPipeline()
    pipeline.df = pd.DataFrame(
        [
            {
                "game_id": "2024020001",
                "date": pd.to_datetime("2024-10-01"),
                "season": 2024,
                "home_team_abbrev": "AAA",
                "away_team_abbrev": "BBB",
                "home_goalie_starter": "Goalie A",
                "away_goalie_starter": "Goalie B",
            },
            {
                "game_id": "2024020002",
                "date": pd.to_datetime("2024-10-03"),
                "season": 2024,
                "home_team_abbrev": "BBB",
                "away_team_abbrev": "AAA",
                "home_goalie_starter": "Goalie B",
                "away_goalie_starter": "Goalie A",
            },
        ]
    )

    pipeline.add_rest_days()
    df = pipeline.df.set_index("game_id")

    assert df.loc["2024020002", "home_rest_days"] == 1
    assert df.loc["2024020002", "away_rest_days"] == 1
    assert df.loc["2024020002", "home_goalie_rest_days"] == 1
    assert df.loc["2024020002", "away_goalie_rest_days"] == 1
