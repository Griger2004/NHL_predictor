import os
import asyncio
import time
import numpy as np
import pandas as pd

from api.client import ApiClient

from config import (
    API_BASE_URL,
    OUTPUT_DIR,
    CSV_FILE,
    TIMEOUT,
    MAX_CONCURRENT_REQUESTS,
    RETRIES,
    MAX_GAMES,
    SEASONS,
    STANDINGS_FIELDS,
    HOME_RENAME,
    AWAY_RENAME,
    GOALIE_STATS,
    GOALIE_MERGE_COLS,
    SEASON_STATS,
    HOME_TEAM_STATS_COLS,
    AWAY_TEAM_STATS_COLS,
)

from utils.helpers import (
    get_num_powerplays,
    calc_num_penalty_kills,
    calc_penalty_kill_pct,
    extract_name,
    extract_category_stat,
    calc_team_save_pct,
    extract_fractional_stat,
    get_starter_goalie,
    extract_all_teams_playing_on_date,
    extract_all_standings_stats,
)


# ===== STAT EXTRACTION =====

def extract_all_basic_team_stats(game_data):
    """Extract all game statistics from game data."""
    if game_data.get("gameState") in ["FUT", "LIVE", "PRE"]:
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
        home_place, home_name, home_abbrev = "Utah", "Mammoth", "UTA"

    if full_away_name in ("Arizona Coyotes", "Utah Utah Hockey Club"):
        away_place, away_name, away_abbrev = "Utah", "Mammoth", "UTA"

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

    return {
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
        "away_giveaways": away_giveaways,
    }


def extract_all_basic_goalie_stats(boxscore_data):
    """Extract all goalie statistics from boxscore data."""
    date_str = boxscore_data.get("gameDate", "")
    season_year = boxscore_data.get("season", 0)

    home = boxscore_data.get("homeTeam", {})
    away = boxscore_data.get("awayTeam", {})
    home_goalies = boxscore_data.get("playerByGameStats", {}).get("homeTeam", {}).get("goalies", [])
    away_goalies = boxscore_data.get("playerByGameStats", {}).get("awayTeam", {}).get("goalies", [])

    home_starter_goalie, _ = get_starter_goalie(home_goalies)
    away_starter_goalie, _ = get_starter_goalie(away_goalies)

    # TODO: Find endpoint that gives TEAM save percentage directly.
    if len(home_goalies) < 2:
        home_save_pct = home_goalies[0].get("savePctg", 0.0) if home_goalies else 0.0
    else:
        home_save_pct = calc_team_save_pct(
            home_goalies[0].get("saves", 0) + home_goalies[1].get("saves", 0),
            home_goalies[0].get("shotsAgainst", 0) + home_goalies[1].get("shotsAgainst", 0),
        )

    if len(away_goalies) < 2:
        away_save_pct = away_goalies[0].get("savePctg", 0.0) if away_goalies else 0.0
    else:
        away_save_pct = calc_team_save_pct(
            away_goalies[0].get("saves", 0) + away_goalies[1].get("saves", 0),
            away_goalies[0].get("shotsAgainst", 0) + away_goalies[1].get("shotsAgainst", 0),
        )

    home_starter_obj = home_goalies[0] if home_goalies and home_goalies[0].get("starter") else (home_goalies[1] if len(home_goalies) > 1 else {})
    away_starter_obj = away_goalies[0] if away_goalies and away_goalies[0].get("starter") else (away_goalies[1] if len(away_goalies) > 1 else {})

    home_abbrev = "UTA" if home.get("abbrev", "") == "ARI" else home.get("abbrev", "")
    away_abbrev = "UTA" if away.get("abbrev", "") == "ARI" else away.get("abbrev", "")

    return {
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


# ===== EWM HELPERS =====

_EWM_CARRY_WEIGHT = 0.7  # 70% last-season form, 30% toward league mean (FiveThirtyEight-style)


def _ewm_with_season_regression(df, entity_col, season_col, stat_col, alpha=0.3):
    """
    Compute shift(1) EWM with regression-to-mean at season boundaries.

    For each entity's first game of a new season the EWM is seeded with:
        0.7 * final_ewm_of_prev_season + 0.3 * prev_season_league_avg
    instead of carrying the raw value across seasons (leakage) or resetting
    to NaN (throws away all historical signal).

    df must be sorted by [entity_col, season_col, "date"] before calling.
    Returns a float Series aligned to df.index.
    """
    result = pd.Series(np.nan, index=df.index, dtype=float)
    seasons = sorted(df[season_col].unique())
    league_avg_by_season = df.groupby(season_col)[stat_col].mean()

    for _, entity_df in df.groupby(entity_col, sort=False):
        entity_df = entity_df.sort_values([season_col, "date"])
        prev_final_ewm = np.nan

        for season in seasons:
            mask = entity_df[season_col] == season
            if not mask.any():
                continue

            group = entity_df[mask]
            values = group[stat_col].to_numpy(dtype=float)
            idx = group.index

            # Seed for this season's first game
            if np.isnan(prev_final_ewm):
                seed = np.nan
            else:
                lg_avg = float(
                    league_avg_by_season[season]
                    if season in league_avg_by_season.index
                    else league_avg_by_season.mean()
                )
                seed = _EWM_CARRY_WEIGHT * prev_final_ewm + (1 - _EWM_CARRY_WEIGHT) * lg_avg

            # shift(1) semantics: ewm_vals[i] is the EWM *before* game i
            ewm_vals = np.full(len(values), np.nan)
            current_ewm = seed

            for i, v in enumerate(values):
                ewm_vals[i] = current_ewm
                if not np.isnan(v):
                    current_ewm = (
                        v if np.isnan(current_ewm)
                        else alpha * v + (1 - alpha) * current_ewm
                    )

            result.loc[idx] = ewm_vals
            prev_final_ewm = current_ewm

    return result


# ===== PIPELINE =====

class NHLPipeline:
    def __init__(self):
        self.df = None

    def _api_client(self):
        return ApiClient(API_BASE_URL, TIMEOUT, MAX_CONCURRENT_REQUESTS, RETRIES)

    def _require_data(self):
        if self.df is None:
            raise ValueError("No data loaded. Run fetch_games() first.")

    @staticmethod
    def _format_elapsed(seconds):
        mins, secs = divmod(seconds, 60)
        return f"{int(mins)}m {secs:.2f}s" if mins >= 1 else f"{secs:.2f}s"

    # --- Async fetch helpers ---

    async def _fetch_season_games(self, season):
        game_ids = [f"{season}02{str(n).zfill(4)}" for n in range(1, MAX_GAMES + 1)]
        print(f"  - Season {season}: requesting up to {len(game_ids)} regular-season game IDs...")
        async with self._api_client() as client:
            results = await asyncio.gather(*[
                client.get_json(f"/v1/wsc/game-story/{gid}") for gid in game_ids
            ])
        rows = []
        for game_data in results:
            if not game_data:
                continue
            row = extract_all_basic_team_stats(game_data)
            if row is None:
                break
            rows.append(row)
        print(f"  - Season {season}: kept {len(rows)} completed games")
        return rows

    async def _fetch_all_goalie_data(self, game_ids):
        async with self._api_client() as client:
            boxscores = await asyncio.gather(*[
                client.get_json(f"/v1/gamecenter/{gid}/boxscore") for gid in game_ids
            ])
        return [extract_all_basic_goalie_stats(b) for b in boxscores if b is not None]

    async def _build_standings_dataframe(self, df):
        games_by_date = dict(tuple(df.groupby("date")))
        unique_dates = list(games_by_date.keys())

        async with self._api_client() as client:
            standings_results = await asyncio.gather(*[
                client.get_json(f"/v1/standings/{d.strftime('%Y-%m-%d')}") for d in unique_dates
            ])

        rows = []
        for date_val, standings_data in zip(unique_dates, standings_results):
            if not standings_data:
                continue

            date_str = date_val.strftime("%Y-%m-%d")
            playing_teams = extract_all_teams_playing_on_date(date_val, df=df)
            standings_by_team = extract_all_standings_stats(standings_data, playing_teams)

            for _, row in games_by_date[date_val].iterrows():
                out_row = {"game_id": row["game_id"], "date": date_str}
                home_stats = standings_by_team.get(row["home_team_abbrev"], {})
                away_stats = standings_by_team.get(row["away_team_abbrev"], {})

                for field in STANDINGS_FIELDS:
                    out_row[f"home_{field}"] = home_stats.get(field)
                    out_row[f"away_{field}"] = away_stats.get(field)

                    # Snapshot current streak so future-game logic can look back one game
                    if field in ["streakCode", "streakCount"]:
                        out_row[f"home_{field}_CURR"] = home_stats.get(field)
                        out_row[f"away_{field}_CURR"] = away_stats.get(field)

                out_row["home_teamAbbrev"] = row["home_team_abbrev"]
                out_row["away_teamAbbrev"] = row["away_team_abbrev"]
                rows.append(out_row)

        return pd.DataFrame(rows)

    # --- Pipeline steps ---

    def fetch_games(self):
        """Step 1: Fetch basic game info for all seasons."""
        print("\n=== STEP 1: Fetch Basic Game Info ===")
        step_start = time.perf_counter()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Output directory ready: {OUTPUT_DIR}")
        if os.path.exists(CSV_FILE):
            os.remove(CSV_FILE)
            print(f"Removed existing output file: {CSV_FILE}")
        else:
            print(f"No existing output file found at: {CSV_FILE}")

        async def _run():
            results = await asyncio.gather(*[self._fetch_season_games(s) for s in SEASONS])
            return [row for season_rows in results for row in season_rows]

        total_requested = len(SEASONS) * MAX_GAMES
        all_rows = asyncio.run(_run())

        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"])
        self.df = df
        print(f"Fetched {len(df)} completed games")
        print(f"Step 1 completed in {self._format_elapsed(time.perf_counter() - step_start)}")

    def add_team_rolling_features(self):
        """Step 2: Compute rolling averages and differentials."""
        self._require_data()
        print("\n=== STEP 2: Compute Rolling Averages ===")

        df = self.df.copy()
        alpha = 0.3

        home_stats = df[HOME_TEAM_STATS_COLS].rename(columns={
            "home_team_abbrev": "team_abbrev",
            "home_gf": "gf",
            "home_ga": "ga",
            "home_sog": "sog",
            "home_win": "win",
            "home_powerplay_pct": "powerplay_pct",
            "home_pk_pct": "pk_pct",
            "home_powerplays": "powerplays",
            "home_pk": "penalty_kills",
            "home_faceoffwin_pct": "faceoffwin_pct",
            "home_pims": "pims",
            "home_hits": "hits",
            "home_blockedshots": "blockedshots",
            "home_giveaways": "giveaways",
            "home_takeaways": "takeaways",
        })

        away_stats = df[AWAY_TEAM_STATS_COLS].rename(columns={
            "away_team_abbrev": "team_abbrev",
            "away_gf": "gf",
            "away_ga": "ga",
            "away_sog": "sog",
            "away_powerplay_pct": "powerplay_pct",
            "away_pk_pct": "pk_pct",
            "away_powerplays": "powerplays",
            "away_pk": "penalty_kills",
            "away_faceoffwin_pct": "faceoffwin_pct",
            "away_pims": "pims",
            "away_hits": "hits",
            "away_blockedshots": "blockedshots",
            "away_giveaways": "giveaways",
            "away_takeaways": "takeaways",
        })
        away_stats["win"] = 1 - away_stats["home_win"]
        away_stats.drop(columns=["home_win"], inplace=True)

        combined = (
            pd.concat([home_stats, away_stats], ignore_index=True)
            .sort_values(["team_abbrev", "season", "date"])
            .reset_index(drop=True)
        )

        stats_to_roll = {
            "gf_ewm": "gf",
            "ga_ewm": "ga",
            "sog_ewm": "sog",
            "wins_l5": "win",
            "powerplay_pct_ewm": "powerplay_pct",
            "penalty_kill_pct_ewm": "pk_pct",
            "powerplays_l5": "powerplays",
            "penalty_kills_l5": "penalty_kills",
            "faceoffwin_pct_ewm": "faceoffwin_pct",
            "pims_ewm": "pims",
            "hits_ewm": "hits",
            "blockedshots_ewm": "blockedshots",
            "giveaways_ewm": "giveaways",
            "takeaways_ewm": "takeaways",
        }

        for new_col, source_col in stats_to_roll.items():
            if new_col == "wins_l5":
                combined[new_col] = (
                    combined.groupby(["team_abbrev", "season"])[source_col]
                    .transform(lambda x: x.rolling(window=5, min_periods=1).sum().shift(1))
                )
            elif new_col in ("powerplays_l5", "penalty_kills_l5"):
                combined[new_col] = (
                    combined.groupby(["team_abbrev", "season"])[source_col]
                    .transform(lambda s: s.rolling(5, min_periods=1).mean().shift(1))
                )
            else:
                combined[new_col] = _ewm_with_season_regression(
                    combined, "team_abbrev", "season", source_col, alpha=alpha
                )

        combined["games_l5"] = (
            combined.groupby(["team_abbrev", "season"])["win"]
            .transform(lambda x: x.rolling(5, min_periods=1).count().shift(1))
        )
        combined["win_pct_l5"] = combined["wins_l5"] / combined["games_l5"]

        merge_cols = ["date", "team_abbrev"] + [
            c for c in combined.columns if c.endswith("_l5") or c.endswith("_ewm")
        ]

        df = (
            df.merge(combined[merge_cols], left_on=["home_team_abbrev", "date"], right_on=["team_abbrev", "date"], how="left")
            .drop(columns=["team_abbrev"])
        )
        df = df.rename(columns={c: f"home_{c}" for c in df.columns if c.endswith("_l5") or c.endswith("_ewm")})

        df = (
            df.merge(combined[merge_cols], left_on=["away_team_abbrev", "date"], right_on=["team_abbrev", "date"], how="left")
            .drop(columns=["team_abbrev"])
        )
        df = df.rename(columns={
            c: f"away_{c}" for c in df.columns
            if (c.endswith("_l5") or c.endswith("_ewm")) and not c.startswith("home_")
        })

        rolling_cols = [c for c in df.columns if c.endswith("_l5") or c.endswith("_ewm")]
        df[rolling_cols] = df[rolling_cols].round(3)
        print("Rolling averages computed.")

        df["home_goal_diff_ewm"] = df["home_gf_ewm"] - df["away_gf_ewm"]
        df["home_ga_diff_ewm"] = df["home_ga_ewm"] - df["away_ga_ewm"]
        df["home_shot_diff_ewm"] = df["home_sog_ewm"] - df["away_sog_ewm"]

        diff_cols = ["home_gf_ewm", "away_gf_ewm", "home_ga_ewm", "away_ga_ewm",
                     "home_sog_ewm", "away_sog_ewm", "home_goal_diff_ewm", "home_ga_diff_ewm", "home_shot_diff_ewm"]
        df[diff_cols] = df[diff_cols].round(3)
        print("Goal/shot differentials computed.")

        self.df = df

    def add_goalie_features(self):
        """Step 3: Fetch goalie data, compute EWMs, and merge."""
        self._require_data()
        print("\n=== STEP 3: Fetch Goalie Data ===")

        game_ids = self.df["game_id"].tolist()
        print(f"Fetching goalie data for {len(game_ids)} games...")
        goalie_data_rows = asyncio.run(self._fetch_all_goalie_data(game_ids))

        goalie_df = pd.DataFrame(goalie_data_rows)
        if goalie_df.empty:
            raise ValueError("Goalie raw data is empty.")

        goalie_df["date"] = pd.to_datetime(goalie_df["date"])
        print(f"Fetched {len(goalie_data_rows)} goalie records")

        print("\n=== STEP 3B: Compute Goalie EWMs ===")
        alpha = 0.3
        goalie_df = goalie_df.sort_values("date").reset_index(drop=True)

        home_goalies = goalie_df[["game_id", "date", "season", *HOME_RENAME.keys()]].rename(columns=HOME_RENAME)
        away_goalies = goalie_df[["game_id", "date", "season", *AWAY_RENAME.keys()]].rename(columns=AWAY_RENAME)

        goalie_long = (
            pd.concat([home_goalies, away_goalies], ignore_index=True)
            .sort_values(["goalie", "season", "date"])
            .reset_index(drop=True)
        )

        for stat in GOALIE_STATS:
            goalie_long[f"{stat}_ewm"] = _ewm_with_season_regression(
                goalie_long, "goalie", "season", stat, alpha=alpha
            )

        goalie_l5 = goalie_long[["game_id", "goalie"] + [f"{s}_ewm" for s in GOALIE_STATS]]

        goalie_df = (
            goalie_df
            .merge(goalie_l5, left_on=["game_id", "home_goalie_starter"], right_on=["game_id", "goalie"], how="left")
            .rename(columns={f"{s}_ewm": f"home_goalie_{s}_ewm" for s in GOALIE_STATS})
            .drop(columns=["goalie"])
        )
        goalie_df = (
            goalie_df
            .merge(goalie_l5, left_on=["game_id", "away_goalie_starter"], right_on=["game_id", "goalie"], how="left")
            .rename(columns={f"{s}_ewm": f"away_goalie_{s}_ewm" for s in GOALIE_STATS})
            .drop(columns=["goalie"])
        )

        team_long = pd.concat([
            goalie_df[["game_id", "date", "season", "home_team_abbrev", "home_save_pct"]]
            .rename(columns={"home_team_abbrev": "team", "home_save_pct": "save_pct"}),
            goalie_df[["game_id", "date", "season", "away_team_abbrev", "away_save_pct"]]
            .rename(columns={"away_team_abbrev": "team", "away_save_pct": "save_pct"}),
        ], ignore_index=True).sort_values(["team", "season", "date"]).reset_index(drop=True)

        team_long["team_save_pct_ewm"] = _ewm_with_season_regression(
            team_long, "team", "season", "save_pct", alpha=alpha
        )

        goalie_df = (
            goalie_df
            .merge(team_long[["game_id", "team", "team_save_pct_ewm"]], left_on=["game_id", "home_team_abbrev"], right_on=["game_id", "team"], how="left")
            .rename(columns={"team_save_pct_ewm": "home_team_save_pct_ewm"})
            .drop(columns=["team"])
        )
        goalie_df = (
            goalie_df
            .merge(team_long[["game_id", "team", "team_save_pct_ewm"]], left_on=["game_id", "away_team_abbrev"], right_on=["game_id", "team"], how="left")
            .rename(columns={"team_save_pct_ewm": "away_team_save_pct_ewm"})
            .drop(columns=["team"])
        )

        goalie_df = (
            goalie_df
            .sort_values("date")
            .groupby("game_id", as_index=False)
            .first()
        )

        self.df = self.df.merge(goalie_df[GOALIE_MERGE_COLS], on="game_id", how="left", validate="one_to_one")

        # Fill debut goalie NaN EWMs with season mean as a neutral prior
        goalie_ewm_cols = [
            c for c in self.df.columns
            if ("home_goalie" in c or "away_goalie" in c) and c.endswith("_ewm")
        ]
        for col in goalie_ewm_cols:
            self.df[col] = self.df[col].fillna(self.df.groupby("season")[col].transform("mean"))

        print("Goalie data merged.")

    def add_standings_features(self):
        """Step 4: Fetch and merge season standings data."""
        self._require_data()
        print("\n=== STEP 4: Fetch Season Standings ===")

        standing_df = asyncio.run(self._build_standings_dataframe(self.df))

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
        standing_df["pointPctg_diff"] = standing_df["home_pointPctg_season"] - standing_df["away_pointPctg_season"]

        standing_df = standing_df.sort_values("date").reset_index(drop=True)

        home_hist = standing_df[["date", "home_teamAbbrev", "home_streakCount_CURR", "home_streakCode_CURR"]].rename(
            columns={"home_teamAbbrev": "team", "home_streakCount_CURR": "streakCount", "home_streakCode_CURR": "streakCode"}
        )
        away_hist = standing_df[["date", "away_teamAbbrev", "away_streakCount_CURR", "away_streakCode_CURR"]].rename(
            columns={"away_teamAbbrev": "team", "away_streakCount_CURR": "streakCount", "away_streakCode_CURR": "streakCode"}
        )

        hist = (
            pd.concat([home_hist, away_hist])
            .sort_values(["team", "date"])
            .reset_index(drop=True)
        )
        hist["prev_streakCount"] = hist.groupby("team")["streakCount"].shift(1)
        hist["prev_streakCode"] = hist.groupby("team")["streakCode"].shift(1)

        standing_df = (
            standing_df
            .merge(hist[["team", "date", "prev_streakCount", "prev_streakCode"]], left_on=["home_teamAbbrev", "date"], right_on=["team", "date"], how="left")
            .merge(hist[["team", "date", "prev_streakCount", "prev_streakCode"]], left_on=["away_teamAbbrev", "date"], right_on=["team", "date"], how="left", suffixes=("_home", "_away"))
        )

        standing_df["home_win_streak"] = np.where(
            standing_df["home_streakCode_CURR"] == "W",
            standing_df["home_streakCount_CURR"] - 1,
            np.where(standing_df["prev_streakCode_home"] == "W", standing_df["prev_streakCount_home"], 0),
        )
        standing_df["away_win_streak"] = np.where(
            standing_df["away_streakCode_CURR"] == "W",
            standing_df["away_streakCount_CURR"] - 1,
            np.where(standing_df["prev_streakCode_away"] == "W", standing_df["prev_streakCount_away"], 0),
        )

        standing_df = (
            standing_df
            .fillna(0)
            .sort_values("date")
            .drop_duplicates(subset=["game_id"], keep="last")
        )

        self.df = self.df.merge(
            standing_df[["game_id"] + SEASON_STATS], on="game_id", how="left", validate="one_to_one"
        )
        print("Season standings merged.")

    def add_rest_days(self):
        """Step 5: Compute team and goalie rest days."""
        self._require_data()
        print("\n=== STEP 5: Compute Rest Days ===")

        df = self.df.copy()

        team_games = pd.concat([
            df[["date", "season", "home_team_abbrev"]].rename(columns={"home_team_abbrev": "team"}),
            df[["date", "season", "away_team_abbrev"]].rename(columns={"away_team_abbrev": "team"}),
        ], ignore_index=True).sort_values(["team", "season", "date"])

        team_games["team_rest_days"] = (
            team_games.groupby(["team", "season"])["date"].diff().dt.days - 1
        ).fillna(3)

        df = (
            df.merge(team_games[["team", "season", "date", "team_rest_days"]], left_on=["home_team_abbrev", "season", "date"], right_on=["team", "season", "date"], how="left")
            .rename(columns={"team_rest_days": "home_rest_days"})
            .drop(columns=["team"])
        )
        df = (
            df.merge(team_games[["team", "season", "date", "team_rest_days"]], left_on=["away_team_abbrev", "season", "date"], right_on=["team", "season", "date"], how="left")
            .rename(columns={"team_rest_days": "away_rest_days"})
            .drop(columns=["team"])
        )

        goalie_games = pd.concat([
            df[["date", "season", "home_team_abbrev", "home_goalie_starter"]].rename(columns={"home_team_abbrev": "team", "home_goalie_starter": "goalie"}),
            df[["date", "season", "away_team_abbrev", "away_goalie_starter"]].rename(columns={"away_team_abbrev": "team", "away_goalie_starter": "goalie"}),
        ], ignore_index=True).sort_values(["goalie", "team", "season", "date"])

        goalie_games["goalie_rest_days"] = (
            goalie_games.groupby(["goalie", "team", "season"])["date"].diff().dt.days - 1
        ).fillna(3)

        df = (
            df.merge(goalie_games[["goalie", "team", "season", "date", "goalie_rest_days"]], left_on=["home_goalie_starter", "home_team_abbrev", "season", "date"], right_on=["goalie", "team", "season", "date"], how="left")
            .rename(columns={"goalie_rest_days": "home_goalie_rest_days"})
            .drop(columns=["goalie", "team"])
        )
        df = (
            df.merge(goalie_games[["goalie", "team", "season", "date", "goalie_rest_days"]], left_on=["away_goalie_starter", "away_team_abbrev", "season", "date"], right_on=["goalie", "team", "season", "date"], how="left")
            .rename(columns={"goalie_rest_days": "away_goalie_rest_days"})
            .drop(columns=["goalie", "team"])
        )

        self.df = df
        print("Rest days computed.")

    def add_head_to_head(self):
        """Step 6: Compute head-to-head statistics."""
        self._require_data()
        print("\n=== STEP 6: Compute Head-to-Head ===")

        df = self.df.copy()
        df["matchup"] = df.apply(
            lambda r: "_".join(sorted([r["home_team_abbrev"], r["away_team_abbrev"]])), axis=1
        )
        df = df.sort_values(["season", "date"]).reset_index(drop=True)

        h2h_long = pd.concat([
            df[["game_id", "date", "season", "matchup", "home_team_abbrev", "away_team_abbrev", "home_gf", "home_win"]]
            .rename(columns={"home_team_abbrev": "team", "away_team_abbrev": "opponent", "home_gf": "gf", "home_win": "win"}),
            df[["game_id", "date", "season", "matchup", "away_team_abbrev", "home_team_abbrev", "away_gf", "home_win"]]
            .assign(win=lambda x: 1 - x["home_win"])
            .rename(columns={"away_team_abbrev": "team", "home_team_abbrev": "opponent", "away_gf": "gf"})
            .drop(columns="home_win"),
        ], ignore_index=True).sort_values(["season", "matchup", "date"])

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

        home_h2h = h2h_long.rename(columns={"team": "home_team_abbrev", "h2h_wins": "home_h2h_wins", "h2h_gf": "home_h2h_gf"})[
            ["game_id", "home_team_abbrev", "home_h2h_wins", "home_h2h_gf"]
        ]
        away_h2h = h2h_long.rename(columns={"team": "away_team_abbrev", "h2h_wins": "away_h2h_wins", "h2h_gf": "away_h2h_gf"})[
            ["game_id", "away_team_abbrev", "away_h2h_wins", "away_h2h_gf"]
        ]

        df = df.merge(home_h2h, on=["game_id", "home_team_abbrev"], how="left")
        df = df.merge(away_h2h, on=["game_id", "away_team_abbrev"], how="left")
        df["home_h2h_wins_diff"] = df["home_h2h_wins"] - df["away_h2h_wins"]

        self.df = df
        print("Head-to-head computed.")

    def save(self, drop_first_season=True, round_numeric=True):
        """Save the dataframe to CSV, with optional final cleanup."""
        if self.df is None:
            raise ValueError("No data to save. Run the pipeline steps first.")

        df = self.df

        if drop_first_season:
            print("\n=== STEP 7: Drop First Season ===")
            before = len(df)
            df = df[df["season"] != 20222023].reset_index(drop=True)
            print(f"Dropped {before - len(df)} games. Remaining: {len(df)}")

        if round_numeric:
            print("\n=== STEP 8: Round Numeric Columns ===")
            df = df.round(3)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_csv(CSV_FILE, index=False)
        self.df = df
        print(f"Saved {len(df)} games to {CSV_FILE}")

    def run(self):
        """Run the full pipeline end to end."""
        print("\n=== PIPELINE TIMING ===")
        pipeline_start = time.perf_counter()

        steps = [
            ("Step 1 - Fetch games", self.fetch_games),
            ("Step 2 - Team rolling features", self.add_team_rolling_features),
            ("Step 3 - Goalie features", self.add_goalie_features),
            ("Step 4 - Standings features", self.add_standings_features),
            ("Step 5 - Rest days", self.add_rest_days),
            ("Step 6 - Head-to-head", self.add_head_to_head),
            ("Step 7/8 - Save output", self.save),
        ]

        for label, func in steps:
            started = time.perf_counter()
            func()
            elapsed = time.perf_counter() - started
            print(f"{label} took {self._format_elapsed(elapsed)}")

        total_elapsed = time.perf_counter() - pipeline_start
        print(f"Total pipeline runtime: {self._format_elapsed(total_elapsed)}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NHL PREDICTOR - DATA SCRAPER")
    print("=" * 60)

    try:
        pipeline = NHLPipeline()
        pipeline.run()
        print("\n" + "=" * 60)
        print("ALL STEPS COMPLETED SUCCESSFULLY!")
        print("=" * 60 + "\n")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
