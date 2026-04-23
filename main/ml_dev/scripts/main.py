import os
import asyncio
import time
import numpy as np
import pandas as pd

from api.client import ApiClient

# DB setup — writes nhl_game_data table after each pipeline run
try:
    from sqlalchemy import create_engine as _create_engine
    _SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_DIR = os.path.abspath(os.path.join(_SCRIPTS_DIR, '..', '..'))
    _DEFAULT_DB_URL = f"sqlite:///{os.path.join(_PROJECT_DIR, 'nhl_predictions.db')}"

    def _build_db_engine():
        url = os.environ.get("DATABASE_URL", _DEFAULT_DB_URL)
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return _create_engine(url, pool_pre_ping=True)

    _db_engine = _build_db_engine()
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
from config import (
    API_BASE_URL, OUTPUT_DIR, CSV_FILE, TIMEOUT, MAX_CONCURRENT_REQUESTS,
    RETRIES, MAX_GAMES, SEASONS, STANDINGS_FIELDS, HOME_RENAME, AWAY_RENAME,
    GOALIE_STATS, GOALIE_MERGE_COLS, SEASON_STATS,
    HOME_TEAM_STATS_COLS, AWAY_TEAM_STATS_COLS,
)
from utils.helpers import (
    get_num_powerplays, calc_num_penalty_kills, calc_penalty_kill_pct,
    extract_name, extract_category_stat, calc_team_save_pct,
    extract_fractional_stat, get_starter_goalie,
    extract_all_teams_playing_on_date, extract_all_standings_stats,
)


# ===== STAT EXTRACTION =====

_UTAH_ALIASES = ("Arizona Coyotes", "Utah Utah Hockey Club")

def _normalize_team(place, name, abbrev):
    if f"{place} {name}".strip() in _UTAH_ALIASES:
        return "Utah", "Mammoth", "UTA"
    return place, name, abbrev


def extract_all_basic_team_stats(game_data):
    if game_data.get("gameState") in ["FUT", "LIVE", "PRE"]:
        return None

    home = game_data.get("homeTeam", {})
    away = game_data.get("awayTeam", {})
    team_stats = game_data.get("summary", {}).get("teamGameStats", [])

    home_place, home_name, home_abbrev = _normalize_team(
        extract_name(home.get("placeName")), extract_name(home.get("name")), home.get("abbrev", "")
    )
    away_place, away_name, away_abbrev = _normalize_team(
        extract_name(away.get("placeName")), extract_name(away.get("name")), away.get("abbrev", "")
    )

    home_fo, away_fo = extract_category_stat(team_stats, "faceoffWinningPctg")
    home_pp_str, away_pp_str = extract_category_stat(team_stats, "powerPlay")
    home_pp_goals, home_pp_opps = get_num_powerplays(home_pp_str)
    away_pp_goals, away_pp_opps = get_num_powerplays(away_pp_str)
    home_pp_pct, away_pp_pct = extract_category_stat(team_stats, "powerPlayPctg")
    home_pims, away_pims = extract_category_stat(team_stats, "pim")
    home_hits, away_hits = extract_category_stat(team_stats, "hits")
    home_bs, away_bs = extract_category_stat(team_stats, "blockedShots")
    home_ta, away_ta = extract_category_stat(team_stats, "takeaways")
    home_gv, away_gv = extract_category_stat(team_stats, "giveaways")

    return {
        "game_id": game_data.get("id"),
        "date": game_data.get("gameDate", ""),
        "season": game_data.get("season", 0),
        "home_team": f"{home_place} {home_name}".strip(),
        "away_team": f"{away_place} {away_name}".strip(),
        "home_team_abbrev": home_abbrev, "away_team_abbrev": away_abbrev,
        "home_win": 1 if home.get("score", 0) > away.get("score", 0) else 0,
        "home_gf": home.get("score", 0), "away_gf": away.get("score", 0),
        "home_ga": away.get("score", 0), "away_ga": home.get("score", 0),
        "home_sog": home.get("sog", 0), "away_sog": away.get("sog", 0),
        "home_faceoffwin_pct": home_fo, "away_faceoffwin_pct": away_fo,
        "home_powerplays": home_pp_opps, "away_powerplays": away_pp_opps,
        "home_powerplay_pct": home_pp_pct, "away_powerplay_pct": away_pp_pct,
        "home_pk": calc_num_penalty_kills(away_pp_opps, away_pp_goals),
        "away_pk": calc_num_penalty_kills(home_pp_opps, home_pp_goals),
        "home_pk_pct": calc_penalty_kill_pct(away_pp_opps, away_pp_goals),
        "away_pk_pct": calc_penalty_kill_pct(home_pp_opps, home_pp_goals),
        "home_pims": home_pims, "away_pims": away_pims,
        "home_hits": home_hits, "away_hits": away_hits,
        "home_blockedshots": home_bs, "away_blockedshots": away_bs,
        "home_takeaways": home_ta, "away_takeaways": away_ta,
        "home_giveaways": home_gv, "away_giveaways": away_gv,
    }


def _calc_team_save_pct(goalies):
    if not goalies:
        return 0.0
    if len(goalies) < 2:
        return goalies[0].get("savePctg", 0.0)
    return calc_team_save_pct(
        goalies[0].get("saves", 0) + goalies[1].get("saves", 0),
        goalies[0].get("shotsAgainst", 0) + goalies[1].get("shotsAgainst", 0),
    )


def _get_starter_obj(goalies):
    if not goalies:
        return {}
    return goalies[0] if goalies[0].get("starter") else (goalies[1] if len(goalies) > 1 else {})


def _goalie_stats_row(prefix, obj):
    return {
        f"{prefix}_goalie_save_pct": obj.get("savePctg", 0.0),
        f"{prefix}_goalie_ga": obj.get("goalsAgainst", 0),
        f"{prefix}_goalie_saves": obj.get("saves", 0),
        f"{prefix}_goalie_evenStrengthShotsAgainst": extract_fractional_stat(obj.get("evenStrengthShotsAgainst", "0/0")),
        f"{prefix}_goalie_powerPlayShotsAgainst": extract_fractional_stat(obj.get("powerPlayShotsAgainst", "0/0")),
        f"{prefix}_goalie_shorthandedShotsAgainst": extract_fractional_stat(obj.get("shorthandedShotsAgainst", "0/0")),
        f"{prefix}_goalie_evenStrengthGoalsAgainst": obj.get("evenStrengthGoalsAgainst", 0),
        f"{prefix}_goalie_powerPlayGoalsAgainst": obj.get("powerPlayGoalsAgainst", 0),
    }


def extract_all_basic_goalie_stats(boxscore_data):
    home = boxscore_data.get("homeTeam", {})
    away = boxscore_data.get("awayTeam", {})
    pbg = boxscore_data.get("playerByGameStats", {})
    home_goalies = pbg.get("homeTeam", {}).get("goalies", [])
    away_goalies = pbg.get("awayTeam", {}).get("goalies", [])

    home_starter, _ = get_starter_goalie(home_goalies)
    away_starter, _ = get_starter_goalie(away_goalies)

    def _abbrev(team):
        a = team.get("abbrev", "")
        return "UTA" if a == "ARI" else a

    return {
        "game_id": boxscore_data.get("id"),
        "date": boxscore_data.get("gameDate", ""),
        "season": boxscore_data.get("season", 0),
        "home_team_abbrev": _abbrev(home), "away_team_abbrev": _abbrev(away),
        "home_goalie_starter": home_starter, "away_goalie_starter": away_starter,
        "home_save_pct": _calc_team_save_pct(home_goalies),
        "away_save_pct": _calc_team_save_pct(away_goalies),
        **_goalie_stats_row("home", _get_starter_obj(home_goalies)),
        **_goalie_stats_row("away", _get_starter_obj(away_goalies)),
    }


# ===== EWM =====

_EWM_CARRY_WEIGHT = 0.7  # 70% last-season form, 30% toward league mean (FiveThirtyEight-style)


def _ewm_with_season_regression(df, entity_col, season_col, stat_col, alpha=0.3):
    """
    Shift(1) EWM with regression-to-mean at season boundaries.
    Seeds the first game of a new season with:
        0.7 * final_ewm_prev_season + 0.3 * prev_season_league_avg
    df must be sorted by [entity_col, season_col, "date"] before calling.
    """
    result = pd.Series(np.nan, index=df.index, dtype=float)
    seasons = sorted(df[season_col].unique())
    lg_avg = df.groupby(season_col)[stat_col].mean()

    for _, entity_df in df.groupby(entity_col, sort=False):
        entity_df = entity_df.sort_values([season_col, "date"])
        prev_ewm = np.nan

        for season in seasons:
            mask = entity_df[season_col] == season
            if not mask.any():
                continue

            group = entity_df[mask]
            values = group[stat_col].to_numpy(dtype=float)

            seed = np.nan if np.isnan(prev_ewm) else (
                _EWM_CARRY_WEIGHT * prev_ewm + (1 - _EWM_CARRY_WEIGHT) * float(
                    lg_avg.get(season, lg_avg.mean())
                )
            )

            ewm_vals = np.full(len(values), np.nan)
            cur = seed
            for i, v in enumerate(values):
                ewm_vals[i] = cur
                if not np.isnan(v):
                    cur = v if np.isnan(cur) else alpha * v + (1 - alpha) * cur

            result.loc[group.index] = ewm_vals
            prev_ewm = cur

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
        print(f"  - Season {season}: requesting {len(game_ids)} game IDs...")
        async with self._api_client() as client:
            results = await asyncio.gather(*[client.get_json(f"/v1/wsc/game-story/{gid}") for gid in game_ids])
        rows = []
        for game_data in results:
            if not game_data:
                continue
            row = extract_all_basic_team_stats(game_data)
            if row is None:
                break
            rows.append(row)
        print(f"  - Season {season}: {len(rows)} completed games")
        return rows

    async def _fetch_all_goalie_data(self, game_ids):
        async with self._api_client() as client:
            boxscores = await asyncio.gather(*[client.get_json(f"/v1/gamecenter/{gid}/boxscore") for gid in game_ids])
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
                out_row = {"game_id": row["game_id"], "date": date_str,
                           "home_teamAbbrev": row["home_team_abbrev"],
                           "away_teamAbbrev": row["away_team_abbrev"]}
                for side, abbrev in [("home", row["home_team_abbrev"]), ("away", row["away_team_abbrev"])]:
                    stats = standings_by_team.get(abbrev, {})
                    for field in STANDINGS_FIELDS:
                        out_row[f"{side}_{field}"] = stats.get(field)
                        if field in ("streakCode", "streakCount"):
                            out_row[f"{side}_{field}_CURR"] = stats.get(field)
                rows.append(out_row)

        return pd.DataFrame(rows)

    # --- Pipeline steps ---

    def fetch_games(self):
        """Step 1: Fetch basic game info for all seasons."""
        print("\n=== STEP 1: Fetch Basic Game Info ===")
        t0 = time.perf_counter()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if os.path.exists(CSV_FILE):
            os.remove(CSV_FILE)
            print(f"Removed existing: {CSV_FILE}")

        async def _fetch_all():
            return await asyncio.gather(*[self._fetch_season_games(s) for s in SEASONS])

        all_season_rows = asyncio.run(_fetch_all())
        all_rows = [row for season_rows in all_season_rows for row in season_rows]

        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"])
        self.df = df
        print(f"Fetched {len(df)} completed games | {self._format_elapsed(time.perf_counter() - t0)}")

    def add_team_rolling_features(self):
        """Step 2: Compute rolling averages and differentials."""
        self._require_data()
        print("\n=== STEP 2: Compute Rolling Averages ===")

        df = self.df.copy()

        home_stats = df[HOME_TEAM_STATS_COLS].rename(columns={
            "home_team_abbrev": "team_abbrev", "home_win": "win",
            "home_gf": "gf", "home_ga": "ga", "home_sog": "sog",
            "home_powerplay_pct": "powerplay_pct", "home_pk_pct": "pk_pct",
            "home_powerplays": "powerplays", "home_pk": "penalty_kills",
            "home_faceoffwin_pct": "faceoffwin_pct", "home_pims": "pims",
            "home_hits": "hits", "home_blockedshots": "blockedshots",
            "home_giveaways": "giveaways", "home_takeaways": "takeaways",
        })
        away_stats = df[AWAY_TEAM_STATS_COLS].rename(columns={
            "away_team_abbrev": "team_abbrev",
            "away_gf": "gf", "away_ga": "ga", "away_sog": "sog",
            "away_powerplay_pct": "powerplay_pct", "away_pk_pct": "pk_pct",
            "away_powerplays": "powerplays", "away_pk": "penalty_kills",
            "away_faceoffwin_pct": "faceoffwin_pct", "away_pims": "pims",
            "away_hits": "hits", "away_blockedshots": "blockedshots",
            "away_giveaways": "giveaways", "away_takeaways": "takeaways",
        })
        away_stats["win"] = 1 - away_stats["home_win"]
        away_stats.drop(columns=["home_win"], inplace=True)

        combined = (
            pd.concat([home_stats, away_stats], ignore_index=True)
            .sort_values(["team_abbrev", "season", "date"])
            .reset_index(drop=True)
        )

        ewm_stats = ["gf", "ga", "sog", "powerplay_pct", "pk_pct", "faceoffwin_pct",
                     "pims", "hits", "blockedshots", "giveaways", "takeaways"]
        for s in ewm_stats:
            combined[f"{s}_ewm"] = _ewm_with_season_regression(combined, "team_abbrev", "season", s)

        for col, src, agg in [
            ("wins_l5", "win", "sum"),
            ("powerplays_l5", "powerplays", "mean"),
            ("penalty_kills_l5", "penalty_kills", "mean"),
        ]:
            combined[col] = combined.groupby(["team_abbrev", "season"])[src].transform(
                lambda x, a=agg: x.rolling(5, min_periods=1).agg(a).shift(1)
            )

        combined["games_l5"] = combined.groupby(["team_abbrev", "season"])["win"].transform(
            lambda x: x.rolling(5, min_periods=1).count().shift(1)
        )
        combined["win_pct_l5"] = combined["wins_l5"] / combined["games_l5"]

        merge_cols = ["date", "team_abbrev"] + [c for c in combined.columns if c.endswith("_l5") or c.endswith("_ewm")]

        for side, abbrev_col in [("home", "home_team_abbrev"), ("away", "away_team_abbrev")]:
            before = set(df.columns)
            df = df.merge(combined[merge_cols], left_on=[abbrev_col, "date"], right_on=["team_abbrev", "date"], how="left").drop(columns=["team_abbrev"])
            new = [c for c in df.columns if c not in before and (c.endswith("_l5") or c.endswith("_ewm"))]
            df = df.rename(columns={c: f"{side}_{c}" for c in new})

        rolling_cols = [c for c in df.columns if c.endswith("_l5") or c.endswith("_ewm")]
        df[rolling_cols] = df[rolling_cols].round(3)

        df["home_goal_diff_ewm"] = (df["home_gf_ewm"] - df["away_gf_ewm"]).round(3)
        df["home_ga_diff_ewm"] = (df["home_ga_ewm"] - df["away_ga_ewm"]).round(3)
        df["home_shot_diff_ewm"] = (df["home_sog_ewm"] - df["away_sog_ewm"]).round(3)

        self.df = df
        print("Rolling averages and differentials computed.")

    def add_goalie_features(self):
        """Step 3: Fetch goalie data, compute EWMs, and merge."""
        self._require_data()
        print("\n=== STEP 3: Fetch & Compute Goalie Features ===")

        game_ids = self.df["game_id"].tolist()
        goalie_data_rows = asyncio.run(self._fetch_all_goalie_data(game_ids))

        goalie_df = pd.DataFrame(goalie_data_rows)
        if goalie_df.empty:
            raise ValueError("Goalie raw data is empty.")
        goalie_df["date"] = pd.to_datetime(goalie_df["date"])
        print(f"Fetched {len(goalie_data_rows)} goalie records")

        # Per-goalie EWMs
        goalie_long = pd.concat([
            goalie_df[["game_id", "date", "season", *HOME_RENAME.keys()]].rename(columns=HOME_RENAME),
            goalie_df[["game_id", "date", "season", *AWAY_RENAME.keys()]].rename(columns=AWAY_RENAME),
        ], ignore_index=True).sort_values(["goalie", "season", "date"]).reset_index(drop=True)

        for stat in GOALIE_STATS:
            goalie_long[f"{stat}_ewm"] = _ewm_with_season_regression(goalie_long, "goalie", "season", stat)

        goalie_ewm = goalie_long[["game_id", "goalie"] + [f"{s}_ewm" for s in GOALIE_STATS]]

        for side, starter_col in [("home", "home_goalie_starter"), ("away", "away_goalie_starter")]:
            goalie_df = (
                goalie_df
                .merge(goalie_ewm, left_on=["game_id", starter_col], right_on=["game_id", "goalie"], how="left")
                .rename(columns={f"{s}_ewm": f"{side}_goalie_{s}_ewm" for s in GOALIE_STATS})
                .drop(columns=["goalie"])
            )

        # Team save pct EWM
        team_long = pd.concat([
            goalie_df[["game_id", "date", "season", "home_team_abbrev", "home_save_pct"]].rename(columns={"home_team_abbrev": "team", "home_save_pct": "save_pct"}),
            goalie_df[["game_id", "date", "season", "away_team_abbrev", "away_save_pct"]].rename(columns={"away_team_abbrev": "team", "away_save_pct": "save_pct"}),
        ], ignore_index=True).sort_values(["team", "season", "date"]).reset_index(drop=True)

        team_long["team_save_pct_ewm"] = _ewm_with_season_regression(team_long, "team", "season", "save_pct")

        for side, abbrev_col in [("home", "home_team_abbrev"), ("away", "away_team_abbrev")]:
            goalie_df = (
                goalie_df
                .merge(team_long[["game_id", "team", "team_save_pct_ewm"]], left_on=["game_id", abbrev_col], right_on=["game_id", "team"], how="left")
                .rename(columns={"team_save_pct_ewm": f"{side}_team_save_pct_ewm"})
                .drop(columns=["team"])
            )

        goalie_df = goalie_df.sort_values("date").groupby("game_id", as_index=False).first()
        self.df = self.df.merge(goalie_df[GOALIE_MERGE_COLS], on="game_id", how="left", validate="one_to_one")

        # Fill debut goalie NaN EWMs with season mean as neutral prior
        goalie_ewm_cols = [c for c in self.df.columns if "goalie" in c and c.endswith("_ewm")]
        for col in goalie_ewm_cols:
            self.df[col] = self.df[col].fillna(self.df.groupby("season")[col].transform("mean"))

        print("Goalie features merged.")

    def add_standings_features(self):
        """Step 4: Fetch and merge season standings data."""
        self._require_data()
        print("\n=== STEP 4: Fetch Season Standings ===")

        standing_df = asyncio.run(self._build_standings_dataframe(self.df))
        standing_df[["home_gamesPlayed", "away_gamesPlayed"]] = (
            standing_df[["home_gamesPlayed", "away_gamesPlayed"]].astype(int)
        )

        for side in ("home", "away"):
            gp = standing_df[f"{side}_gamesPlayed"]
            standing_df[f"{side}_win_pct_season"] = (standing_df[f"{side}_homeWins"] + standing_df[f"{side}_roadWins"]) / gp
            standing_df[f"{side}_gf_per_game_season"] = standing_df[f"{side}_goalsForPctg"]
            standing_df[f"{side}_pointPctg_season"] = standing_df[f"{side}_pointPctg"]
        standing_df["home_home_win_pct"] = standing_df["home_homeWins"] / standing_df["home_homeGamesPlayed"]
        standing_df["away_away_win_pct"] = standing_df["away_roadWins"] / standing_df["away_roadGamesPlayed"]
        standing_df["pointPctg_diff"] = standing_df["home_pointPctg_season"] - standing_df["away_pointPctg_season"]

        standing_df = standing_df.sort_values("date").reset_index(drop=True)

        # Build prior-game streak by shifting each team's streak back one game
        hist = pd.concat([
            standing_df[["date", "home_teamAbbrev", "home_streakCount_CURR", "home_streakCode_CURR"]].rename(
                columns={"home_teamAbbrev": "team", "home_streakCount_CURR": "streakCount", "home_streakCode_CURR": "streakCode"}),
            standing_df[["date", "away_teamAbbrev", "away_streakCount_CURR", "away_streakCode_CURR"]].rename(
                columns={"away_teamAbbrev": "team", "away_streakCount_CURR": "streakCount", "away_streakCode_CURR": "streakCode"}),
        ]).sort_values(["team", "date"]).reset_index(drop=True)

        hist["prev_streakCount"] = hist.groupby("team")["streakCount"].shift(1)
        hist["prev_streakCode"] = hist.groupby("team")["streakCode"].shift(1)

        standing_df = (
            standing_df
            .merge(hist[["team", "date", "prev_streakCount", "prev_streakCode"]], left_on=["home_teamAbbrev", "date"], right_on=["team", "date"], how="left")
            .merge(hist[["team", "date", "prev_streakCount", "prev_streakCode"]], left_on=["away_teamAbbrev", "date"], right_on=["team", "date"], how="left", suffixes=("_home", "_away"))
        )

        for side, prev_code, prev_count in [
            ("home", "prev_streakCode_home", "prev_streakCount_home"),
            ("away", "prev_streakCode_away", "prev_streakCount_away"),
        ]:
            standing_df[f"{side}_win_streak"] = np.where(
                standing_df[f"{side}_streakCode_CURR"] == "W",
                standing_df[f"{side}_streakCount_CURR"] - 1,
                np.where(standing_df[prev_code] == "W", standing_df[prev_count], 0),
            )

        standing_df = standing_df.fillna(0).sort_values("date").drop_duplicates(subset=["game_id"], keep="last")
        self.df = self.df.merge(standing_df[["game_id"] + SEASON_STATS], on="game_id", how="left", validate="one_to_one")
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
        team_games["team_rest_days"] = (team_games.groupby(["team", "season"])["date"].diff().dt.days - 1).fillna(3)

        for side, abbrev_col in [("home", "home_team_abbrev"), ("away", "away_team_abbrev")]:
            df = (
                df.merge(team_games[["team", "season", "date", "team_rest_days"]], left_on=[abbrev_col, "season", "date"], right_on=["team", "season", "date"], how="left")
                .rename(columns={"team_rest_days": f"{side}_rest_days"})
                .drop(columns=["team"])
            )

        goalie_games = pd.concat([
            df[["date", "season", "home_team_abbrev", "home_goalie_starter"]].rename(columns={"home_team_abbrev": "team", "home_goalie_starter": "goalie"}),
            df[["date", "season", "away_team_abbrev", "away_goalie_starter"]].rename(columns={"away_team_abbrev": "team", "away_goalie_starter": "goalie"}),
        ], ignore_index=True).sort_values(["goalie", "team", "season", "date"])
        goalie_games["goalie_rest_days"] = (goalie_games.groupby(["goalie", "team", "season"])["date"].diff().dt.days - 1).fillna(3)

        for side, starter_col, abbrev_col in [
            ("home", "home_goalie_starter", "home_team_abbrev"),
            ("away", "away_goalie_starter", "away_team_abbrev"),
        ]:
            df = (
                df.merge(goalie_games[["goalie", "team", "season", "date", "goalie_rest_days"]], left_on=[starter_col, abbrev_col, "season", "date"], right_on=["goalie", "team", "season", "date"], how="left")
                .rename(columns={"goalie_rest_days": f"{side}_goalie_rest_days"})
                .drop(columns=["goalie", "team"])
            )

        self.df = df
        print("Rest days computed.")

    def add_head_to_head(self):
        """Step 6: Compute head-to-head statistics."""
        self._require_data()
        print("\n=== STEP 6: Compute Head-to-Head ===")

        df = self.df.copy()
        df["matchup"] = df.apply(lambda r: "_".join(sorted([r["home_team_abbrev"], r["away_team_abbrev"]])), axis=1)
        df = df.sort_values(["season", "date"]).reset_index(drop=True)

        h2h_long = pd.concat([
            df[["game_id", "date", "season", "matchup", "home_team_abbrev", "away_team_abbrev", "home_gf", "home_win"]]
            .rename(columns={"home_team_abbrev": "team", "away_team_abbrev": "opponent", "home_gf": "gf", "home_win": "win"}),
            df[["game_id", "date", "season", "matchup", "away_team_abbrev", "home_team_abbrev", "away_gf", "home_win"]]
            .assign(win=lambda x: 1 - x["home_win"])
            .rename(columns={"away_team_abbrev": "team", "home_team_abbrev": "opponent", "away_gf": "gf"})
            .drop(columns="home_win"),
        ], ignore_index=True).sort_values(["season", "matchup", "date"])

        h2h_long["h2h_wins"] = h2h_long.groupby(["season", "matchup", "team"])["win"].transform(
            lambda s: s.cumsum().shift(1)
        ).fillna(0)
        h2h_long["h2h_gf"] = h2h_long.groupby(["season", "matchup", "team"])["gf"].transform(
            lambda s: s.expanding().mean().shift(1)
        ).fillna(0).round(3)

        for side in ("home", "away"):
            abbrev_col = f"{side}_team_abbrev"
            h2h = h2h_long.rename(columns={"team": abbrev_col, "h2h_wins": f"{side}_h2h_wins", "h2h_gf": f"{side}_h2h_gf"})[
                ["game_id", abbrev_col, f"{side}_h2h_wins", f"{side}_h2h_gf"]
            ]
            df = df.merge(h2h, on=["game_id", abbrev_col], how="left")

        df["home_h2h_wins_diff"] = df["home_h2h_wins"] - df["away_h2h_wins"]
        self.df = df
        print("Head-to-head computed.")

    def save(self, drop_first_season=True, round_numeric=True):
        """Save dataframe to CSV with optional cleanup."""
        if self.df is None:
            raise ValueError("No data to save.")
        df = self.df
        if drop_first_season:
            before = len(df)
            df = df[df["season"] != 20222023].reset_index(drop=True)
            print(f"\nDropped first season: {before - len(df)} games removed, {len(df)} remaining.")
        if round_numeric:
            df = df.round(3)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_csv(CSV_FILE, index=False)
        self.df = df
        print(f"Saved {len(df)} games to {CSV_FILE}")

        if DB_AVAILABLE:
            try:
                df_db = df.drop(columns=["matchup"], errors="ignore")
                df_db.to_sql(
                    "nhl_game_data", _db_engine,
                    if_exists="replace", index=False,
                )
                print(f"Saved {len(df_db)} games to nhl_game_data DB table")
            except Exception as e:
                print(f"  Warning: DB write failed — {e}")

    def run(self):
        """Run the full pipeline end to end."""
        t0 = time.perf_counter()
        steps = [
            ("Fetch games", self.fetch_games),
            ("Team rolling features", self.add_team_rolling_features),
            ("Goalie features", self.add_goalie_features),
            ("Standings features", self.add_standings_features),
            ("Rest days", self.add_rest_days),
            ("Head-to-head", self.add_head_to_head),
            ("Save output", self.save),
        ]
        for label, func in steps:
            t = time.perf_counter()
            func()
            print(f"  [{label}] {self._format_elapsed(time.perf_counter() - t)}")
        print(f"\nTotal pipeline runtime: {self._format_elapsed(time.perf_counter() - t0)}")


if __name__ == "__main__":
    print("\n" + "=" * 60 + "\nNHL PREDICTOR - DATA PIPELINE\n" + "=" * 60)
    try:
        NHLPipeline().run()
        print("\n" + "=" * 60 + "\nALL STEPS COMPLETED SUCCESSFULLY!\n" + "=" * 60)
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
