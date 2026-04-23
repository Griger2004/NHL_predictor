import os
from sqlalchemy import create_engine, text

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
_DEFAULT_DB_URL = f"sqlite:///{os.path.join(PROJECT_DIR, 'nhl_predictions.db')}"


def _make_engine():
    url = os.environ.get("DATABASE_URL", _DEFAULT_DB_URL)
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return create_engine(url, pool_pre_ping=True)


engine = _make_engine()


def init_db():
    """Create all tables if they don't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS games (
                game_id        INTEGER PRIMARY KEY,
                game_date      TEXT    NOT NULL,
                season         INTEGER,
                home_team      TEXT    NOT NULL,
                away_team      TEXT    NOT NULL,
                home_team_name TEXT,
                away_team_name TEXT,
                game_time_utc  TEXT,
                game_state     TEXT,
                home_score     INTEGER DEFAULT 0,
                away_score     INTEGER DEFAULT 0,
                created_at     TEXT    DEFAULT (datetime('now')),
                updated_at     TEXT    DEFAULT (datetime('now'))
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS goalie_states (
                game_id        INTEGER NOT NULL REFERENCES games(game_id),
                side           TEXT    NOT NULL CHECK(side IN ('home', 'away')),
                goalie_name    TEXT,
                detection_tier TEXT    CHECK(detection_tier IN ('LIVE', 'PRE', 'FUT')),
                recorded_at    TEXT    DEFAULT (datetime('now')),
                PRIMARY KEY (game_id, side)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id             INTEGER NOT NULL REFERENCES games(game_id),
                predicted_at        TEXT    DEFAULT (datetime('now')),
                home_goalie         TEXT,
                away_goalie         TEXT,
                home_goalie_tier    TEXT,
                away_goalie_tier    TEXT,
                pred_home_win       INTEGER NOT NULL,
                prob_home_win       REAL    NOT NULL,
                prob_away_win       REAL    NOT NULL,
                confidence          REAL    NOT NULL,
                feature_snapshot    TEXT,
                model_version       TEXT,
                invalidation_reason TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS game_results (
                game_id         INTEGER PRIMARY KEY REFERENCES games(game_id),
                home_score      INTEGER,
                away_score      INTEGER,
                actual_home_win INTEGER NOT NULL,
                finalized_at    TEXT    DEFAULT (datetime('now'))
            )
        """))


def upsert_game(game_id, game_date, season, home_team, away_team,
                home_team_name, away_team_name, game_time_utc,
                game_state, home_score, away_score):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO games
                (game_id, game_date, season, home_team, away_team,
                 home_team_name, away_team_name, game_time_utc,
                 game_state, home_score, away_score)
            VALUES
                (:gid, :date, :season, :home_team, :away_team,
                 :home_team_name, :away_team_name, :game_time_utc,
                 :state, :home_score, :away_score)
            ON CONFLICT(game_id) DO UPDATE SET
                game_state    = excluded.game_state,
                home_score    = excluded.home_score,
                away_score    = excluded.away_score,
                updated_at    = datetime('now')
        """), {
            "gid": game_id, "date": game_date, "season": season,
            "home_team": home_team, "away_team": away_team,
            "home_team_name": home_team_name, "away_team_name": away_team_name,
            "game_time_utc": game_time_utc, "state": game_state,
            "home_score": home_score, "away_score": away_score,
        })


def upsert_final_result(game_id, home_score, away_score):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO game_results (game_id, home_score, away_score, actual_home_win)
            VALUES (:gid, :hs, :as, :win)
            ON CONFLICT(game_id) DO NOTHING
        """), {
            "gid": game_id, "hs": home_score, "as": away_score,
            "win": int(home_score > away_score),
        })


def get_latest_prediction(game_id):
    """Return the most recent prediction row for a game with its stored goalie states."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT p.*,
                   gs_home.goalie_name    AS stored_home_goalie,
                   gs_home.detection_tier AS stored_home_tier,
                   gs_away.goalie_name    AS stored_away_goalie,
                   gs_away.detection_tier AS stored_away_tier,
                   g.home_team, g.away_team, g.home_team_name, g.away_team_name,
                   g.game_date, g.game_time_utc, g.game_state
            FROM predictions p
            JOIN games g ON g.game_id = p.game_id
            LEFT JOIN goalie_states gs_home
                ON gs_home.game_id = p.game_id AND gs_home.side = 'home'
            LEFT JOIN goalie_states gs_away
                ON gs_away.game_id = p.game_id AND gs_away.side = 'away'
            WHERE p.game_id = :gid
            ORDER BY p.predicted_at DESC
            LIMIT 1
        """), {"gid": game_id}).mappings().first()
        return dict(row) if row else None


def get_predictions_for_date(date_str):
    """Return the latest prediction per game for a given date."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT p.id, p.game_id, p.predicted_at,
                   p.home_goalie, p.away_goalie,
                   p.pred_home_win, p.prob_home_win, p.prob_away_win, p.confidence,
                   p.invalidation_reason,
                   g.home_team, g.away_team, g.home_team_name, g.away_team_name,
                   g.game_date, g.game_time_utc, g.game_state
            FROM predictions p
            JOIN games g ON g.game_id = p.game_id
            WHERE g.game_date = :date
              AND p.id IN (
                  SELECT MAX(id) FROM predictions GROUP BY game_id
              )
            ORDER BY g.game_time_utc
        """), {"date": date_str}).mappings().all()
        return [dict(r) for r in rows]


def get_prediction_history_for_date(date_str):
    """Return all prediction attempts (including repredictions) for a given date."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT p.game_id, p.predicted_at,
                   p.home_goalie, p.away_goalie,
                   p.home_goalie_tier, p.away_goalie_tier,
                   p.pred_home_win, p.prob_home_win, p.prob_away_win, p.confidence,
                   p.invalidation_reason,
                   g.home_team, g.away_team, g.game_state
            FROM predictions p
            JOIN games g ON g.game_id = p.game_id
            WHERE g.game_date = :date
            ORDER BY p.game_id, p.predicted_at
        """), {"date": date_str}).mappings().all()
        return [dict(r) for r in rows]


def get_accuracy_stats():
    """Return overall accuracy across all finalized games."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            WITH latest AS (
                SELECT game_id, MAX(id) AS max_id
                FROM predictions
                GROUP BY game_id
            ),
            latest_preds AS (
                SELECT p.* FROM predictions p
                JOIN latest l ON l.game_id = p.game_id AND l.max_id = p.id
            )
            SELECT
                COUNT(*)  AS total_games,
                SUM(CASE WHEN lp.pred_home_win = gr.actual_home_win THEN 1 ELSE 0 END) AS correct,
                ROUND(
                    100.0 * SUM(CASE WHEN lp.pred_home_win = gr.actual_home_win THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(*), 0),
                    1
                ) AS accuracy_pct
            FROM latest_preds lp
            JOIN game_results gr ON gr.game_id = lp.game_id
        """)).mappings().first()
        return dict(row) if row else {"total_games": 0, "correct": 0, "accuracy_pct": 0}
