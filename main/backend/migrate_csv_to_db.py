"""
One-time migration: load nhl_data.csv into the nhl_game_data table.

Run from the backend directory:
    python migrate_csv_to_db.py

Uses DATABASE_URL env var if set; otherwise defaults to the project-level SQLite file.
"""

import os
import sys
import pandas as pd
from db import engine

CSV_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'ml_dev', 'scripts', 'generated', 'data', 'nhl_data.csv'
)


def main():
    csv_path = os.path.normpath(CSV_PATH)
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        print("Run main.py first to generate the data file.")
        sys.exit(1)

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # matchup is a display string — not useful in the DB
    df = df.drop(columns=["matchup"], errors="ignore")

    print(f"Writing to nhl_game_data table ...")
    df.to_sql(
        "nhl_game_data",
        engine,
        if_exists="replace",
        index=False,
    )

    # Verify
    from sqlalchemy import text
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM nhl_game_data")).scalar()
    print(f"  Done — {count} rows in nhl_game_data")


if __name__ == "__main__":
    main()
