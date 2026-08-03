import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

MOVIES_CSV = "data/movies_final.csv"
MOVIES_PARQUET = "data/movies_final.parquet"
OLD_INTERACTIONS_DB = "data/interactions.db"
NEW_DB = "data/movies.db"

def migrate():

    # -------------- Movies dataset Migration --------------
    print("Starting migration...")
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(NEW_DB)
    conn.execute("PRAGMA journal_mode=WAL;")

    print("Migrating movies...")
    if Path(MOVIES_PARQUET).exists():
        df = pd.read_parquet(MOVIES_PARQUET)
        print(f"Loaded {len(df):,} movies from Parquet")
    elif Path(MOVIES_CSV).exists():
        df = pd.read_csv(MOVIES_CSV, encoding="utf-8")
        print(f"Loaded {len(df):,} movies from CSV")
    else:
        raise FileNotFoundError("No movies CSV or Parquet file found.")

    for col, default in [
        ("tmdb_rating",        0.0),
        ("tmdb_votes",         0),
        ("poster_url",         ""),
        ("original_language",  ""),
        ("source",             "wikipedia"),
    ]:
        if col not in df.columns:
            df[col] = default

    df.to_sql("movies", conn, if_exists="replace", index=False)

    conn.executescript("""
    CREATE INDEX IF NOT EXISTS idx_movies_title
        ON movies(title);
    CREATE INDEX IF NOT EXISTS idx_movies_year
        ON movies(release_year);
    CREATE INDEX IF NOT EXISTS idx_movies_genre
        ON movies(genre);
    CREATE INDEX IF NOT EXISTS idx_movies_language
        ON movies(original_language); """)

    print(f"Saved {len(df):,} movies to {NEW_DB}")

    # -------------- Interactions dataset Migration --------------
    if Path(OLD_INTERACTIONS_DB).exists():
        print("Migrating interactions...")

        old_conn = sqlite3.connect(OLD_INTERACTIONS_DB)

        users = pd.read_sql_query("SELECT * FROM users", old_conn)
        users.to_sql("users", old_conn, if_exists="replace", index=False)
        print(f"  Saved {len(users):,} users to {NEW_DB}")

        interactions = pd.read_sql_query("SELECT * FROM interactions", old_conn)
        interactions.to_sql("interactions", conn, if_exists="replace", index=False)
        print(f"  Saved {len(interactions):,} interactions to {NEW_DB}")

        conn.executescript("""CREATE INDEX IF NOT EXISTS idx_user_id
                        ON interactions(user_id);
                    CREATE INDEX IF NOT EXISTS idx_movie_id
                        ON interactions(movie_id);
                    CREATE INDEX IF NOT EXISTS idx_timestamp
                        ON interactions(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_action
                        ON interactions(action);""")

        old_conn.close()

    else:
            print("\n2. No interactions database found — creating empty tables...")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id    TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS interactions (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id  TEXT    NOT NULL,
                    movie_id INTEGER NOT NULL,
                    title    TEXT    NOT NULL,
                    action   TEXT    NOT NULL,
                    rating   REAL,
                    query    TEXT,
                    timestamp TEXT   NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );
            """)
            
    print("\n3. Verifying migration...")

    movie_count  = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
    user_count   = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    inter_count  = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]

    print(f"   Movies:       {movie_count:,}")
    print(f"   Users:        {user_count:,}")
    print(f"   Interactions: {inter_count:,}")

    # Sample check
    sample = conn.execute(
        "SELECT title, release_year, genre FROM movies LIMIT 3"
    ).fetchall()
    print(f"\n   Sample movies:")
    for row in sample:
        print(f"     {row[0]} ({row[1]}) — {row[2]}")

    conn.close()
    print(f"\nMigration complete → {NEW_DB}")

if __name__ == "__main__":
    migrate()