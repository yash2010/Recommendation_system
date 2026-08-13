import sqlite3
import os
from pathlib import Path
from datetime import datetime

DB_PATH = os.environ.get("DB_PATH", "data/moviesdat.db")

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db() -> None:

    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.executescript("""
            -- Users table — one row per unique user
            CREATE TABLE IF NOT EXISTS users (
                user_id     TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL
            );

            -- Interactions table — every click, rating, and watch
            CREATE TABLE IF NOT EXISTS interactions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT    NOT NULL,
                movie_id    INTEGER NOT NULL,
                title       TEXT    NOT NULL,
                action      TEXT    NOT NULL,    -- 'click', 'rate', 'watch'
                rating      REAL,               -- NULL for clicks and watches
                query       TEXT,               -- search query that led here
                timestamp   TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            -- Indexes for fast lookups
            CREATE INDEX IF NOT EXISTS idx_user_id
                ON interactions(user_id);

            CREATE INDEX IF NOT EXISTS idx_movie_id
                ON interactions(movie_id);

            CREATE INDEX IF NOT EXISTS idx_timestamp
                ON interactions(timestamp);

            CREATE INDEX IF NOT EXISTS idx_action
                ON interactions(action);
        """)

    print(f"Database ready at {DB_PATH}")

def ensure_user(conn: sqlite3.Connection, user_id: str) -> None:
    """
    Insert user if they don't exist yet.
    INSERT OR IGNORE silently skips if user_id already exists.
    This avoids checking first — one DB call instead of two.
    """
    conn.execute(
        "INSERT OR IGNORE INTO users (user_id, created_at) VALUES (?, ?)",
        (user_id, datetime.utcnow().isoformat())
    )

def log_interaction(
    user_id:  str,
    movie_id: int,
    title:    str,
    action:   str,           # 'click', 'rate', 'watch'
    rating:   float = None,  # only for action='rate'
    query:    str   = None,  # search query that led here
) -> int:

    with get_connection() as conn:
        ensure_user(conn, user_id)

        cursor = conn.execute(
            """
            INSERT INTO interactions
                (user_id, movie_id, title, action, rating, query, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                movie_id,
                title,
                action,
                rating,   
                query,
                datetime.utcnow().isoformat()
            )
        )
        return cursor.lastrowid   

def get_user_history(
    user_id: str,
    limit:   int  = 50,
    action:  str  = None,   # optional filter: 'click', 'rate', 'watch'
) -> list[dict]:

    with get_connection() as conn:
        if action:
            rows = conn.execute(
                """
                SELECT * FROM interactions
                WHERE user_id = ? AND action = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (user_id, action, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM interactions
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (user_id, limit)
            ).fetchall()

        return [dict(row) for row in rows]


def get_user_ratings(user_id: str) -> list[dict]:

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT movie_id, title, rating, timestamp
            FROM interactions
            WHERE user_id = ?
              AND action = 'rate'
              AND rating IS NOT NULL
            ORDER BY timestamp DESC
            """,
            (user_id,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_all_ratings() -> list[dict]:
   
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT user_id, movie_id, title, rating, timestamp
            FROM interactions
            WHERE action = 'rate'
              AND rating IS NOT NULL
            ORDER BY timestamp
            """
        ).fetchall()
        return [dict(row) for row in rows]


def get_all_clicks() -> list[dict]:
 
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT user_id, movie_id, title, query, timestamp
            FROM interactions
            WHERE action = 'click'
            ORDER BY timestamp
            """
        ).fetchall()
        return [dict(row) for row in rows]


def get_movie_stats(movie_id: int) -> dict:
 
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*)        AS total_ratings,
                AVG(rating)     AS avg_rating,
                MIN(rating)     AS min_rating,
                MAX(rating)     AS max_rating
            FROM interactions
            WHERE movie_id = ?
              AND action = 'rate'
              AND rating IS NOT NULL
            """,
            (movie_id,)
        ).fetchone()

        return {
            "movie_id":      movie_id,
            "total_ratings": row["total_ratings"],
            "avg_rating":    round(row["avg_rating"], 2) if row["avg_rating"] else None,
            "min_rating":    row["min_rating"],
            "max_rating":    row["max_rating"],
        }


def get_stats() -> dict:

    with get_connection() as conn:
        total_interactions = conn.execute(
            "SELECT COUNT(*) FROM interactions"
        ).fetchone()[0]

        total_users = conn.execute(
            "SELECT COUNT(*) FROM users"
        ).fetchone()[0]

        total_ratings = conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE action = 'rate'"
        ).fetchone()[0]

        total_clicks = conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE action = 'click'"
        ).fetchone()[0]

        avg_rating = conn.execute(
            "SELECT AVG(rating) FROM interactions WHERE action = 'rate'"
        ).fetchone()[0]

        most_rated = conn.execute(
            """
            SELECT title, COUNT(*) as count
            FROM interactions
            WHERE action = 'rate'
            GROUP BY movie_id
            ORDER BY count DESC
            LIMIT 5
            """
        ).fetchall()

        return {
            "total_interactions": total_interactions,
            "total_users":        total_users,
            "total_ratings":      total_ratings,
            "total_clicks":       total_clicks,
            "avg_rating":         round(avg_rating, 2) if avg_rating else None,
            "most_rated_movies":  [dict(r) for r in most_rated],
        }


def export_for_training() -> dict:

    with get_connection() as conn:
        ratings = conn.execute(
            """
            SELECT user_id, movie_id, rating, timestamp
            FROM interactions
            WHERE action = 'rate' AND rating IS NOT NULL
            ORDER BY timestamp
            """
        ).fetchall()

        clicks = conn.execute(
            """
            SELECT user_id, movie_id, timestamp
            FROM interactions
            WHERE action = 'click'
            ORDER BY timestamp
            """
        ).fetchall()

        users = conn.execute(
            "SELECT user_id FROM users ORDER BY created_at"
        ).fetchall()

        movies = conn.execute(
            """
            SELECT DISTINCT movie_id
            FROM interactions
            ORDER BY movie_id
            """
        ).fetchall()

        return {
            "ratings": [dict(r) for r in ratings],
            "clicks":  [dict(r) for r in clicks],
            "users":   [r["user_id"] for r in users],
            "movies":  [r["movie_id"] for r in movies],
        }
