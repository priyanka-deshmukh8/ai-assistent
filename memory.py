
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "assistant_memory.db"
MAX_HISTORY_MESSAGES = 10


def init_db():
    """Create the conversation history table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_exchange(user_message: str, assistant_message: str):
    """Save one completed conversation exchange."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.executemany(
            "INSERT INTO conversation (role, content) VALUES (?, ?)",
            [
                ("user", user_message),
                ("assistant", assistant_message),
            ],
        )


def load_history(limit: int = MAX_HISTORY_MESSAGES) -> list[dict]:
    """Load recent messages in chronological order."""
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT role, content
            FROM (
                SELECT id, role, content
                FROM conversation
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (limit,),
        ).fetchall()

    return [
        {"role": role, "content": content}
        for role, content in rows
    ]


def clear_history():
    """Delete saved conversation history."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("DELETE FROM conversation")
