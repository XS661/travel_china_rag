import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

from . import config

DB_PATH = config.DATA_DIR / "conversations.db"

MAX_HISTORY_TURNS = config.MAX_HISTORY_TURNS if hasattr(config, "MAX_HISTORY_TURNS") else 6
MAX_CONTEXT_CHARS = config.MAX_CONTEXT_CHARS if hasattr(config, "MAX_CONTEXT_CHARS") else 6000
SESSION_TTL_DAYS = 30


def _ensure_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_city TEXT,
            metadata TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            detected_city TEXT,
            sources_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at ASC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at DESC)"
    )
    conn.commit()
    conn.close()


def create_session(user_id: str | None = None) -> str:
    _ensure_db()
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions (session_id, user_id, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?)",
        (session_id, user_id, now, now, json.dumps({}, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()
    return session_id


def get_or_create_session(session_id: str, user_id: str | None = None) -> str:
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if row:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if user_id:
            conn.execute(
                "UPDATE sessions SET updated_at = ?, user_id = COALESCE(user_id, ?) WHERE session_id = ?",
                (now, user_id, session_id),
            )
        else:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id)
            )
        conn.commit()
        conn.close()
        return session_id
    conn.close()
    return create_session(user_id)


def append_message(
    session_id: str,
    role: str,
    content: str,
    detected_city: str | None = None,
    sources: list[dict] | None = None,
) -> None:
    _ensure_db()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages (session_id, role, content, detected_city, sources_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            session_id,
            role,
            content,
            detected_city,
            json.dumps(sources or [], ensure_ascii=False),
            now,
        ),
    )
    if detected_city:
        conn.execute(
            "UPDATE sessions SET updated_at = ?, last_city = ? WHERE session_id = ?",
            (now, detected_city, session_id),
        )
    else:
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id)
        )
    conn.commit()
    conn.close()


def get_recent_messages(session_id: str, max_turns: int | None = None) -> list[dict]:
    _ensure_db()
    turns = max_turns or MAX_HISTORY_TURNS
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT role, content, detected_city, sources_json FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
        (session_id, turns * 2),
    ).fetchall()
    conn.close()
    result = []
    for row in reversed(rows):
        result.append(
            {
                "role": row["role"],
                "content": row["content"],
                "detected_city": row["detected_city"],
                "sources": json.loads(row["sources_json"] or "[]"),
            }
        )
    return result


def extract_context(messages: list[dict]) -> dict:
    context = {
        "cities": [],
        "keywords": set(),
        "summary": "",
    }
    cities = []
    texts = []
    for msg in messages:
        if msg["role"] == "user":
            texts.append(msg["content"])
        if msg.get("detected_city") and msg["detected_city"] not in cities:
            cities.append(msg["detected_city"])

    context["cities"] = cities
    context["summary"] = " ".join(texts[-3:]) if texts else ""
    return context


def get_last_city(session_id: str) -> str | None:
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT last_city FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def list_user_sessions(user_id: str, limit: int = 20) -> list[dict]:
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT s.session_id, s.created_at, s.updated_at, s.last_city,
               (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id) as msg_count,
               (SELECT content FROM messages m WHERE m.session_id = s.session_id AND role = 'user' ORDER BY created_at ASC LIMIT 1) as first_question
        FROM sessions s
        WHERE s.user_id = ?
        ORDER BY s.updated_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [
        {
            "session_id": row["session_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_city": row["last_city"],
            "message_count": row["msg_count"],
            "first_question": row["first_question"],
        }
        for row in rows
    ]


def delete_session(session_id: str) -> None:
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def cleanup_expired_sessions() -> int:
    _ensure_db()
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=SESSION_TTL_DAYS)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))
    deleted = cur.rowcount
    conn.execute(
        "DELETE FROM messages WHERE session_id NOT IN (SELECT session_id FROM sessions)"
    )
    conn.commit()
    conn.close()
    return deleted