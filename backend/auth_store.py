import base64
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT_DIR / "data" / "users.db"
SECRET_KEY = "travel-qa-demo-secret-key-v1"


def _ensure_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_history (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT,
            detected_city TEXT,
            timestamp TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_history_user_id ON user_history(user_id, created_at DESC)"
    )
    conn.commit()
    conn.close()


def _normalize_username(username: str) -> str:
    return (username or "").strip()


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(8)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200000,
    )
    return f"pbkdf2_sha256$200000${salt}${digest.hex()}"


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations, salt, expected_hex = password_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            int(iterations),
        )
        return hmac.compare_digest(digest.hex(), expected_hex)
    except (ValueError, TypeError):
        return False


def _user_row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "password_hash": row["password_hash"],
        "created_at": row["created_at"],
    }


def get_user(user_id: str) -> dict | None:
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return _user_row_to_dict(row)


def get_user_by_username(username: str) -> dict | None:
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?",
        (username.strip(),),
    ).fetchone()
    conn.close()
    return _user_row_to_dict(row)


def register_user(username: str, password: str) -> dict:
    _ensure_db()
    cleaned_name = _normalize_username(username)
    if not re.fullmatch(r"[A-Za-z0-9_\u4e00-\u9fa5]{3,30}", cleaned_name):
        raise ValueError("用户名只能包含中文、字母、数字和下划线，长度 3-30")
    if len(password or "") < 6:
        raise ValueError("密码长度至少为 6 位")
    if get_user_by_username(cleaned_name):
        raise ValueError("用户名已存在")

    user_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (user_id, cleaned_name, _hash_password(password), created_at),
    )
    conn.commit()
    conn.close()
    return get_user(user_id) or {
        "id": user_id,
        "username": cleaned_name,
        "password_hash": _hash_password(password),
        "created_at": created_at,
    }


def authenticate_user(username: str, password: str) -> dict | None:
    user = get_user_by_username(_normalize_username(username))
    if not user or not _verify_password(password, user["password_hash"]):
        return None
    return user


def create_token(user: dict) -> str:
    payload = {
        "sub": user["id"],
        "username": user["username"],
        "exp": int((datetime.now(timezone.utc) + timedelta(days=7)).timestamp()),
    }
    payload_json = json.dumps(
        payload, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload_json).decode("utf-8").rstrip("=")
    signature = hmac.new(
        SECRET_KEY.encode("utf-8"), encoded.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{encoded}.{signature}"


def verify_token(token: str) -> dict:
    if not token:
        raise ValueError("未提供登录令牌")
    try:
        encoded, signature = token.split(".", 1)
    except ValueError as exc:
        raise ValueError("登录令牌格式错误") from exc

    expected = hmac.new(
        SECRET_KEY.encode("utf-8"), encoded.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("登录令牌无效")

    padded = encoded + "=" * (-len(encoded) % 4)
    payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    exp = int(payload.get("exp", 0))
    if exp < int(datetime.now(timezone.utc).timestamp()):
        raise ValueError("登录令牌已过期")

    user = get_user(payload.get("sub", ""))
    if not user:
        raise ValueError("用户不存在")

    return {
        "id": user["id"],
        "username": user["username"],
        "created_at": user["created_at"],
    }


def list_user_history(user_id: str) -> list[dict]:
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM user_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 20",
        (user_id,),
    ).fetchall()
    conn.close()
    return [
        {
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "detected_city": row["detected_city"],
            "timestamp": row["timestamp"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def append_user_history(
    *,
    user_id: str,
    question: str,
    answer: str | None = None,
    detected_city: str | None = None,
    timestamp: str | None = None,
) -> dict:
    _ensure_db()
    if not user_id:
        raise ValueError("用户不存在")
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    history_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO user_history (id, user_id, question, answer, detected_city, timestamp, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (history_id, user_id, question.strip(), answer, detected_city, ts, created_at),
    )
    conn.commit()
    conn.close()
    return {
        "id": history_id,
        "question": question.strip(),
        "answer": answer,
        "detected_city": detected_city,
        "timestamp": ts,
        "created_at": created_at,
    }


def clear_user_history(user_id: str) -> None:
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM user_history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
