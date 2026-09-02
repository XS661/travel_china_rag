import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import jieba

ROOT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT_DIR / "knowledge"
UPLOAD_DIR = ROOT_DIR / "uploads"
DB_PATH = ROOT_DIR / "data" / "contributions.db"

STOP_WORDS = {
    "的",
    "了",
    "在",
    "是",
    "我",
    "有",
    "和",
    "就",
    "不",
    "人",
    "都",
    "一",
    "一个",
    "上",
    "也",
    "很",
    "到",
    "说",
    "要",
    "去",
    "你",
    "会",
    "着",
    "没有",
    "看",
    "好",
    "自己",
    "这",
    "里",
    "于",
    "为",
    "从",
    "以",
    "但",
    "呢",
    "啊",
    "呢",
    "吧",
    "我们",
    "他们",
    "它们",
}


def _ensure_storage() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS contributions (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            username TEXT,
            city TEXT NOT NULL,
            title TEXT,
            category TEXT,
            sub_category TEXT,
            source TEXT,
            source_type TEXT,
            file_name TEXT,
            content TEXT NOT NULL,
            notes TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            review_note TEXT,
            approved_entry TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)

    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(contributions)").fetchall()
    }
    if "user_id" not in columns:
        conn.execute("ALTER TABLE contributions ADD COLUMN user_id TEXT")
    if "username" not in columns:
        conn.execute("ALTER TABLE contributions ADD COLUMN username TEXT")

    conn.commit()
    conn.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify_city(city: str) -> str:
    text = re.sub(r"[^A-Za-z0-9\u4e00-\u9fa5]+", "_", city.strip())
    text = text.strip("_")
    return text.lower() or "custom_city"


def save_submission(
    *,
    city: str,
    title: str,
    content: str,
    category: str = "",
    sub_category: str = "",
    source: str = "用户亲身经历",
    source_type: str = "text",
    file_name: str | None = None,
    notes: str = "",
    user_id: str | None = None,
    username: str | None = None,
) -> dict:
    _ensure_storage()
    submission_id = str(uuid.uuid4())
    created_at = _now_iso()

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO contributions (
            id, user_id, username, city, title, category, sub_category, source, source_type, file_name,
            content, notes, status, review_note, approved_entry, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', '', '', ?, ?)
        """,
        (
            submission_id,
            user_id,
            username or "",
            city.strip(),
            title.strip(),
            category.strip(),
            sub_category.strip(),
            source.strip() or "用户亲身经历",
            source_type,
            file_name,
            content.strip(),
            notes.strip(),
            created_at,
            created_at,
        ),
    )
    conn.commit()
    conn.close()

    return get_submission(submission_id)


def get_submission(submission_id: str) -> dict | None:
    _ensure_storage()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM contributions WHERE id = ?", (submission_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    item = dict(row)
    item["approved_entry"] = (
        json.loads(item["approved_entry"]) if item["approved_entry"] else None
    )
    return item


def get_user_submission(user_id: str, submission_id: str) -> dict | None:
    _ensure_storage()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM contributions WHERE id = ? AND user_id = ?",
        (submission_id, user_id),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    item = dict(row)
    item["approved_entry"] = (
        json.loads(item["approved_entry"]) if item["approved_entry"] else None
    )
    return item


def delete_user_submission(user_id: str, submission_id: str) -> bool:
    _ensure_storage()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "DELETE FROM contributions WHERE id = ? AND user_id = ?",
        (submission_id, user_id),
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def list_submissions(
    status: str | None = None, user_id: str | None = None
) -> list[dict]:
    _ensure_storage()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if status and user_id:
        rows = conn.execute(
            "SELECT * FROM contributions WHERE status = ? AND user_id = ? ORDER BY created_at DESC",
            (status, user_id),
        ).fetchall()
    elif status:
        rows = conn.execute(
            "SELECT * FROM contributions WHERE status = ? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
    elif user_id:
        rows = conn.execute(
            "SELECT * FROM contributions WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM contributions ORDER BY created_at DESC"
        ).fetchall()
    conn.close()

    out = []
    for row in rows:
        item = dict(row)
        item["approved_entry"] = (
            json.loads(item["approved_entry"]) if item["approved_entry"] else None
        )
        out.append(item)
    return out


def update_submission_status(
    submission_id: str,
    *,
    status: str,
    review_note: str = "",
    approved_entry: dict | None = None,
) -> dict | None:
    _ensure_storage()
    updated_at = _now_iso()
    approved_json = (
        json.dumps(approved_entry, ensure_ascii=False) if approved_entry else ""
    )

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        UPDATE contributions
        SET status = ?, review_note = ?, approved_entry = ?, updated_at = ?
        WHERE id = ?
        """,
        (status, review_note, approved_json, updated_at, submission_id),
    )
    conn.commit()
    conn.close()
    return get_submission(submission_id)


def _extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    tokens = []
    for token in jieba.cut(cleaned):
        if token.strip() and len(token) > 1 and token not in STOP_WORDS:
            tokens.append(token)
    deduped = []
    seen = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            deduped.append(token)
        if len(deduped) >= max_keywords:
            break
    return deduped


def _infer_category(title: str, content: str, fallback: str = "其他") -> str:
    combined = f"{title} {content}".lower()
    if any(
        k in combined
        for k in ["火锅", "餐厅", "美食", "小吃", "酒店", "住宿", "早餐", "晚餐"]
    ):
        return "美食"
    if any(
        k in combined
        for k in [
            "地铁",
            "火车",
            "高铁",
            "公交",
            "机场",
            "交通",
            "租车",
            "路线",
            "打车",
        ]
    ):
        return "交通"
    if any(
        k in combined
        for k in [
            "景点",
            "博物馆",
            "公园",
            "长城",
            "故宫",
            "寺庙",
            "山",
            "水",
            "游玩",
            "攻略",
        ]
    ):
        return "景点"
    if any(
        k in combined for k in ["行程", "游玩", "三天", "攻略", "路线", "景点", "推荐"]
    ):
        return "行程"
    return fallback or "其他"


def prepare_knowledge_entry(payload: dict) -> dict:
    city = (payload.get("city") or "未知城市").strip()
    title = (payload.get("title") or f"{city}旅游体验").strip()
    content = (payload.get("content") or "").strip()
    if not content:
        raise ValueError("投稿内容不能为空")

    category = (
        payload.get("category") or _infer_category(title, content)
    ).strip() or "其他"
    sub_category = (payload.get("sub_category") or "").strip()
    source = (payload.get("source") or "用户亲身经历").strip() or "用户亲身经历"
    source_url = (payload.get("source_url") or "").strip()
    submission_id = (payload.get("submission_id") or payload.get("id") or "").strip()
    user_id = (payload.get("user_id") or "").strip()
    username = (payload.get("username") or "").strip()

    keywords = _extract_keywords(f"{title} {content} {category} {sub_category}")
    entry = {
        "id": f"user_{_slugify_city(city)}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}",
        "domain": "全国旅游",
        "city": city,
        "category": category,
        "sub_category": sub_category,
        "title": title,
        "content": content,
        "keywords": keywords,
        "source": source,
        "source_url": source_url,
        "submission_id": submission_id,
        "user_id": user_id,
        "username": username,
        "chunk_id": 1,
        "city_tag": category,
    }
    return entry


def list_community_posts(username: str | None = None) -> list[dict]:
    _ensure_storage()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if username:
        rows = conn.execute(
            "SELECT * FROM contributions WHERE status = 'approved' AND username = ? ORDER BY created_at DESC",
            (username,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM contributions WHERE status = 'approved' ORDER BY created_at DESC"
        ).fetchall()
    conn.close()

    out = []
    for row in rows:
        item = dict(row)
        item["approved_entry"] = (
            json.loads(item["approved_entry"]) if item["approved_entry"] else None
        )
        out.append(item)
    return out


def get_public_submission(submission_id: str) -> dict | None:
    _ensure_storage()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM contributions WHERE id = ? AND status = 'approved'",
        (submission_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    item = dict(row)
    item["approved_entry"] = (
        json.loads(item["approved_entry"]) if item["approved_entry"] else None
    )
    return item


def append_entry_to_knowledge(entry: dict) -> dict:
    city = (entry.get("city") or "未知城市").strip()
    if not city:
        raise ValueError("知识条目必须包含城市信息")

    slug = _slugify_city(city)
    city_file = KNOWLEDGE_DIR / f"{slug}.json"
    city_file.parent.mkdir(parents=True, exist_ok=True)

    if city_file.exists():
        with open(city_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
    else:
        data = []

    meta = next((item for item in data if item.get("id") == "_meta"), None)
    if meta is None:
        data.insert(
            0,
            {
                "id": "_meta",
                "domain": "全国旅游",
                "city": city,
                "category": "_meta",
                "sub_category": "",
                "title": "",
                "content": "",
                "keywords": [city],
                "source": "系统自动生成",
                "source_url": "",
                "chunk_id": 0,
                "city_tag": entry.get("category", "其他"),
                "aliases": [city],
            },
        )

    normalized = prepare_knowledge_entry(
        {
            **entry,
            "submission_id": entry.get("submission_id") or entry.get("id") or "",
            "user_id": entry.get("user_id") or "",
            "username": entry.get("username") or "",
        }
    )
    if not any(existing.get("id") == normalized["id"] for existing in data):
        data.append(normalized)

    with open(city_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    try:
        import city_detector
        import retriever

        city_detector._metadata_loaded = False
        city_detector._load_city_metadata()
        retriever._knowledge_cache = None
        retriever._city_knowledge_cache = None
        retriever._bm25_corpus = None
        retriever._bm25_index = None
        retriever._inverted_index_cache.clear()
        retriever._vector_corpus = None
        retriever._vector_status = "not_loaded"
    except Exception:
        pass

    return normalized


def review_contribution(
    *,
    city: str,
    title: str,
    content: str,
    category: str = "",
    sub_category: str = "",
    source: str = "用户亲身经历",
    filename: str | None = None,
) -> dict:
    cleaned = (content or "").strip()
    if not city or not city.strip():
        return {"status": "rejected", "reason": "城市不能为空", "entry": None}
    if not cleaned:
        return {"status": "rejected", "reason": "投稿内容不能为空", "entry": None}

    suspicious = re.search(
        r"黄赌毒|赌博|色情|违法|暴力|攻击|危害|诈骗|非法", cleaned, re.IGNORECASE
    )
    if suspicious:
        return {
            "status": "rejected",
            "reason": "内容包含不适合知识库的敏感信息，已拒绝",
            "entry": None,
        }

    if len(cleaned) < 20:
        return {
            "status": "rejected",
            "reason": "内容过短，无法形成有效的旅游知识点",
            "entry": None,
        }

    final_title = (title or f"{city}旅游体验").strip() or f"{city}旅游体验"
    final_category = (
        category or _infer_category(final_title, cleaned)
    ).strip() or "其他"
    final_sub_category = (sub_category or "").strip()

    dynamic_source = source.strip() or "用户亲身经历"
    if filename:
        dynamic_source = f"{dynamic_source}（附件：{filename}）"

    payload = {
        "city": city.strip(),
        "title": final_title,
        "content": cleaned,
        "category": final_category,
        "sub_category": final_sub_category,
        "source": dynamic_source,
    }

    try:
        from openai import OpenAI
        from generator import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

        if not DEEPSEEK_API_KEY:
            raise ValueError("No API key")

        client = OpenAI(
            api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=30
        )
        prompt = (
            "你是旅游知识审核助手。请根据用户上传的亲身经历或文案，整理成一条适合旅游知识库的事实型条目。"
            "请严格输出一段 JSON，字段必须包含：city、title、content、category、sub_category、source。"
            "要求：内容必须是事实型、可用于旅游问答，不要添加编造信息，不要出现营销语气。"
            "如果原文含附件名或真实经历，保留为可信来源描述。\n\n"
            f"城市：{city}\n标题：{title}\n分类：{category}\n子分类：{sub_category}\n来源：{source}\n\n原文：\n{cleaned}"
        )
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200,
        )
        text = response.choices[0].message.content or ""
        json_block = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(json_block.group(0) if json_block else text)

        if not parsed.get("city") or not parsed.get("content"):
            raise ValueError("AI audit did not produce a valid entry")

        cleaned_entry = prepare_knowledge_entry(
            {
                "city": parsed.get("city") or city,
                "title": parsed.get("title") or final_title,
                "content": parsed.get("content") or cleaned,
                "category": parsed.get("category") or final_category,
                "sub_category": parsed.get("sub_category") or final_sub_category,
                "source": parsed.get("source") or dynamic_source,
            }
        )
        return {
            "status": "approved",
            "reason": "AI 已整理并审核通过，已纳入知识库",
            "entry": cleaned_entry,
            "review_note": "大模型已整理并审核通过",
        }
    except Exception:
        fallback_entry = prepare_knowledge_entry(payload)
        return {
            "status": "approved",
            "reason": "上传内容已自动整理并校验通过，已纳入知识库",
            "entry": fallback_entry,
            "review_note": "已按规则自动整理并校验合规性",
        }
