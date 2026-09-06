import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone

import jieba

from . import config

KNOWLEDGE_DIR = config.KNOWLEDGE_DIR
UPLOAD_DIR = config.UPLOAD_DIR
DB_PATH = config.DATA_DIR / "contributions.db"

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


# ============================================================
# 数据质量（M3-A）：长文切片 + 上传去重
# ============================================================

def _split_content_into_chunks(
    content: str, max_chars: int = 800, min_chars: int = 200
) -> list[str]:
    """把长文本切成适合知识库的块。

    切分优先级：段落边界（\\n）→ 句子边界（。！？；）→ 字符硬切；
    尾部小块低于 min_chars 时并入前一块，避免碎块污染检索。
    """
    if len(content) <= max_chars:
        return [content]

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n".join(buf))
            buf, buf_len = [], 0

    for para in re.split(r"\n\s*\n|\n", content):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars and buf_len + len(para) + 1 <= max_chars:
            buf.append(para)
            buf_len += len(para) + 1
            continue
        flush()
        if len(para) <= max_chars:
            buf.append(para)
            buf_len = len(para) + 1
            continue
        # 单段超长：按句子拆
        for sentence in re.split(r"(?<=[。！？；.!?;])", para):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_chars:
                flush()
                for i in range(0, len(sentence), max_chars):
                    piece = sentence[i : i + max_chars].strip()
                    if piece:
                        chunks.append(piece)
            elif buf_len + len(sentence) + 1 <= max_chars:
                buf.append(sentence)
                buf_len += len(sentence) + 1
            else:
                flush()
                buf.append(sentence)
                buf_len = len(sentence) + 1
    flush()

    # 合并过小的尾块到前一块
    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < min_chars and len(merged[-1]) + len(c) <= max_chars * 1.5:
            merged[-1] += "\n" + c
        else:
            merged.append(c)
    return merged


def prepare_knowledge_entries(payload: dict) -> list[dict]:
    """构造入库条目列表；超长投稿自动按段落/句子切成多个 chunk。

    切片后的条目共享同一 id（M2 向量快照按城市块+块内位置定位，可安全复用），
    chunk_id 从 1 递增；第 2 块起的标题追加「（第 N 部分）」后缀，避免多条
    同标题干扰排序。
    """
    base = prepare_knowledge_entry(payload)
    content = base["content"]
    chunks = _split_content_into_chunks(
        content, config.CHUNK_MAX_CHARS, config.CHUNK_MIN_CHARS
    )
    if len(chunks) == 1:
        return [base]

    entries = []
    for i, piece in enumerate(chunks):
        entry = dict(base)
        entry["content"] = piece
        entry["chunk_id"] = i + 1
        if i > 0:
            entry["title"] = f"{base['title']}（第{i + 1}部分）"
        entries.append(entry)
    return entries


def _entry_tokens(text: str) -> set[str]:
    """分词并过滤停用词/单字词，用作去重比较的词集"""
    return {
        t
        for t in jieba.lcut(text)
        if t.strip() and len(t) >= 2 and t not in STOP_WORDS
    }


def find_similar_entry(
    entries: list[dict],
    city: str,
    title: str,
    content: str,
    threshold: float = 0.6,
) -> dict | None:
    """在同城市条目中查找与 (title, content) 高度相似的条目。

    相似度 = 词集交集大小 / 两词集较小者大小（Jaccard 变体），
    只与同城市条目比较，避免跨城误伤。未命中返回 None。
    """
    target = _entry_tokens(f"{title} {content}")
    if not target:
        return None
    best: dict | None = None
    best_overlap = 0.0
    for entry in entries:
        if entry.get("city") != city or entry.get("id") == "_meta":
            continue
        other = _entry_tokens(f"{entry.get('title', '')} {entry.get('content', '')}")
        if not other:
            continue
        overlap = len(target & other) / min(len(target), len(other))
        if overlap > best_overlap:
            best_overlap, best = overlap, entry
    return best if best_overlap >= threshold else None


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

    normalized = prepare_knowledge_entry(
        {
            **entry,
            "submission_id": entry.get("submission_id") or entry.get("id") or "",
            "user_id": entry.get("user_id") or "",
            "username": entry.get("username") or "",
        }
    )

    # M3-A：兜底去重——防止绕过审核直接入库；同一投稿的切片块之间互不比
    if config.DEDUP_ENABLED:
        try:
            from . import retriever as _retriever

            candidates = [
                e
                for e in _retriever.knowledge_base.load()
                if e.get("id") != normalized["id"]
                and (
                    not normalized.get("submission_id")
                    or e.get("submission_id") != normalized.get("submission_id")
                )
            ]
            similar = find_similar_entry(
                candidates,
                city,
                normalized["title"],
                normalized["content"],
                threshold=config.DEDUP_OVERLAP_THRESHOLD,
            )
        except Exception as e:
            print(f"[WARNING] 入库去重检查失败，跳过：{e}")
            similar = None
        if similar is not None:
            print(
                f"[INFO] 去重跳过入库：与「{similar.get('title')}」高度相似（{city}）"
            )
            return {
                **normalized,
                "skipped": True,
                "skip_reason": f"与「{similar.get('title')}」高度相似",
            }

    # M3-B：写入 SQLite 事务（WAL 多读单写 + busy 等待，天然并发安全；
    # UNIQUE (city, id, chunk_id) 允许同投稿切片多条共存，同 id 不重复写）
    try:
        from . import knowledge_db

        knowledge_db.ensure_db(KNOWLEDGE_DIR)
        inserted = knowledge_db.insert_entries([normalized])
    except Exception as e:
        print(f"[WARNING] 知识库写入失败：{e}")
        raise
    if inserted == 0:
        return {
            **normalized,
            "skipped": True,
            "skip_reason": "相同条目已存在",
        }

    try:
        from . import city_detector
        from . import retriever

        city_detector._metadata_loaded = False
        city_detector._load_city_metadata()
        # 统一刷新检索服务的全部内存缓存（磁盘向量快照保留）
        retriever.knowledge_base.reload()
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

    # M3-A：上传去重——与同城市已有知识条目比较，高度相似直接拒绝（不再调用 LLM）
    if config.DEDUP_ENABLED:
        try:
            from . import retriever as _retriever

            similar = find_similar_entry(
                _retriever.knowledge_base.load(),
                city.strip(),
                final_title,
                cleaned,
                threshold=config.DEDUP_OVERLAP_THRESHOLD,
            )
        except Exception as e:
            print(f"[WARNING] 去重检查失败，跳过：{e}")
            similar = None
        if similar is not None:
            return {
                "status": "rejected",
                "reason": (
                    f"内容与已有知识「{similar.get('title') or similar.get('id')}」"
                    "高度相似，疑似重复投稿，已拒绝"
                ),
                "entry": None,
                "similar_id": similar.get("id", ""),
            }

    try:
        from openai import OpenAI

        if not config.DEEPSEEK_API_KEY:
            raise ValueError("No API key")

        client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            timeout=30,
        )
        prompt = (
            "你是旅游知识审核助手。请根据用户上传的亲身经历或文案，整理成一条适合旅游知识库的事实型条目。"
            "请严格输出一段 JSON，字段必须包含：city、title、content、category、sub_category、source。"
            "要求：内容必须是事实型、可用于旅游问答，不要添加编造信息，不要出现营销语气。"
            "如果原文含附件名或真实经历，保留为可信来源描述。\n\n"
            f"城市：{city}\n标题：{title}\n分类：{category}\n子分类：{sub_category}\n来源：{source}\n\n原文：\n{cleaned}"
        )
        response = client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200,
        )
        text = response.choices[0].message.content or ""
        json_block = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(json_block.group(0) if json_block else text)

        if not parsed.get("city") or not parsed.get("content"):
            raise ValueError("AI audit did not produce a valid entry")

        cleaned_entries = prepare_knowledge_entries(
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
            "entry": cleaned_entries[0],
            "entries": cleaned_entries,
            "review_note": "大模型已整理并审核通过",
        }
    except Exception:
        fallback_entries = prepare_knowledge_entries(payload)
        return {
            "status": "approved",
            "reason": "上传内容已自动整理并校验通过，已纳入知识库",
            "entry": fallback_entries[0],
            "entries": fallback_entries,
            "review_note": "已按规则自动整理并校验合规性",
        }
