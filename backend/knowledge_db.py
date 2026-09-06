"""知识库 SQLite 存储层（M3-B，B+ 架构）。

运行时唯一读写目标：backend/data/knowledge.db（gitignored）。
backend/knowledge/*.json 仅作为首次自举种子与归档（纳入 git 管理）：
- DB 缺失或为空、且 KNOWLEDGE_DIR 存在 JSON 时，自动全量迁移（自举），
  因此克隆仓库/新环境零手工步骤即可运行；
- 投稿/审核写入全部走 SQLite 事务，天然并发安全，不再读改写 JSON。

表结构：
- knowledge_entries：知识条目，UNIQUE (city, id, chunk_id)
  —— 同一投稿切片共享 id，chunk_id 递增（修复 JSON 路径"同 id 只写第一条"的问题）
- city_meta：城市元数据（由 JSON 的 _meta 条目迁移，供 city_detector）
- db_meta：库级元数据（schema 版本等）

迁移幂等：既有 (city, id, chunk_id) 冲突条目自动跳过并计数，
用于处理历史数据中的重复 id 条目。
"""

import json
import sqlite3
from pathlib import Path

import jieba

from . import config

KNOWLEDGE_DB_PATH = config.DATA_DIR / "knowledge.db"
SCHEMA_VERSION = 2

# 中文检索停用词：高频虚词与疑问词，避免 OR 语义下
# "的/了/是" 之类单字命中全库造成假相关
STOPWORDS: frozenset[str] = frozenset(
    (
        "的", "了", "是", "在", "和", "与", "及", "或", "而", "之", "也",
        "都", "很", "更", "最", "不", "没", "有", "个", "我", "你", "他",
        "她", "它", "这", "那", "哪", "等", "吧", "吗", "呢", "啊", "哦",
        "从", "对", "到", "为", "于", "以", "把", "被", "让", "给", "向",
        "由", "并", "且", "但", "却", "又", "就", "才", "只", "还", "请",
        "问", "想", "要", "什么", "怎么", "如何", "哪里", "多少", "可以",
        "需要", "请问", "一下",
    )
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_entries (
    rid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL,
    city TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    keywords TEXT NOT NULL DEFAULT '[]',
    category TEXT NOT NULL DEFAULT '',
    sub_category TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    submission_id TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '',
    username TEXT NOT NULL DEFAULT '',
    chunk_id INTEGER NOT NULL DEFAULT 1,
    city_tag TEXT NOT NULL DEFAULT '',
    domain TEXT NOT NULL DEFAULT '全国旅游',
    UNIQUE (city, id, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_entries_city ON knowledge_entries(city);

CREATE TABLE IF NOT EXISTS city_meta (
    city TEXT PRIMARY KEY,
    keywords TEXT NOT NULL DEFAULT '[]',
    aliases TEXT NOT NULL DEFAULT '[]',
    tag TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS db_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- FTS5 全文索引：存储 jieba 分词后的检索文本（title+content+keywords+分类），
-- 提供持久化的 BM25（bm25() 排名函数），中文检索引擎不再依赖内存重建
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    rid UNINDEXED,
    search_text
);
"""


def get_connection() -> sqlite3.Connection:
    """打开知识库连接（WAL 模式，多读单写，busy 等待）"""
    conn = sqlite3.connect(KNOWLEDGE_DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.execute(
        "INSERT OR REPLACE INTO db_meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


# ------------------------------ FTS5 辅助 ------------------------------

def _fts_search_text(entry: dict) -> str:
    """构建 FTS 索引文本（与 retriever._build_search_text 一致：标题+内容+关键词+分类）"""
    parts = [
        entry.get("title", ""),
        entry.get("content", ""),
        " ".join(entry.get("keywords") or []),
        entry.get("category", ""),
        entry.get("sub_category", ""),
    ]
    return " ".join(filter(None, parts))


def _fts_tokens(entry: dict) -> str:
    """jieba 分词后以空格连接，作为 FTS5 的索引词串"""
    return " ".join(t for t in jieba.lcut(_fts_search_text(entry)) if t.strip())


def _sync_fts_for(conn: sqlite3.Connection, entry: dict) -> None:
    """按 (city, id, chunk_id) 回查 rid，同步一条 FTS 索引"""
    row = conn.execute(
        "SELECT rid FROM knowledge_entries WHERE city=? AND id=? AND chunk_id=?",
        (entry.get("city", ""), entry.get("id", ""), _to_chunk_id(entry)),
    ).fetchone()
    if row is not None:
        conn.execute(
            "INSERT OR REPLACE INTO knowledge_fts(rid, search_text) VALUES (?, ?)",
            (row["rid"], _fts_tokens(entry)),
        )


def _clear_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM knowledge_fts")


def fts_bm25_scores(tokens: list[str]) -> dict[int, float]:
    """FTS5 bm25 评分查询：返回 {rid: 分数}，分数越大越相关。

    匹配策略：AND 优先（全部查询词命中，与 BM25Okapi 的强相关文档排序一致），
    AND 无结果时降级 OR（保召回，避免多词查询整库落空）。
    注意：FTS5 bm25() 输出为负值（越小越相关），这里取反后归一化到 (0,1]。
    """
    query_tokens = [
        t.strip() for t in tokens if t.strip() and t not in STOPWORDS
    ]
    if not query_tokens:
        return {}
    conn = get_connection()
    try:
        for op in ("AND", "OR"):
            match = f" {op} ".join(f'"{t}"' for t in query_tokens)
            rows = conn.execute(
                "SELECT f.rid AS rid, -bm25(knowledge_fts) AS s"
                " FROM knowledge_fts f JOIN knowledge_entries e ON e.rid = f.rid"
                " WHERE knowledge_fts MATCH ?"
                " ORDER BY s DESC",
                (match,),
            ).fetchall()
            if rows:
                raw = {r["rid"]: float(r["s"]) for r in rows}
                # min-max 归一化到 (0,1]：单调变换不改变排序，
                # 但让分数展示、round 截断和 score>0 过滤都可靠
                lo, hi = min(raw.values()), max(raw.values())
                span = hi - lo
                if span > 0:
                    return {rid: (v - lo) / span for rid, v in raw.items()}
                return {rid: 1.0 for rid in raw}
        return {}
    finally:
        conn.close()


def _to_chunk_id(entry: dict) -> int:
    try:
        return int(entry.get("chunk_id") or 1)
    except (TypeError, ValueError):
        return 1


# ------------------------------ 行 <-> 条目 ------------------------------

_ROW_FIELDS = (
    "id",
    "city",
    "title",
    "content",
    "keywords",
    "category",
    "sub_category",
    "source",
    "source_url",
    "submission_id",
    "user_id",
    "username",
    "chunk_id",
    "city_tag",
    "domain",
)


def row_to_entry(row: sqlite3.Row) -> dict:
    entry = dict(row)
    try:
        entry["keywords"] = json.loads(entry.get("keywords") or "[]")
    except (json.JSONDecodeError, TypeError):
        entry["keywords"] = []
    return entry


def _entry_to_row(entry: dict) -> tuple:
    return (
        entry.get("id", ""),
        entry.get("city", ""),
        entry.get("title", ""),
        entry.get("content", ""),
        json.dumps(entry.get("keywords") or [], ensure_ascii=False),
        entry.get("category", ""),
        entry.get("sub_category", ""),
        entry.get("source", ""),
        entry.get("source_url", ""),
        entry.get("submission_id", ""),
        entry.get("user_id", ""),
        entry.get("username", ""),
        _to_chunk_id(entry),  # 历史数据 chunk_id 可能残留字符串，防御性转 int
        entry.get("city_tag", ""),
        entry.get("domain", "全国旅游"),
    )


# ------------------------------ JSON 解析 ------------------------------

def parse_city_file(path: Path) -> tuple[list[dict], dict | None]:
    """解析单个城市 JSON：返回 (条目列表, meta dict|无)。

    兼容旧的对象数组格式（含 _meta 首条）与 {_meta, items} 包装格式。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = None
    if isinstance(data, list):
        items = [e for e in data if isinstance(e, dict)]
        meta = next((e for e in items if e.get("id") == "_meta"), None)
        entries = [e for e in items if e.get("id") != "_meta"]
    elif isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            entries = [e for e in items if isinstance(e, dict)]
        else:
            entries = [data] if data.get("id") != "_meta" else []
        meta = data.get("_meta")
        if not isinstance(meta, dict) and data.get("id") == "_meta":
            meta = data
    else:
        entries = []

    if meta is None:
        # 无 _meta 时沿用旧行为：以文件名（城市 slug）作为城市名
        meta = {
            "city": path.stem,
            "keywords": [],
            "aliases": [],
            "city_tag": "",
        }
    else:
        meta = {
            "city": meta.get("city") or path.stem,
            "keywords": meta.get("keywords") or [],
            "aliases": meta.get("aliases") or [],
            "city_tag": meta.get("city_tag") or "",
        }
    return entries, meta


# ------------------------------ 迁移 / 自举 ------------------------------

def migrate_from_json(knowledge_dir: Path) -> dict:
    """一次性全量迁移：knowledge/*.json → SQLite。幂等（UNIQUE 去重）。"""
    counts = {"entries": 0, "duplicates_skipped": 0, "meta": 0, "files": 0}
    conn = get_connection()
    init_schema(conn)
    _clear_fts(conn)  # 全量重建 FTS 索引，避免残留脏数据
    try:
        for json_file in sorted(knowledge_dir.glob("*.json")):
            try:
                entries, meta = parse_city_file(json_file)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[WARNING] 迁移跳过 {json_file.name}: {e}")
                continue
            counts["files"] += 1
            for entry in entries:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO knowledge_entries (%s) VALUES (%s)"
                    % (
                        ", ".join(_ROW_FIELDS),
                        ", ".join("?" * len(_ROW_FIELDS)),
                    ),
                    _entry_to_row(entry),
                )
                if cur.rowcount == 0:
                    counts["duplicates_skipped"] += 1
                else:
                    counts["entries"] += 1
                    _sync_fts_for(conn, entry)
            conn.execute(
                "INSERT OR REPLACE INTO city_meta(city, keywords, aliases, tag)"
                " VALUES (?, ?, ?, ?)",
                (
                    meta["city"],
                    json.dumps(meta["keywords"], ensure_ascii=False),
                    json.dumps(meta["aliases"], ensure_ascii=False),
                    meta["city_tag"],
                ),
            )
            counts["meta"] += 1
        conn.commit()
    finally:
        conn.close()
    print(
        f"[INFO] 知识库迁移完成：{counts['files']} 个文件，"
        f"{counts['entries']} 条条目，skip 重复 {counts['duplicates_skipped']}，"
        f"{counts['meta']} 个城市元数据"
    )
    return counts


def ensure_db(knowledge_dir: Path | None = None) -> None:
    """确保 DB 可用；DB 为空且存在 JSON 种子时自动迁移（自举）。"""
    if not KNOWLEDGE_DB_PATH.parent.exists():
        KNOWLEDGE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection()
    try:
        has_table = conn.execute(
            "SELECT name FROM sqlite_master"
            " WHERE type='table' AND name='knowledge_entries'"
        ).fetchone()
        if not has_table:
            init_schema(conn)
        count = conn.execute("SELECT COUNT(*) FROM knowledge_entries").fetchone()[0]
    finally:
        conn.close()

    if count == 0 and knowledge_dir is not None and knowledge_dir.exists():
        migrate_from_json(knowledge_dir)


def rebuild(knowledge_dir: Path) -> dict:
    """清空并全量重建（用于维护/归档同步）"""
    conn = get_connection()
    try:
        conn.execute("DROP TABLE IF EXISTS knowledge_entries")
        conn.execute("DROP TABLE IF EXISTS knowledge_fts")
        conn.execute("DROP TABLE IF EXISTS city_meta")
        conn.execute("DROP TABLE IF EXISTS db_meta")
        conn.commit()
    finally:
        conn.close()
    return migrate_from_json(knowledge_dir)


# ------------------------------ 读 ------------------------------

def fetch_all_entries() -> list[dict]:
    """全量读取知识条目（按城市、插入序排列，保持城市块连续）"""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM knowledge_entries ORDER BY city, rid"
        ).fetchall()
        return [row_to_entry(r) for r in rows]
    finally:
        conn.close()


def fetch_city_meta() -> dict[str, dict]:
    """读取全部城市元数据：{城市名: {keywords, aliases, city_tag}}"""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM city_meta ORDER BY city").fetchall()
        out: dict[str, dict] = {}
        for r in rows:
            out[r["city"]] = {
                "keywords": json.loads(r["keywords"] or "[]"),
                "aliases": json.loads(r["aliases"] or "[]"),
                "city_tag": r["tag"] or "",
            }
        return out
    finally:
        conn.close()


def count_entries() -> int:
    conn = get_connection()
    try:
        return conn.execute("SELECT COUNT(*) FROM knowledge_entries").fetchone()[0]
    finally:
        conn.close()


# ------------------------------ 写 ------------------------------

def insert_entries(entries: list[dict]) -> int:
    """事务性写入条目；返回实际插入条数（UNIQUE 冲突跳过）"""
    if not entries:
        return 0
    conn = get_connection()
    inserted = 0
    try:
        for entry in entries:
            cur = conn.execute(
                "INSERT OR IGNORE INTO knowledge_entries (%s) VALUES (%s)"
                % (
                    ", ".join(_ROW_FIELDS),
                    ", ".join("?" * len(_ROW_FIELDS)),
                ),
                _entry_to_row(entry),
            )
            inserted += cur.rowcount
            if cur.rowcount > 0:
                _sync_fts_for(conn, entry)
            # 新城市自动补城市元数据（与旧 JSON 路径生成的 _meta 一致）
            conn.execute(
                "INSERT OR IGNORE INTO city_meta(city, keywords, aliases, tag)"
                " VALUES (?, ?, ?, ?)",
                (
                    entry.get("city", ""),
                    json.dumps([entry.get("city", "")], ensure_ascii=False),
                    json.dumps([entry.get("city", "")], ensure_ascii=False),
                    entry.get("category", "其他"),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return inserted


# ------------------------------ 导出（归档/审计） ------------------------------

def dump_to_json(out_dir: Path, knowledge_dir: Path | None = None) -> dict:
    """把 SQLite 内容导出为与旧格式一致的城市 JSON 文件（备份/归档用）"""
    ensure_db(knowledge_dir)
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM knowledge_entries ORDER BY city, rid"
        ).fetchall()
    finally:
        conn.close()

    by_city: dict[str, list[dict]] = {}
    for row in rows:
        by_city.setdefault(row["city"], []).append(row_to_entry(row))

    metas = fetch_city_meta()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = 0
    for city in sorted(by_city):
        meta = metas.get(city, {})
        payload = [
            {
                "id": "_meta",
                "domain": "全国旅游",
                "city": city,
                "category": "_meta",
                "sub_category": "",
                "title": "",
                "content": "",
                "keywords": meta.get("keywords", [city]),
                "source": "系统自动生成",
                "source_url": "",
                "chunk_id": 0,
                "city_tag": meta.get("city_tag", "其他"),
                "aliases": meta.get("aliases", [city]),
            }
        ]
        payload.extend(by_city[city])
        with open(out_dir / f"{city}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        files += 1
    return {"files": files, "entries": len(rows), "cities": len(by_city)}