"""
检索模块
实现四种检索方案：
- 方案A：jieba 分词 + 关键词匹配 + 倒排索引
- 方案B：BM25 检索（rank-bm25）
- 方案C：文本向量相似度检索（sentence-transformers 稠密向量）
- 方案D：BM25 + 文本向量的混合检索（加权融合）
支持城市过滤，优先匹配指定城市的知识片段

向量索引使用增量式持久化快照（backend/data/vector_index/）：
新增知识条目入库后，下一次向量/混合检索只会对新条目做编码，
不再全库重建。删除快照目录、更换 EMBEDDING_MODEL_NAME 或
提升 VECTOR_INDEX_VERSION 均可强制全量重建。
"""

import json
import os
import threading
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi
import numpy as np

from city_detector import COVERED_CITIES, CITY_TAGS, _load_city_metadata

# 知识库目录
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# 全局缓存
_knowledge_cache: list[dict] | None = None
_city_knowledge_cache: dict[str, list[dict]] | None = None
# BM25 相关缓存
_bm25_corpus: list[list[str]] | None = None
_bm25_index: BM25Okapi | None = None
# 倒排索引缓存（按候选集缓存：key 为城市名或 "__all__"，避免跨城市复用错索引）
_inverted_index_cache: dict[str, dict[str, set[int]]] = {}
# 向量检索相关缓存
_vector_model = None
_vector_corpus: np.ndarray | None = None
_vector_city_ids: dict[str, list[str]] | None = None  # 城市 -> 条目 id 列表（快照对齐基准）
_vector_status: str = "not_loaded"  # not_loaded / ready / unavailable
_vector_lock = threading.Lock()
_VECTOR_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
_VECTOR_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
# 向量索引持久化目录与版本（版本/模型变更 → 自动全量重建）
VECTOR_INDEX_DIR = Path(__file__).parent / "data" / "vector_index"
VECTOR_INDEX_VERSION = 1
# jieba 分词词典初始化标志
_jieba_initialized: bool = False


def _init_jieba():
    """初始化 jieba 分词，加载城市名到词典"""
    global _jieba_initialized
    if _jieba_initialized:
        return
    # 将城市名和别名加入 jieba 词典，确保能被正确分词
    from city_detector import get_all_city_names

    for name in get_all_city_names():
        jieba.add_word(name)
    _jieba_initialized = True


def load_knowledge_base() -> list[dict]:
    """加载全部知识库到内存，并缓存"""
    global _knowledge_cache
    if _knowledge_cache is not None:
        return _knowledge_cache

    all_entries = []
    if not KNOWLEDGE_DIR.exists():
        print(f"[WARNING] 知识库目录不存在: {KNOWLEDGE_DIR}")
        return all_entries

    for json_file in sorted(KNOWLEDGE_DIR.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                entries = json.load(f)
                if isinstance(entries, list):
                    # 过滤掉 _meta 元数据条目，不作为可检索的知识
                    entries = [
                        e
                        for e in entries
                        if isinstance(e, dict) and e.get("id") != "_meta"
                    ]
                    all_entries.extend(entries)
                elif isinstance(entries, dict):
                    # 兼容 {_meta, items} 格式；普通单条对象仍可直接加载。
                    items = entries.get("items")
                    if isinstance(items, list):
                        all_entries.extend(e for e in items if isinstance(e, dict))
                    elif entries.get("id") != "_meta":
                        all_entries.append(entries)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARNING] 读取知识库文件失败 {json_file}: {e}")

    _knowledge_cache = all_entries
    print(f"[INFO] 知识库加载完成：共 {len(all_entries)} 条记录")
    return all_entries


def get_knowledge_by_city() -> dict[str, list[dict]]:
    """按城市分组获取知识库"""
    global _city_knowledge_cache
    if _city_knowledge_cache is not None:
        return _city_knowledge_cache

    all_entries = load_knowledge_base()
    city_map: dict[str, list[dict]] = {}
    for entry in all_entries:
        city = entry.get("city", "未知")
        if city not in city_map:
            city_map[city] = []
        city_map[city].append(entry)

    _city_knowledge_cache = city_map
    return city_map


def _build_search_text(entry: dict) -> str:
    """构建用于检索的文本（合并标题、内容、关键词）"""
    parts = [
        entry.get("title", ""),
        entry.get("content", ""),
        " ".join(entry.get("keywords", [])),
        entry.get("category", ""),
        entry.get("sub_category", ""),
    ]
    return " ".join(filter(None, parts))


def _get_tokenized_corpus(entries: list[dict]) -> list[list[str]]:
    """对所有条目进行分词"""
    _init_jieba()
    corpus = []
    for entry in entries:
        text = _build_search_text(entry)
        tokens = list(jieba.cut(text))
        corpus.append(tokens)
    return corpus


def _init_bm25(entries: list[dict]) -> tuple[list[list[str]], BM25Okapi]:
    """初始化 BM25 索引（若尚未初始化）"""
    global _bm25_corpus, _bm25_index
    if _bm25_corpus is None or _bm25_index is None:
        _bm25_corpus = _get_tokenized_corpus(entries)
        _bm25_index = BM25Okapi(_bm25_corpus)
    return _bm25_corpus, _bm25_index


def _init_inverted_index(
    entries: list[dict], cache_key: str = "__all__"
) -> dict[str, set[int]]:
    """构建倒排索引（按候选集缓存，不同城市互不污染）"""
    global _inverted_index_cache
    if cache_key in _inverted_index_cache:
        return _inverted_index_cache[cache_key]

    _init_jieba()
    index: dict[str, set[int]] = {}
    for idx, entry in enumerate(entries):
        text = _build_search_text(entry)
        tokens = set(jieba.cut(text))
        for token in tokens:
            if token not in index:
                index[token] = set()
            index[token].add(idx)

    _inverted_index_cache[cache_key] = index
    return index


# ============================================================
# 方案 A：关键词匹配检索（jieba 分词 + 倒排索引）
# ============================================================


def search_keyword(
    question: str,
    city: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    关键词匹配检索

    Args:
        question: 用户问题
        city: 限定城市（None 表示全库搜索）
        top_k: 返回 Top-K 结果

    Returns:
        检索结果列表，每项包含原知识条目 + score 字段
    """
    _init_jieba()
    all_entries = load_knowledge_base()
    if not all_entries:
        return []

    # 城市过滤
    cache_key = "__all__"
    if city and city in COVERED_CITIES:
        candidates = [e for e in all_entries if e.get("city") == city]
        if not candidates:
            candidates = all_entries  # fallback to all
        else:
            cache_key = city
    else:
        candidates = all_entries

    # 构建倒排索引（按候选集缓存，不同城市互不污染）
    inverted = _init_inverted_index(candidates, cache_key)

    # 对问题分词
    question_tokens = list(jieba.cut(question))
    # 过滤停用词（简单处理）
    stop_words = {
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
    }
    question_keywords = [
        t for t in question_tokens if t.strip() and t not in stop_words
    ]

    if not question_keywords:
        return candidates[:top_k]

    # 计算匹配得分
    scores: dict[int, float] = {}
    for token in question_keywords:
        if token in inverted:
            for doc_idx in inverted[token]:
                scores[doc_idx] = scores.get(doc_idx, 0) + 1

    # 按分数排序
    scored_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 返回 top_k 结果
    results = []
    for idx, score in scored_indices[:top_k]:
        entry = dict(candidates[idx])
        entry["score"] = (
            round(score / len(question_keywords), 4) if question_keywords else 0
        )
        results.append(entry)

    return results


# ============================================================
# 方案 B：BM25 检索（rank-bm25）
# ============================================================


def search_bm25(
    question: str,
    city: str | None = None,
    top_k: int = 5,
    city_boost: float = 1.5,
) -> list[dict]:
    """
    BM25 检索

    Args:
        question: 用户问题
        city: 限定城市（None 表示全库搜索，但匹配城市会有加权）
        top_k: 返回 Top-K 结果
        city_boost: 城市匹配加权系数（仅当 city 非空时生效）

    Returns:
        检索结果列表，每项包含原知识条目 + score 字段
    """
    _init_jieba()
    all_entries = load_knowledge_base()
    if not all_entries:
        return []

    # 初始化 BM25（全库）
    _init_bm25(all_entries)

    # 对问题分词
    question_tokens = list(jieba.cut(question))

    # BM25 评分
    bm25_scores = _bm25_index.get_scores(question_tokens)

    # 构建带分数的条目列表
    scored_entries = []
    for idx, entry in enumerate(all_entries):
        score = float(bm25_scores[idx])

        # 城市优先：若识别到城市且条目城市匹配，加权
        if city and entry.get("city") == city:
            score *= city_boost

        # 若有城市且条目城市不匹配，降权
        if city and entry.get("city") != city:
            score *= 0.5

        scored_entries.append((score, entry))

    # 按分数降序排序
    scored_entries.sort(key=lambda x: x[0], reverse=True)

    # 返回 top_k
    results = []
    for score, entry in scored_entries[:top_k]:
        result = dict(entry)
        result["score"] = round(score, 4)
        results.append(result)

    return results


# ============================================================
# 方案 C：文本向量相似度检索（sentence-transformers）
# ============================================================


def get_vector_status() -> str:
    """返回向量检索可用状态：not_loaded / ready / unavailable"""
    return _vector_status


def invalidate_vector_cache() -> None:
    """知识库新增条目后调用：仅清空内存中的向量语料。

    磁盘快照保留，下次向量/混合检索时自动加载快照，
    并且只对新增条目做增量编码，不再全库重建。
    """
    global _vector_corpus, _vector_city_ids, _vector_status
    with _vector_lock:
        _vector_corpus = None
        _vector_city_ids = None
        _vector_status = "not_loaded"


# ============================================================
# 向量索引持久化（增量式快照）
# ============================================================


def _load_vector_model():
    """加载向量模型（已加载则直接复用）；失败返回 None"""
    global _vector_model
    if _vector_model is not None:
        return _vector_model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        try:
            _vector_model = SentenceTransformer(
                _VECTOR_MODEL_NAME, local_files_only=True
            )
        except Exception:
            # 缓存未命中，尝试自动下载（国内网络建议设置 HF_ENDPOINT=https://hf-mirror.com）
            print("[INFO] 本地缓存未命中，尝试自动下载模型...")
            _vector_model = SentenceTransformer(_VECTOR_MODEL_NAME)
        return _vector_model
    except Exception as e:
        print(f"[WARNING] 向量模型加载失败：{e}")
        print(
            "[HINT] 首次使用需先下载模型（国内网络可先设置 "
            "HF_ENDPOINT=https://hf-mirror.com 或配置代理）："
            f'python -c "from sentence_transformers import SentenceTransformer; '
            f"SentenceTransformer('{_VECTOR_MODEL_NAME}')\""
        )
        return None


def _snapshot_paths() -> tuple[Path, Path]:
    """返回向量快照文件路径（向量矩阵 + manifest）"""
    return VECTOR_INDEX_DIR / "embeddings.npy", VECTOR_INDEX_DIR / "manifest.json"


def _group_ids_by_city(entries: list[dict]) -> dict[str, list[str]]:
    """按城市分组（组内保持条目顺序），作为快照对齐的基准。

    知识库按城市分文件组织，同一城市的条目在全局列表中连续出现，
    因此“城市块顺序 + 块内顺序”与全局条目顺序一一对应。
    """
    city_ids: dict[str, list[str]] = {}
    for entry in entries:
        city = entry.get("city", "未知")
        city_ids.setdefault(city, []).append(entry.get("id") or "")
    return city_ids


def _load_vector_snapshot() -> tuple[np.ndarray | None, dict[str, list[str]] | None]:
    """读取磁盘快照；缺失、版本不符或损坏时返回 (None, None)"""
    emb_path, manifest_path = _snapshot_paths()
    if not emb_path.exists() or not manifest_path.exists():
        return None, None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        embeddings = np.load(emb_path)
    except Exception as e:
        print(f"[WARNING] 向量快照读取失败，将执行全量重建：{e}")
        return None, None

    cities = manifest.get("cities")
    if (
        manifest.get("model") != _VECTOR_MODEL_NAME
        or manifest.get("version") != VECTOR_INDEX_VERSION
        or not isinstance(embeddings, np.ndarray)
        or embeddings.ndim != 2
        or not isinstance(cities, list)
    ):
        return None, None

    city_ids: dict[str, list[str]] = {}
    offset = 0
    for block in cities:
        ids = block.get("ids")
        if not isinstance(ids, list) or block.get("count") != len(ids):
            return None, None
        city_ids[block.get("city", "未知")] = ids
        offset += len(ids)
    if offset != embeddings.shape[0]:
        return None, None
    return np.asarray(embeddings, dtype=np.float32), city_ids


def _save_vector_snapshot(
    embeddings: np.ndarray, city_ids: dict[str, list[str]]
) -> None:
    """原子写快照（临时文件 + rename），并发写不会损坏"""
    VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    emb_path, manifest_path = _snapshot_paths()
    manifest = {
        "model": _VECTOR_MODEL_NAME,
        "version": VECTOR_INDEX_VERSION,
        "cities": [
            {"city": city, "count": len(ids), "ids": ids}
            for city, ids in city_ids.items()
        ],
    }
    tmp_emb = emb_path.with_name(f"{emb_path.stem}.{os.getpid()}.tmp.npy")
    tmp_manifest = manifest_path.with_name(
        f"{manifest_path.stem}.{os.getpid()}.tmp.json"
    )
    np.save(tmp_emb, embeddings)
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    os.replace(tmp_emb, emb_path)
    os.replace(tmp_manifest, manifest_path)


def _encode_incremental(
    entries: list[dict],
    old_emb: np.ndarray,
    old_city_ids: dict[str, list[str]],
    model,
) -> tuple[np.ndarray, int]:
    """按城市块增量编码：旧条目从快照矩阵复用，仅新条目编码。

    返回 (与 entries 严格按位置对齐的新语料矩阵, 新增条目数)。
    依赖前提：当前各城市的条目列表是旧快照对应城市列表的“后缀扩展”。
    """
    # 旧矩阵各城市块的起始行
    block_start: dict[str, int] = {}
    offset = 0
    for city, ids in old_city_ids.items():
        block_start[city] = offset
        offset += len(ids)

    # 找出“新条目”：在其所属城市块内的位置 >= 旧块长度
    new_indices: list[int] = []
    city_pos: dict[str, int] = {}
    for idx, entry in enumerate(entries):
        city = entry.get("city", "未知")
        pos = city_pos.get(city, 0)
        city_pos[city] = pos + 1
        if pos >= len(old_city_ids.get(city, [])):
            new_indices.append(idx)

    if new_indices:
        new_vecs = model.encode(
            [_build_search_text(entries[i]) for i in new_indices],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        new_vecs = np.asarray(new_vecs, dtype=np.float32)
    else:
        new_vecs = np.empty((0, old_emb.shape[1]), dtype=np.float32)

    corpus = np.empty((len(entries), old_emb.shape[1]), dtype=np.float32)
    new_vec_by_idx = dict(zip(new_indices, new_vecs))
    city_pos = {}
    for idx, entry in enumerate(entries):
        city = entry.get("city", "未知")
        pos = city_pos.get(city, 0)
        city_pos[city] = pos + 1
        if idx in new_vec_by_idx:
            corpus[idx] = new_vec_by_idx[idx]
        else:
            corpus[idx] = old_emb[block_start[city] + pos]
    return corpus, len(new_indices)


def _init_vector_index(entries: list[dict]) -> np.ndarray | None:
    """构建/加载向量索引（首次调用会下载模型）。

    优先级：
    1. 磁盘快照与当前库完全一致 → 直接加载，零编码
    2. 快照是当前库的“按城市前缀” → 只增量编码新增条目
    3. 其余情况（版本/模型变更、数据被改动）→ 全量重建
    """
    global _vector_corpus, _vector_city_ids, _vector_status
    if _vector_corpus is not None:
        return _vector_corpus
    with _vector_lock:
        if _vector_corpus is not None:
            return _vector_corpus

        current_city_ids = _group_ids_by_city(entries)
        snapshot_emb, snapshot_city_ids = _load_vector_snapshot()

        # 1) 快照命中：当前库与快照完全一致，零编码加载
        if snapshot_emb is not None and snapshot_city_ids == current_city_ids:
            if _load_vector_model() is None:
                _vector_status = "unavailable"
                return None
            _vector_corpus = snapshot_emb
            _vector_city_ids = current_city_ids
            _vector_status = "ready"
            print(
                f"[INFO] 向量索引从快照加载：{len(entries)} 条"
                f"（{len(current_city_ids)} 个城市）"
            )
            return _vector_corpus

        # 2) 增量更新：旧快照是当前库的“按城市前缀”
        if snapshot_emb is not None and all(
            current_city_ids.get(city, [])[: len(ids)] == ids
            for city, ids in snapshot_city_ids.items()
        ):
            model = _load_vector_model()
            if model is None:
                _vector_status = "unavailable"
                return None
            try:
                corpus, new_count = _encode_incremental(
                    entries, snapshot_emb, snapshot_city_ids, model
                )
                _save_vector_snapshot(corpus, current_city_ids)
                _vector_corpus = corpus
                _vector_city_ids = current_city_ids
                _vector_status = "ready"
                print(
                    f"[INFO] 向量索引增量更新：新增 {new_count} 条"
                    f"（共 {len(entries)} 条，{len(current_city_ids)} 个城市）"
                )
                return _vector_corpus
            except Exception as e:
                print(f"[WARNING] 向量索引增量更新失败：{e}")
                _vector_status = "unavailable"
                return None

        # 3) 全量重建
        try:
            model = _load_vector_model()
            if model is None:
                _vector_status = "unavailable"
                return None
            texts = [_build_search_text(entry) for entry in entries]
            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
            corpus = np.asarray(vectors, dtype=np.float32)
            _save_vector_snapshot(corpus, current_city_ids)
            _vector_corpus = corpus
            _vector_city_ids = current_city_ids
            _vector_status = "ready"
            print(
                f"[INFO] 向量索引全量构建完成：共 {len(entries)} 条，"
                f"维度 {corpus.shape[1]}"
            )
        except Exception as e:
            _vector_status = "unavailable"
            print(f"[WARNING] 向量模型加载失败：{e}")
            return None
        return _vector_corpus


def _get_corpus_for(entries: list[dict]) -> np.ndarray | None:
    """获取与 entries 严格位置对齐的向量语料（带对齐防御）"""
    corpus = _init_vector_index(entries)
    if corpus is not None and len(corpus) != len(entries):
        print("[WARNING] 向量语料与条目数不一致，强制重建向量索引")
        invalidate_vector_cache()
        corpus = _init_vector_index(entries)
    return corpus


def _embed_query(question: str) -> np.ndarray | None:
    """对用户问题编码（bge 模型需加检索指令前缀）"""
    try:
        if _vector_model is None:
            return None
        vec = _vector_model.encode(
            _VECTOR_QUERY_PREFIX + question,
            normalize_embeddings=True,
        )
        return np.asarray(vec, dtype=np.float32)
    except Exception as e:
        print(f"[WARNING] 问题编码失败：{e}")
        return None


def _apply_city_boost(
    scored: list[tuple[float, dict]],
    city: str | None,
    city_boost: float,
) -> list[tuple[float, dict]]:
    """城市加权：匹配城市加权，其他城市降权"""
    boosted = []
    for score, entry in scored:
        if city and entry.get("city") == city:
            score *= city_boost
        elif city:
            score *= 0.5
        boosted.append((score, entry))
    return boosted


def search_vector(
    question: str,
    city: str | None = None,
    top_k: int = 5,
    city_boost: float = 1.5,
) -> list[dict]:
    """
    文本向量相似度检索（稠密向量余弦相似度）

    Args:
        question: 用户问题
        city: 限定城市（None 表示全库搜索，但匹配城市会有加权）
        top_k: 返回 Top-K 结果
        city_boost: 城市匹配加权系数（仅当 city 非空时生效）

    Returns:
        检索结果列表；向量模型不可用时返回空列表
    """
    all_entries = load_knowledge_base()
    if not all_entries:
        return []

    corpus = _get_corpus_for(all_entries)
    if corpus is None:
        return []

    q_vec = _embed_query(question)
    if q_vec is None:
        return []

    sims = corpus @ q_vec
    scored_entries = _apply_city_boost(
        [(float(sims[idx]), entry) for idx, entry in enumerate(all_entries)],
        city,
        city_boost,
    )
    scored_entries.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, entry in scored_entries[:top_k]:
        result = dict(entry)
        result["score"] = round(score, 4)
        results.append(result)
    return results


# ============================================================
# 方案 D：BM25 + 文本向量混合检索（加权融合）
# ============================================================


def search_hybrid(
    question: str,
    city: str | None = None,
    top_k: int = 5,
    w_bm25: float = 0.5,
    w_vector: float = 0.5,
    city_boost: float = 1.5,
) -> list[dict]:
    """
    BM25 + 文本向量混合检索

    将 BM25 分数归一化到 [0,1] 后与向量余弦相似度加权求和。
    向量模型不可用时自动降级为纯 BM25。

    Args:
        question: 用户问题
        city: 限定城市
        top_k: 返回 Top-K 结果
        w_bm25: BM25 权重（默认 0.5）
        w_vector: 向量相似度权重（默认 0.5）
        city_boost: 城市匹配加权系数

    Returns:
        检索结果列表
    """
    _init_jieba()
    all_entries = load_knowledge_base()
    if not all_entries:
        return []

    _init_bm25(all_entries)
    question_tokens = list(jieba.cut(question))
    bm25_scores = _bm25_index.get_scores(question_tokens)

    corpus = _get_corpus_for(all_entries)
    if corpus is None:
        print("[INFO] 向量模型不可用，混合检索降级为 BM25")
        return search_bm25(question, city=city, top_k=top_k, city_boost=city_boost)

    q_vec = _embed_query(question)
    if q_vec is None:
        return search_bm25(question, city=city, top_k=top_k, city_boost=city_boost)

    # BM25 分数 min-max 归一化到 [0,1]
    bm25_min = float(np.min(bm25_scores))
    bm25_max = float(np.max(bm25_scores))
    if bm25_max > bm25_min:
        bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
    else:
        bm25_norm = np.zeros_like(bm25_scores)

    vec_scores = corpus @ q_vec
    fused = w_bm25 * bm25_norm + w_vector * vec_scores

    scored_entries = _apply_city_boost(
        [(float(fused[idx]), entry) for idx, entry in enumerate(all_entries)],
        city,
        city_boost,
    )
    scored_entries.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, entry in scored_entries[:top_k]:
        result = dict(entry)
        result["score"] = round(score, 4)
        results.append(result)
    return results


# ============================================================
# 统一检索接口
# ============================================================


def search_knowledge(
    question: str,
    city: str | None = None,
    top_k: int = 5,
    method: str = "bm25",
) -> list[dict]:
    """
    统一检索入口

    Args:
        question: 用户问题
        city: 限定城市
        top_k: 返回 Top-K 结果
        method: 检索方法 ("keyword" / "bm25" / "vector" / "hybrid")

    Returns:
        检索结果列表
    """
    if method == "keyword":
        return search_keyword(question, city=city, top_k=top_k)
    if method == "vector":
        return search_vector(question, city=city, top_k=top_k)
    if method == "hybrid":
        return search_hybrid(question, city=city, top_k=top_k)
    return search_bm25(question, city=city, top_k=top_k)


def compare_methods(
    question: str,
    city: str | None = None,
    top_k: int = 3,
) -> list[dict]:
    """运行全部检索方式，返回带耗时与状态的对比结果"""
    import time

    methods = [
        ("keyword", "关键词匹配"),
        ("bm25", "BM25"),
        ("vector", "向量相似度"),
        ("hybrid", "BM25+向量"),
    ]

    results = []
    for method, label in methods:
        start = time.perf_counter()
        entries = []
        status = "ok"
        try:
            entries = search_knowledge(question, city=city, top_k=top_k, method=method)
            if method in ("vector", "hybrid") and get_vector_status() == "unavailable":
                status = "unavailable"
                entries = []
        except Exception as e:
            status = "error"
            print(f"[WARNING] 检索方式 {method} 执行失败：{e}")
        latency_ms = int((time.perf_counter() - start) * 1000)
        results.append(
            {
                "method": method,
                "label": label,
                "status": status,
                "latency_ms": latency_ms,
                "results": entries,
            }
        )
    return results


def get_city_info() -> list[dict]:
    """获取所有已覆盖城市的基本信息（标签从知识库_meta自动读取）"""
    _load_city_metadata()
    city_map = get_knowledge_by_city()
    info_list = []
    for city in COVERED_CITIES:
        entries = city_map.get(city, [])
        categories = list(set(e.get("category", "") for e in entries))
        info_list.append(
            {
                "city": city,
                "tags": CITY_TAGS.get(city, ""),
                "entry_count": len(entries),
                "categories": categories,
            }
        )
    return info_list


def get_knowledge_page(
    city: str | None = None,
    category: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> dict:
    """分页获取知识库条目"""
    all_entries = load_knowledge_base()

    # 过滤
    if city:
        all_entries = [e for e in all_entries if e.get("city") == city]
    if category:
        all_entries = [e for e in all_entries if e.get("category") == category]

    total = len(all_entries)
    start = (page - 1) * page_size
    end = start + page_size
    page_entries = all_entries[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if total > 0 else 0,
        "entries": page_entries,
    }
