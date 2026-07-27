"""
检索模块
实现两种检索方案：
- 方案A：jieba 分词 + 关键词匹配 + 倒排索引
- 方案B：BM25 检索（rank-bm25）
支持城市过滤，优先匹配指定城市的知识片段
"""

import json
import os
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from city_detector import COVERED_CITIES, CITY_TAGS, _load_city_metadata

# 知识库目录
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# 全局缓存
_knowledge_cache: list[dict] | None = None
_city_knowledge_cache: dict[str, list[dict]] | None = None
# BM25 相关缓存
_bm25_corpus: list[list[str]] | None = None
_bm25_index: BM25Okapi | None = None
# 倒排索引缓存
_inverted_index: dict[str, set[int]] | None = None
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
                    entries = [e for e in entries if e.get("id") != "_meta"]
                    all_entries.extend(entries)
                elif isinstance(entries, dict):
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


def _init_inverted_index(entries: list[dict]) -> dict[str, set[int]]:
    """构建倒排索引（若尚未初始化）"""
    global _inverted_index
    if _inverted_index is not None:
        return _inverted_index

    _init_jieba()
    _inverted_index = {}
    for idx, entry in enumerate(entries):
        text = _build_search_text(entry)
        tokens = set(jieba.cut(text))
        for token in tokens:
            if token not in _inverted_index:
                _inverted_index[token] = set()
            _inverted_index[token].add(idx)

    return _inverted_index


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
    if city and city in COVERED_CITIES:
        candidates = [e for e in all_entries if e.get("city") == city]
        if not candidates:
            candidates = all_entries  # fallback to all
    else:
        candidates = all_entries

    # 构建倒排索引
    inverted = _init_inverted_index(candidates)

    # 对问题分词
    question_tokens = list(jieba.cut(question))
    # 过滤停用词（简单处理）
    stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
    question_keywords = [t for t in question_tokens if t.strip() and t not in stop_words]

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
        entry["score"] = round(score / len(question_keywords), 4) if question_keywords else 0
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
        method: 检索方法 ("keyword" 或 "bm25")

    Returns:
        检索结果列表
    """
    if method == "keyword":
        return search_keyword(question, city=city, top_k=top_k)
    else:
        return search_bm25(question, city=city, top_k=top_k)


def get_city_info() -> list[dict]:
    """获取所有已覆盖城市的基本信息（标签从知识库_meta自动读取）"""
    _load_city_metadata()
    city_map = get_knowledge_by_city()
    info_list = []
    for city in COVERED_CITIES:
        entries = city_map.get(city, [])
        categories = list(set(e.get("category", "") for e in entries))
        info_list.append({
            "city": city,
            "tags": CITY_TAGS.get(city, ""),
            "entry_count": len(entries),
            "categories": categories,
        })
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
