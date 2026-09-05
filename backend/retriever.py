"""检索模块（统一检索服务）

实现四种检索方案，统一由 KnowledgeBase 类管理全部缓存与索引状态：
- 方案A：jieba 分词 + 关键词匹配 + 倒排索引
- 方案B：BM25 检索（rank-bm25）
- 方案C：文本向量相似度检索（sentence-transformers 稠密向量，增量式快照持久化）
- 方案D：BM25 + 文本向量的混合检索（加权融合）

模块级维护一个 KnowledgeBase 单例（knowledge_base），并保留同名薄封装函数
作为稳定的公开 API；新增/测试时也可直接使用单例对象。

向量索引增量式持久化（backend/data/vector_index/）：
新增知识条目入库后，下一次向量/混合检索只会对新条目做编码，不再全库重建。
删除快照目录、更换 EMBEDDING_MODEL_NAME 或提升 VECTOR_INDEX_VERSION
均可强制全量重建。
"""

import json
import os
import threading
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .city_detector import CITY_TAGS, COVERED_CITIES, _load_city_metadata, get_all_city_names

# 知识库与向量快照目录（类内按使用时机读取模块全局，便于测试时替换）
KNOWLEDGE_DIR = config.KNOWLEDGE_DIR
VECTOR_INDEX_DIR = config.VECTOR_INDEX_DIR
VECTOR_INDEX_VERSION = config.VECTOR_INDEX_VERSION

_VECTOR_MODEL_NAME = config.EMBEDDING_MODEL_NAME
_VECTOR_QUERY_PREFIX = config.VECTOR_QUERY_PREFIX

_STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这",
}


class KnowledgeBase:
    """知识库检索服务的状态容器：加载、索引、四种检索策略、向量快照持久化。"""

    def __init__(self) -> None:
        # 知识库缓存
        self._entries: list[dict] | None = None
        self._by_city: dict[str, list[dict]] | None = None
        # BM25 缓存
        self._bm25_corpus: list[list[str]] | None = None
        self._bm25_index = None  # BM25Okapi | None
        # 倒排索引缓存（按候选集缓存：key 为城市名或 "__all__"）
        self._inverted_index_cache: dict[str, dict[str, set[int]]] = {}
        # 向量检索缓存
        self._vector_model = None
        self._vector_corpus: np.ndarray | None = None
        self._vector_city_ids: dict[str, list[str]] | None = None
        self._vector_status: str = "not_loaded"  # not_loaded / ready / unavailable
        self._vector_lock = threading.Lock()
        # jieba 分词词典初始化标志
        self._jieba_initialized: bool = False

    # ============================================================
    # 分词初始化
    # ============================================================

    def _ensure_jieba(self) -> None:
        """初始化 jieba 分词，加载城市名到词典"""
        if self._jieba_initialized:
            return
        for name in get_all_city_names():
            jieba.add_word(name)
        self._jieba_initialized = True

    # ============================================================
    # 知识库加载
    # ============================================================

    def load(self) -> list[dict]:
        """加载全部知识库到内存，并缓存"""
        if self._entries is not None:
            return self._entries

        all_entries: list[dict] = []
        if not KNOWLEDGE_DIR.exists():
            print(f"[WARNING] 知识库目录不存在: {KNOWLEDGE_DIR}")
            all_entries = []
        else:
            for json_file in sorted(KNOWLEDGE_DIR.glob("*.json")):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            # 过滤掉 _meta 元数据条目，不作为可检索的知识
                            data = [
                                e
                                for e in data
                                if isinstance(e, dict) and e.get("id") != "_meta"
                            ]
                            all_entries.extend(data)
                        elif isinstance(data, dict):
                            # 兼容 {_meta, items} 格式；普通单条对象仍可直接加载。
                            items = data.get("items")
                            if isinstance(items, list):
                                all_entries.extend(
                                    e for e in items if isinstance(e, dict)
                                )
                            elif data.get("id") != "_meta":
                                all_entries.append(data)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"[WARNING] 读取知识库文件失败 {json_file}: {e}")

        self._entries = all_entries
        print(f"[INFO] 知识库加载完成：共 {len(all_entries)} 条记录")
        return all_entries

    def get_by_city(self) -> dict[str, list[dict]]:
        """按城市分组获取知识库"""
        if self._by_city is not None:
            return self._by_city

        city_map: dict[str, list[dict]] = {}
        for entry in self.load():
            city = entry.get("city", "未知")
            city_map.setdefault(city, []).append(entry)

        self._by_city = city_map
        return city_map

    def clear_knowledge_caches(self) -> None:
        """清空知识库/倒排/BM25 内存缓存（知识库内容变更后调用）"""
        self._entries = None
        self._by_city = None
        self._bm25_corpus = None
        self._bm25_index = None
        self._inverted_index_cache.clear()

    def reload(self) -> None:
        """知识库变更（新条目入库）后整体刷新：清空内存缓存，
        磁盘向量快照保留，下次向量/混合检索只对新增条目做增量编码。"""
        self.clear_knowledge_caches()
        self.invalidate_vector_cache()

    # ============================================================
    # 文本与分词工具
    # ============================================================

    @staticmethod
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

    def _tokenized_corpus(self, entries: list[dict]) -> list[list[str]]:
        """对所有条目进行分词"""
        self._ensure_jieba()
        corpus = []
        for entry in entries:
            text = self._build_search_text(entry)
            tokens = list(jieba.cut(text))
            corpus.append(tokens)
        return corpus

    # ============================================================
    # 索引构建
    # ============================================================

    def _ensure_bm25(self, entries: list[dict]) -> None:
        """初始化 BM25 索引（若尚未初始化）"""
        if self._bm25_corpus is None or self._bm25_index is None:
            self._bm25_corpus = self._tokenized_corpus(entries)
            self._bm25_index = BM25Okapi(self._bm25_corpus)

    def _inverted_index(
        self, entries: list[dict], cache_key: str = "__all__"
    ) -> dict[str, set[int]]:
        """构建倒排索引（按候选集缓存，不同城市互不污染）"""
        if cache_key in self._inverted_index_cache:
            return self._inverted_index_cache[cache_key]

        self._ensure_jieba()
        index: dict[str, set[int]] = {}
        for idx, entry in enumerate(entries):
            text = self._build_search_text(entry)
            tokens = set(jieba.cut(text))
            for token in tokens:
                index.setdefault(token, set()).add(idx)

        self._inverted_index_cache[cache_key] = index
        return index

    # ============================================================
    # 方案 A：关键词匹配检索（jieba 分词 + 倒排索引）
    # ============================================================

    def search_keyword(
        self,
        question: str,
        city: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """关键词匹配检索

        Args:
            question: 用户问题
            city: 限定城市（None 表示全库搜索）
            top_k: 返回 Top-K 结果

        Returns:
            检索结果列表，每项包含原知识条目 + score 字段
        """
        self._ensure_jieba()
        all_entries = self.load()
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
        inverted = self._inverted_index(candidates, cache_key)

        # 对问题分词，过滤停用词
        question_tokens = list(jieba.cut(question))
        question_keywords = [
            t for t in question_tokens if t.strip() and t not in _STOP_WORDS
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
        self,
        question: str,
        city: str | None = None,
        top_k: int = 5,
        city_boost: float = 1.5,
    ) -> list[dict]:
        """BM25 检索

        Args:
            question: 用户问题
            city: 限定城市（None 表示全库搜索，但匹配城市会有加权）
            top_k: 返回 Top-K 结果
            city_boost: 城市匹配加权系数（仅当 city 非空时生效）

        Returns:
            检索结果列表，每项包含原知识条目 + score 字段
        """
        self._ensure_jieba()
        all_entries = self.load()
        if not all_entries:
            return []

        # 初始化 BM25（全库）
        self._ensure_bm25(all_entries)

        # 对问题分词
        question_tokens = list(jieba.cut(question))

        # BM25 评分
        bm25_scores = self._bm25_index.get_scores(question_tokens)

        # 构建带分数的条目列表（城市优先加权）
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
    # 向量模型与快照持久化
    # ============================================================

    @property
    def vector_status(self) -> str:
        """向量检索可用状态：not_loaded / ready / unavailable"""
        return self._vector_status

    def _load_vector_model(self):
        """加载向量模型（已加载则直接复用）；失败返回 None。
        测试可替换本方法注入假模型。"""
        if self._vector_model is not None:
            return self._vector_model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        try:
            try:
                self._vector_model = SentenceTransformer(
                    _VECTOR_MODEL_NAME, local_files_only=True
                )
            except Exception:
                # 缓存未命中，尝试自动下载（国内网络建议设置 HF_ENDPOINT=https://hf-mirror.com）
                print("[INFO] 本地缓存未命中，尝试自动下载模型...")
                self._vector_model = SentenceTransformer(_VECTOR_MODEL_NAME)
            return self._vector_model
        except Exception as e:
            print(f"[WARNING] 向量模型加载失败：{e}")
            print(
                "[HINT] 首次使用需先下载模型（国内网络可先设置 "
                "HF_ENDPOINT=https://hf-mirror.com 或配置代理）："
                f'python -c "from sentence_transformers import SentenceTransformer; '
                f"SentenceTransformer('{_VECTOR_MODEL_NAME}')\""
            )
            return None

    def invalidate_vector_cache(self) -> None:
        """知识库新增条目后调用：仅清空内存中的向量语料。

        磁盘快照保留，下次向量/混合检索时自动加载快照，
        并且只对新增条目做增量编码，不再全库重建。
        """
        with self._vector_lock:
            self._vector_corpus = None
            self._vector_city_ids = None
            self._vector_status = "not_loaded"

    def _snapshot_paths(self) -> tuple[Path, Path]:
        """返回向量快照文件路径（向量矩阵 + manifest）"""
        return (
            VECTOR_INDEX_DIR / "embeddings.npy",
            VECTOR_INDEX_DIR / "manifest.json",
        )

    @staticmethod
    def _group_ids_by_city(entries: list[dict]) -> dict[str, list[str]]:
        """按城市分组（组内保持条目顺序），作为快照对齐的基准。

        知识库按城市分文件组织，同一城市的条目在全局列表中连续出现，
        因此"城市块顺序 + 块内顺序"与全局条目顺序一一对应。
        """
        city_ids: dict[str, list[str]] = {}
        for entry in entries:
            city = entry.get("city", "未知")
            city_ids.setdefault(city, []).append(entry.get("id") or "")
        return city_ids

    def _load_vector_snapshot(self) -> tuple[np.ndarray | None, dict[str, list[str]] | None]:
        """读取磁盘快照；缺失、版本不符或损坏时返回 (None, None)"""
        emb_path, manifest_path = self._snapshot_paths()
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
        self, embeddings: np.ndarray, city_ids: dict[str, list[str]]
    ) -> None:
        """原子写快照（临时文件 + rename），并发写不会损坏"""
        VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        emb_path, manifest_path = self._snapshot_paths()
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
        self,
        entries: list[dict],
        old_emb: np.ndarray,
        old_city_ids: dict[str, list[str]],
        model,
    ) -> tuple[np.ndarray, int]:
        """按城市块增量编码：旧条目从快照矩阵复用，仅新条目编码。

        返回 (与 entries 严格按位置对齐的新语料矩阵, 新增条目数)。
        依赖前提：当前各城市的条目列表是旧快照对应城市列表的"后缀扩展"。
        """
        # 旧矩阵各城市块的起始行
        block_start: dict[str, int] = {}
        offset = 0
        for city, ids in old_city_ids.items():
            block_start[city] = offset
            offset += len(ids)

        # 找出"新条目"：在其所属城市块内的位置 >= 旧块长度
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
                [self._build_search_text(entries[i]) for i in new_indices],
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

    def init_vector_index(self, entries: list[dict]) -> np.ndarray | None:
        """构建/加载向量索引（首次调用会下载模型）。

        优先级：
        1. 磁盘快照与当前库完全一致 → 直接加载，零编码
        2. 快照是当前库的"按城市前缀" → 只增量编码新增条目
        3. 其余情况（版本/模型变更、数据被改动）→ 全量重建
        """
        if self._vector_corpus is not None:
            return self._vector_corpus
        with self._vector_lock:
            if self._vector_corpus is not None:
                return self._vector_corpus

            current_city_ids = self._group_ids_by_city(entries)
            snapshot_emb, snapshot_city_ids = self._load_vector_snapshot()

            # 1) 快照命中：当前库与快照完全一致，零编码加载
            if snapshot_emb is not None and snapshot_city_ids == current_city_ids:
                if self._load_vector_model() is None:
                    self._vector_status = "unavailable"
                    return None
                self._vector_corpus = snapshot_emb
                self._vector_city_ids = current_city_ids
                self._vector_status = "ready"
                print(
                    f"[INFO] 向量索引从快照加载：{len(entries)} 条"
                    f"（{len(current_city_ids)} 个城市）"
                )
                return self._vector_corpus

            # 2) 增量更新：旧快照是当前库的"按城市前缀"
            if snapshot_emb is not None and all(
                current_city_ids.get(city, [])[: len(ids)] == ids
                for city, ids in snapshot_city_ids.items()
            ):
                model = self._load_vector_model()
                if model is None:
                    self._vector_status = "unavailable"
                    return None
                try:
                    corpus, new_count = self._encode_incremental(
                        entries, snapshot_emb, snapshot_city_ids, model
                    )
                    self._save_vector_snapshot(corpus, current_city_ids)
                    self._vector_corpus = corpus
                    self._vector_city_ids = current_city_ids
                    self._vector_status = "ready"
                    print(
                        f"[INFO] 向量索引增量更新：新增 {new_count} 条"
                        f"（共 {len(entries)} 条，{len(current_city_ids)} 个城市）"
                    )
                    return self._vector_corpus
                except Exception as e:
                    print(f"[WARNING] 向量索引增量更新失败：{e}")
                    self._vector_status = "unavailable"
                    return None

            # 3) 全量重建
            try:
                model = self._load_vector_model()
                if model is None:
                    self._vector_status = "unavailable"
                    return None
                texts = [self._build_search_text(entry) for entry in entries]
                vectors = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32,
                )
                corpus = np.asarray(vectors, dtype=np.float32)
                self._save_vector_snapshot(corpus, current_city_ids)
                self._vector_corpus = corpus
                self._vector_city_ids = current_city_ids
                self._vector_status = "ready"
                print(
                    f"[INFO] 向量索引全量构建完成：共 {len(entries)} 条，"
                    f"维度 {corpus.shape[1]}"
                )
            except Exception as e:
                self._vector_status = "unavailable"
                print(f"[WARNING] 向量模型加载失败：{e}")
                return None
            return self._vector_corpus

    def _get_corpus_for(self, entries: list[dict]) -> np.ndarray | None:
        """获取与 entries 严格位置对齐的向量语料（带对齐防御）"""
        corpus = self.init_vector_index(entries)
        if corpus is not None and len(corpus) != len(entries):
            print("[WARNING] 向量语料与条目数不一致，强制重建向量索引")
            self.invalidate_vector_cache()
            corpus = self.init_vector_index(entries)
        return corpus

    def _embed_query(self, question: str) -> np.ndarray | None:
        """对用户问题编码（bge 模型需加检索指令前缀）"""
        try:
            if self._vector_model is None:
                return None
            vec = self._vector_model.encode(
                _VECTOR_QUERY_PREFIX + question,
                normalize_embeddings=True,
            )
            return np.asarray(vec, dtype=np.float32)
        except Exception as e:
            print(f"[WARNING] 问题编码失败：{e}")
            return None

    # ============================================================
    # 方案 C：文本向量相似度检索（sentence-transformers）
    # ============================================================

    def search_vector(
        self,
        question: str,
        city: str | None = None,
        top_k: int = 5,
        city_boost: float = 1.5,
    ) -> list[dict]:
        """文本向量相似度检索（稠密向量余弦相似度）

        向量模型不可用时返回空列表。
        """
        all_entries = self.load()
        if not all_entries:
            return []

        corpus = self._get_corpus_for(all_entries)
        if corpus is None:
            return []

        q_vec = self._embed_query(question)
        if q_vec is None:
            return []

        sims = corpus @ q_vec
        scored_entries = self._apply_city_boost(
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

    @staticmethod
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

    # ============================================================
    # 方案 D：BM25 + 文本向量混合检索（加权融合）
    # ============================================================

    def search_hybrid(
        self,
        question: str,
        city: str | None = None,
        top_k: int = 5,
        w_bm25: float = 0.5,
        w_vector: float = 0.5,
        city_boost: float = 1.5,
    ) -> list[dict]:
        """BM25 + 文本向量混合检索。

        将 BM25 分数归一化到 [0,1] 后与向量余弦相似度加权求和；
        向量模型不可用时自动降级为纯 BM25。
        """
        self._ensure_jieba()
        all_entries = self.load()
        if not all_entries:
            return []

        self._ensure_bm25(all_entries)
        question_tokens = list(jieba.cut(question))
        bm25_scores = self._bm25_index.get_scores(question_tokens)

        corpus = self._get_corpus_for(all_entries)
        if corpus is None:
            print("[INFO] 向量模型不可用，混合检索降级为 BM25")
            return self.search_bm25(question, city=city, top_k=top_k, city_boost=city_boost)

        q_vec = self._embed_query(question)
        if q_vec is None:
            return self.search_bm25(question, city=city, top_k=top_k, city_boost=city_boost)

        # BM25 分数 min-max 归一化到 [0,1]
        bm25_min = float(np.min(bm25_scores))
        bm25_max = float(np.max(bm25_scores))
        if bm25_max > bm25_min:
            bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_norm = np.zeros_like(bm25_scores)

        vec_scores = corpus @ q_vec
        fused = w_bm25 * bm25_norm + w_vector * vec_scores

        scored_entries = self._apply_city_boost(
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

    def search(
        self,
        question: str,
        city: str | None = None,
        top_k: int = 5,
        method: str = "bm25",
    ) -> list[dict]:
        """统一检索入口。

        Args:
            question: 用户问题
            city: 限定城市
            top_k: 返回 Top-K 结果
            method: 检索方法 ("keyword" / "bm25" / "vector" / "hybrid")
        """
        if method == "keyword":
            return self.search_keyword(question, city=city, top_k=top_k)
        if method == "vector":
            return self.search_vector(question, city=city, top_k=top_k)
        if method == "hybrid":
            return self.search_hybrid(question, city=city, top_k=top_k)
        return self.search_bm25(question, city=city, top_k=top_k)

    def compare(
        self,
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
                entries = self.search(question, city=city, top_k=top_k, method=method)
                if method in ("vector", "hybrid") and self.vector_status == "unavailable":
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

    # ============================================================
    # 视图查询（城市信息 / 知识库浏览）
    # ============================================================

    def city_info(self) -> list[dict]:
        """获取所有已覆盖城市的基本信息（标签从知识库 _meta 自动读取）"""
        _load_city_metadata()
        city_map = self.get_by_city()
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

    def knowledge_page(
        self,
        city: str | None = None,
        category: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """分页获取知识库条目"""
        all_entries = self.load()

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


# ============================================================
# 模块级单例与公开 API（薄封装，保持既有调用习惯）
# ============================================================

knowledge_base = KnowledgeBase()


def load_knowledge_base() -> list[dict]:
    return knowledge_base.load()


def get_knowledge_by_city() -> dict[str, list[dict]]:
    return knowledge_base.get_by_city()


def search_keyword(question: str, city: str | None = None, top_k: int = 5) -> list[dict]:
    return knowledge_base.search_keyword(question, city=city, top_k=top_k)


def search_bm25(
    question: str, city: str | None = None, top_k: int = 5, city_boost: float = 1.5
) -> list[dict]:
    return knowledge_base.search_bm25(question, city=city, top_k=top_k, city_boost=city_boost)


def search_vector(
    question: str, city: str | None = None, top_k: int = 5, city_boost: float = 1.5
) -> list[dict]:
    return knowledge_base.search_vector(question, city=city, top_k=top_k, city_boost=city_boost)


def search_hybrid(
    question: str,
    city: str | None = None,
    top_k: int = 5,
    w_bm25: float = 0.5,
    w_vector: float = 0.5,
    city_boost: float = 1.5,
) -> list[dict]:
    return knowledge_base.search_hybrid(
        question, city=city, top_k=top_k, w_bm25=w_bm25, w_vector=w_vector, city_boost=city_boost
    )


def search_knowledge(
    question: str,
    city: str | None = None,
    top_k: int = 5,
    method: str = "bm25",
) -> list[dict]:
    return knowledge_base.search(question, city=city, top_k=top_k, method=method)


def compare_methods(question: str, city: str | None = None, top_k: int = 3) -> list[dict]:
    return knowledge_base.compare(question, city=city, top_k=top_k)


def get_city_info() -> list[dict]:
    return knowledge_base.city_info()


def get_knowledge_page(
    city: str | None = None,
    category: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> dict:
    return knowledge_base.knowledge_page(
        city=city, category=category, page=page, page_size=page_size
    )


def get_vector_status() -> str:
    return knowledge_base.vector_status


def invalidate_vector_cache() -> None:
    knowledge_base.invalidate_vector_cache()