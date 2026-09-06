"""混合检索（M1）测试：RRF 融合 / 召回池与重排 / 城市硬过滤

覆盖：
1. RRF 倒数排名融合的计分与排序语义
2. 城市硬过滤：单城市问题限定城市；多城市/对比类问题回退全库
3. search_bm25 / search_vector 的城市硬过滤
4. search_hybrid 端到端：RRF 结果、城市限定、score/bm25_score/vector_score 字段
5. 重排器注入：按重排分数改变最终排序
6. 向量通道不可用时的 BM25 降级

运行（仓库根目录）：
    uv run python -m unittest discover -s backend/tests -v
"""

import unittest

import numpy as np

from backend import city_detector, retriever


def _entry(city, eid, title, content):
    return {
        "id": eid,
        "domain": "全国旅游",
        "city": city,
        "category": "景点",
        "sub_category": "",
        "title": title,
        "content": content,
        "keywords": [],
        "source": "测试来源",
        "chunk_id": 1,
    }


class FakeReranker:
    """模拟 CrossEncoder：predict 返回按池顺序给出的分数"""

    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs, **kwargs):
        self.pairs = list(pairs)
        return np.asarray(self.scores, dtype=np.float32)


class HybridRRFTests(unittest.TestCase):
    def setUp(self):
        self.kb = retriever.KnowledgeBase()
        self.entries = [
            _entry("甲市", "j-1", "甲市火锅", "本地火锅 麻辣鲜香 推荐"),
            _entry("甲市", "j-2", "甲市长城", "古长城 徒步 景点 门票"),
            _entry("乙市", "y-1", "乙市海滩", "海滩 海鲜 美食 度假"),
            _entry("乙市", "y-2", "乙市博物馆", "博物馆 展览 历史"),
        ]
        self.kb.load = lambda: self.entries

        # 向量通道：矩阵行与 entries 一一对应（one-hot 使 sims = 第一列）
        self.matrix = np.array(
            [[0.9, 0.0], [0.1, 0.0], [0.8, 0.0], [0.4, 0.0]],
            dtype=np.float32,
        )
        self.kb._get_corpus_for = lambda entries: self.matrix
        self.kb._embed_query = lambda q: np.array([1.0, 0.0], dtype=np.float32)
        self.kb._load_reranker = lambda: None  # 默认不重排

        # 测试城市加入覆盖列表（COVERED_CITIES 由 city_detector 维护）
        self._saved_cities = list(city_detector.COVERED_CITIES)
        for city in ("甲市", "乙市"):
            if city not in city_detector.COVERED_CITIES:
                city_detector.COVERED_CITIES.append(city)

    def tearDown(self):
        city_detector.COVERED_CITIES.clear()
        city_detector.COVERED_CITIES.extend(self._saved_cities)

    # ---------- RRF ----------

    def test_rrf_fuse_semantics(self):
        """两个通道都命中的文档排名更靠前；单通道排名不能压过双通道"""
        scores = retriever.KnowledgeBase._rrf_fuse_indices([0, 1, 2], [1, 2, 3])
        self.assertGreater(scores[1], scores[2])  # 双通道名次更高
        self.assertGreater(scores[2], scores[0])  # 双通道 > 仅第一通道第1名
        self.assertGreater(scores[0], scores[3])  # 单通道第1 > 单通道第3

    # ---------- 城市硬过滤 ----------

    def test_scope_single_city(self):
        scope, filtered = self.kb._scope_candidates(self.entries, "甲市", "甲市 火锅")
        self.assertTrue(filtered)
        self.assertEqual(scope, [0, 1])

    def test_scope_multi_city_question_falls_back_to_all(self):
        for question in ("甲市和乙市哪个好", "甲市与乙市对比"):
            scope, filtered = self.kb._scope_candidates(self.entries, "甲市", question)
            self.assertFalse(filtered)
            self.assertIsNone(scope)

    def test_scope_unknown_or_missing_city_falls_back(self):
        self.assertEqual(self.kb._scope_candidates(self.entries, "丙市", "火锅"), (None, False))
        self.assertEqual(self.kb._scope_candidates(self.entries, None, "火锅"), (None, False))

    # ---------- 单通道 + 城市硬过滤 ----------

    def test_bm25_hard_filters_to_city(self):
        results = self.kb.search_bm25("火锅", city="甲市", top_k=5)
        self.assertTrue(results)
        self.assertTrue(all(r["city"] == "甲市" for r in results))
        self.assertEqual(results[0]["id"], "j-1")

    def test_vector_hard_filters_to_city(self):
        results = self.kb.search_vector("随便", city="甲市", top_k=5)
        self.assertTrue(results)
        self.assertTrue(all(r["city"] == "甲市" for r in results))
        self.assertEqual(results[0]["id"], "j-1")  # 甲市里向量分最高

    # ---------- 混合检索（RRF） ----------

    def test_hybrid_rrf_within_city_scope(self):
        results = self.kb.search_hybrid("火锅", city="甲市", top_k=5, rerank=False)
        self.assertTrue(all(r["city"] == "甲市" for r in results))
        self.assertEqual(results[0]["id"], "j-1")
        # 同时暴露两通道分数与新 score 字段
        self.assertIn("bm25_score", results[0])
        self.assertIn("vector_score", results[0])

    def test_hybrid_reranker_reorders_pool(self):
        # RRF 池：[j-1(0), j-2(1)]；重排器反转为 [j-2, j-1]
        fake = FakeReranker([0.1, 0.9])
        self.kb._load_reranker = lambda: fake
        results = self.kb.search_hybrid("火锅", city="甲市", top_k=1, rerank=True)
        self.assertEqual(results[0]["id"], "j-2")
        self.assertEqual(len(fake.pairs), 2)  # 召回池内的文档对都被重排

    def test_hybrid_degrades_to_bm25_without_vector(self):
        self.kb._get_corpus_for = lambda entries: None
        results = self.kb.search_hybrid("火锅", city="甲市", top_k=5, rerank=False)
        self.assertTrue(all(r["city"] == "甲市" for r in results))
        self.assertEqual(results[0]["id"], "j-1")
        self.assertNotIn("vector_score", results[0])  # 降级路径只保留 score


if __name__ == "__main__":
    unittest.main()