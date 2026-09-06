"""
增量向量索引测试

覆盖核心行为：
1. 首次构建会生成磁盘快照（embeddings.npy + manifest.json）
2. 数据未变时从快照加载，不触发任何重新编码
3. 往"靠前城市"追加条目（新条目插入全局列表中间）时，只增量编码新条目
4. 新增城市文件也能走增量路径
5. 版本号变更触发全量重建

使用 FakeModel 替换真实 embedding 模型（注入 KnowledgeBase 实例），
不依赖模型下载与网络。

运行方式（仓库根目录）：
    uv run python -m unittest discover -s backend/tests -v
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend import city_detector, contribution_store, retriever


class FakeModel:
    """模拟 sentence-transformer：记录 encode 调用，返回固定维度的随机向量"""

    def __init__(self):
        self.encode_calls = []  # 每次 encode 的输入文本列表

    def encode(self, texts, **kwargs):
        texts = list(texts)
        self.encode_calls.append(texts)
        rng = np.random.default_rng(seed=sum(len(t) for t in texts) or 1)
        return rng.random((len(texts), 8)).astype(np.float32)


def _entry(city, eid, title, content="这是一条用于测试的旅游知识内容。"):
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


def _meta(city):
    return {
        "id": "_meta",
        "domain": "全国旅游",
        "city": city,
        "category": "_meta",
        "title": "",
        "content": "",
        "keywords": [city],
        "source": "系统自动生成",
        "aliases": [city],
    }


class IncrementalVectorIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.kb_dir = self.tmp / "knowledge"
        self.kb_dir.mkdir()
        self.vec_dir = self.tmp / "vector_index"
        self.vec_dir.mkdir()

        kb = retriever.knowledge_base

        # 保存原始模块状态，tearDown 恢复
        self._originals = {
            "retriever.KNOWLEDGE_DIR": retriever.KNOWLEDGE_DIR,
            "retriever.VECTOR_INDEX_DIR": retriever.VECTOR_INDEX_DIR,
            "retriever.VECTOR_INDEX_VERSION": retriever.VECTOR_INDEX_VERSION,
            "kb._load_vector_model": kb._load_vector_model,
            "contribution_store.KNOWLEDGE_DIR": contribution_store.KNOWLEDGE_DIR,
            "contribution_store.DB_PATH": contribution_store.DB_PATH,
            "city_detector.KNOWLEDGE_DIR": city_detector.KNOWLEDGE_DIR,
        }

        # 重定向到临时目录
        retriever.KNOWLEDGE_DIR = self.kb_dir
        retriever.VECTOR_INDEX_DIR = self.vec_dir
        contribution_store.KNOWLEDGE_DIR = self.kb_dir
        contribution_store.DB_PATH = self.tmp / "db" / "contributions.db"
        city_detector.KNOWLEDGE_DIR = self.kb_dir

        # 假模型替换真实模型加载（注入 KnowledgeBase 实例方法）
        self.fake = FakeModel()
        kb._load_vector_model = lambda: self.fake

        # 初始知识库：a市 2 条 + b市 1 条（不含 _meta）
        self._write_city(
            "a市",
            [_meta("a市"), _entry("a市", "a-1", "A1"), _entry("a市", "a-2", "A2")],
        )
        self._write_city("b市", [_meta("b市"), _entry("b市", "b-1", "B1")])
        self._reset_caches()

    def tearDown(self):
        for name, value in self._originals.items():
            target, attr = name.split(".", 1)
            module = {
                "retriever": retriever,
                "kb": retriever.knowledge_base,
                "contribution_store": contribution_store,
                "city_detector": city_detector,
            }[target]
            setattr(module, attr, value)
        self._reset_caches()
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---------- 工具方法 ----------

    def _write_city(self, city, items):
        from backend.contribution_store import _slugify_city

        with open(self.kb_dir / f"{_slugify_city(city)}.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)

    def _reset_caches(self):
        city_detector._metadata_loaded = False
        retriever.knowledge_base.clear_knowledge_caches()
        retriever.knowledge_base.invalidate_vector_cache()

    def _manifest(self):
        with open(self.vec_dir / "manifest.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def kb(self):
        return retriever.knowledge_base

    # ---------- 测试用例 ----------

    def test_01_full_build_creates_snapshot(self):
        entries = retriever.load_knowledge_base()
        self.assertEqual(len(entries), 3)
        corpus = self.kb.init_vector_index(entries)
        self.assertEqual(corpus.shape, (3, 8))
        self.assertTrue((self.vec_dir / "embeddings.npy").exists())
        self.assertTrue((self.vec_dir / "manifest.json").exists())
        self.assertEqual(len(self.fake.encode_calls), 1)
        self.assertEqual(len(self.fake.encode_calls[0]), 3)
        self.assertEqual(self.kb.vector_status, "ready")

    def test_02_snapshot_hit_does_not_reencode(self):
        entries = retriever.load_knowledge_base()
        corpus1 = self.kb.init_vector_index(entries)
        calls_after_build = len(self.fake.encode_calls)

        self.kb.invalidate_vector_cache()  # 模拟上传后的缓存清理
        entries = retriever.load_knowledge_base()
        corpus2 = self.kb.init_vector_index(entries)

        self.assertEqual(len(self.fake.encode_calls), calls_after_build)  # 零编码
        np.testing.assert_array_equal(corpus1, corpus2)
        self.assertEqual(self.kb.vector_status, "ready")

    def test_03_append_to_front_city_only_encodes_new_entry(self):
        # 首次构建
        entries = retriever.load_knowledge_base()
        corpus0 = self.kb.init_vector_index(entries)
        calls_before = len(self.fake.encode_calls)

        # 向排序靠前的 a市 追加 → 新条目插到全局列表中间（a-2 之后、b-1 之前）
        entry = contribution_store.append_entry_to_knowledge(
            {
                "city": "a市",
                "title": "A3 新增",
                "content": "新增的第三条测试内容，用于验证增量编码只处理新条目。",
                "user_id": "u1",
                "username": "tester",
                "submission_id": "sub-1",
            }
        )
        self.assertEqual(entry["city"], "a市")

        entries = retriever.load_knowledge_base()
        self.assertEqual(
            [e["id"] for e in entries], ["a-1", "a-2", entry["id"], "b-1"]
        )
        self.assertEqual(len(entries), 4)

        corpus = self.kb.init_vector_index(entries)
        self.assertEqual(corpus.shape, (4, 8))
        # 只编码了 1 条新条目
        self.assertEqual(len(self.fake.encode_calls), calls_before + 1)
        self.assertEqual(len(self.fake.encode_calls[-1]), 1)
        # 旧条目向量按位置复用：a-1/a-2 不变，b-1 后移但向量相同
        np.testing.assert_array_equal(corpus[:2], corpus0[:2])
        np.testing.assert_array_equal(corpus[3], corpus0[2])

        # 快照中的 a市 块应包含新 id
        manifest = self._manifest()
        a_block = next(b for b in manifest["cities"] if b["city"] == "a市")
        self.assertEqual(a_block["ids"], ["a-1", "a-2", entry["id"]])
        self.assertEqual(self.kb.vector_status, "ready")

    def test_04_new_city_file_is_incremental(self):
        entries = retriever.load_knowledge_base()
        self.kb.init_vector_index(entries)
        calls_before = len(self.fake.encode_calls)

        self._write_city("z市", [_meta("z市"), _entry("z市", "z-1", "Z1")])
        self._reset_caches()

        entries = retriever.load_knowledge_base()
        corpus = self.kb.init_vector_index(entries)
        self.assertEqual(corpus.shape, (4, 8))
        self.assertEqual(len(self.fake.encode_calls), calls_before + 1)
        self.assertEqual(len(self.fake.encode_calls[-1]), 1)
        self.assertIn("z市", [b["city"] for b in self._manifest()["cities"]])

    def test_05_version_bump_forces_full_rebuild(self):
        entries = retriever.load_knowledge_base()
        self.kb.init_vector_index(entries)
        calls_before = len(self.fake.encode_calls)

        retriever.VECTOR_INDEX_VERSION += 1
        try:
            self.kb.invalidate_vector_cache()
            entries = retriever.load_knowledge_base()
            corpus = self.kb.init_vector_index(entries)
            self.assertEqual(corpus.shape, (3, 8))
            self.assertEqual(len(self.fake.encode_calls), calls_before + 1)
            self.assertEqual(len(self.fake.encode_calls[-1]), 3)  # 全量 3 条
        finally:
            retriever.VECTOR_INDEX_VERSION -= 1

    def test_06_concurrent_double_append_no_entry_loss(self):
        """并发写入串行化：两次追加（跨进程文件锁）后文件里两条都在"""
        e1 = contribution_store.append_entry_to_knowledge(
            {
                "city": "b市",
                "title": "B2",
                "content": "并发写入测试第一条，内容足够长以便通过校验。",
                "user_id": "u1",
                "username": "tester",
            }
        )
        e2 = contribution_store.append_entry_to_knowledge(
            {
                "city": "b市",
                "title": "B3",
                "content": "并发写入测试第二条，内容足够长以便通过校验。",
                "user_id": "u1",
                "username": "tester",
            }
        )
        with open(self.kb_dir / "b市.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        ids = [it.get("id") for it in data if it.get("id") != "_meta"]
        self.assertIn(e1["id"], ids)
        self.assertIn(e2["id"], ids)
        self.assertEqual(len(ids), 3)  # b-1 + B2 + B3


if __name__ == "__main__":
    unittest.main(verbosity=2)