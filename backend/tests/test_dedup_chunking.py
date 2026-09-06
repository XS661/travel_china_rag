"""M3-A 数据质量测试：投稿长文切片 + 上传去重。

运行方式（仓库根目录）：
    uv run python -m unittest discover -s backend/tests -v
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from backend import city_detector, contribution_store, retriever
from backend.contribution_store import (
    _slugify_city,
    find_similar_entry,
    prepare_knowledge_entries,
    prepare_knowledge_entry,
)


def _para(word: str, n: int) -> str:
    return "".join(f"{word}的旅游心得是值得记录。" for _ in range(n))


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


class ChunkingTests(unittest.TestCase):
    def test_short_content_single_chunk(self):
        chunks = contribution_store._split_content_into_chunks("短文本", 800, 200)
        self.assertEqual(chunks, ["短文本"])

    def test_paragraph_boundary_split(self):
        c1 = _para("甲", 60)  # 每句 11 字，60 句 ≈ 660 字
        c2 = _para("乙", 60)
        chunks = contribution_store._split_content_into_chunks(c1 + "\n" + c2, 800, 200)
        self.assertEqual(len(chunks), 2)
        self.assertIn("甲", chunks[0])
        self.assertIn("乙", chunks[1])

    def test_long_paragraph_sentence_split(self):
        content = _para("丙", 120)  # 1320 字，无换行 → 按句子切
        chunks = contribution_store._split_content_into_chunks(content, 800, 200)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(c) <= 800 for c in chunks))
        self.assertTrue(all("丙" in c for c in chunks))

    def test_trailing_small_chunk_merged(self):
        body = _para("丁", 75)  # 825 字
        tail = "结尾总结。"
        chunks = contribution_store._split_content_into_chunks(
            body + "\n" + tail, 800, 200
        )
        self.assertEqual(len(chunks), 1)  # 尾块过小 → 并入前一块


class PrepareEntriesTests(unittest.TestCase):
    def test_short_content_single_entry(self):
        payload = {"city": "成都", "title": "火锅", "content": "宽窄巷子吃火锅。"}
        entries = prepare_knowledge_entries(payload)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["chunk_id"], 1)

    def test_long_content_splits_into_chunks(self):
        payload = {"city": "北京", "title": "深度攻略", "content": _para("攻略", 160)}
        entries = prepare_knowledge_entries(payload)
        self.assertGreater(len(entries), 1)
        # 同 id、chunk_id 递增、后续块标题带后缀
        self.assertEqual(len({e["id"] for e in entries}), 1)
        self.assertEqual(
            [e["chunk_id"] for e in entries], list(range(1, len(entries) + 1))
        )
        self.assertEqual(entries[0]["title"], "深度攻略")
        self.assertIn("第2部分", entries[1]["title"])
        self.assertTrue(all(e["content"] for e in entries))


class DedupTests(unittest.TestCase):
    def test_find_similar_entry_hits_same_city(self):
        entries = [
            _entry("甲市", "t1", "故宫攻略", "故宫是北京最著名的景点，需要提前预约门票。"),
            _entry("甲市", "t2", "长城攻略", "长城在北京郊区，适合徒步。"),
            _entry("乙市", "t3", "外滩", "外滩是上海的著名景点。"),
        ]
        dup = "故宫是北京最著名的旅游景点，去之前一定要提前预约门票。"
        hit = find_similar_entry(entries, "甲市", "故宫门票预约", dup, threshold=0.6)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["id"], "t1")

    def test_find_similar_entry_no_false_positive(self):
        entries = [_entry("甲市", "t1", "故宫攻略", "故宫是北京最著名的景点，需要提前预约门票。")]
        hit = find_similar_entry(
            entries, "甲市", "长城攻略", "长城在北京郊区，适合徒步半天。", threshold=0.6
        )
        self.assertIsNone(hit)

    def test_find_similar_entry_ignores_other_city(self):
        entries = [_entry("乙市", "t3", "外滩", "外滩是上海的著名景点，夜景很美。")]
        hit = find_similar_entry(
            entries, "甲市", "外滩夜景", "外滩是上海的著名景点，夜景特别美。", threshold=0.6
        )
        self.assertIsNone(hit)


class ReviewDedupIntegrationTests(unittest.TestCase):
    """review_contribution 去重分支（临时知识库隔离）"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.kb_dir = self.tmp / "knowledge"
        self.kb_dir.mkdir()
        self._originals = {
            "retriever.KNOWLEDGE_DIR": retriever.KNOWLEDGE_DIR,
            "city_detector.KNOWLEDGE_DIR": city_detector.KNOWLEDGE_DIR,
            "contribution_store.KNOWLEDGE_DIR": contribution_store.KNOWLEDGE_DIR,
        }
        retriever.KNOWLEDGE_DIR = self.kb_dir
        city_detector.KNOWLEDGE_DIR = self.kb_dir
        contribution_store.KNOWLEDGE_DIR = self.kb_dir

        # 写一条已有知识入库
        entry = _entry(
            "甲市", "k-1", "甲市美食攻略", "甲市的火锅非常有名，推荐去老城区吃正宗麻辣火锅。"
        )
        with open(self.kb_dir / f"{_slugify_city('甲市')}.json", "w", encoding="utf-8") as f:
            json.dump([entry], f, ensure_ascii=False)
        retriever.knowledge_base.clear_knowledge_caches()
        city_detector._metadata_loaded = False

    def tearDown(self):
        for name, value in self._originals.items():
            target, attr = name.split(".", 1)
            mod = {
                "retriever": retriever,
                "city_detector": city_detector,
                "contribution_store": contribution_store,
            }[target]
            setattr(mod, attr, value)
        retriever.knowledge_base.clear_knowledge_caches()
        city_detector._metadata_loaded = False
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_review_rejects_duplicate_content(self):
        result = contribution_store.review_contribution(
            city="甲市",
            title="甲市美食推荐",
            content="甲市的火锅特别有名，强烈推荐去老城区吃正宗的麻辣火锅。",
        )
        self.assertEqual(result["status"], "rejected")
        self.assertIn("相似", result["reason"])

    def test_review_approves_distinct_content(self):
        result = contribution_store.review_contribution(
            city="甲市",
            title="甲市爬山路线",
            content="甲市郊区的山里有一条徒步路线，风景很好，适合周末去放松。",
        )
        self.assertEqual(result["status"], "approved")
        self.assertTrue(result["entry"])


if __name__ == "__main__":
    unittest.main()