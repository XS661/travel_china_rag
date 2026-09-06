"""投稿、认证与社区接口测试（基于 FastAPI TestClient）。

运行方式（仓库根目录）：
    uv run python -m unittest discover -s backend/tests -v
"""

import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from backend.contribution_store import prepare_knowledge_entry
from backend.main import app
from backend.routers.ask import filter_relevant_sources
from backend.schemas import Source

# ---------------------------------------------------------------------------
# AuthFlowTests 隔离：把知识库 JSON 目录与 SQLite 数据库重定向到临时目录，
# 避免测试把投稿写入真实的 backend/knowledge/ 与 backend/data/*.db。
# （路由处理器在请求时读取模块级路径常量，import 之后改值即生效）
# ---------------------------------------------------------------------------
_ORIG_PATHS: dict[str, Path] = {}
_TMP_DIR: str | None = None


def setUpModule():
    global _TMP_DIR
    from backend import auth_store, contribution_store

    _TMP_DIR = tempfile.mkdtemp(prefix="travel_qa_test_")
    _ORIG_PATHS["contribution_store.KNOWLEDGE_DIR"] = contribution_store.KNOWLEDGE_DIR
    _ORIG_PATHS["contribution_store.DB_PATH"] = contribution_store.DB_PATH
    _ORIG_PATHS["auth_store.DB_PATH"] = auth_store.DB_PATH

    contribution_store.KNOWLEDGE_DIR = Path(_TMP_DIR) / "knowledge"
    contribution_store.KNOWLEDGE_DIR.mkdir()
    contribution_store.DB_PATH = Path(_TMP_DIR) / "data" / "contributions.db"
    auth_store.DB_PATH = Path(_TMP_DIR) / "data" / "users.db"


def tearDownModule():
    from backend import auth_store, contribution_store

    contribution_store.KNOWLEDGE_DIR = _ORIG_PATHS["contribution_store.KNOWLEDGE_DIR"]
    contribution_store.DB_PATH = _ORIG_PATHS["contribution_store.DB_PATH"]
    auth_store.DB_PATH = _ORIG_PATHS["auth_store.DB_PATH"]
    if _TMP_DIR is not None:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)


class ContributionStoreTests(unittest.TestCase):
    def test_prepare_knowledge_entry_handles_city_and_content(self):
        item = prepare_knowledge_entry(
            {
                "city": "成都",
                "title": "成都火锅打卡路线",
                "content": "我在宽窄巷子附近吃了火锅，晚上适合去春熙路散步。",
                "source": "用户亲身经历",
                "category": "美食",
                "sub_category": "火锅",
            }
        )

        self.assertEqual(item["city"], "成都")
        self.assertEqual(item["title"], "成都火锅打卡路线")
        self.assertIn("火锅", item["content"])
        self.assertTrue(item["keywords"])
        self.assertEqual(item["source"], "用户亲身经历")

    def test_source_model_accepts_source_url_and_identifier(self):
        item = Source(
            id="1",
            title="宝藏录像店Random Play",
            source="用户亲身经历（附件：experience.txt）",
            source_url="https://example.com/source/1",
            city="新艾利都",
        )

        self.assertEqual(item.source_url, "https://example.com/source/1")
        self.assertIn("用户亲身经历", item.source)

    def test_filter_relevant_sources_for_answer(self):
        results = [
            {"title": "蒙德城景点推荐", "score": 0.92},
            {"title": "冒菜美食攻略", "score": 0.85},
            {"title": "巴渝文化介绍", "score": 0.75},
        ]

        filtered = filter_relevant_sources(
            results,
            "蒙德城适合散步和拍照，推荐风神丘陵和蒙德城遗址。[来源1]",
        )

        self.assertEqual([item["title"] for item in filtered], ["蒙德城景点推荐"])


class AuthFlowTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_register_and_login_flow(self):
        username = f"traveler_{uuid.uuid4().hex[:6]}"
        resp = self.client.post(
            "/api/register",
            json={"username": username, "password": "Secret123!"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["user"]["username"], username)
        self.assertIn("token", payload)

        me = self.client.get(
            "/api/me",
            headers={"Authorization": f"Bearer {payload['token']}"},
        )
        self.assertEqual(me.status_code, 200, me.text)
        self.assertEqual(me.json()["username"], username)

    def test_contribute_requires_login(self):
        resp = self.client.post(
            "/api/contribute",
            data={
                "city": "成都",
                "title": "成都美食体验",
                "content": "我在春熙路附近吃了火锅，味道非常好，适合晚上逛街。",
                "source": "用户亲身经历",
                "source_type": "text",
                "notes": "测试",
            },
        )
        self.assertEqual(resp.status_code, 401, resp.text)
        self.assertIn("登录", resp.json()["detail"])

    def test_history_is_scoped_by_user(self):
        username = f"history_{uuid.uuid4().hex[:6]}"
        reg = self.client.post(
            "/api/register",
            json={"username": username, "password": "Secret123!"},
        )
        token = reg.json()["token"]

        history = self.client.get(
            "/api/history",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(history.status_code, 200, history.text)
        self.assertEqual(history.json(), [])

        created = self.client.post(
            "/api/history",
            json={
                "question": "成都怎么玩",
                "answer": "推荐宽窄巷子和春熙路",
                "detected_city": "成都",
                "timestamp": "2026-08-31T00:00:00Z",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(created.status_code, 200, created.text)
        self.assertEqual(created.json()["question"], "成都怎么玩")

        listed = self.client.get(
            "/api/history",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(listed.status_code, 200, listed.text)
        self.assertEqual(len(listed.json()), 1)
        self.assertEqual(listed.json()[0]["question"], "成都怎么玩")

    def test_my_contributions_are_scoped_by_user(self):
        username = f"posts_{uuid.uuid4().hex[:6]}"
        reg = self.client.post(
            "/api/register",
            json={"username": username, "password": "Secret123!"},
        )
        token = reg.json()["token"]

        resp = self.client.post(
            "/api/contribute",
            data={
                "city": "长沙",
                "title": "长沙夜游体验",
                "content": "我在黄兴南路附近散步，晚上可以吃麻辣小龙虾，夜景很适合拍照。",
                "source": "用户亲身经历",
                "source_type": "text",
                "notes": "测试贴文",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)

        mine = self.client.get(
            "/api/my-contributions",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(mine.status_code, 200, mine.text)
        self.assertTrue(len(mine.json()) >= 1)
        self.assertEqual(mine.json()[0]["city"], "长沙")
        self.assertEqual(mine.json()[0]["username"], username)

    def test_my_contribution_detail_and_delete(self):
        username = f"detail_{uuid.uuid4().hex[:6]}"
        reg = self.client.post(
            "/api/register",
            json={"username": username, "password": "Secret123!"},
        )
        token = reg.json()["token"]

        created = self.client.post(
            "/api/contribute",
            data={
                "city": "杭州",
                "title": "杭州西湖慢游",
                "content": "我从断桥走到苏堤，傍晚看西湖很安静，适合慢慢散步。",
                "source": "用户亲身经历",
                "source_type": "text",
                "notes": "细节测试",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(created.status_code, 200, created.text)
        post_id = created.json()["submission_id"]

        detail = self.client.get(
            f"/api/my-contributions/{post_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual(detail.json()["city"], "杭州")
        self.assertIn("西湖", detail.json()["content"])

        deleted = self.client.delete(
            f"/api/my-contributions/{post_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(deleted.status_code, 200, deleted.text)

        check = self.client.get(
            f"/api/my-contributions/{post_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(check.status_code, 404, check.text)

    def test_public_community_post_listing_and_lookup(self):
        username = f"community_{uuid.uuid4().hex[:6]}"
        reg = self.client.post(
            "/api/register",
            json={"username": username, "password": "Secret123!"},
        )
        token = reg.json()["token"]

        created = self.client.post(
            "/api/contribute",
            data={
                "city": "新艾利都",
                "title": "随机店铺打卡",
                "content": "我在六分街附近的录像店里发现了很有氛围的店铺，适合散步和拍照。",
                "source": "用户亲身经历",
                "source_type": "text",
                "notes": "社区测试",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(created.status_code, 200, created.text)
        post_id = created.json()["submission_id"]

        feed = self.client.get("/api/community")
        self.assertEqual(feed.status_code, 200, feed.text)
        self.assertTrue(any(item["id"] == post_id for item in feed.json()))

        detail = self.client.get(f"/api/community/{post_id}")
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual(detail.json()["username"], username)
        self.assertEqual(detail.json()["city"], "新艾利都")


if __name__ == "__main__":
    unittest.main()