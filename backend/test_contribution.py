import unittest
import uuid

from fastapi.testclient import TestClient

from main import Source, app
from contribution_store import prepare_knowledge_entry


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


if __name__ == "__main__":
    unittest.main()
