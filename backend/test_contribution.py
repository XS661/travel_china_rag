import unittest

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


if __name__ == "__main__":
    unittest.main()
