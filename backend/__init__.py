"""走遍中国 · 智能旅游助手后端包。

按层组织：
- config.py     环境变量与路径常量（唯一读取 .env 的位置）
- schemas.py    请求/响应 Pydantic 模型
- deps.py       FastAPI 依赖（鉴权等）
- routers/      路由层（ask / auth / knowledge / community）
- retriever.py  检索服务（KnowledgeBase 单例：关键词 / BM25 / 向量 / 混合）
- generator.py  LLM 答案生成与降级
- city_detector.py  城市识别
- auth_store.py / contribution_store.py  数据访问层（SQLite / JSON）
"""