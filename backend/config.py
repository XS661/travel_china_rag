"""全局配置：环境变量、路径常量。

集中读取 backend/.env（仅此处调用 load_dotenv），其余模块统一从这里取配置，
避免"模块导入时各自加载 .env"这类 import 副作用。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 目录约定
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
KNOWLEDGE_DIR = BACKEND_DIR / "knowledge"
DATA_DIR = BACKEND_DIR / "data"
UPLOAD_DIR = BACKEND_DIR / "uploads"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# 加载 backend/.env（显式绝对路径，与工作目录无关）
load_dotenv(BACKEND_DIR / ".env")

# ---------------- DeepSeek LLM ----------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# ---------------- 向量检索 ----------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
VECTOR_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
VECTOR_INDEX_DIR = DATA_DIR / "vector_index"
# 向量快照格式版本：更换向量模型或跳版本可强制全量重建
VECTOR_INDEX_VERSION = 1

# ---------------- 认证 ----------------
# 演示用内置密钥；生产环境请通过环境变量 SECRET_KEY 覆盖
SECRET_KEY = os.getenv("SECRET_KEY", "travel-qa-demo-secret-key-v1")
TOKEN_TTL_DAYS = 7