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

# ---------------- 混合检索（M1：RRF + 召回池 + 重排） ----------------
# 每条通道的召回池大小：先各自取 top-recall_k，融合后精排取 top_k
HYBRID_RECALL_K = 50
# CrossEncoder 重排模型（可选能力）。下载体积较大（约 1.1GB），
# 设为空字符串可禁用重排，混合检索退化为纯 RRF 排序
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")

# ---------------- 数据质量（M3-A） ----------------
# 投稿长文切片：content 超过 max 字时按段落/句子边界切成多个 chunk
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "800"))
# 切出的尾部小块低于 min 字时并入前一块（避免碎块）
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "200"))
# 投稿去重：与同城市已有条目的词集重叠度 ≥ 阈值时拒绝入库
DEDUP_ENABLED = os.getenv("DEDUP_ENABLED", "true").lower() in ("1", "true", "yes")
DEDUP_OVERLAP_THRESHOLD = float(os.getenv("DEDUP_OVERLAP_THRESHOLD", "0.6"))

# ---------------- 认证 ----------------
# 演示用内置密钥；生产环境请通过环境变量 SECRET_KEY 覆盖
SECRET_KEY = os.getenv("SECRET_KEY", "travel-qa-demo-secret-key-v1")
TOKEN_TTL_DAYS = 7

# ---------------- 上下文记忆 ----------------
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "15"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
SESSION_TTL_DAYS = 30