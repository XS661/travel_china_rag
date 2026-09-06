"""FastAPI 应用入口：组装应用、托管静态页面。

启动方式（仓库根目录，任选其一）：
    uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
    uv run python -m backend.main
    python backend/main.py          # 兼容写法：直接以脚本方式运行
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    # 直接 `python backend/main.py` 运行时，本文件被当作顶层脚本（非包模块），
    # 相对导入不可用：这里把仓库根目录加入 sys.path，再以包模块方式启动。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.main import main

    sys.exit(main())

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .routers import ask, auth, community, knowledge


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预加载知识库（retriever 懒加载兜底）"""
    from .retriever import load_knowledge_base

    entries = load_knowledge_base()
    print(f"[启动] 知识库已加载，共 {len(entries)} 条记录")
    cities = sorted({e.get("city", "") for e in entries})
    print(f"[启动] 覆盖城市：{', '.join(cities)}")
    yield


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""
    app = FastAPI(
        title="走遍中国 · 智能旅游助手 API",
        description="面向全国的智能旅游问答与推荐系统后端",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS 配置：允许前端跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 业务路由
    app.include_router(auth.router)
    app.include_router(knowledge.router)
    app.include_router(ask.router)
    app.include_router(community.router)

    # ============================================================
    # 静态文件服务（托管前端页面）
    # ============================================================
    app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")

    @app.get("/")
    async def index():
        """返回前端首页"""
        return FileResponse(config.FRONTEND_DIR / "index.html")

    @app.get("/compare")
    @app.get("/compare/")
    async def compare_page():
        """对比模式页面：与首页同一前端，前端根据路径自动开启对比模式"""
        return FileResponse(config.FRONTEND_DIR / "index.html")

    return app


app = create_app()


def main() -> None:
    """命令行入口：启动 uvicorn 开发服务器"""
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()