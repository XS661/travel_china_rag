"""
旅游出行问答与推荐系统 — FastAPI 后端入口
提供 /api/ask、/api/health、/api/cities、/api/knowledge 接口
"""

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from city_detector import extract_city_from_text, COVERED_CITIES
from contribution_store import (
    append_entry_to_knowledge,
    get_submission,
    list_submissions,
    review_contribution,
    save_submission,
    update_submission_status,
)
from retriever import search_knowledge, get_city_info, get_knowledge_page
from generator import call_llm, fallback_format, DEEPSEEK_MODEL

app = FastAPI(
    title="走遍中国 · 智能旅游助手 API",
    description="面向全国的智能旅游问答与推荐系统后端",
    version="1.0.0",
)

# CORS 配置：允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 请求/响应模型
# ============================================================


class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1, max_length=500)
    city: str | None = Field(default=None, description="用户选择的城市")
    top_k: int = Field(default=5, description="检索返回数量", ge=1, le=10)
    method: str = Field(
        default="bm25", description="检索方法：keyword / bm25 / vector / hybrid"
    )
    raw: bool = Field(
        default=False, description="true 时只返回检索结果（含分数），不调用 LLM"
    )


class Source(BaseModel):
    id: str
    title: str
    source: str
    city: str
    score: float | None = None


class AskResponse(BaseModel):
    question: str
    detected_city: str | None
    answer: str
    sources: list[Source]
    retrieval_method: str
    model: str


class CityInfo(BaseModel):
    city: str
    tags: str
    entry_count: int
    categories: list[str]


class KnowledgePage(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    entries: list[dict]


class ContributionResponse(BaseModel):
    status: str
    submission_id: str | None = None
    review_note: str | None = None
    entry: dict | None = None
    reason: str | None = None


# ============================================================
# 启动事件
# ============================================================


@app.on_event("startup")
async def startup():
    """启动时预加载知识库"""
    from retriever import load_knowledge_base

    entries = load_knowledge_base()
    print(f"[启动] 知识库已加载，共 {len(entries)} 条记录")
    cities = set(e.get("city", "") for e in entries)
    print(f"[启动] 覆盖城市：{', '.join(sorted(cities))}")


# ============================================================
# API 路由
# ============================================================


@app.get("/api/health")
async def health():
    """健康检查"""
    from retriever import load_knowledge_base
    from retriever import get_vector_status

    entries = load_knowledge_base()
    return {
        "status": "ok",
        "knowledge_count": len(entries),
        "covered_cities": COVERED_CITIES,
        "vector_retrieval": get_vector_status(),
    }


@app.get("/api/cities")
async def get_cities() -> list[CityInfo]:
    """获取已覆盖的城市列表与简介"""
    info = get_city_info()
    return [CityInfo(**item) for item in info]


@app.get("/api/knowledge")
async def browse_knowledge(
    city: str | None = None,
    category: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> KnowledgePage:
    """分页浏览知识库条目"""
    result = get_knowledge_page(
        city=city, category=category, page=page, page_size=page_size
    )
    return KnowledgePage(**result)


@app.get("/api/contributions")
async def contributions(status: str | None = None):
    """获取用户知识贡献记录（数据库）"""
    return list_submissions(status=status)


@app.post("/api/contribute", response_model=ContributionResponse)
async def contribute_knowledge(
    city: str = Form(..., min_length=1, max_length=50),
    title: str = Form("", max_length=120),
    content: str = Form("", max_length=5000),
    source: str = Form("用户亲身经历", max_length=120),
    source_type: str = Form("text", max_length=40),
    notes: str = Form("", max_length=2000),
    file: UploadFile | None = File(default=None),
):
    """用户上传亲身经历/文案/附件，审核后纳入知识库"""
    text_from_file = ""
    file_name = None
    if file is not None:
        file_name = file.filename or "upload"
        raw = await file.read()
        if raw:
            candidate = raw.decode("utf-8", errors="ignore")
            if candidate.strip():
                text_from_file = candidate.strip()

    merged_content = (content or "").strip() or text_from_file
    if not merged_content.strip():
        raise HTTPException(status_code=400, detail="请填写文案或上传文本文件")

    submission = save_submission(
        city=city,
        title=title or (f"{city}旅游体验"),
        content=merged_content,
        category="",
        sub_category="",
        source=source,
        source_type=source_type,
        file_name=file_name,
        notes=notes,
    )

    review_result = review_contribution(
        city=city,
        title=title or f"{city}旅游体验",
        content=merged_content,
        category="",
        sub_category="",
        source=source,
        filename=file_name,
    )

    if review_result["status"] == "approved":
        entry = review_result["entry"]
        append_entry_to_knowledge(entry)
        update_submission_status(
            submission["id"],
            status="approved",
            review_note=review_result.get("review_note", "AI 审核通过"),
            approved_entry=entry,
        )
        return ContributionResponse(
            status="approved",
            submission_id=submission["id"],
            review_note=review_result.get("review_note", "AI 审核通过"),
            entry=entry,
            reason=review_result.get("reason", "审核通过"),
        )

    update_submission_status(
        submission["id"],
        status="rejected",
        review_note=review_result.get("reason", "审核未通过"),
        approved_entry=None,
    )
    return ContributionResponse(
        status="rejected",
        submission_id=submission["id"],
        review_note=review_result.get("reason", "审核未通过"),
        entry=None,
        reason=review_result.get("reason", "审核未通过"),
    )


@app.post("/api/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """核心问答接口"""
    question = req.question.strip()
    city = req.city
    top_k = req.top_k
    method = req.method

    # 1. 空问题检查已在 Pydantic 校验中处理

    # 2. 城市识别（前端传了用前端，否则从问题文本提取）
    detected_city = city if city else extract_city_from_text(question)

    # 3. 知识库检索
    search_results = search_knowledge(
        question,
        city=detected_city,
        top_k=top_k,
        method=method,
    )

    # 4. 检索无结果 → 返回提示
    if not search_results and not req.raw:
        return AskResponse(
            question=question,
            detected_city=detected_city,
            answer=(
                f"该地区暂未收录，当前已覆盖城市：{', '.join(COVERED_CITIES)}。\n"
                f"请尝试搜索以上城市的相关问题。"
            ),
            sources=[],
            retrieval_method=method,
            model="none",
        )

    # 5. 原始检索模式（对比模式使用）：只返回检索结果，不调用 LLM
    if req.raw:
        return AskResponse(
            question=question,
            detected_city=detected_city,
            answer="",
            sources=[
                Source(
                    id=r.get("id", ""),
                    title=r.get("title", ""),
                    source=r.get("source", ""),
                    city=r.get("city", ""),
                    score=r.get("score"),
                )
                for r in search_results
            ],
            retrieval_method=method,
            model="none",
        )

    # 6. LLM 调用（含异常降级）
    model_used = DEEPSEEK_MODEL
    try:
        answer = call_llm(question, search_results, timeout=30)
    except ValueError as e:
        # API Key 未配置等配置问题
        answer = f"[配置提示] {str(e)}\n\n" + fallback_format(search_results)
        model_used = "fallback"
    except Exception as e:
        # 网络超时、API 错误等 → 降级
        print(f"[WARNING] LLM 调用失败: {e}")
        answer = fallback_format(search_results)
        model_used = "fallback"

    # 7. 构造来源列表
    sources = [
        Source(
            id=r.get("id", ""),
            title=r.get("title", ""),
            source=r.get("source", ""),
            city=r.get("city", ""),
            score=r.get("score"),
        )
        for r in search_results
    ]

    return AskResponse(
        question=question,
        detected_city=detected_city,
        answer=answer,
        sources=sources,
        retrieval_method=method,
        model=model_used,
    )


# ============================================================
# 静态文件服务（托管前端页面）
# ============================================================

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# 静态资源托管（index.html 引用的 style.css / app.js）
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def index():
    """返回前端首页"""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/compare")
@app.get("/compare/")
async def compare_page():
    """对比模式页面：与首页同一前端，前端根据路径自动开启对比模式"""
    return FileResponse(FRONTEND_DIR / "index.html")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
