"""
旅游出行问答与推荐系统 — FastAPI 后端入口
提供 /api/ask、/api/health、/api/cities、/api/knowledge 接口
"""

import re
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from auth_store import (
    append_user_history,
    authenticate_user,
    clear_user_history,
    create_token,
    list_user_history,
    register_user,
    verify_token,
)
from city_detector import extract_city_from_text, COVERED_CITIES
from contribution_store import (
    append_entry_to_knowledge,
    get_public_submission,
    get_submission,
    list_community_posts,
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
    source_url: str | None = None
    city: str
    score: float | None = None
    user_id: str | None = None
    username: str | None = None


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


class AuthUser(BaseModel):
    id: str
    username: str
    created_at: str


class LoginRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码", min_length=6, max_length=128)


class AuthResponse(BaseModel):
    token: str
    user: AuthUser


class HistoryEntry(BaseModel):
    id: str
    question: str
    answer: str | None = None
    detected_city: str | None = None
    timestamp: str
    created_at: str


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
# 权限认证
# ============================================================


def _get_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="请先登录")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="登录令牌格式错误，请重新登录")
    return token


async def get_current_user(
    authorization: str | None = Header(default=None),
) -> dict:
    try:
        return verify_token(_get_bearer_token(authorization))
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


# ============================================================
# 答案来源过滤
# ============================================================


def _tokenize_for_relevance(text: str) -> set[str]:
    """将文本拆成关键词集合，便于比较 answer 与来源的相关性。"""
    if not text:
        return set()
    cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", " ", text.lower())
    return {token for token in cleaned.split() if len(token) >= 1}


def _filter_relevant_sources(search_results: list[dict], answer: str) -> list[dict]:
    """保留与当前回答直接相关的来源。优先遵循答案中的 [来源N] 引用。"""
    if not search_results:
        return []

    cited_indices = []
    for match in re.finditer(r"\[来源\s*(\d+)\]", answer or ""):
        try:
            idx = int(match.group(1)) - 1
        except ValueError:
            continue
        if 0 <= idx < len(search_results):
            cited_indices.append(idx)

    if cited_indices:
        deduped = []
        seen = set()
        for idx in cited_indices:
            if idx not in seen:
                deduped.append(search_results[idx])
                seen.add(idx)
        return deduped

    answer_tokens = _tokenize_for_relevance(answer or "")
    if not answer_tokens:
        return search_results[: min(3, len(search_results))]

    ranked = []
    for idx, result in enumerate(search_results):
        candidate_text = " ".join(
            filter(
                None,
                [
                    result.get("title", ""),
                    result.get("content", ""),
                    result.get("city", ""),
                    result.get("source", ""),
                    result.get("category", ""),
                    result.get("sub_category", ""),
                ],
            )
        )
        result_tokens = _tokenize_for_relevance(candidate_text)
        overlap = len(answer_tokens & result_tokens)
        title_tokens = _tokenize_for_relevance(str(result.get("title", "")))
        title_overlap = len(answer_tokens & title_tokens)
        score = overlap * 2 + title_overlap
        ranked.append((score, idx, result))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [item[2] for item in ranked if item[0] > 0]
    if selected:
        return selected[: min(3, len(selected))]
    return search_results[: min(3, len(search_results))]


@app.post("/api/register", response_model=AuthResponse)
async def register(req: LoginRequest):
    """注册新用户并返回令牌"""
    try:
        user = register_user(req.username, req.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    token = create_token(user)
    return AuthResponse(
        token=token,
        user=AuthUser(
            id=user["id"], username=user["username"], created_at=user["created_at"]
        ),
    )


@app.post("/api/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    """用户登录并返回令牌"""
    user = authenticate_user(req.username, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_token(user)
    return AuthResponse(
        token=token,
        user=AuthUser(
            id=user["id"], username=user["username"], created_at=user["created_at"]
        ),
    )


@app.get("/api/me", response_model=AuthUser)
async def me(current_user: dict = Depends(get_current_user)):
    """返回当前登录用户信息"""
    return AuthUser(**current_user)


@app.get("/api/history", response_model=list[HistoryEntry])
async def get_history(current_user: dict = Depends(get_current_user)):
    """获取当前用户的历史记录"""
    return list_user_history(current_user["id"])


@app.post("/api/history", response_model=HistoryEntry)
async def add_history_entry(
    entry: dict,
    current_user: dict = Depends(get_current_user),
):
    """追加当前用户的历史记录"""
    record = append_user_history(
        user_id=current_user["id"],
        question=entry.get("question", "").strip(),
        answer=entry.get("answer"),
        detected_city=entry.get("detected_city"),
        timestamp=entry.get("timestamp"),
    )
    return HistoryEntry(**record)


@app.delete("/api/history")
async def clear_history(current_user: dict = Depends(get_current_user)):
    """清空当前用户的历史记录"""
    clear_user_history(current_user["id"])
    return {"status": "ok", "user_id": current_user["id"]}


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


@app.get("/api/community")
async def community_posts(username: str | None = None):
    """公共社区：按作者或全部展示已审核的帖子"""
    return list_community_posts(username=username)


@app.get("/api/community/{submission_id}")
async def community_post_detail(submission_id: str):
    """公开社区帖子详情"""
    item = get_public_submission(submission_id)
    if item is None:
        raise HTTPException(status_code=404, detail="帖子不存在或尚未审核通过")
    return item


@app.get("/api/my-contributions")
async def my_contributions(current_user: dict = Depends(get_current_user)):
    """获取当前用户自己的帖子列表"""
    return list_submissions(user_id=current_user["id"])


@app.get("/api/my-contributions/{submission_id}")
async def my_contribution_detail(
    submission_id: str,
    current_user: dict = Depends(get_current_user),
):
    """获取当前用户自己的某一篇帖子详情"""
    from contribution_store import get_user_submission

    item = get_user_submission(current_user["id"], submission_id)
    if item is None:
        raise HTTPException(status_code=404, detail="帖子不存在或不属于当前用户")
    return item


@app.delete("/api/my-contributions/{submission_id}")
async def delete_my_contribution(
    submission_id: str,
    current_user: dict = Depends(get_current_user),
):
    """删除当前用户自己的某一篇帖子"""
    from contribution_store import delete_user_submission

    deleted = delete_user_submission(current_user["id"], submission_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="帖子不存在或不属于当前用户")
    return {"status": "deleted", "submission_id": submission_id}


@app.post("/api/contribute", response_model=ContributionResponse)
async def contribute_knowledge(
    city: str = Form(..., min_length=1, max_length=50),
    title: str = Form("", max_length=120),
    content: str = Form("", max_length=5000),
    source: str = Form("用户亲身经历", max_length=120),
    source_type: str = Form("text", max_length=40),
    notes: str = Form("", max_length=2000),
    file: UploadFile | None = File(default=None),
    current_user: dict = Depends(get_current_user),
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
        user_id=current_user["id"],
        username=current_user["username"],
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
        entry["user_id"] = current_user["id"]
        entry["username"] = current_user["username"]
        entry["submission_id"] = submission["id"]
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
                    id=r.get("submission_id") or r.get("id", ""),
                    title=r.get("title", ""),
                    source=r.get("source", ""),
                    source_url=r.get("source_url") or r.get("sourceUrl") or "",
                    city=r.get("city", ""),
                    score=r.get("score"),
                    user_id=r.get("user_id"),
                    username=r.get("username"),
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

    # 7. 只保留与当前回答直接相关的来源
    relevant_results = _filter_relevant_sources(search_results, answer)
    sources = [
        Source(
            id=r.get("submission_id") or r.get("id", ""),
            title=r.get("title", ""),
            source=r.get("source", ""),
            source_url=r.get("source_url") or r.get("sourceUrl") or "",
            city=r.get("city", ""),
            score=r.get("score"),
            user_id=r.get("user_id"),
            username=r.get("username"),
        )
        for r in relevant_results
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
