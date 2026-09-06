"""请求/响应 Pydantic 模型。"""

from pydantic import BaseModel, Field


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