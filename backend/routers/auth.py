"""认证与历史记录接口：注册 / 登录 / 当前用户 / 问答历史。"""

from fastapi import APIRouter, Depends, HTTPException

from ..auth_store import (
    append_user_history,
    authenticate_user,
    clear_user_history,
    create_token,
    list_user_history,
    register_user,
)
from ..deps import get_current_user
from ..schemas import AuthResponse, AuthUser, HistoryEntry, LoginRequest

router = APIRouter(tags=["认证"])


@router.post("/api/register", response_model=AuthResponse)
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


@router.post("/api/login", response_model=AuthResponse)
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


@router.get("/api/me", response_model=AuthUser)
async def me(current_user: dict = Depends(get_current_user)):
    """返回当前登录用户信息"""
    return AuthUser(**current_user)


@router.get("/api/history", response_model=list[HistoryEntry])
async def get_history(current_user: dict = Depends(get_current_user)):
    """获取当前用户的历史记录"""
    return list_user_history(current_user["id"])


@router.post("/api/history", response_model=HistoryEntry)
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


@router.delete("/api/history")
async def clear_history(current_user: dict = Depends(get_current_user)):
    """清空当前用户的历史记录"""
    clear_user_history(current_user["id"])
    return {"status": "ok", "user_id": current_user["id"]}