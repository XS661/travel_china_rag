"""FastAPI 共享依赖：鉴权等。"""

from fastapi import Header, HTTPException

from .auth_store import verify_token


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
    """解析 Bearer Token 返回当前用户 dict；失败抛 401。"""
    try:
        return verify_token(_get_bearer_token(authorization))
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc