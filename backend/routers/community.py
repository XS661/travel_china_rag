"""用户投稿与社区接口：投稿审核入库、社区帖子、我的投稿。"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..contribution_store import (
    append_entry_to_knowledge,
    delete_user_submission,
    get_public_submission,
    get_user_submission,
    list_community_posts,
    list_submissions,
    review_contribution,
    save_submission,
    update_submission_status,
)
from ..deps import get_current_user
from ..schemas import ContributionResponse

router = APIRouter(tags=["社区"])


@router.get("/api/contributions")
async def contributions(status: str | None = None):
    """获取用户知识贡献记录（数据库）"""
    return list_submissions(status=status)


@router.get("/api/community")
async def community_posts(username: str | None = None):
    """公共社区：按作者或全部展示已审核的帖子"""
    return list_community_posts(username=username)


@router.get("/api/community/{submission_id}")
async def community_post_detail(submission_id: str):
    """公开社区帖子详情"""
    item = get_public_submission(submission_id)
    if item is None:
        raise HTTPException(status_code=404, detail="帖子不存在或尚未审核通过")
    return item


@router.get("/api/my-contributions")
async def my_contributions(current_user: dict = Depends(get_current_user)):
    """获取当前用户自己的帖子列表"""
    return list_submissions(user_id=current_user["id"])


@router.get("/api/my-contributions/{submission_id}")
async def my_contribution_detail(
    submission_id: str,
    current_user: dict = Depends(get_current_user),
):
    """获取当前用户自己的某一篇帖子详情"""
    item = get_user_submission(current_user["id"], submission_id)
    if item is None:
        raise HTTPException(status_code=404, detail="帖子不存在或不属于当前用户")
    return item


@router.delete("/api/my-contributions/{submission_id}")
async def delete_my_contribution(
    submission_id: str,
    current_user: dict = Depends(get_current_user),
):
    """删除当前用户自己的某一篇帖子"""
    deleted = delete_user_submission(current_user["id"], submission_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="帖子不存在或不属于当前用户")
    return {"status": "deleted", "submission_id": submission_id}


@router.post("/api/contribute", response_model=ContributionResponse)
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