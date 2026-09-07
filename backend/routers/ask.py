"""核心问答接口：城市识别 → 上下文记忆 → 知识库检索 → LLM 生成 → 来源过滤。"""

import re

from fastapi import APIRouter

from .. import config
from ..city_detector import COVERED_CITIES, extract_city_from_text
from ..conversation_store import (
    append_message,
    create_session,
    extract_context,
    get_last_city,
    get_or_create_session,
    get_recent_messages,
)
from ..generator import call_llm, fallback_format
from ..retriever import search_knowledge
from ..schemas import AskRequest, AskResponse, Source

router = APIRouter(tags=["问答"])


def _source_from_result(result: dict) -> Source:
    """把检索结果条目（知识 JSON / 用户贡献条目）规范化为 Source 模型。"""
    return Source(
        id=result.get("submission_id") or result.get("id", ""),
        title=result.get("title", ""),
        source=result.get("source", ""),
        source_url=result.get("source_url") or "",
        city=result.get("city", ""),
        score=result.get("score"),
        user_id=result.get("user_id"),
        username=result.get("username"),
    )


def _tokenize_for_relevance(text: str) -> set[str]:
    if not text:
        return set()
    cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", " ", text.lower())
    return {token for token in cleaned.split() if len(token) >= 1}


def filter_relevant_sources(search_results: list[dict], answer: str) -> list[dict]:
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


def _build_history_for_llm(messages: list[dict]) -> list[dict]:
    """把会话消息转为 LLM 可用的 messages 数组（排除 sources 等额外字段）。"""
    out = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            out.append({"role": role, "content": content})
    return out


@router.post("/api/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """核心问答接口（支持上下文记忆）"""
    question = req.question.strip()
    city = req.city
    top_k = req.top_k
    method = req.method

    session_id = None
    history_messages = []
    context_cities: list[str] = []

    # 1. 会话管理：无论前端是否传了 session_id，都确保有一个可用的
    if req.session_id:
        session_id = get_or_create_session(req.session_id, user_id=req.user_id)
    else:
        session_id = create_session(user_id=req.user_id)

    raw_history = get_recent_messages(session_id, max_turns=config.MAX_HISTORY_TURNS)
    if raw_history:
        history_messages = _build_history_for_llm(raw_history)
        ctx = extract_context(raw_history)
        context_cities = ctx["cities"]

    # 2. 城市识别（前端 → 问题文本 → 会话上下文，三级回退）
    detected_city = city if city else extract_city_from_text(question)

    # 如果当前问题没提到城市，但历史有明确城市，补上
    if not detected_city and context_cities:
        detected_city = context_cities[-1]

    # 如果还没有，再查一次会话 last_city
    if not detected_city and session_id:
        last = get_last_city(session_id)
        if last:
            detected_city = last

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
            session_id=session_id,
            context_cities=context_cities,
        )

    # 5. 原始检索模式（对比模式使用）：只返回检索结果，不调用 LLM
    if req.raw:
        return AskResponse(
            question=question,
            detected_city=detected_city,
            answer="",
            sources=[_source_from_result(r) for r in search_results],
            retrieval_method=method,
            model="none",
            session_id=session_id,
            context_cities=context_cities,
        )

    # 6. LLM 调用（含异常降级）
    model_used = config.DEEPSEEK_MODEL
    try:
        answer = call_llm(
            question=question,
            search_results=search_results,
            timeout=30,
            history_messages=history_messages if history_messages else None,
        )
    except ValueError as e:
        answer = f"[配置提示] {str(e)}\n\n" + fallback_format(search_results)
        model_used = "fallback"
    except Exception as e:
        print(f"[WARNING] LLM 调用失败: {e}")
        answer = fallback_format(search_results)
        model_used = "fallback"

    # 7. 只保留与当前回答直接相关的来源
    relevant_results = filter_relevant_sources(search_results, answer)
    sources = [_source_from_result(r) for r in relevant_results]

    # 8. 保存本轮对话到会话
    append_message(session_id, "user", question, detected_city=detected_city)
    append_message(
        session_id,
        "assistant",
        answer,
        detected_city=detected_city,
        sources=[
            {"title": s.title, "source": s.source, "city": s.city}
            for s in relevant_results
        ],
    )

    return AskResponse(
        question=question,
        detected_city=detected_city,
        answer=answer,
        sources=sources,
        retrieval_method=method,
        model=model_used,
        session_id=session_id,
        context_cities=context_cities,
    )