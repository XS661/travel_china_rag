"""核心问答接口：城市识别 → 知识库检索 → LLM 生成 → 来源过滤。"""

import re

from fastapi import APIRouter

from .. import config
from ..city_detector import COVERED_CITIES, extract_city_from_text
from ..generator import call_llm, fallback_format
from ..retriever import search_knowledge
from ..schemas import AskRequest, AskResponse, Source

router = APIRouter(tags=["问答"])


def _source_from_result(result: dict) -> Source:
    """把检索结果条目（知识 JSON / 用户贡献条目）规范化为 Source 模型。

    统一的字段契约：来源链接一律读 source_url（用户投稿条目也写 source_url），
    id 优先使用投稿 submission_id，便于前端跳转社区详情。
    """
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
    """将文本拆成关键词集合，便于比较 answer 与来源的相关性。"""
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


@router.post("/api/ask", response_model=AskResponse)
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
            sources=[_source_from_result(r) for r in search_results],
            retrieval_method=method,
            model="none",
        )

    # 6. LLM 调用（含异常降级）
    model_used = config.DEEPSEEK_MODEL
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
    relevant_results = filter_relevant_sources(search_results, answer)
    sources = [_source_from_result(r) for r in relevant_results]

    return AskResponse(
        question=question,
        detected_city=detected_city,
        answer=answer,
        sources=sources,
        retrieval_method=method,
        model=model_used,
    )