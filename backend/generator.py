"""
答案生成模块
- LLM 调用（DeepSeek API，兼容 OpenAI SDK）
- 提示词构造
- 降级方案（API 失败时直接返回检索片段原文）
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 环境变量（使用文件所在目录的绝对路径，避免工作目录不同导致找不到）
load_dotenv(Path(__file__).parent / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

from city_detector import COVERED_CITIES

# ============================================================
# 提示词模板
# ============================================================

SYSTEM_PROMPT = """你是一个专业的全国旅游知识问答助手。你了解中国各大热门旅游城市的景点、美食、交通、文化和旅游攻略。请严格依据以下【参考资料】回答用户问题。

【规则】
1. 只基于【参考资料】中的内容回答，不要凭空编造。
2. 如果资料不足以回答问题，请明确回复：
   "抱歉，根据当前知识库暂无法确定答案，建议您查阅当地旅游官方网站获取最新信息。"
3. 回答时尽量分点说明，结构清晰，便于阅读。
4. 每个要点末尾标注参考来源编号，如 [来源1]。
5. 对于行程规划类问题，给出具体时间安排和理由。
6. 回答中不要涉及医疗、法律、金融等专业建议。"""

COMPARE_PROMPT = """你是一个专业的全国旅游规划助手。用户正在对比多个旅游目的地，请根据以下资料进行对比分析。

【要求】
1. 以表格或分点形式对比各城市的特色、消费、交通、适合人群等。
2. 给出每种选择的适用场景和推荐理由。
3. 如果用户有具体偏好（如预算、季节），请据此重点推荐。
4. 标注参考来源编号。

【参考资料】
{references}

【用户需求】
{question}

请分析："""


def build_prompt(question: str, search_results: list[dict]) -> str:
    """
    构造发给 LLM 的提示词

    Args:
        question: 用户问题
        search_results: 检索到的知识片段列表

    Returns:
        完整的提示词字符串
    """
    if not search_results:
        return _build_no_result_prompt(question)

    # 构造参考资料文本
    reference_parts = []
    for i, result in enumerate(search_results, 1):
        city = result.get("city", "未知")
        title = result.get("title", "无标题")
        content = result.get("content", "")
        source = result.get("source", "未知来源")
        reference_parts.append(
            f"[来源{i}] 《{title}》({city}) - 来源：{source}\n{content}"
        )

    references = "\n\n---\n\n".join(reference_parts)

    prompt = f"""{SYSTEM_PROMPT}

【参考资料】
{references}

【用户问题】
{question}

请回答："""
    return prompt


def _build_no_result_prompt(question: str) -> str:
    """无检索结果时的提示词"""
    return f"""{SYSTEM_PROMPT}

【参考资料】
（暂无相关参考资料）

【用户问题】
{question}

注意：当前知识库中没有找到与用户问题相关的资料。请在回答中如实告知用户，并建议查阅当地旅游官方网站。
已覆盖城市：{', '.join(COVERED_CITIES)}。请回答："""


def build_compare_prompt(question: str, search_results: list[dict]) -> str:
    """构造多城市对比专用提示词"""
    if not search_results:
        return _build_no_result_prompt(question)

    reference_parts = []
    for i, result in enumerate(search_results, 1):
        city = result.get("city", "未知")
        title = result.get("title", "无标题")
        content = result.get("content", "")
        source = result.get("source", "未知来源")
        reference_parts.append(
            f"[来源{i}] 《{title}》({city}) - 来源：{source}\n{content}"
        )

    references = "\n\n---\n\n".join(reference_parts)

    return COMPARE_PROMPT.format(references=references, question=question)


# ============================================================
# LLM 调用
# ============================================================

def call_llm(
    question: str,
    search_results: list[dict],
    timeout: int = 30,
) -> str:
    """
    调用 DeepSeek API 生成回答

    Args:
        question: 用户问题
        search_results: 检索到的知识片段
        timeout: 超时时间（秒）

    Returns:
        LLM 生成的回答文本

    Raises:
        Various exceptions on API failure
    """
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "DEEPSEEK_API_KEY 未配置，请在 backend/.env 文件中设置 API Key。\n"
            "可复制 .env.example 为 .env 并填入你的 Key。"
        )

    from openai import OpenAI

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        timeout=timeout,
    )

    # 检测是否为对比类问题
    is_compare = any(kw in question for kw in ["对比", "比较", "不同", "区别", "哪个更", "还是"])
    if is_compare:
        prompt = build_compare_prompt(question, search_results)
    else:
        prompt = build_prompt(question, search_results)

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=2000,
    )

    answer = response.choices[0].message.content
    return answer.strip() if answer else ""


def fallback_format(search_results: list[dict]) -> str:
    """
    降级方案：API 不可用时直接格式化检索结果返回

    Args:
        search_results: 检索到的知识片段

    Returns:
        格式化后的纯文本回答
    """
    if not search_results:
        return (
            "抱歉，当前知识库中未找到相关信息。\n\n"
            f"已覆盖城市：{', '.join(COVERED_CITIES)}\n"
            "建议您查阅当地旅游官方网站获取最新信息。"
        )

    parts = [
        "⚠️ AI 生成暂不可用，以下为知识库检索结果（降级模式）：\n"
    ]

    for i, result in enumerate(search_results, 1):
        title = result.get("title", "无标题")
        city = result.get("city", "")
        content = result.get("content", "")
        source = result.get("source", "")
        parts.append(f"【{i}】{title}（{city}）")
        parts.append(f"{content}")
        if source:
            parts.append(f"——来源：{source}")
        parts.append("")

    return "\n".join(parts)
