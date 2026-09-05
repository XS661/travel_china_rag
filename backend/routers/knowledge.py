"""知识库浏览接口：健康检查 / 城市列表 / 知识条目分页。"""

from fastapi import APIRouter

from ..city_detector import COVERED_CITIES, _load_city_metadata
from ..retriever import get_city_info, get_knowledge_page, get_vector_status, load_knowledge_base
from ..schemas import CityInfo, KnowledgePage

router = APIRouter(tags=["知识库"])


@router.get("/api/health")
async def health():
    """健康检查"""
    entries = load_knowledge_base()
    # 确保城市元数据已加载，避免 covered_cities 为空数组
    _load_city_metadata()
    return {
        "status": "ok",
        "knowledge_count": len(entries),
        "covered_cities": COVERED_CITIES,
        "vector_retrieval": get_vector_status(),
    }


@router.get("/api/cities")
async def get_cities() -> list[CityInfo]:
    """获取已覆盖的城市列表与简介"""
    info = get_city_info()
    return [CityInfo(**item) for item in info]


@router.get("/api/knowledge")
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