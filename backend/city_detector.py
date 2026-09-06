"""
城市名识别模块
从用户问题中提取城市名，支持城市别名映射（如"帝都" → "北京"）

所有城市信息（列表、别名、标签）均从知识库的 _meta 元数据自动读取
（M3-B 起由 SQLite city_meta 表提供，首次自动从 JSON _meta 迁移），
新增城市只需正常入库，无需修改本模块。
"""

import json

from . import config

KNOWLEDGE_DIR = config.KNOWLEDGE_DIR

# === 以下变量在 _load_city_metadata() 中自动填充 ===
COVERED_CITIES: list[str] = []  # 知识库覆盖的城市名列表
CITY_ALIASES: dict[str, str] = {}  # 别名 → 城市名映射
CITY_TAGS: dict[str, str] = {}  # 城市名 → 标签
_metadata_loaded: bool = False


def _read_city_meta_from_sqlite() -> dict[str, dict]:
    """从 SQLite city_meta 表读取城市元数据（DB 缺失时返回空）"""
    try:
        from .knowledge_db import ensure_db, fetch_city_meta

        ensure_db(KNOWLEDGE_DIR)
        return fetch_city_meta()
    except Exception as e:
        print(f"[WARNING] 城市元数据从 SQLite 读取失败：{e}")
        return {}


def _read_city_meta_from_json() -> dict[str, dict]:
    """回退路径：从 knowledge/*.json 的 _meta 条目扫描读取"""
    meta_map: dict[str, dict] = {}
    if not KNOWLEDGE_DIR.exists():
        return meta_map
    for json_file in sorted(KNOWLEDGE_DIR.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        if isinstance(data, list):
            meta = next(
                (
                    entry
                    for entry in data
                    if isinstance(entry, dict) and entry.get("id") == "_meta"
                ),
                None,
            )
        elif isinstance(data, dict):
            meta = data.get("_meta")
            if not isinstance(meta, dict) and data.get("id") == "_meta":
                meta = data
        else:
            meta = None
        city_name = meta.get("city", "") if meta else json_file.stem
        meta_map[city_name] = {
            "keywords": meta.get("keywords", []) if meta else [],
            "aliases": meta.get("aliases", []) if meta else [],
            "city_tag": meta.get("city_tag", "") if meta else "",
        }
    return meta_map


def _load_city_metadata():
    """加载城市元数据：SQLite city_meta → 回退 JSON _meta 扫描"""
    global COVERED_CITIES, CITY_ALIASES, CITY_TAGS, _metadata_loaded
    if _metadata_loaded:
        return

    meta_map = _read_city_meta_from_sqlite()
    if not meta_map:
        meta_map = _read_city_meta_from_json()
    if not meta_map:
        return

    cities = list(meta_map.keys())
    aliases: dict[str, str] = {}
    tags: dict[str, str] = {}
    for city, meta in meta_map.items():
        for alias in meta.get("aliases", []):
            aliases[alias] = city
        tags[city] = meta.get("city_tag", "")

    # 使用 clear + extend/update 而非重新赋值，
    # 确保其他模块 import 的引用也能看到更新后的值
    COVERED_CITIES.clear()
    COVERED_CITIES.extend(cities)
    CITY_ALIASES.clear()
    CITY_ALIASES.update(aliases)
    CITY_TAGS.clear()
    CITY_TAGS.update(tags)
    _metadata_loaded = True


def extract_city_from_text(text: str, city_list: list[str] | None = None) -> str | None:
    """
    从文本中识别城市名。

    优先级：
    1. 先匹配别名映射表
    2. 再匹配城市全名（优先匹配更长的城市名，如"西安"在"西"之前）

    Args:
        text: 用户输入的文本
        city_list: 要匹配的城市列表，默认使用 COVERED_CITIES

    Returns:
        识别到的城市名，未识别则返回 None
    """
    _load_city_metadata()

    if city_list is None:
        city_list = COVERED_CITIES

    if not text or not text.strip():
        return None

    text_lower = text.strip()

    # 1. 别名匹配（优先级高）
    for alias, city in CITY_ALIASES.items():
        if alias in text_lower:
            if city in city_list:
                return city

    # 2. 城市全名匹配（按长度降序，优先匹配更长的名）
    sorted_cities = sorted(city_list, key=len, reverse=True)
    for city in sorted_cities:
        if city in text_lower:
            return city

    return None


def get_all_city_names() -> list[str]:
    """返回所有别名和城市名的集合，用于分词词典扩展"""
    _load_city_metadata()
    names = list(COVERED_CITIES)
    names.extend(CITY_ALIASES.keys())
    return names


def is_city_covered(city: str) -> bool:
    """检查城市是否在知识库覆盖范围内"""
    _load_city_metadata()
    return city in COVERED_CITIES
