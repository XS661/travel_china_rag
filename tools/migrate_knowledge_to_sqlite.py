#!/usr/bin/env python3
"""知识库 SQLite 迁移/维护工具（M3-B）。

用法（仓库根目录）：
    uv run python -m tools.migrate_knowledge_to_sqlite init     # 确保 DB 可用（空库自动从 JSON 自举迁移）
    uv run python -m tools.migrate_knowledge_to_sqlite rebuild  # 清空后从 JSON 全量重建（维护/校验用）
    uv run python -m tools.migrate_knowledge_to_sqlite count    # 统计 DB 条目数/城市数
    uv run python -m tools.migrate_knowledge_to_sqlite dump <目录>  # 导出为旧格式城市 JSON（归档/审计）
    uv run python -m tools.migrate_knowledge_to_sqlite shadow <db路径>  # 影子验证：迁移到指定临时库后对比 JSON 计数
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import config, knowledge_db  # noqa: E402


def _json_entry_count(knowledge_dir: Path) -> int:
    total = 0
    for json_file in sorted(knowledge_dir.glob("*.json")):
        try:
            entries, _ = knowledge_db.parse_city_file(json_file)
            total += len(entries)
        except Exception:
            continue
    return total


def cmd_init(args) -> int:
    knowledge_db.ensure_db(config.KNOWLEDGE_DIR)
    print(f"知识库 DB: {knowledge_db.KNOWLEDGE_DB_PATH}")
    print(f"条目数:   {knowledge_db.count_entries()}")
    print(f"城市元数据: {len(knowledge_db.fetch_city_meta())}")
    return 0


def cmd_rebuild(args) -> int:
    counts = knowledge_db.rebuild(config.KNOWLEDGE_DIR)
    print("重建完成:", counts)
    return 0


def cmd_count(args) -> int:
    print(f"条目数: {knowledge_db.count_entries()}")
    print(f"城市元数据: {len(knowledge_db.fetch_city_meta())}")
    return 0


def cmd_dump(args) -> int:
    out = knowledge_db.dump_to_json(Path(args.out_dir), config.KNOWLEDGE_DIR)
    print("导出完成:", out)
    return 0


def cmd_shadow(args) -> int:
    """影子验证：把知识库导到临时 DB 路径，核对迁移计数（不触碰生产库）"""
    target = Path(args.db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    orig = knowledge_db.KNOWLEDGE_DB_PATH
    knowledge_db.KNOWLEDGE_DB_PATH = target
    try:
        counts = knowledge_db.rebuild(config.KNOWLEDGE_DIR)
    finally:
        knowledge_db.KNOWLEDGE_DB_PATH = orig
    json_count = _json_entry_count(config.KNOWLEDGE_DIR)
    print(f"JSON 原始条目: {json_count}")
    print(f"DB 迁移条目:   {counts['entries']}")
    print(f"跳过重复:      {counts['duplicates_skipped']}")
    print(f"校验: {json_count} = {counts['entries']} + {counts['duplicates_skipped']} "
          f"→ {'通过' if json_count == counts['entries'] + counts['duplicates_skipped'] else '不通过'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库 SQLite 迁移/维护工具")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="确保 DB 可用（空库自动从 JSON 自举）")
    sub.add_parser("rebuild", help="清空后从 JSON 全量重建")
    sub.add_parser("count", help="统计 DB 条目/城市数")
    p_dump = sub.add_parser("dump", help="导出为旧格式城市 JSON")
    p_dump.add_argument("out_dir", help="导出目录")
    p_shadow = sub.add_parser("shadow", help="影子验证迁移（不触碰生产库）")
    p_shadow.add_argument("db_path", help="临时 DB 文件路径")
    args = parser.parse_args()

    handlers = {
        "init": cmd_init,
        "rebuild": cmd_rebuild,
        "count": cmd_count,
        "dump": cmd_dump,
        "shadow": cmd_shadow,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())