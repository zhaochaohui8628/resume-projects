"""实体类型常量与标注工具（6 类白名单，与旧 ner 一致）。"""
from __future__ import annotations

VALID_TYPES = ["工程类型", "工序", "设备", "参数", "规范编号", "危大类别"]
TYPE_SET = frozenset(VALID_TYPES)

# 词典匹配的类型优先级（同词串多类型冲突时取高优先级；规范编号由正则先行，不在表内）
TYPE_PRIORITY = {"设备": 0, "工程类型": 1, "危大类别": 2, "参数": 3, "工序": 4}

# LLM 判定结果三态
KEEP = "keep"
FIX = "fix"
DROP = "drop"


def span_of(text: str, start: int, end: int) -> str:
    return text[start:end]
