"""LLM 输出解析与确定性兜底（P2）。

parse_llm_json：从 LLM 原始输出提取 JSON（容忍 markdown 代码块/前后杂散文本）；
sanitize_entities：类型白名单过滤 + span 越界/反向修复 + 排序去重——保证进训练集的
实体永远合法，LLM 输出再脏也不污染数据。
"""
from __future__ import annotations

import json
import re

from ner2.src.common.types import TYPE_SET

_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def parse_llm_json(raw: str) -> dict | None:
    """容错解析：优先 JSON 块，其次全文。失败返回 None。"""
    if not raw:
        return None
    s = raw.strip()
    m = _JSON_BLOCK.search(s)
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # 兜底：截取第一对花括号
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            obj = json.loads(s[i:j + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def sanitize_entities(text: str, entities: list[dict]) -> list[dict]:
    """确定性兜底：类型白名单 + span 合法化 + 排序去重。"""
    n = len(text)
    out = []
    for e in entities or []:
        if not isinstance(e, dict):
            continue
        t = e.get("type")
        if t not in TYPE_SET:
            continue
        try:
            s, en = int(e["start"]), int(e["end"])
        except (KeyError, TypeError, ValueError):
            continue
        s, en = max(0, s), min(n, en)
        if s >= en:
            continue
        out.append({"type": t, "start": s, "end": en})
    # 按 start 排序，重叠保留最长（先到先得按 start 升序、长度降序）
    out.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    kept, last_end = [], -1
    for e in out:
        if e["start"] < last_end:
            continue
        kept.append(e)
        last_end = e["end"]
    return kept


def normalize_judgment(text: str, obj: dict | None, rule_qid: str) -> dict:
    """把 LLM 判定对象规范化为统一 schema（judgments.jsonl 行）。

    schema: {"qid","text","verdict","entities","issues","reason","raw"}
    verdict 非法时降级为 keep（保守：宁留规则标签，由后续冲突过滤兜底）。
    """
    if obj is None:
        return {"qid": rule_qid, "text": text, "verdict": "keep",
                "entities": [], "issues": ["LLM 输出无法解析"], "reason": "parse-fail",
                "raw": None}
    verdict = obj.get("verdict")
    if verdict not in ("keep", "fix", "drop"):
        verdict = "keep"
    ents = sanitize_entities(text, obj.get("entities"))
    issues = [str(x) for x in (obj.get("issues") or [])][:10]
    return {
        "qid": rule_qid,
        "text": text,
        "verdict": verdict,
        "entities": ents,
        "issues": issues,
        "reason": str(obj.get("reason") or ""),
        "raw": obj,
    }
