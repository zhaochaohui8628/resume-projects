"""冲突样本判定（P3 前置）：规则弱标签 vs LLM 复核标签的差异分析。

判定维度：
- keep   → 一致（无冲突）
- fix    → 需修正：边界修正 / 类型修正 / 漏标补充（LLM 补的实体）
- drop   → 冲突：LLM 认为规则标签错误（整句 drop 或剔除特定实体）

对每条 rule entity，找 LLM 侧"匹配实体"（类型相同且 span 重叠）；
找不到 → 该实体被 LLM 剔除 = 冲突。
LLM 侧多出的实体（无 rule 匹配）→ 漏标补充 = 非冲突，是清洗收益。
"""
from __future__ import annotations

from ner2.src.common.types import KEEP, FIX, DROP


def _overlap(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def same_span(a: dict, b: dict) -> bool:
    return a["start"] == b["start"] and a["end"] == b["end"]


def analyze(rule_entities: list[dict], llm_entities: list[dict]) -> dict:
    """返回 {verdict, conflicts, fixes, additions}。

    conflicts: 被 LLM 剔除/改类型的规则实体（附 reason 分类）
    fixes:     规则实体被修正的项（边界或类型变化）
    additions: LLM 补充的实体（漏标修复，清洗收益）
    """
    conflicts, fixes, additions = [], [], []
    rule = list(rule_entities)
    llm = list(llm_entities)
    # 规则侧逐条对账
    for re_ in rule:
        matched = [le for le in llm if le["type"] == re_["type"] and _overlap(re_, le)]
        if not matched:
            # 类型不同但重叠（类型修正）也算冲突
            type_fix = [le for le in llm if le["type"] != re_["type"] and _overlap(re_, le)]
            if type_fix:
                conflicts.append({"type": "type", "rule": re_, "llm": type_fix[0]})
            else:
                conflicts.append({"type": "removed", "rule": re_})
        elif not any(same_span(re_, le) for le in matched):
            fixes.append({"rule": re_, "llm": matched[0]})
    # LLM 侧多出（无重叠匹配）
    for le in llm:
        if not any(_overlap(le, re_) for re_ in rule):
            additions.append(le)
    if conflicts:
        # 类型修正：LLM 已给出修正后标签 → 保留为 fix（非冲突剔除）
        if any(c["type"] == "type" for c in conflicts):
            verdict = FIX
        # 纯剔除：全部规则实体被删且 LLM 无补充 → 整句弱标错误 → drop
        elif all(c["type"] == "removed" for c in conflicts) \
                and len(conflicts) == len(rule) and not additions:
            verdict = DROP
        else:
            verdict = FIX
    elif fixes or additions:
        verdict = FIX
    else:
        verdict = KEEP
    return {"verdict": verdict, "conflicts": conflicts,
            "fixes": fixes, "additions": additions}


def apply_judgment(rule: dict, judgment: dict) -> dict:
    """按判定结果生成一条银标样本（silver 集行）。

    keep  → 规则标签原样
    fix   → LLM 复核标签（已含修正+补充）
    drop  → 剔除整句
    """
    v = judgment.get("verdict", KEEP)
    if v == DROP:
        return None
    if v == FIX:
        ents = judgment.get("entities") or []
    else:
        ents = [{k: e[k] for k in ("type", "start", "end")} for e in rule.get("entities", [])]
    return {"text": rule["text"], "entities": ents,
            "src": "silver", "verdict": v, "qid": rule.get("qid")}
