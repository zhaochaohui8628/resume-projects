"""真实施工方案用例集（11 本真实专项方案 → 33 个用例）。

来源：data/raw/plans_internal（6 本）+ data/raw/plans_xproj（5 本），
由用户上传的真实方案经「概况 + 编制依据区」抽取为 plan 片段
（agent/bench/data/real_fragments/），标注见 data/real_cases.jsonl。

标注流程（力求正确性）：
  1) 机械标注 A：rules_checker 对片段输出（data/rule_engine_mechanical.json）
  2) 人工标注 B：结合全文核对 + 建筑规范常识（本文件标注数据）
  3) 交叉验证：A vs B 不一致项标记 issue（规则盲区 / 存疑版本号 / 类型口径），
     输出 REVIEW_GUIDE.md 供领域专家复核

加载：load_real_cases() -> list[BenchCase]（plan 从片段文件读入内存）。
"""
from __future__ import annotations

import json
import os

from .cases import BenchCase

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
FRAG_DIR = os.path.join(DATA_DIR, "real_fragments")
_CACHE: list[BenchCase] | None = None


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_real_cases(include_param: bool = True) -> list[BenchCase]:
    """加载真实方案用例；plan 从片段文件读取。

    - real_cases.jsonl        33 条人工精标（原「概况+编制依据」片段，多为封面/目录）
    - real_cases_topics.jsonl 56 条主题切片
    - real_cases_params.jsonl **新增**：以「含施工参数正文句」为中心重切的片段
      （原片段经 is_dirty 过滤后大量保留 0 句 → NER 无"参数"实体 → 三元组空
      → 判档/技术路无输入 → FNR 恒 1.0。本文件解决该问题）
    """
    global _CACHE
    if _CACHE is not None:
        return list(_CACHE)
    files = ["real_cases.jsonl", "real_cases_topics.jsonl"]
    if include_param:
        files.append("real_cases_params.jsonl")
    cases: list[BenchCase] = []
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        for row in _load_jsonl(path):
            plan = ""
            pf = row.get("plan_file")
            if pf:
                try:
                    with open(os.path.join(DATA_DIR, pf), encoding="utf-8") as f:
                        plan = f.read()
                except Exception:
                    plan = ""
            cases.append(BenchCase(
                id=row["id"],
                category=row["category"],
                desc=row["desc"],
                query=row.get("query", ""),
                plan=plan,
                expect=row.get("expect", {"survive": True, "has_output": True}),
                golden_risks=row.get("golden_risks", []),
                golden_entities=row.get("golden_entities", []),
                golden_rag=row.get("golden_rag"),
            ))
    _CACHE = cases
    return list(cases)


def real_case_ids() -> list[str]:
    return [c.id for c in load_real_cases()]


def stats() -> dict:
    cases = load_real_cases()
    from collections import Counter
    cats = Counter(c.category for c in cases)
    return {
        "total": len(cases),
        "by_category": dict(cats),
        "with_risks": sum(1 for c in cases if c.golden_risks),
        "with_entities": sum(1 for c in cases if c.golden_entities),
        "with_rag": sum(1 for c in cases if c.golden_rag),
    }