"""评估口径（P5 沿用旧约定）：严格 = 类型+span 完全一致；宽容 = 类型一致 + span 重叠。

预测不重复计：以金标实体为基准，每个 gold span 最多命中一次。
句子级聚合，F1 = 2PR/(P+R)。
"""
from __future__ import annotations

from collections import Counter


def _overlap(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def exact(a: dict, b: dict) -> bool:
    return a["type"] == b["type"] and a["start"] == b["start"] and a["end"] == b["end"]


def evaluate(preds: list[dict], golds: list[dict],
             relaxed: bool = False) -> dict:
    """preds/golds 均为实体列表（含 type/start/end）。

    返回 {tp, fp, fn, precision, recall, f1}。gold 每个实体最多命中一次，
    预测不可重复使用（匹配到的 pred 从池中移除）。
    """
    pred_pool = list(preds)
    tp = 0
    matched = set()
    for g in golds:
        hit = None
        for i, p in enumerate(pred_pool):
            if i in matched:
                continue
            ok = (not relaxed and exact(p, g)) or \
                 (relaxed and p["type"] == g["type"] and _overlap(p, g))
            if ok:
                hit = i
                break
        if hit is not None:
            matched.add(hit)
            tp += 1
    fn = len(golds) - tp
    fp = len(preds) - len(matched)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def label_set_difference(rule: list[dict], llm: list[dict]) -> dict:
    """清洗前后标签集差异：{rule_only, llm_only, shared, jaccard}。

    rule_only = 规则有、清洗后无（冲突/剔除）；llm_only = 清洗后新增（漏标修复）。
    """
    rk = {(e["type"], e["start"], e["end"]) for e in rule}
    lk = {(e["type"], e["start"], e["end"]) for e in llm}
    inter = rk & lk
    union = rk | lk
    return {
        "rule_only": len(rk - lk),
        "llm_only": len(lk - rk),
        "shared": len(inter),
        "jaccard": round(len(inter) / len(union), 4) if union else 1.0,
    }
