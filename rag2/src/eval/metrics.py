"""检索评估指标（零依赖）：hit@k / recall@k / MRR / nDCG@k。

口径统一在**条款级**：`clause_id = f"{规范编号}::{条款号}"`。gold 允许多值
（一条问句可能对应多个可接受条款），命中判定为"预测列表前 k 中出现任一 gold"。
"""
from __future__ import annotations

import math


def _first_hit_rank(pred: list[str], gold: set[str]) -> int | None:
    for i, p in enumerate(pred, 1):
        if p in gold:
            return i
    return None


def hit_at_k(pred: list[str], gold: set[str], k: int) -> float:
    r = _first_hit_rank(pred[:k], gold)
    return 1.0 if r else 0.0


def recall_at_k(pred: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    got = len({p for p in pred[:k] if p in gold})
    return got / len(gold)


def mrr(pred: list[str], gold: set[str]) -> float:
    r = _first_hit_rank(pred, gold)
    return 1.0 / r if r else 0.0


def ndcg_at_k(pred: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for i, p in enumerate(pred[:k], 1):
        if p in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def aggregate(rows: list[dict], k: int = 5) -> dict:
    """rows: [{"pred": [...], "gold": [...]}, ...]"""
    n = len(rows)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "hit@k": sum(hit_at_k(r["pred"], set(r["gold"]), k) for r in rows) / n,
        f"hit@{k}": sum(hit_at_k(r["pred"], set(r["gold"]), k) for r in rows) / n,
        "recall@k": sum(recall_at_k(r["pred"], set(r["gold"]), k) for r in rows) / n,
        "mrr": sum(mrr(r["pred"], set(r["gold"])) for r in rows) / n,
        f"ndcg@{k}": sum(ndcg_at_k(r["pred"], set(r["gold"]), k) for r in rows) / n,
    }
