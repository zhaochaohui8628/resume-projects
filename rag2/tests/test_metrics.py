"""评估指标单测（零依赖）+ 融合归一化单测。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.eval.metrics import aggregate, hit_at_k, mrr, ndcg_at_k, recall_at_k  # noqa: E402
from src.retrieval.hybrid import winsor_minmax  # noqa: E402


def test_hit_and_mrr():
    gold = {"A::1.1"}
    assert hit_at_k(["X::1", "A::1.1"], gold, 5) == 1.0
    assert hit_at_k(["X::1"], gold, 5) == 0.0
    assert mrr(["X::1", "A::1.1"], gold) == 0.5
    assert mrr(["A::1.1"], gold) == 1.0
    assert mrr(["X::1"], gold) == 0.0


def test_recall_multi_gold():
    gold = {"A::1", "B::2"}
    assert recall_at_k(["A::1"], gold, 5) == 0.5
    assert recall_at_k(["A::1", "B::2"], gold, 5) == 1.0


def test_ndcg_bounds():
    gold = {"A::1"}
    assert ndcg_at_k(["A::1"], gold, 5) == 1.0
    assert 0 < ndcg_at_k(["X", "A::1"], gold, 5) < 1
    assert ndcg_at_k(["X"], gold, 5) == 0.0


def test_aggregate_shapes():
    rows = [{"pred": ["A::1"], "gold": ["A::1"]}, {"pred": ["Z"], "gold": ["A::1"]}]
    m = aggregate(rows, k=5)
    assert m["n"] == 2 and abs(m["hit@5"] - 0.5) < 1e-9 and abs(m["mrr"] - 0.5) < 1e-9


def test_winsor_minmax_clamps():
    v = winsor_minmax([0.0, 5.0, 50.0, 500.0], 0.0, 100.0)
    assert list(np.round(v, 3)) == [0.0, 0.05, 0.5, 1.0]
    # 分母为 0 时退化为全 0，不产生 NaN
    assert list(winsor_minmax([1.0, 2.0], 3.0, 3.0)) == [0.0, 0.0]
