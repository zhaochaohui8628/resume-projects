"""标定 BM25 的温莎归一边界（winsor_lo/hi）。

**必须用真实 query 的命中分分布标定，不能用文档 self-score 分布**：
self-score 是全词自匹配，量级远高于真实 query 命中分（旧项目实测 p10=267 vs
真实命中 p50=32），拿它当上界会把所有真实命中截断成 0，融合直接失效。

做法：取标定 query 集（默认用人工撰写的 Phase1 训练 query），逐条跑 BM25 top-K，
汇总命中分数取分位数。lo 取 0（BM25+ 的 idf 恒正、分数非负），hi 取 p99。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import data_dir  # noqa: E402
from src.index.bm25 import BM25Index  # noqa: E402


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return s[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(data_dir("phase1", "train.jsonl")))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--lo-pct", type=float, default=1.0)
    ap.add_argument("--hi-pct", type=float, default=99.0)
    a = ap.parse_args()

    bm = BM25Index.load(data_dir("index", "bm25.json"))
    scores: list[float] = []
    n = 0
    for line in open(a.queries, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        hits = bm.search(row["query"], top_k=a.top_k)
        scores.extend(s for _uid, s in hits)
        n += 1
    lo = pct(scores, a.lo_pct)
    hi = pct(scores, a.hi_pct)
    print(f"query {n} 条，命中分样本 {len(scores)}")
    print(f"  min {min(scores):.2f} p1 {lo:.2f} p10 {pct(scores,10):.2f} "
          f"p50 {pct(scores,50):.2f} p90 {pct(scores,90):.2f} max {max(scores):.2f} "
          f"mean {statistics.mean(scores):.2f}")
    bm.winsor_lo, bm.winsor_hi = lo, hi
    bm.save(data_dir("index", "bm25.json"))
    with open(data_dir("index", "bm25_calib.json"), "w", encoding="utf-8") as f:
        json.dump({"queries": n, "samples": len(scores), "lo": lo, "hi": hi,
                   "p10": pct(scores, 10), "p50": pct(scores, 50), "p99": pct(scores, 99),
                   "source": str(a.queries)}, f, ensure_ascii=False, indent=2)
    print(f"写入 winsor_lo={lo:.2f} winsor_hi={hi:.2f}")


if __name__ == "__main__":
    main()
