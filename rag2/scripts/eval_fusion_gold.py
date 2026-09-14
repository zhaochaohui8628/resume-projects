"""黄金集融合方式对照：RRF vs 凸组合（各扫超参），总体 + 场景分层。

## 为什么单独做这个实验

融合策略（RRF / 凸组合）此前只在**旧语料 val 43 条**上对照过（RRF 0.846 < 凸组合 0.895），
而 val 的问句是"照着条款写"的，与真实口语问句分布不同——黄金集（500 条自然语言问句）
上已经出现过 α 结论反转（val 0.3 → 黄金集 0.9）。因此融合方式必须在黄金集上重测，
旧语料结论不再作为定稿依据。

## 配置矩阵

- 基线：BM25-only（凸组合 α=1.0）、纯向量（α=0.0）
- RRF：`Σ 1/(k + rank)`，k ∈ {10, 20, 60, 100}
- 凸组合：温莎截断 min-max 全局归一后 α·bm25 + (1-α)·dense，α ∈ 0.0~1.0 步长 0.1

指标：hit@5 / MRR / nDCG@5（cand=20，top_k=10 召回后按 k=5 计）。

用法：
  python scripts/eval_fusion_gold.py
  python scripts/eval_fusion_gold.py --cand 20 --tag fusion_gold
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import data_dir  # noqa: E402
from src.eval.metrics import aggregate  # noqa: E402
from src.index.bm25 import BM25Index  # noqa: E402
from src.index.dense import FaissStore  # noqa: E402
from src.retrieval.hybrid import Corpus, EncoderPair, HybridRetriever  # noqa: E402

RRF_KS = (10, 20, 60, 100)
ALPHAS = tuple(round(0.1 * i, 1) for i in range(11))   # 0.0 ~ 1.0
SCENARIOS = ("长尾", "模糊", "对抗")


def build(corpus, bm, store, enc, cand: int, mode: str, alpha: float, rrf_k: int):
    return HybridRetriever(corpus, bm, store, enc, alpha=alpha,
                           fuse_mode=mode, rrf_k=rrf_k, cand=cand)


def evaluate(retriever, rows) -> dict:
    det = []
    for row in rows:
        gold = {row["clause_id"]}
        pred = [h["clause_id"] for h in retriever.search(row["query"], top_k=10)]
        det.append({"gold": list(gold), "pred": pred,
                    "hit@k": int(any(p in gold for p in pred[:5]))})
    m = aggregate(det, k=5)
    return {"hit@5": round(m["hit@5"], 4), "mrr": round(m["mrr"], 4),
            "ndcg@5": round(m["ndcg@5"], 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(data_dir("phase7", "gold_eval.jsonl")))
    ap.add_argument("--index", default="dual_mix.faiss")
    ap.add_argument("--query-tower", default="data/models/dual_mix/query_encoder")
    ap.add_argument("--doc-tower", default=None)
    ap.add_argument("--cand", type=int, default=20)
    ap.add_argument("--tag", default="fusion_gold")
    a = ap.parse_args()

    corpus = Corpus(data_dir("corpus", "clauses.jsonl"), data_dir("index", "units.jsonl"))
    bm = BM25Index.load(data_dir("index", "bm25.json"))
    store = FaissStore.load(data_dir("index", a.index))
    enc = EncoderPair(a.doc_tower or "data/models/dual_mix/doc_encoder", a.query_tower)

    rows = [json.loads(l) for l in open(a.gold, encoding="utf-8") if l.strip()]
    print(f"黄金集 {len(rows)} 条 | scenario {dict(Counter(r['scenario'] for r in rows))} "
          f"| cand={a.cand}")
    print(f"\n{'配置':<26} {'hit@5':>7} {'MRR':>8} {'nDCG@5':>8}")
    print("-" * 52)

    results: list[dict] = []

    def run(label: str, mode: str, alpha: float, rrf_k: int) -> dict:
        r = build(corpus, bm, store, enc, a.cand, mode, alpha, rrf_k)
        m = evaluate(r, rows)
        print(f"{label:<26} {m['hit@5']:>7.3f} {m['mrr']:>8.4f} {m['ndcg@5']:>8.4f}")
        rec = {"label": label, "mode": mode, "alpha": alpha, "rrf_k": rrf_k, **m}
        results.append(rec)
        return rec

    # ---- 基线 ----
    run("BM25-only (α=1.0)", "convex", 1.0, 60)
    run("纯向量 (α=0.0)", "convex", 0.0, 60)

    # ---- RRF ----
    print("-" * 52)
    best_rrf = None
    for k in RRF_KS:
        rec = run(f"RRF k={k}", "rrf", 0.0, k)
        if best_rrf is None or rec["mrr"] > best_rrf["mrr"]:
            best_rrf = rec

    # ---- 凸组合 ----
    print("-" * 52)
    best_convex = None
    for al in ALPHAS:
        rec = run(f"凸组合 α={al}", "convex", al, 60)
        if best_convex is None or rec["mrr"] > best_convex["mrr"]:
            best_convex = rec

    print("-" * 52)
    delta = best_convex["mrr"] - best_rrf["mrr"]
    print(f"最优 RRF   : {best_rrf['label']:<16} MRR={best_rrf['mrr']:.4f}")
    print(f"最优凸组合 : {best_convex['label']:<16} MRR={best_convex['mrr']:.4f}")
    print(f"差值       : {delta:+.4f}（{delta / best_rrf['mrr'] * 100:+.1f}% 相对 RRF）")

    # ---- 最优配置的场景分层 ----
    breakdown: dict[str, dict] = {}
    print(f"\n=== 场景分层（各自最优配置）===")
    for name, rec in (("RRF", best_rrf), ("凸组合", best_convex)):
        r = build(corpus, bm, store, enc, a.cand, rec["mode"], rec["alpha"], rec["rrf_k"])
        per = {}
        for sc in SCENARIOS:
            sub = [x for x in rows if x["scenario"] == sc]
            if sub:
                per[sc] = evaluate(r, sub)
        breakdown[name] = {"config": rec, "per_scenario": per}
        print(f"\n  [{name}] {rec['label']}")
        print(f"    {'场景':<6} {'n':>4} {'hit@5':>7} {'MRR':>8} {'nDCG@5':>8}")
        for sc, m in per.items():
            n = sum(1 for x in rows if x["scenario"] == sc)
            print(f"    {sc:<6} {n:>4} {m['hit@5']:>7.3f} {m['mrr']:>8.4f} {m['ndcg@5']:>8.4f}")

    out = {
        "gold_set": a.gold,
        "n": len(rows),
        "cand": a.cand,
        "index": a.index,
        "all_results": results,
        "best_rrf": best_rrf,
        "best_convex": best_convex,
        "delta_mrr": round(delta, 4),
        "delta_pct_vs_rrf": round(delta / best_rrf["mrr"] * 100, 2),
        "breakdown": breakdown,
    }
    dst = data_dir("phase7", f"{a.tag}.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 → {dst}")


if __name__ == "__main__":
    main()
