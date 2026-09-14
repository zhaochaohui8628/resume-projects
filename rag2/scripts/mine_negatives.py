"""Phase 2：难负例候选挖掘（BM25 ∪ 微调双塔 多路召回取并集 topN）。

流程：
  1. 对每条 query 分别取 BM25 topN 与 稠密 topN（按 **clause_id** 去重，不是 unit）；
  2. 取并集，剔除正样本本身与同 clause_id 的滑窗兄弟；
  3. 用与检索一致的凸组合（温莎截断 min-max 全局归一）给并集打分排序，截断到 topN；
  4. 落盘候选，连同「该候选来自哪条通路、各通路名次、双方原文」，供 LLM 语义清洗逐条判定。

topN 取值：默认 30。理由——top10 里多为显著不相关（"易负例"，清洗价值低），
真正需要判别"适用条件/数值口径是否冲突"的难负例主要落在 rank 10~30；
再往上（>30）召回质量下降、清洗成本线性上升。实际值待本脚本跑完看分布后敲定。

用法：
  python scripts/mine_negatives.py \
    --queries data/phase1/train.jsonl \
    --index dual_mix.faiss --query-tower data/models/dual_mix/query_encoder \
    --topn 30 --alpha 0.9 --out data/phase2/candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import BASE_TOWER_MODEL, data_dir  # noqa: E402
from src.index.bm25 import BM25Index  # noqa: E402
from src.index.dense import FaissStore  # noqa: E402
from src.retrieval.hybrid import Corpus, EncoderPair, winsor_minmax  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(data_dir("phase1", "train.jsonl")))
    ap.add_argument("--index", default="dual_mix.faiss")
    ap.add_argument("--doc-tower", default=None)
    ap.add_argument("--query-tower", default=None)
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--fuse-mode", choices=("convex", "rrf"), default="convex",
                    help="并集排序方式：convex=温莎归一凸组合（需 --alpha）；"
                         "rrf=Σ 1/(k+rank)（需 --rrf-k）。召回始终是双路无融合并集，"
                         "融合只用于排序截断。")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--min-rank", type=int, default=1, help="只保留最差名次 >= 该值的候选（跳过太容易的）")
    ap.add_argument("--out", default=str(data_dir("phase2", "candidates.jsonl")))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    corpus = Corpus(data_dir("corpus", "clauses.jsonl"), data_dir("index", "units.jsonl"))
    bm = BM25Index.load(data_dir("index", "bm25.json"))
    store = FaissStore.load(data_dir("index", a.index))
    enc = EncoderPair(a.doc_tower or BASE_TOWER_MODEL, a.query_tower)

    rows = [json.loads(l) for l in open(a.queries, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_cand = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            q, pos = row["query"], row["pos_id"]
            # ---- 两路召回（按 clause_id 去重，保留最好名次）----
            bm_hits = bm.search(q, top_k=a.topn * 3)     # 只调一次，下面复用
            bm_rank: dict[str, int] = {}
            for r, (uid, _s) in enumerate(bm_hits, 1):
                bm_rank.setdefault(corpus.uid2clause[uid], r)
            qv = enc.query.encode([q])
            idx, sc = store.search(qv, top_k=a.topn * 3)
            dn_rank: dict[str, int] = {}
            dn_score: dict[str, float] = {}
            for r, (j, s) in enumerate(zip(idx[0], sc[0]), 1):
                if j < 0:
                    continue
                cid = corpus.units[j]["clause_id"]
                if cid not in dn_rank:
                    dn_rank[cid] = r
                    dn_score[cid] = float(s)

            # ---- 归一 + 凸组合（与检索链路口径一致）----
            nb = winsor_minmax([s for _u, s in bm_hits], bm.winsor_lo, bm.winsor_hi)
            bm_score: dict[str, float] = {}
            for (uid, _s), v in zip(bm_hits, nb):
                cid = corpus.uid2clause[uid]
                if cid not in bm_score or v > bm_score[cid]:
                    bm_score[cid] = float(v)
            nd = winsor_minmax(list(dn_score.values()), -1.0, 1.0) if dn_score else []
            dn_norm = {c: float(v) for c, v in zip(dn_score.keys(), nd)}

            union = set(bm_rank) | set(dn_rank)
            union.discard(pos)
            if a.fuse_mode == "rrf":
                # RRF 只看名次，两路各自缺位时按 10**9 计（贡献 ≈ 0）
                fused = {c: (1.0 / (a.rrf_k + bm_rank.get(c, 10 ** 9))
                             + 1.0 / (a.rrf_k + dn_rank.get(c, 10 ** 9)))
                         for c in union}
            else:
                fused = {c: a.alpha * bm_score.get(c, 0.0)
                         + (1 - a.alpha) * dn_norm.get(c, 0.0) for c in union}
            ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:a.topn]

            cands = []
            for cid, fs in ranked:
                best = min(bm_rank.get(cid, 10 ** 9), dn_rank.get(cid, 10 ** 9))
                if best < a.min_rank:
                    continue
                cands.append({
                    "cand_id": cid,
                    "source": corpus.meta(cid).get("source", ""),
                    "clause_no": corpus.meta(cid).get("clause_no", ""),
                    "bm25_rank": bm_rank.get(cid),
                    "dense_rank": dn_rank.get(cid),
                    "dense_cos": round(dn_score.get(cid, 0.0), 4),
                    "fused": round(fs, 4),
                    "text": corpus.text(cid),
                })
            n_cand += len(cands)
            f.write(json.dumps({
                "query": q,
                "pos_id": pos,
                "pos_source": corpus.meta(pos).get("source", ""),
                "pos_text": corpus.text(pos),
                "candidates": cands,
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(rows)} 候选均值 {n_cand/(i+1):.1f} "
                      f"({time.time()-t0:.0f}s)")
    print(f"完成 {len(rows)} 条 query，候选 {n_cand} 个（均值 {n_cand/max(1,len(rows)):.1f}）-> {a.out}")


if __name__ == "__main__":
    main()
