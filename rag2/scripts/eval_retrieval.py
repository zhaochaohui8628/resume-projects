"""检索评估：在给定期望集上跑 BM25 / 向量 / 融合三路对照，输出条款级指标。

用法示例：
  python scripts/eval_retrieval.py --tag base --mode bm25
  python scripts/eval_retrieval.py --tag base --mode vector --index tower_base.faiss
  python scripts/eval_retrieval.py --tag dual --mode convex --alpha 0.9 \
      --index dual_mix.faiss --query-tower data/models/dual_mix/query_encoder
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import BASE_TOWER_MODEL, data_dir  # noqa: E402
from src.eval.metrics import aggregate  # noqa: E402
from src.retrieval.hybrid import Corpus, EncoderPair, HybridRetriever  # noqa: E402
from src.index.bm25 import BM25Index  # noqa: E402
from src.index.dense import FaissStore  # noqa: E402


def load_retriever(mode: str, index_file: str, doc_tower: str | None,
                   query_tower: str | None, alpha: float, cand: int,
                   rrf_k: int = 60, device: str | None = None):
    corpus = Corpus(data_dir("corpus", "clauses.jsonl"), data_dir("index", "units.jsonl"))
    bm = BM25Index.load(data_dir("index", "bm25.json"))
    store = FaissStore.load(data_dir("index", index_file))
    enc = EncoderPair(doc_tower or BASE_TOWER_MODEL, query_tower, device=device)
    return HybridRetriever(corpus, bm, store, enc, alpha=alpha,
                           fuse_mode=("rrf" if mode == "rrf" else "convex"),
                           rrf_k=rrf_k, cand=cand)


def _device_of(r: HybridRetriever) -> str:
    """记录实际编码设备，便于区分「指标差异是设备造成的还是代码造成的」。"""
    try:
        return str(next(r.encoder.query.model.parameters()).device)
    except Exception:  # noqa: BLE001
        return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=str(data_dir("phase1", "val.jsonl")))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["bm25", "vector", "convex", "rrf"], default="convex")
    ap.add_argument("--index", default="tower_base.faiss")
    ap.add_argument("--doc-tower", default=None)
    ap.add_argument("--query-tower", default=None)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--cand", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--device", default=None,
                    help="cpu / cuda；缺省自动（有 GPU 就用 GPU）。"
                         "注意：设备不同会带来 fp32 数值抖动，指标可能有千分位差异")
    ap.add_argument("--dump", default=None, help="逐条结果落盘路径")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.testset, encoding="utf-8") if l.strip()]

    # 防呆：dual_*.faiss 是用**微调后**的 doc 塔建的，两塔必须成对使用，缺任一个都会**静默偏低**
    # （不报错，指标落在"看着还挺合理"的区间，极易被当成真实结论写进报告）：
    #   · 缺 --query-tower → 拿基座 query 塔查微调索引；
    #   · 缺 --doc-tower   → 拿基座 doc 塔编码，与索引所在的微调向量空间不一致
    #     （实测漏给时 hit@5 0.86 → 0.79，比 BM25 单跑还差）。
    # 故按 index 名（dual_mix / dual_gold / ...）成对补齐两塔。
    if a.index.startswith("dual"):
        tag = a.index.split(".")[0]
        for role, attr in (("query", "query_tower"), ("doc", "doc_tower")):
            if getattr(a, attr):
                continue
            guess = data_dir("models", tag, f"{role}_encoder")
            if guess.is_dir():
                setattr(a, attr, str(guess))
                print(f"[防呆] index={a.index} 为微调索引，已自动补 --{role}-tower {guess}",
                      file=sys.stderr)

    r = load_retriever(a.mode, a.index, a.doc_tower, a.query_tower, a.alpha, a.cand,
                       a.rrf_k, a.device)

    details, t0 = [], time.time()
    for row in rows:
        q = row["query"]
        gold = set(row["golds"])
        if a.mode == "bm25":
            hits = r.bm25_recall(q, a.cand)
            pred = []
            seen = set()
            for uid, _s in hits:
                cid = r.corpus.uid2clause[uid]
                if cid not in seen:
                    seen.add(cid)
                    pred.append(cid)
        elif a.mode == "vector":
            hits = r.dense_recall(q, a.cand)
            pred, seen = [], set()
            for uid, _s in hits:
                cid = r.corpus.uid2clause[uid]
                if cid not in seen:
                    seen.add(cid)
                    pred.append(cid)
        else:
            pred = [h["clause_id"] for h in
                    r.search(q, top_k=max(a.top_k, a.cand), alpha=a.alpha)]
        details.append({"query": q, "gold": list(gold), "pred": pred[:max(a.top_k, a.cand)],
                        "hit@k": int(any(p in gold for p in pred[:a.top_k]))})
    el = time.time() - t0
    m = aggregate(details, k=a.top_k)
    out = {"tag": a.tag, "mode": a.mode, "index": a.index, "alpha": a.alpha,
           "cand": a.cand, "top_k": a.top_k, "device": _device_of(r), "n": len(details),
           "hit@k": m[f"hit@{a.top_k}"], "mrr": m["mrr"],
           "ndcg@k": m[f"ndcg@{a.top_k}"], "recall@k": m["recall@k"],
           "sec_per_query": round(el / max(1, len(details)), 4)}
    print(json.dumps(out, ensure_ascii=False))
    res_dir = data_dir("eval")
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / f"{a.tag}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
