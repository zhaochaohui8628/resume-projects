"""端到端链路评估：双路融合召回 top20 → 教师 CE 精排 → 报 hit@1/3/5 / MRR。

链路：query → BM25 + dual_mix 向量（RRF 融合，cand=20）→ 取 top20 候选 →
CE（cross_v2_ep4）重打分 → 按分重排 → 截断 top5(top10 算 MRR，与历史口径一致)。

输出与"无精排的融合召回"对照，量化精排增益。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import data_dir  # noqa: E402
from src.retrieval.hybrid import Corpus, EncoderPair, HybridRetriever  # noqa: E402
from src.index.bm25 import BM25Index  # noqa: E402
from src.index.dense import FaissStore  # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(data_dir("phase7", "gold_eval_clean.jsonl")))
    ap.add_argument("--ce", default=str(data_dir("models", "cross_v2_ep4")))
    ap.add_argument("--doc-tower", default="data/models/dual_mix/doc_encoder")
    ap.add_argument("--query-tower", default="data/models/dual_mix/query_encoder")
    ap.add_argument("--faiss", default=str(data_dir("index", "dual_mix.faiss")))
    ap.add_argument("--recall-topk", type=int, default=20,
                    help="融合召回取 top20 候选交给精排")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="pipeline_rerank_gold.json",
                    help="输出文件名（存到 data/eval/ 下）")
    a = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # ---- 检索链路 ----
    corpus = Corpus(data_dir("corpus", "clauses.jsonl"), data_dir("index", "units.jsonl"))
    bm = BM25Index.load(data_dir("index", "bm25.json"))
    store = FaissStore.load(a.faiss)
    enc = EncoderPair(a.doc_tower, a.query_tower)
    retriever = HybridRetriever(corpus, bm, store, enc, fuse_mode="rrf", rrf_k=10, cand=20)

    rows = [json.loads(l) for l in open(a.gold, encoding="utf-8") if l.strip()]

    # ---- 精排器 ----
    tok = AutoTokenizer.from_pretrained(a.ce)
    model = AutoModelForSequenceClassification.from_pretrained(a.ce).to(a.device).eval()

    def rerank(query: str, cands: list[dict]) -> list[dict]:
        """cands: [{clause_id, text, ...}] -> 按 CE logit 降序返回新列表（带 score）"""
        texts = [c["text"] for c in cands]
        logits = []
        with torch.no_grad():
            for i in range(0, len(texts), a.batch_size):
                chunk = texts[i:i + a.batch_size]
                encr = tok([query] * len(chunk), chunk, padding=True, truncation=True,
                           max_length=a.max_len, return_tensors="pt").to(a.device)
                logits.extend(model(**encr).logits.squeeze(-1).float().cpu().tolist())
        ordered = sorted(zip(cands, logits), key=lambda x: -x[1])
        return [dict(c, score=float(s)) for c, s in ordered]

    # ---- 评估 ----
    det_base, det_rer = [], []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        gold = set([row["clause_id"]])
        hits = retriever.search(row["query"], top_k=a.recall_topk)
        cands = [{"clause_id": h["clause_id"], "text": h.get("text") or corpus.text(h["clause_id"])}
                 for h in hits]
        # 无精排基线（融合召回前10）
        base10 = [c["clause_id"] for c in cands[:10]]
        # 精排
        rer = rerank(row["query"], cands)
        rer10 = [c["clause_id"] for c in rer[:10]]
        rer5 = [c["clause_id"] for c in rer[:5]]
        det_base.append({"gold": list(gold), "pred": base10,
                         "hit@k": int(any(g in gold for g in base10[:5]))})
        det_rer.append({"gold": list(gold), "pred": rer10,
                        "hit@k": int(any(g in gold for g in rer5)),
                        "hit1": int(rer10[0] in gold)})
        if i % 100 == 0:
            print(f"  评估 {i}/{len(rows)}（{time.time()-t0:.0f}s）", flush=True)

    from src.eval.metrics import aggregate

    mb = aggregate(det_base, k=5)
    mr = aggregate(det_rer, k=5)
    hit1 = sum(d["hit1"] for d in det_rer) / max(1, len(det_rer))

    print("=" * 64)
    print(f"评估集: {len(rows)} 条 | 融合召回 top{a.recall_topk} -> CE 精排 -> top5")
    print(f"教师 CE: {a.ce}")
    print(f"--- 融合召回 top10（无精排，基线）---")
    print(f"   hit@5 {mb['hit@5']:.4f} | MRR {mb['mrr']:.4f}")
    print(f"--- + CE 精排 ---")
    print(f"   hit@1 {hit1:.4f} | hit@5 {mr['hit@5']:.4f} | MRR {mr['mrr']:.4f}")
    print("=" * 64)

    out = {
        "gold": a.gold, "ce": a.ce, "n": len(rows),
        "recall_topk": a.recall_topk,
        "fusion_no_rerank": {"hit@5": round(mb["hit@5"], 4), "mrr": round(mb["mrr"], 4)},
        "with_rerank": {"hit@1": round(hit1, 4), "hit@5": round(mr["hit@5"], 4),
                        "mrr": round(mr["mrr"], 4)},
        "sec_per_query": round((time.time() - t0) / max(1, len(rows)), 4),
    }
    d = data_dir("eval")
    d.mkdir(parents=True, exist_ok=True)
    with open(d / a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已存 -> {d / a.out}")


if __name__ == "__main__":
    main()
