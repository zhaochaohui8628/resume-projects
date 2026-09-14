"""精排评估：对 (query, 候选) 打分成对打分，报 Hit@1 / MRR / 平均正负分差。

输入 JSONL 每行：{"query": ..., "pos_text": ..., "neg_texts": [...]}
（与 phase1/phase2 数据的字段一致，可直接复用 val.jsonl 展开后的形态）

用法：
  python scripts/eval_rerank.py --data data/phase2/ce_dev.jsonl --tag base
  python scripts/eval_rerank.py --data data/phase2/ce_dev.jsonl --tag cross_p1 \
      --model data/models/cross_p1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import BASE_CROSS_MODEL, data_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=BASE_CROSS_MODEL)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(dev).eval()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    hit1, rr, gaps, lat = 0, 0.0, [], []
    with torch.no_grad():
        for r in rows:
            q = r["query"]
            docs = [r["pos_text"]] + list(r["neg_texts"])
            t0 = time.time()
            logits = []
            for i in range(0, len(docs), a.batch_size):
                chunk = docs[i:i + a.batch_size]
                enc = tok([q] * len(chunk), chunk, padding=True, truncation=True,
                          max_length=a.max_len, return_tensors="pt").to(dev)
                logits.extend(model(**enc).logits.squeeze(-1).float().cpu().tolist())
            lat.append(time.time() - t0)
            order = sorted(range(len(logits)), key=lambda i: -logits[i])
            rank = order.index(0) + 1
            hit1 += int(rank == 1)
            rr += 1.0 / rank
            gaps.append(logits[0] - max(logits[1:]) if len(logits) > 1 else 0.0)
    n = max(1, len(rows))
    out = {"tag": a.tag, "model": a.model, "n": len(rows),
           "hit@1": hit1 / n, "mrr": rr / n,
           "mean_pos_minus_best_neg": round(sum(gaps) / n, 4),
           "sec_per_query": round(sum(lat) / n, 4), "device": dev}
    print(json.dumps(out, ensure_ascii=False))
    d = data_dir("eval")
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"rerank_{a.tag}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
