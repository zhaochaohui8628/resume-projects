"""Pooling A/B 验证：同一份双塔权重，CLS vs mean(mask) 两种 pooling 的检索能力对比。

动机：双塔训练/导出已从 CLS 改为 mean pooling，但现有 dual_* 模型权重都是 CLS 训练出来的。
直接换 mean 推理 = 训练/推理不一致，能力可能下降。必须先用数据量化，才能决定
「是重训双塔再挖难负例」还是「保持现状」。

做法：用 transformers 手工加载 doc/query 塔权重，分别以 CLS / mean 两种 pooling 编码
黄金评估集 500 条 query + 全部索引单元，各自建 FAISS，测纯向量 MRR / hit@5 / nDCG@5。
全程不用 SentenceTransformer（它读目录 1_Pooling 配置，无法在脚本内切换 pooling），
保证单一变量是 pooling 方式本身。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.paths import data_dir
from src.eval.metrics import aggregate
from src.index.dense import FaissStore
from src.retrieval.hybrid import Corpus


def encode(model: AutoModel, tok: AutoTokenizer, texts: list[str],
           pool: str, device: str, batch: int = 64, max_len: int = 512) -> np.ndarray:
    vecs: list[np.ndarray] = []
    for i in range(0, len(texts), batch):
        b = texts[i:i + batch]
        enc = tok(b, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        h = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        if pool == "cls":
            v = h[:, 0]
        else:  # mean（mask 加权，pad 位置不参与）
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        v = F.normalize(v, dim=-1)
        vecs.append(v.cpu().numpy())
    return np.vstack(vecs).astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-tower", default="data/models/dual_mix/doc_encoder")
    ap.add_argument("--query-tower", default="data/models/dual_mix/query_encoder")
    ap.add_argument("--eval", default="data/phase7/gold_eval.jsonl")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="data/phase7/pooling_ab.json")
    a = ap.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = [json.loads(l) for l in open(a.eval, encoding="utf-8") if l.strip()]
    queries = [r["query"] for r in rows]
    golds = [set([r["clause_id"]]) for r in rows]

    corpus = Corpus(data_dir("corpus", "clauses.jsonl"), data_dir("index", "units.jsonl"))
    docs = [u["text"] for u in corpus.units]
    print(f"doc 单元 {len(docs)} | eval query {len(queries)} | device {device}",
          flush=True)

    doc_m = AutoModel.from_pretrained(a.doc_tower).to(device).eval()
    qry_m = AutoModel.from_pretrained(a.query_tower).to(device).eval()
    tok = AutoTokenizer.from_pretrained(a.query_tower)

    results: dict[str, dict] = {}
    for pool in ("cls", "mean"):
        print(f"--- pooling = {pool} ---", flush=True)
        dv = encode(doc_m, tok, docs, pool, device)
        qv = encode(qry_m, tok, queries, pool, device)
        store = FaissStore().build(dv)
        det = []
        for i, q in enumerate(qv):
            idx, _ = store.search(q.reshape(1, -1), top_k=10)
            pred = [corpus.uid2clause[corpus.units[j]["uid"]] for j in idx[0]]
            det.append({"gold": list(golds[i]), "pred": pred,
                        "hit@k": int(any(p in golds[i] for p in pred[:5]))})
        m = aggregate(det, k=5)
        results[pool] = {"hit@5": round(m["hit@5"], 4), "mrr": round(m["mrr"], 4),
                         "ndcg@5": round(m["ndcg@5"], 4)}
        print(f"  {pool}: hit@5 {m['hit@5']:.4f} | MRR {m['mrr']:.4f} | "
              f"nDCG@5 {m['ndcg@5']:.4f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps({"model": a.doc_tower, "n_eval": len(rows),
                    "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"已写入 {a.out}", flush=True)


if __name__ == "__main__":
    main()
