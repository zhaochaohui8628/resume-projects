# -*- coding: utf-8 -*-
"""黄金集对比评估：dual_mix(基座) vs dual_distill_v2(蒸馏学生)。

纯向量 MRR/hit@k —— 判断蒸馏是否带来增益（gold_eval 500 条，与训练数据零重叠）。

⚠ 依赖已清理的产物：`data/models/dual_distill_v2/` 与 `data/index/dual_distill_v2.faiss`
  已随 2026-09-13 项目清理删除。要用本脚本，先重建：
    1) & $PY rag2/scripts/gen_distill_data.py       # 生成蒸馏数据
    2) & $PY rag2/src/train/distill/train.py        # 训练学生塔 -> data/models/dual_distill_v2
    3) & $PY rag2/scripts/build_dual_index.py ...   # 用学生 doc 塔重建索引
  （蒸馏是 P4 的备选对照路线，未纳入最终交付链路）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import data_dir
from src.retrieval.hybrid import Corpus, EncoderPair, HybridRetriever
from src.index.bm25 import BM25Index
from src.index.dense import FaissStore
from src.eval.metrics import aggregate


def eval_one(name: str, doc_dir: str, query_dir: str, index_path: str, cand: int = 20) -> dict:
    corpus = Corpus(data_dir("corpus", "clauses.jsonl"), data_dir("index", "units.jsonl"))
    bm = BM25Index.load(data_dir("index", "bm25.json"))
    store = FaissStore.load(index_path)
    enc = EncoderPair(doc_dir, query_dir)
    r = HybridRetriever(corpus, bm, store, enc, fuse_mode="rrf", rrf_k=10, cand=cand)
    rows = [json.loads(l) for l in open(data_dir("phase7", "gold_eval.jsonl"), encoding="utf-8") if l.strip()]
    det = []
    for row in rows:
        gold = set([row["clause_id"]])
        pred = [h["clause_id"] for h in r.search(row["query"], top_k=10)]
        det.append({"gold": list(gold), "pred": pred,
                    "hit@k": int(any(p in gold for p in pred[:5]))})
    m = aggregate(det, k=5)
    print(f"{name}: hit@5 {m['hit@5']:.4f}  MRR {m['mrr']:.4f}")
    return {"name": name, "hit@5": m["hit@5"], "mrr": m["mrr"]}


if __name__ == "__main__":
    results = []
    # 基座（dual_mix CLS，历史基线）
    results.append(eval_one("dual_mix(基线)", "data/models/dual_mix/doc_encoder",
                            "data/models/dual_mix/query_encoder",
                            data_dir("index", "dual_mix.faiss")))
    # 蒸馏学生 v2（新训）
    results.append(eval_one("dual_distill_v2(蒸馏)", "data/models/dual_distill_v2/doc_encoder",
                            "data/models/dual_distill_v2/query_encoder",
                            data_dir("index", "dual_distill_v2.faiss")))
    print("---")
    for r in results:
        print(f"{r['name']}: MRR {r['mrr']:.4f} hit@5 {r['hit@5']:.4f}")
