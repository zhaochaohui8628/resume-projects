"""子块尺寸 A/B：同一语料、同一 query，只变子块长度，看指标怎么动。

## 为什么要做这个实验

需求给定子块窗口 = 700/100，但编码器 `bge-small-zh-v1.5` 的**位置编码上限是 512**。
中文约 1 字 1 token，700 字的子块会被静默截断（超长部分对向量没有任何贡献）。
旧实现用 420/80 正是为了留这个余量，注释里写着"bge 上限 512 token"。
两者冲突，必须用数字说话：改用 700 之后指标掉的部分，到底是"语料变干净但变小了"
还是"子块超长被截断"。

做法：不落盘，全部在内存里做——对每个尺寸重建单元 → 用微调 doc 塔编码 →
建 FAISS → 与**同一个 BM25** 做凸组合融合 → 在 val 集上算 hit@k / MRR。
BM25 不随子块尺寸变化（同一份 units 文本…其实会变），所以两路都按各自尺寸重建，
保证比的是"整套检索链在那个尺寸下的表现"。

⚠️ **阈值与尺寸分离**：`make_children_spans(text, size, overlap)` 的 `size` 只影响
"超过 `MAX_PARENT_LEN` 的过长条款"被切成多大，**不改变"是否切分"的判据**（默认
threshold=`MAX_PARENT_LEN`，见 `src/corpus/clause_split.py`）。所以这里的 A/B
比的是"同一阈值下不同子块尺寸"，不是"阈值跟着 size 走"。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import data_dir  # noqa: E402
from src.corpus.clause_split import make_children_spans  # noqa: E402
from src.eval.metrics import aggregate  # noqa: E402
from src.index.bm25 import BM25Index  # noqa: E402
from src.index.dense import DenseEncoder, FaissStore  # noqa: E402
from src.retrieval.hybrid import Corpus, EncoderPair, HybridRetriever  # noqa: E402


def build_units(clauses: list[dict], size: int, overlap: int) -> list[dict]:
    units = []
    for c in clauses:
        text, cid = c["text"], c["id"]
        m = c["metadata"]
        for k, (s, e) in enumerate(make_children_spans(text, size, overlap)):
            if not text[s:e].strip():
                continue
            units.append({"uid": f"{cid}#{k}", "clause_id": cid, "text": text[s:e],
                          "start": s, "end": e, "parent_len": len(text),
                          "source": m["source"], "clause_no": m["clause_no"]})
    return units


def run(size: int, overlap: int, *, clauses_path: str, doc_tower: str, query_tower: str,
        alpha: float, rows: list[dict], top_k: int, cand: int, device: str) -> dict:
    clauses = [json.loads(l) for l in open(clauses_path, encoding="utf-8") if l.strip()]
    units = build_units(clauses, size, overlap)
    bm = BM25Index().build([u["uid"] for u in units], [u["text"] for u in units])
    enc = DenseEncoder(doc_tower, max_seq_length=512, device=device)
    vecs = enc.encode([u["text"] for u in units])
    store = FaissStore().build(vecs)

    corpus = Corpus(data_dir("corpus", "clauses.jsonl"), data_dir("index", "units.jsonl"))
    corpus.units = units
    corpus.uid2clause = {u["uid"]: u["clause_id"] for u in units}
    corpus.uid2span = {u["uid"]: (u["start"], u["end"]) for u in units}
    pair = EncoderPair(doc_tower, query_tower, device=device)
    r = HybridRetriever(corpus, bm, store, pair, alpha=alpha, fuse_mode="convex", cand=cand)

    details = []
    for row in rows:
        gold = set(row["golds"])
        pred = [h["clause_id"] for h in r.search(row["query"], top_k=max(top_k, cand))]
        details.append({"gold": list(gold), "pred": pred[:max(top_k, cand)],
                        "hit@k": int(any(p in gold for p in pred[:top_k]))})
    m = aggregate(details, k=top_k)
    over = sum(1 for u in units if len(u["text"]) > 512)
    return {"size": size, "overlap": overlap, "units": len(units),
            "mean_len": round(sum(len(u["text"]) for u in units) / len(units), 1),
            "over512_pct": round(over / len(units) * 100, 1),
            "hit@k": round(m[f"hit@{top_k}"], 4), "mrr": round(m["mrr"], 4),
            "ndcg": round(m[f"ndcg@{top_k}"], 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="420,512,700")
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--clauses", default=str(data_dir("corpus", "clauses.jsonl")),
                    help="父块文件；可指向 _baseline_clauses.jsonl 做「同代码跑旧语料」对照")
    ap.add_argument("--tag", default="ab_chunk_size")
    ap.add_argument("--doc-tower", default="data/models/dual_mix/doc_encoder")
    ap.add_argument("--query-tower", default="data/models/dual_mix/query_encoder")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--cand", type=int, default=10)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in
            open(data_dir("phase1", "val.jsonl"), encoding="utf-8") if l.strip()]
    print(f"val {len(rows)} 条 | 融合 α={a.alpha} | 父块 {a.clauses}")
    header = f"{'size':>6} {'overlap':>8} {'units':>7} {'均值长':>7} {'>512占比':>9} {'hit@5':>7} {'MRR':>7} {'nDCG':>7}"
    print(header)
    print("-" * len(header))
    out = []
    for s in (int(x) for x in a.sizes.split(",")):
        r = run(s, a.overlap, clauses_path=a.clauses, doc_tower=a.doc_tower,
                query_tower=a.query_tower, alpha=a.alpha, rows=rows, top_k=a.top_k,
                cand=a.cand, device=a.device)
        out.append(r)
        print(f"{r['size']:>6} {r['overlap']:>8} {r['units']:>7} {r['mean_len']:>7} "
              f"{r['over512_pct']:>8}% {r['hit@k']:>7} {r['mrr']:>7} {r['ndcg']:>7}")
    Path(data_dir("eval")).mkdir(parents=True, exist_ok=True)
    with open(data_dir("eval", f"{a.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
