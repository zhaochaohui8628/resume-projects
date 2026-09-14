"""把黄金集 query 组装成双塔训练数据（InfoNCE 用）：正例 + hard negatives。

## 为什么需要 hard negatives

只靠批内负例（in-batch）时，一个 batch 里其他 query 的正例对该 query 而言往往是
"完全不相关的条款"——模型轻松就能分开，学不到区分相似条款的能力。黄金集上纯向量
MRR 只有 0.698（BM25 有 0.854），说明向量塔缺的正是这个能力。

所以这里做 **hard negative mining**：用当前已微调的检索塔（现行 `dual_mix`，检索链路与线上一致，
凸组合 α=0.9）召回每条 query 的 top-k，剔除 gold 自身后取前 N 个作显式难负例。

## 假阴性风险与处理

被召回的非 gold 条款里，理论上可能混着"其实也能回答该问句"的条款（假阴性），
把它们当负例会教模型打压合理答案。缓解措施：
  1. 必须排除 gold 自身的 clause_id（正例与负例不重叠）；
  2. `--exclude-same-source` 可开启"排除与 gold 同规范的候选"——负例变简单但零假阴性；
  3. P2 试标实测假阴性率极低（50 对里 0 例），默认关闭该选项以保留难负例难度。

## 输出格式

`{query, pos_id, pos_text, neg_ids, neg_texts, scenario}` —— 与 phase1 的 train.jsonl
字段对齐，可直接喂给 `src/train/dual_tower/train.py`（它读 query/pos_text/neg_texts）。

用法：
  python scripts/build_gold_train_data.py \
    --gold data/phase7/gold_train.jsonl \
    --out  data/phase7/dual_train_gold.jsonl \
    --query-tower data/models/dual_mix/query_encoder \
    --topk 12 --n-neg 4
  # 混合旧数据（防灾难性遗忘，做对照）
  python scripts/build_gold_train_data.py --mix data/phase1/train.jsonl \
    --out data/phase7/dual_train_mix.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import data_dir  # noqa: E402
from src.index.bm25 import BM25Index  # noqa: E402
from src.index.dense import FaissStore  # noqa: E402
from src.retrieval.hybrid import Corpus, EncoderPair, HybridRetriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(data_dir("phase7", "gold_train.jsonl")))
    ap.add_argument("--out", default=str(data_dir("phase7", "dual_train_gold.jsonl")))
    ap.add_argument("--mix", default=None, help="并入的旧数据（phase1/train.jsonl）")
    ap.add_argument("--index", default="dual_mix.faiss")
    ap.add_argument("--doc-tower", default="data/models/dual_mix/doc_encoder")
    ap.add_argument("--query-tower", default="data/models/dual_mix/query_encoder")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--n-neg", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--cand", type=int, default=20)
    ap.add_argument("--exclude-same-source", action="store_true",
                    help="排除与 gold 同规范的候选（零假阴性，但负例变简单）")
    ap.add_argument("--max-text", type=int, default=1200,
                    help="正/负例文本截断长度（训练 max_len=224 token，存全文既浪费又慢；"
                         "取 MAX_PARENT_LEN=1200 与回扩口径一致）")
    a = ap.parse_args()

    def txt(cid: str) -> str:
        return corpus.text(cid)[:a.max_text]

    corpus = Corpus(data_dir("corpus", "clauses.jsonl"), data_dir("index", "units.jsonl"))
    bm = BM25Index.load(data_dir("index", "bm25.json"))
    store = FaissStore.load(data_dir("index", a.index))
    enc = EncoderPair(a.doc_tower, a.query_tower)
    r = HybridRetriever(corpus, bm, store, enc, alpha=a.alpha,
                        fuse_mode="convex", cand=a.cand)

    gold_rows = [json.loads(l) for l in open(a.gold, encoding="utf-8") if l.strip()]
    print(f"黄金集训练 query {len(gold_rows)} 条 | topk={a.topk} n_neg={a.n_neg} "
          f"| 排除同规范={a.exclude_same_source}")

    out: list[dict] = []
    n_short = 0          # 负例不足的条数
    stat_neg: Counter[int] = Counter()
    for row in gold_rows:
        q, pos_id = row["query"], row["clause_id"]
        pos_src = corpus.meta(pos_id).get("source", "")
        pos_text = txt(pos_id)
        hits = r.search(q, top_k=a.topk)
        neg_ids: list[str] = []
        for h in hits:
            cid = h["clause_id"]
            if cid == pos_id:
                continue
            if a.exclude_same_source and corpus.meta(cid).get("source", "") == pos_src:
                continue
            neg_ids.append(cid)
            if len(neg_ids) >= a.n_neg:
                break
        if len(neg_ids) < a.n_neg:
            n_short += 1
        stat_neg[len(neg_ids)] += 1
        out.append({
            "query": q,
            "pos_id": pos_id,
            "pos_text": pos_text,
            "neg_ids": neg_ids,
            "neg_texts": [txt(c) for c in neg_ids],
            "scenario": row.get("scenario", ""),
        })

    n_mix = 0
    if a.mix:
        mix_rows = [json.loads(l) for l in open(a.mix, encoding="utf-8") if l.strip()]
        for m in mix_rows:
            if "pos_text" not in m or "neg_texts" not in m:
                raise SystemExit(f"{a.mix} 缺少 pos_text/neg_texts 字段，无法并入")
            m.setdefault("scenario", "phase1")
            out.append(m)
        n_mix = len(mix_rows)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(f"输出 {len(out)} 条 -> {a.out}"
          + (f"（含并入旧数据 {n_mix} 条）" if n_mix else ""))
    print(f"负例数分布 {dict(sorted(stat_neg.items()))}；不足 {a.n_neg} 个的 {n_short} 条")
    print(f"scenario 分布 {dict(Counter(o['scenario'] for o in out))}")
    L = sorted(len(o["pos_text"]) for o in out)
    n = len(L)
    print(f"正例文本长度 p50={L[n//2]} p90={L[int(n*0.9)]} max={L[-1]}")


if __name__ == "__main__":
    main()
