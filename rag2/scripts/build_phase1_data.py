"""装配 Phase 1 冷启动数据：我撰写的 query 对 + 条款正样本 → 训练/验证集。

数据来源说明：**query 文本与"哪条条款是它的答案"这两个字段全部由人（模型）逐条撰写**，
脚本只做三件机械的事：
  1. 校验 clause_id 真实存在于语料（防笔误）；
  2. 按 clause_id 分组切分 train/val（同一条款的多个问句不会跨集合，杜绝泄漏）；
  3. 为每条 query 配 k 个**跨规范**的简单负例（Phase 1 只需要"容易的"负例）。
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import data_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-k", type=int, default=4)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    clauses = {}
    for line in open(data_dir("corpus", "clauses.jsonl"), encoding="utf-8"):
        if line.strip():
            c = json.loads(line)
            clauses[c["id"]] = c

    rows = []
    bad = []
    for fp in sorted(glob.glob(str(data_dir("authoring", "queries_*.jsonl")))):
        # 文件名决定问句语域：queries_batch_* = 直问式；queries_natural_* = 口语式
        register = "natural" if "natural" in Path(fp).name else "direct"
        for line in open(fp, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            cid, q = d["clause_id"], d["query"].strip()
            if cid not in clauses:
                bad.append(d)
                continue
            if len(q) < 6:
                bad.append(d)
                continue
            rows.append({"query": q, "clause_id": cid, "register": register})
    print(f"有效 query {len(rows)}；无效 {len(bad)}")
    for b in bad[:10]:
        print("  ✗", b)

    # 同 query 去重
    seen = set()
    uniq = []
    for r in rows:
        k = (r["query"], r["clause_id"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    rows = uniq
    print(f"去重后 {len(rows)}；覆盖条款 {len({r['clause_id'] for r in rows})}")

    # 按条款分组切分
    by_clause: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_clause[r["clause_id"]].append(r)
    cids = sorted(by_clause)
    rng.shuffle(cids)
    n_val = max(1, int(len(cids) * args.val_ratio))
    val_cids = set(cids[:n_val])
    train_cids = [c for c in cids if c not in val_cids]

    by_source: dict[str, list[str]] = defaultdict(list)
    for cid, c in clauses.items():
        by_source[c["metadata"]["source"]].append(cid)

    def pack(cid: str, r: dict) -> dict:
        pos = clauses[cid]
        src = pos["metadata"]["source"]
        cands: list[str] = []
        for s, ids in by_source.items():
            if s != src:
                cands.append(rng.choice(ids))
        negs = rng.sample(cands, min(args.neg_k, len(cands)))
        return {
            "query": r["query"],
            "pos_id": cid,
            "pos_text": pos["text"],
            "neg_ids": negs,
            "neg_texts": [clauses[n]["text"] for n in negs],
            "source": src,
            "clause_no": pos["metadata"]["clause_no"],
            "register": r["register"],
        }

    train, val = [], []
    for cid in train_cids:
        for r in by_clause[cid]:
            train.append(pack(cid, r))
    for cid in sorted(val_cids):
        for r in by_clause[cid]:
            val.append({"query": r["query"], "golds": [cid], "register": r["register"]})

    rng.shuffle(train)
    rng.shuffle(val)
    data_dir("phase1").mkdir(parents=True, exist_ok=True)
    for name, obj in (("train.jsonl", train), ("val.jsonl", val)):
        with open(data_dir("phase1", name), "w", encoding="utf-8") as f:
            for r in obj:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter as _C
    print(f"train {len(train)} 条（{len(train_cids)} 条款）语域 {dict(_C(r['register'] for r in train))}")
    print(f"val   {len(val)} 条（{len(val_cids)} 条款）语域 {dict(_C(r['register'] for r in val))}")
    print("覆盖规范：", len({r['source'] for r in train} | {clauses[r['golds'][0]]['metadata']['source'] for r in val}))


if __name__ == "__main__":
    main()
