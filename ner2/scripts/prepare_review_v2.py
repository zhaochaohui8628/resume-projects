"""P2 扩量辅助：把已判定的样本（judgments.jsonl，按 text 复用）映射到新抽样池，
列出"未判定"样本清单（供模型逐条补判）。

用法：
    python ner2/scripts/prepare_review_v2.py \
        --pool data/phase2/review_pool_v2.jsonl \
        --prior data/phase2/judgments.jsonl \
        --out data/phase2/pool_v2_unjudged.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import JUDGMENTS, PHASE2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(PHASE2, "review_pool_v2.jsonl"))
    ap.add_argument("--prior", default=JUDGMENTS)
    ap.add_argument("--out", default=os.path.join(PHASE2, "pool_v2_unjudged.jsonl"))
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
    prior = {}
    for l in open(args.prior, encoding="utf-8"):
        j = json.loads(l)
        prior[j["text"]] = j  # 按 text 复用（判定与 qid 无关）

    reused, unjudged = [], []
    for r in pool:
        if r["text"] in prior:
            reused.append(r["text"])
        else:
            unjudged.append(r)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in unjudged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"pool {len(pool)} | 复用旧判定 {len(reused)} | 待补判 {len(unjudged)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
