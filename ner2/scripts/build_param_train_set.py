"""参数专项 Step4：组装训练/验证集（专项 + 通用回放）。

- 训练集 = `param_specialist_full.jsonl`（参数专项，含负样本）+ **通用回放**（从 silver_train /
  gold_train_portion 抽样）—— 回放用于**防止其它类型被遗忘**（专项句子以外，模型仍需认识
  设备/工序/工程类型等），这是"低学习率不学崩"之外的第二道保险。
- 验证集 = `param_specialist_full_dev.jsonl`（专项 dev，供早停看参数是否真变好）+ silver_val（通用）。
  早停指标 = 合并验证集上的整体严格 F1；**训练全程不使用 gold_test**。

用法：`python ner2/scripts/build_param_train_set.py`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import GOLD_TRAIN_PORTION, PHASE6, SILVER_TRAIN, SILVER_VAL  # noqa: E402


def _h(t: str) -> str:
    return hashlib.md5(t.strip().encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-replay", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=PHASE6)
    args = ap.parse_args()

    spec = [json.loads(l) for l in open(os.path.join(args.outdir, "param_specialist_full.jsonl"),
                                       encoding="utf-8") if l.strip()]
    spec_dev = [json.loads(l) for l in open(
        os.path.join(args.outdir, "param_specialist_full_dev.jsonl"), encoding="utf-8") if l.strip()]
    spec_hashes = {_h(r["text"]) for r in spec + spec_dev}

    replay_pool = []
    for p in (SILVER_TRAIN, GOLD_TRAIN_PORTION):
        if os.path.exists(p):
            replay_pool += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    replay_pool = [r for r in replay_pool if _h(r["text"]) not in spec_hashes]
    random.Random(args.seed).shuffle(replay_pool)
    replay = replay_pool[:args.n_replay]
    replay = [{"text": r["text"], "entities": r.get("entities", []), "src": "replay"} for r in replay]

    val_extra = [json.loads(l) for l in open(SILVER_VAL, encoding="utf-8") if l.strip()]
    val = spec_dev + [{"text": r["text"], "entities": r.get("entities", []), "src": "silver_val"}
                      for r in val_extra]

    train = spec + replay
    for name, rows in (("param_train.jsonl", train), ("param_val.jsonl", val),
                       ("param_replay.jsonl", replay)):
        p = os.path.join(args.outdir, name)
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[set] {name}: {len(rows)} 句 / "
              f"{sum(len(r['entities']) for r in rows)} 实体")
    print(f"[set] 训练集构成：专项 {len(spec)} + 回放 {len(replay)} = {len(train)}"
          f"（专项占比 {len(spec)/len(train):.0%}）")
    print(f"[set] 验证集构成：专项 dev {len(spec_dev)} + silver_val {len(val_extra)} = {len(val)}")


if __name__ == "__main__":
    main()
