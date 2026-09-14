"""P5 黄金集划分：500 条并入训练集，其余作测试集。

输入：data/phase5/gold_consensus.jsonl（多专家投票共识，1043 句）
输出：
    gold_test.jsonl          测试集（≥500 句），与其它集合**零重叠**
    gold_train_portion.jsonl 并入训练集的 500 句
    silver_train.jsonl       追加上述 500 句后的训练集（原文件备份为 _pre_gold.jsonl）

零重叠校验（硬约束）：gold_test 与 {silver_train, silver_val, gold_eval_v2, gold_train_portion}
逐句 hash 互斥，任何重叠即报错退出。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import (LEGACY_GOLD, PHASE2, SILVER_TRAIN, SILVER_VAL, ROOT)

PHASE5 = os.path.join(ROOT, "ner2", "data", "phase5")


def _h(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def _load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def _dump(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus", default=os.path.join(PHASE5, "gold_consensus.jsonl"))
    ap.add_argument("--train-portion", type=int, default=500)
    ap.add_argument("--outdir", default=PHASE5)
    args = ap.parse_args()

    rows = _load(args.consensus)
    # 去重（同句只留一条）
    seen, uniq = set(), []
    for r in rows:
        h = _h(r["text"])
        if h in seen:
            continue
        seen.add(h)
        uniq.append(r)

    # 已有集合（互斥基准）
    occupied = {}
    for p, tag in ((SILVER_TRAIN, "silver_train"), (SILVER_VAL, "silver_val"),
                   (LEGACY_GOLD, "gold_eval_v2")):
        for r in _load(p):
            occupied[_h(r["text"])] = tag
    uniq = [r for r in uniq if _h(r["text"]) not in occupied]
    uniq.sort(key=lambda r: _h(r["text"]))          # 稳定排序，划分可复现

    train_portion = uniq[: args.train_portion]
    test = uniq[args.train_portion:]

    # 硬校验：test 与 train_portion 零重叠
    tr_hashes = {_h(r["text"]) for r in train_portion}
    bad = [r["text"] for r in test if _h(r["text"]) in tr_hashes]
    if bad:
        raise SystemExit(f"[error] 测试集与训练并入句重叠 {len(bad)} 条")

    _dump(os.path.join(args.outdir, "gold_test.jsonl"), test)
    _dump(os.path.join(args.outdir, "gold_train_portion.jsonl"), train_portion)

    # 训练集追加（备份原文件）
    tr_rows = _load(SILVER_TRAIN)
    tr_bak = SILVER_TRAIN.replace(".jsonl", "_pre_gold.jsonl")
    if not os.path.exists(tr_bak):
        shutil.copyfile(SILVER_TRAIN, tr_bak)
    old = {_h(r["text"]) for r in tr_rows}
    added = 0
    with open(SILVER_TRAIN, "a", encoding="utf-8") as f:
        for r in train_portion:
            if _h(r["text"]) in old:
                continue
            f.write(json.dumps({"text": r["text"], "entities": r["entities"],
                                "src": "gold_portion", "cid": r["cid"]},
                               ensure_ascii=False) + "\n")
            added += 1

    stats = {
        "consensus_uniq": len(uniq),
        "train_portion": len(train_portion),
        "gold_test": len(test),
        "appended_to_silver_train": added,
        "silver_train_total": len(tr_rows) + added,
        "test_zero_overlap": True,
    }
    json.dump(stats, open(os.path.join(args.outdir, "split_stats.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
