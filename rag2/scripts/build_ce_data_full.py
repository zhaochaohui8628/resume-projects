"""组装教师(CrossEncoder)训练数据：正例 + 简单负例 + 难负例 三类混合。

## 背景（用户 2026-09-12 明确要求）

教师模型（CrossEncoder）训练数据**不能全是难负例**——否则模型学不到"简单区分"，无法收敛。
应为 **正例 + 负例 + 难负例** 三类：

1. **正例**：双塔全部清洗后训练数据的正样本（query → pos_text，label=1）。
   来源 = phase1 train（422 条）+ 黄金集 train（1000 条，经 dual_mix 挖掘对齐后的正例）。
2. **简单负例**：双塔训练数据自带的 neg_texts（phase1 每条约 4 个，跨规范随机负例），label=0。
   **0 对**（用户 2026-09-12 定稿：教师数据只留正例+难负例，`--simple-cap 0`）。
3. **难负例**：P2 混合召回（双路 RRF）清洗后的 true_neg 对（ce_pairs.jsonl），label=0。

训练时 batch 内**适当配置难负例比例**（`--hard-ratio`，默认 0.4）：每个 batch 里难负例
占总样本的 40%，其余 60% 为正例+简单负例（正:简 ≈ 1:1），保证模型既学简单区分又精修难区分。

## 输出

- `ce_train.jsonl` / `ce_dev.jsonl`：`{query, doc, label}`（label 1=正，0=负）。
  按 query 分组切 train/dev（防泄漏）。
- `ce_pairs_full.jsonl`：`{qid, query, pos_text, neg_text, margin?, kind}`（kind=hard|simple），
  供蒸馏阶段使用（教师对全部数据打分后，学生用分数蒸馏）。

用法：
  python scripts/build_ce_data_full.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import data_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-train", default=str(data_dir("phase1", "train.jsonl")))
    ap.add_argument("--gold-train", default=str(data_dir("phase7", "gold_train.jsonl")))
    ap.add_argument("--hard-pairs", default=str(data_dir("phase2", "ce_pairs.jsonl")))
    ap.add_argument("--out-train", default=str(data_dir("phase2", "ce_train_full.jsonl")))
    ap.add_argument("--out-dev", default=str(data_dir("phase2", "ce_dev_full.jsonl")))
    ap.add_argument("--out-pairs", default=str(data_dir("phase2", "ce_pairs_full.jsonl")))
    ap.add_argument("--dev-ratio", type=float, default=0.12)
    ap.add_argument("--hard-ratio", type=float, default=0.4,
                    help="难负例在 CE 训练样本中的占比（batch 级配比，默认 0.4）")
    ap.add_argument("--simple-cap", type=int, default=0,
                    help="简单负例最大使用量（用户 2026-09-12 定稿：教师数据只留正例+难负例，简单负例=0）")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)

    # ---- 1) 正例：phase1 + 黄金集 的正样本 ----
    # gold_train.jsonl 只有 {query, clause_id, scenario}，pos_text 需从语料查
    from src.common.paths import data_dir as _dd
    corpus = {}
    for line in open(_dd("corpus", "clauses.jsonl"), encoding="utf-8"):
        if line.strip():
            c = json.loads(line)
            corpus[c["id"]] = c["text"]

    pos: list[dict] = []          # {query, doc, label}
    seen_pos: set[tuple[str, str]] = set()
    for path, tag in ((a.phase1_train, "p1"), (a.gold_train, "gold")):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            q = r["query"]
            if "pos_text" in r:
                pt = r["pos_text"]
            elif "clause_id" in r:
                pt = corpus.get(r["clause_id"], "")
            else:
                continue
            if not pt or (q, pt) in seen_pos:
                continue
            seen_pos.add((q, pt))
            pos.append({"query": q, "doc": pt, "label": 1})

    # ---- 2) 简单负例：phase1 自带 neg_texts ----
    simple: list[dict] = []       # {query, doc, label}
    for line in open(a.phase1_train, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        for nt in r.get("neg_texts", []):
            simple.append({"query": r["query"], "doc": nt, "label": 0})

    # ---- 3) 难负例：P2 清洗后的 true_neg 对 ----
    hard: list[dict] = []
    for line in open(a.hard_pairs, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        hard.append({"query": r["query"], "doc": r["neg_text"], "label": 0})

    # ---- 4) 简单负例适量 cap（用户：200 对即可，不用全部）----
    if len(simple) > a.simple_cap:
        rng.shuffle(simple)
        simple = simple[:a.simple_cap]

    # ---- 5) 按 hard-ratio 组装训练样本 ----
    # 正例+简单负例 合计占 (1-hard_ratio)，其中正:简 ≈ 1:1
    n_hard = len(hard)
    n_rest_total = int(n_hard * (1 - a.hard_ratio) / a.hard_ratio) if a.hard_ratio > 0 else len(pos) + len(simple)
    n_pos = min(len(pos), n_rest_total // 2)
    n_simple = min(len(simple), n_rest_total - n_pos)

    # 采样正例/简单负例/难负例
    pos_sel = rng.sample(pos, n_pos)
    simple_sel = simple  # 已在上面 cap；此处保持全部已选（n_simple 可能小于 len(simple)，再裁）
    if len(simple_sel) > n_simple:
        simple_sel = rng.sample(simple_sel, n_simple)
    hard_sel = hard

    all_rows = pos_sel + simple_sel + hard_sel
    rng.shuffle(all_rows)

    # 按 query 分组切 train/dev
    qkeys = {}
    for i, row in enumerate(all_rows):
        # 用 (query) 作分组键（同一 query 的正/负不分家）
        qkeys.setdefault(row["query"], []).append(row)
    qs = list(qkeys.keys())
    rng.shuffle(qs)
    n_dev_q = max(1, int(len(qs) * a.dev_ratio))
    dev_qs = set(qs[:n_dev_q])

    train_rows, dev_rows = [], []
    for q, rows in qkeys.items():
        target = dev_rows if q in dev_qs else train_rows
        target.extend(rows)

    # ---- 6) 蒸馏配对文件：把正例和它的负例(简单+难)配对 ----
    hard_keys = {id(r) for r in hard_sel}          # 用 id 判断是否难负例
    pairs: list[dict] = []
    pos_by_q: dict[str, str] = {}
    for p in pos:
        pos_by_q.setdefault(p["query"], p["doc"])
    for row in all_rows:
        if row["label"] == 1:
            continue
        pt = pos_by_q.get(row["query"])
        if pt:
            pairs.append({
                "query": row["query"],
                "pos_text": pt,
                "neg_text": row["doc"],
                "kind": "hard" if id(row) in hard_keys else "simple",
            })
    # 去重（同一 query 可能多条相同 neg）
    seen_pairs = set()
    pairs_uniq = []
    for p in pairs:
        k = (p["query"], p["neg_text"])
        if k in seen_pairs:
            continue
        seen_pairs.add(k)
        pairs_uniq.append(p)

    for path, rows in ((a.out_train, train_rows), (a.out_dev, dev_rows)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(a.out_pairs).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_pairs, "w", encoding="utf-8") as f:
        for p in pairs_uniq:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"正例 {len(pos)}（phase1+gold 去重）| 简单负例 {len(simple)} | 难负例 {len(hard)}")
    print(f"组装：正 {n_pos} / 简单负 {n_simple} / 难负 {n_hard}（难负占比 "
          f"{n_hard/max(1,n_pos+n_simple+n_hard):.2f}）")
    print(f"ce_train_full {len(train_rows)} 行 / ce_dev_full {len(dev_rows)} 行")
    print(f"蒸馏配对 {len(pairs_uniq)} 对 -> {a.out_pairs}")
    print(f"训练集 label 分布 {dict(Counter(r['label'] for r in train_rows))}")


if __name__ == "__main__":
    main()
