"""把 P2 清洗判定组装成 CrossEncoder 训练/验证数据。

## 输入

1. `data/phase2/judgments.jsonl` —— 清洗判定：
   `{qid, cand_id, verdict, difficulty, batch, reason}`
   - `true_neg`  → 真难负例，进 CE 负例池
   - `false_neg` → **假阴性，必须剔除**（候选其实也能回答，当负例会教模型打压正确答案）
2. `data/phase2/judge_A_samesource.jsonl`（+ B 批）——判定清单，提供 query / pos_text / cand_text。

## 输出

- `ce_train.jsonl` / `ce_dev.jsonl`：`{query, doc, label}`（label 1=正例，0=负例）。
  按 **query 分组**切分（同一 query 的正负例不跨 train/dev，避免泄漏）。
- `ce_pairs.jsonl`：`{qid, query, pos_id, pos_text, neg_id, neg_text, difficulty}`
  蒸馏阶段用它算教师分数（pos/neg 成对，供 margin ranking loss 直接用）。

用法：
  python scripts/build_ce_data.py
  python scripts/build_ce_data.py --dev-ratio 0.15
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


def load_pairs(judge_files: list[str]) -> dict[tuple[str, str], dict]:
    """(qid, cand_id) -> 清单记录。"""
    out: dict[tuple[str, str], dict] = {}
    for f in judge_files:
        p = Path(f)
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[(r["qid"], r["cand_id"])] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", default=str(data_dir("phase2", "judgments.jsonl")))
    ap.add_argument("--judge-files", nargs="*", default=[
        str(data_dir("phase2", "judge_A_samesource.jsonl")),
        str(data_dir("phase2", "judge_B_band.jsonl")),
    ])
    ap.add_argument("--out-train", default=str(data_dir("phase2", "ce_train.jsonl")))
    ap.add_argument("--out-dev", default=str(data_dir("phase2", "ce_dev.jsonl")))
    ap.add_argument("--out-pairs", default=str(data_dir("phase2", "ce_pairs.jsonl")))
    ap.add_argument("--dev-ratio", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    jud = [json.loads(l) for l in open(a.judgments, encoding="utf-8") if l.strip()]
    pool = load_pairs(a.judge_files)
    rng = random.Random(a.seed)

    pairs: list[dict] = []
    n_missing = 0
    for j in jud:
        if j["verdict"] != "true_neg":
            continue                       # false_neg 一律剔除
        rec = pool.get((j["qid"], j["cand_id"]))
        if rec is None:
            n_missing += 1
            continue
        pairs.append({
            "qid": j["qid"],
            "query": rec["query"],
            "pos_id": rec["pos_id"],
            "pos_text": rec["pos_text"],
            "neg_id": rec["cand_id"],
            "neg_text": rec["cand_text"],
            "difficulty": j.get("difficulty", ""),
        })

    # 按 query 分组切 train / dev（防泄漏）
    qids = sorted({p["qid"] for p in pairs})
    rng.shuffle(qids)
    n_dev = max(1, int(len(qids) * a.dev_ratio))
    dev_q = set(qids[:n_dev])

    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    for p in pairs:
        rows = train_rows if p["qid"] not in dev_q else dev_rows
        rows.append({"query": p["query"], "doc": p["pos_text"], "label": 1})
        rows.append({"query": p["query"], "doc": p["neg_text"], "label": 0})

    for path, rows in ((a.out_train, train_rows), (a.out_dev, dev_rows),
                       (a.out_pairs, pairs)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"判定 {len(jud)} 条（true_neg {sum(1 for j in jud if j['verdict']=='true_neg')} / "
          f"false_neg {sum(1 for j in jud if j['verdict']=='false_neg')}）")
    if n_missing:
        print(f"  ⚠ {n_missing} 条判定在清单里找不到对应记录（清单已重建？）")
    print(f"配对 {len(pairs)} 对（query {len(qids)} 条，dev {len(dev_q)} 条）")
    print(f"ce_train {len(train_rows)} 行 -> {a.out_train}")
    print(f"ce_dev   {len(dev_rows)} 行 -> {a.out_dev}")
    print(f"ce_pairs {len(pairs)} 行 -> {a.out_pairs}（蒸馏用，pos/neg 成对）")
    print(f"difficulty 分布 {dict(Counter(p['difficulty'] for p in pairs))}")


if __name__ == "__main__":
    main()
