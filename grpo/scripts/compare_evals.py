"""跨模型全量对比（用已落盘评测明细重算，不重跑生成）。

用法：
    python grpo/scripts/compare_evals.py <明细1.json> <明细2.json> ...

每个明细由 evaluate.py 带 EVAL_OUT 产出：[{id, task_type, response, breakdown}, ...]。
本脚本把各模型对同一批样本的得分横向排列，输出 markdown 表格。
"""
from __future__ import annotations

import io
import json
import os
import statistics as st
import sys

DIMS = ("format", "cot", "basis", "answer")


def main():
    paths = sys.argv[1:]
    if not paths:
        print("用法: compare_evals.py <明细1> [明细2 ...]")
        return
    data = {}
    ids = None
    for p in paths:
        rows = json.load(io.open(p, encoding="utf-8"))
        tag = os.path.basename(p).replace("_eval.json", "").replace(".json", "")
        data[tag] = {r["id"]: r for r in rows}
        ids = list(data[tag].keys())
        print(f"[load] {tag}: {len(rows)} 条")

    # 汇总表
    print("\n| 模型 | n | reward | format | cot | basis | answer | 满分率 | 答对率 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tag, m in data.items():
        n = len(m)
        tot = [r["breakdown"]["total"] for r in m.values()]
        full = sum(1 for t in tot if t >= 1.0)
        ans_ok = sum(1 for r in m.values() if r["breakdown"]["answer"] >= 0.9)
        print(f"| {tag} | {n} | {st.mean(tot):.3f} | "
              + " | ".join(f"{st.mean([r['breakdown'][d] for r in m.values()]):.3f}" for d in DIMS)
              + f" | {full / n:.2%} | {ans_ok / n:.2%} |")

    # 逐条对比（仅展示各模型得分不同的行）
    if len(data) >= 2:
        print("\n#### 逐条差异")
        tags = list(data.keys())
        print("| id | task | " + " | ".join(f"{t} (reward)" for t in tags) + " |")
        print("|---|---|" + "|".join("---" for _ in tags) + "|")
        for i in ids:
            vals = [data[t].get(i) for t in tags]
            if any(v is None for v in vals):
                continue
            diffs = [v["breakdown"]["total"] for v in vals]
            if max(diffs) - min(diffs) < 1e-9:
                continue
            print("| " + i + " | " + str(vals[0]["task_type"]) + " | "
                  + " | ".join(f"{v['breakdown']['total']:.3f}" for v in vals) + " |")


if __name__ == "__main__":
    main()
