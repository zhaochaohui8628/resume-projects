"""P2 抽样：从弱标数据中筛出"模型逐条判定"的子集。

策略（与 rag2 P2 分层试标一致）：每类实体至少 N 条 + 可疑模式全保留：
  ①附图/表/章节标题（附图、图、表、平面图、详图 等）——词典词串常非实体语境；
  ②短实体（span ≤ 2 字符）——边界偏短/误标高发；
  ③类型歧义词（词串命中多类的，如"基坑降水"）；
  ④多实体句（≥3 实体）——漏标/冲突高发；
  ⑤危大类别、规范编号（样本稀少，全保留）。
输出：data/phase2/review_pool.jsonl（每行含 qid/text/entities/why 抽样原因）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import PHASE2, WEAK

_TITLE_RE = re.compile(r"附图|平面图|详图|立面图|示意图|系统图|流程图|说明|附表|一览表")
_AMBIG_WORDS = {"基坑降水", "基坑支护", "地下连续墙", "土方开挖", "起重吊装",
                "脚手架", "拆除", "注浆", "灌注", "模板", "支护桩", "钢支撑"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weak", default=WEAK)
    ap.add_argument("--per-type", type=int, default=45, help="每类实体保底条数")
    ap.add_argument("--max-pool", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(PHASE2, "review_pool.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.weak, encoding="utf-8")]
    rng = random.Random(args.seed)

    def reasons(r):
        out = []
        if _TITLE_RE.search(r["text"]):
            out.append("title")
        short = [e for e in r["entities"] if e["end"] - e["start"] <= 2]
        if short:
            out.append("short")
        for e in r["entities"]:
            if r["text"][e["start"]:e["end"]] in _AMBIG_WORDS:
                out.append("ambig")
                break
        if len(r["entities"]) >= 3:
            out.append("multi")
        if any(e["type"] in ("危大类别", "规范编号") for e in r["entities"]):
            out.append("rare")
        return out

    # ① 可疑模式：全保留（上限控制）
    suspicious, normal = [], []
    for r in rows:
        rs = reasons(r)
        (suspicious if rs else normal).append((r, rs))

    # ② 每类实体保底：从 normal 按类型分层抽
    by_type = defaultdict(list)
    for r, _ in normal:
        for e in r["entities"]:
            by_type[e["type"]].append(r)
    pool, seen = [], set()
    pick = {}
    for t, items in by_type.items():
        rng.shuffle(items)
        for r in items:
            if r["qid"] in seen:
                continue
            seen.add(r["qid"])
            pool.append((r, ["type:" + t]))
            if sum(1 for _ in pool if _[0]["qid"] in seen) >= args.per_type:
                break
            if len(pool) >= args.max_pool:
                break
        if len(pool) >= args.max_pool:
            break

    # 可疑优先填满
    for r, rs in suspicious:
        if len(pool) >= args.max_pool:
            break
        if r["qid"] not in seen:
            seen.add(r["qid"])
            pool.append((r, rs))

    # 剩余配额从 normal 随机补
    for r, _ in normal:
        if len(pool) >= args.max_pool:
            break
        if r["qid"] not in seen:
            seen.add(r["qid"])
            pool.append((r, ["random"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r, rs in pool:
            out = {"qid": r["qid"], "text": r["text"],
                   "entities": r["entities"], "why": rs}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    c = Counter(w for _, rs in pool for w in rs)
    tc = Counter(e["type"] for _, _rs in pool for e in _["entities"])
    print(f"pool {len(pool)} | 抽样原因 {dict(c)}")
    print(f"实体类型覆盖 {dict(tc)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
