"""P5 黄金测试集候选抽取（多源异构）。

要求（用户 2026-09-11 定稿）：
  - 1000+ 句、**多源异构**施工方案（11 份方案：内部 6 + 外部项目 5）；
  - 排除已被 silver / gold_eval_v2 使用的句子（避免训练-测试泄漏）；
  - 分层：覆盖 6 类实体 + 各方案来源，可疑模式优先。

输出：data/phase5/gold_candidates.jsonl
    {"cid","text","source","why":[...],"rule_entities":[...]}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import (LEGACY_GOLD, LEGACY_LEXICON, PHASE1, PHASE2, PLANS_DIR,
                                   PLANS_XPROJ_DIR, ROOT)
from ner2.src.weak.remote_label import RemoteLabeler, is_dirty, split_long

PHASE5 = os.path.join(ROOT, "ner2", "data", "phase5")
SILVER = [os.path.join(PHASE2, "silver_train.jsonl"), os.path.join(PHASE2, "silver_val.jsonl")]

_TITLE_RE = re.compile(r"附图|平面图|详图|立面图|示意图|系统图|流程图|说明|附表|一览表|工况")


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def used_hashes() -> set:
    """已被 silver / gold_eval_v2 / 判定池 使用的句子 hash（候选须排除）。"""
    out = set()
    for p in SILVER + [LEGACY_GOLD]:
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                out.add(_md5(json.loads(line)["text"]))
    # 判定池（review_pool 及其 v2）也排除：这些句子已进入 P2 流程
    for name in ("review_pool.jsonl", "review_pool_v2.jsonl", "judgments.jsonl", "judgments_v2.jsonl"):
        p = os.path.join(PHASE2, name)
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                out.add(_md5(json.loads(line)["text"]))
    return out


def iter_sentences(dirs):
    """多源语料 -> (text, source)。"""
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".txt"):
                continue
            src = fn[:-4]
            for ln in open(os.path.join(d, fn), encoding="utf-8"):
                ln = ln.strip()
                if not ln or is_dirty(ln):
                    continue
                if len(ln) <= 240:
                    yield ln, src
                else:
                    for sub in split_long(ln):
                        yield sub, src


def reasons(r):
    out = []
    if _TITLE_RE.search(r["text"]):
        out.append("title")
    if any(e["end"] - e["start"] <= 2 for e in r["entities"]):
        out.append("short")
    if len(r["entities"]) >= 3:
        out.append("multi")
    if any(e["type"] in ("危大类别", "规范编号") for e in r["entities"]):
        out.append("rare")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(PHASE5, "gold_candidates.jsonl"))
    args = ap.parse_args()

    lex = json.load(open(LEGACY_LEXICON, encoding="utf-8"))
    labeler = RemoteLabeler(lex)
    used = used_hashes()
    rng = random.Random(args.seed)

    dirs = [PLANS_DIR, PLANS_XPROJ_DIR]
    seen = set()
    suspicious, normal = [], []
    src_cnt = Counter()
    for text, src in iter_sentences(dirs):
        h = _md5(text)
        if h in used or h in seen:
            continue
        seen.add(h)
        ents = labeler.tag(text)
        if not ents:
            continue  # 黄金集只收含实体的句子
        r = {"text": text, "source": src, "entities": ents}
        rs = reasons(r)
        (suspicious if rs else normal).append((r, rs))
        src_cnt[src] += 1

    pool = []
    # 可疑模式优先（上限 60%）
    rng.shuffle(suspicious)
    cap = int(args.target * 0.6)
    pool.extend(suspicious[:cap])
    # 余量按来源均衡补（多源异构保障）
    by_src = defaultdict(list)
    for r, _ in normal:
        by_src[r["source"]].append(r)
    for r, _ in suspicious[cap:]:
        by_src[r["source"]].append(r)
    order = sorted(by_src)
    idx = 0
    while len(pool) < args.target and any(by_src[s] for s in order):
        s = order[idx % len(order)]
        if by_src[s]:
            pool.append((by_src[s].pop(), ["src:" + s]))
        idx += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, (r, rs) in enumerate(pool):
            out = {"cid": f"g{i:05d}", "text": r["text"], "source": r["source"],
                   "why": rs, "rule_entities": r["entities"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    tc = Counter(e["type"] for r, _ in pool for e in r["entities"])
    sc = Counter(r["source"] for r, _ in pool)
    print(f"候选 {len(pool)} 句 | 覆盖来源 {len(sc)} 个 | 实体 {sum(tc.values())}")
    print("来源分布:", dict(sc))
    print("实体类型:", dict(tc))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
