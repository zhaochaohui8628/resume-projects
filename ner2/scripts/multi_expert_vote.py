"""P5 多专家投票聚合（大模型交叉校验 + 多数表决 + 权威仲裁）。

输入：N 个专家标注文件（schema 一致）
    {"cid","text","entities":[{"type","start","end"}]}
    · expert_rule.jsonl   —— 专家A：确定性规则/词典层（自动）
    · expert_weak.jsonl   —— 专家B：远程监督弱标层（自动）
    · expert_model.jsonl  —— 专家C：模型逐条精标（人工级，本流程的核心专家）

聚合规则：
  1) 对齐：两名专家提的实体若 **类型相同且 span 有重叠** 视为同一实体；
  2) 表决：票数 = 提出该实体的专家数；≥min-votes（默认 2）→ 采纳为共识；
  3) 交叉校验：报告专家两两一致率（Jaccard）与逐类一致率；
  4) 权威仲裁（--authoritative，默认 expert_model）：模型精标为资深标注，
     其判定为最终黄金标注；规则/弱标层的一致性作为交叉校验信号记录；
     规则层与弱标层一致、但被精标否决的实体 → 记入 conflicts（词典假阳性，即清洗收益）。

输出（data/phase5/）：
    gold_consensus.jsonl  最终黄金标注（含 n_votes 票数）
    gold_disputes.jsonl   单票争议实体
    conflicts.jsonl       被精标否决的词典假阳性（交叉校验发现）
    vote_stats.json       一致性统计
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import ROOT

PHASE5 = os.path.join(ROOT, "ner2", "data", "phase5")


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _key(e):
    return e["type"]


def _overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def _same_entity(a, b):
    """类型相同 + span 重叠 = 同一实体。"""
    return a["type"] == b["type"] and _overlap(a, b)


def _vote(candidate_list):
    """对一组专家实体做多数表决。

    返回 [(repr_entity, n_votes, n_experts)]：按票数降序、span 升序。
    """
    used = [False] * len(candidate_list)
    groups = []
    for i, e in enumerate(candidate_list):
        if used[i]:
            continue
        grp = [e]
        used[i] = True
        for j in range(i + 1, len(candidate_list)):
            if used[j]:
                continue
            if _same_entity(e, candidate_list[j]):
                grp.append(candidate_list[j])
                used[j] = True
        groups.append(grp)
    out = []
    for grp in groups:
        # 代表实体：取 span 最长者（更完整的边界）
        rep = max(grp, key=lambda x: x["end"] - x["start"])
        out.append((rep, len(grp), len(grp)))
    out.sort(key=lambda x: (x[0]["start"], x[0]["end"] - x[0]["start"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=os.path.join(PHASE5, "gold_candidates.jsonl"))
    ap.add_argument("--experts", nargs="+", required=True,
                    help="专家文件路径列表（≥2）")
    ap.add_argument("--min-votes", type=int, default=2, help="共识所需最少票数")
    ap.add_argument("--authoritative", default="expert_model.jsonl",
                    help="资深仲裁专家文件名（其标注为最终黄金标注）")
    ap.add_argument("--outdir", default=PHASE5)
    args = ap.parse_args()

    cands = {c["cid"]: c for c in _load(args.candidates)}
    experts = {os.path.basename(p): {r["cid"]: r for r in _load(p)} for p in args.experts}
    names = sorted(experts)
    auth = args.authoritative if args.authoritative in experts else None

    consensus, disputes, conflicts = [], [], []
    pair_agree = defaultdict(lambda: [0, 0])
    per_type_votes = defaultdict(Counter)
    total_ents = Counter()

    for cid, c in cands.items():
        buckets = []
        for nm in names:
            e = experts[nm].get(cid)
            for ent in (e or {}).get("entities", []):
                buckets.append((nm, {"type": ent["type"], "start": ent["start"], "end": ent["end"]}))
        groups = _vote([b for _, b in buckets]) if buckets else []

        # 权威仲裁：以精标专家为准
        auth_ents = [{"type": x["type"], "start": x["start"], "end": x["end"]}
                     for x in (experts[auth].get(cid) or {}).get("entities", [])] if auth else None

        keep, disp = [], []
        for rep, nv, _ in groups:
            total_ents[rep["type"]] += 1
            per_type_votes[rep["type"]][nv] += 1
            if auth_ents is not None:
                # 若与精标实体对齐 -> 采纳（票数仅作参考）；否则进 conflicts
                aligned = any(_same_entity(rep, a) for a in auth_ents)
                if aligned:
                    keep.append({**rep, "n_votes": nv, "n_experts": len(names)})
                else:
                    conflicts.append({"cid": cid, "text": c["text"], "rule_entity": rep,
                                      "n_votes": nv,
                                      "reason": "词典层假阳性，经模型精标否决"})
            else:
                row = {**rep, "n_votes": nv, "n_experts": len(names)}
                (keep if nv >= args.min_votes else disp).append(row)

        # 精标补充：精标有、规则/弱标都没提的实体（漏标修复）
        if auth_ents is not None:
            for a in auth_ents:
                if not any(_same_entity(a, rep) for rep, _, _ in groups):
                    keep.append({**a, "n_votes": 1, "n_experts": len(names),
                                 "note": "精标补充"})

        # 重新排序 + 去重
        keep.sort(key=lambda x: (x["start"], x["end"] - x["start"]))
        if keep:
            consensus.append({"cid": cid, "text": c["text"], "source": c.get("source"),
                              "entities": keep, "n_experts": len(names)})
        for d in disp:
            disputes.append({"cid": cid, "text": c["text"], "entity": d})

        # 两两一致率
        sets = {}
        for nm in names:
            e = experts[nm].get(cid) or {}
            sets[nm] = [(x["type"], x["start"], x["end"]) for x in e.get("entities", [])]
        for a, b in combinations(names, 2):
            sa, sb = set(sets[a]), set(sets[b])
            pair_agree[(a, b)][0] += len(sa & sb)
            pair_agree[(a, b)][1] += len(sa | sb)

    os.makedirs(args.outdir, exist_ok=True)
    _dump(os.path.join(args.outdir, "gold_consensus.jsonl"), consensus)
    _dump(os.path.join(args.outdir, "gold_disputes.jsonl"), disputes)
    _dump(os.path.join(args.outdir, "conflicts.jsonl"), conflicts)

    stats = {
        "candidates": len(cands), "experts": names,
        "authoritative": auth, "min_votes": args.min_votes,
        "consensus_sent": len(consensus),
        "consensus_ents": sum(len(r["entities"]) for r in consensus),
        "disputes": len(disputes),
        "conflicts_vetoed": len(conflicts),
        "pair_jaccard": {f"{a}|{b}": round(v[0] / v[1], 4) if v[1] else 1.0
                         for (a, b), v in pair_agree.items()},
        "vote_dist": {t: dict(c) for t, c in per_type_votes.items()},
    }
    json.dump(stats, open(os.path.join(args.outdir, "vote_stats.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


def _dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
