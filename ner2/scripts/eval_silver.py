"""P3 辅助评估：清洗前后质量对照 + 泄漏检查 + 实体分布。

输出（ner2/data/phase2/）：
    eval_silver_report.md  清洗收益报告（规则 vs 银标的标签集差异、类型分布、修正样例）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import (CONFLICTS, JUDGMENTS, LEGACY_GOLD, PHASE2,
                                   SILVER_TRAIN, SILVER_VAL, WEAK)
from ner2.src.eval.metrics import label_set_difference
from ner2.src.weak.lexicon import load_legacy_lexicon
from ner2.src.weak.remote_label import RemoteLabeler


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _gold_hash_set(path):
    """金标句子 hash 集合（泄漏检查用）。"""
    if not os.path.exists(path):
        return set()
    import hashlib
    out = set()
    for r in _load(path):
        out.add(hashlib.md5(r["text"].encode("utf-8")).hexdigest())
    return out


def _ent_counter(samples):
    c = Counter()
    for s in samples:
        for e in s.get("entities", []):
            c[e["type"]] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=PHASE2)
    ap.add_argument("--judgments", default=JUDGMENTS,
                    help="判定文件（默认 judgments.jsonl；扩量后传 judgments_v2.jsonl）")
    args = ap.parse_args()

    weak = _load(WEAK) if os.path.exists(WEAK) else []
    judgments = _load(args.judgments) if os.path.exists(args.judgments) else []
    train = _load(SILVER_TRAIN) if os.path.exists(SILVER_TRAIN) else []
    val = _load(SILVER_VAL) if os.path.exists(SILVER_VAL) else []
    conflicts = _load(CONFLICTS) if os.path.exists(CONFLICTS) else []

    lex = load_legacy_lexicon()
    labeler = RemoteLabeler(lex)

    # 对比口径：仅限定"被判定过的样本"（规则弱标 vs 银标），避免全量 vs 抽样的不公平比较
    judged_texts = {j["text"] for j in judgments}
    rule_judged = [{"text": r["text"], "entities": labeler.tag(r["text"])}
                   for r in weak if r["text"] in judged_texts]
    rule_ents = _ent_counter(rule_judged)
    silver_ents = _ent_counter(train + val)

    # 泄漏检查：银标训练句 vs 金标
    gold_hashes = _gold_hash_set(LEGACY_GOLD)
    leak = [s["text"] for s in train if
            __import__("hashlib").md5(s["text"].encode("utf-8")).hexdigest() in gold_hashes]

    # 标签集差异（限定被判定样本：规则 vs 银标，实体级）
    rule_all = [e for s in rule_judged for e in s["entities"]]
    silver_all = [e for s in train + val for e in s.get("entities", [])]
    diff = label_set_difference(rule_all, silver_all)

    # fix 修正样例（取前 8 条，展示规则标签 -> 银标标签）
    fix_examples = []
    for j in judgments:
        if j.get("verdict") == "fix" and j.get("entities"):
            rule_e = labeler.tag(j["text"])
            fix_examples.append({
                "text": j["text"],
                "rule": [(e["type"], j["text"][e["start"]:e["end"]]) for e in rule_e],
                "llm": [(e["type"], j["text"][e["start"]:e["end"]]) for e in j["entities"]],
                "reason": j.get("reason", ""),
            })
            if len(fix_examples) >= 8:
                break

    verdict = Counter(j.get("verdict") for j in judgments)

    lines = [
        "# P3 清洗评估报告（ner2）", "",
        f"- 弱标句 {len(weak)} | 判定 {len(judgments)} | silver_train {len(train)} | silver_val {len(val)} | conflicts {len(conflicts)}", "",
        "## 判定分布", "",
        "| verdict | 数量 | 占比 |",
        "|---|---|---|",
    ]
    for v in ("keep", "fix", "drop"):
        n = verdict.get(v, 0)
        lines.append(f"| {v} | {n} | {n/max(1,len(judgments)):.1%} |")
    lines += ["", "## 标签集差异（被判定样本：规则弱标 vs 银标，实体级）", "",
              f"- 规则独有（被剔除/冲突）: {diff['rule_only']}",
              f"- 银标新增（漏标修复）: {diff['llm_only']}",
              f"- 共有: {diff['shared']}",
              f"- Jaccard: {diff['jaccard']}", "",
              "## 实体分布（被判定样本：规则 -> 银标）", "",
              "| 类型 | 规则 | 银标 | 变化 |", "|---|---|---|---|"]
    all_types = sorted(set(rule_ents) | set(silver_ents))
    for t in all_types:
        a, b = rule_ents.get(t, 0), silver_ents.get(t, 0)
        lines.append(f"| {t} | {a} | {b} | {b-a:+d} |")
    lines += ["", "## 泄漏检查（silver_train vs gold_eval_v2）", "",
              f"- 金标句 {len(gold_hashes)} | 泄漏 {len(leak)}", "",
              "## fix 修正样例（规则 -> 银标）", ""]
    for ex in fix_examples:
        lines.append(f"- 句: {ex['text'][:60]}")
        lines.append(f"  - 规则: {ex['rule']}")
        lines.append(f"  - 银标: {ex['llm']}")
        lines.append(f"  - 理由: {ex['reason'][:80]}")
    lines.append("")

    report = os.path.join(args.outdir, "eval_silver_report.md")
    open(report, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[:40]))
    print(f"-> {report}")


if __name__ == "__main__":
    main()
