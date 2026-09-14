"""P3 冲突过滤 + 银标构建。

输入：judgments.jsonl（模型手工判定 / llm_review.py 产物，schema 一致）
输出（ner2/data/phase2/）：
    silver_train.jsonl  训练集（keep 原样 + fix 用修正标签；drop 剔除）
    silver_val.jsonl    留出验证集（默认 15%，按句子 hash 稳定划分）
    conflicts.jsonl     冲突样本明细（drop 与被剔除实体，供人工回看）
    silver_stats.json   统计（判定分布 / 冲突 / 边界修正幅度）

判定映射（conflict.analyze）：
    keep  → 规则标签原样入 silver
    fix   → LLM 修正后标签入 silver（修正本身就是清洗收益）
    drop  → 整句剔除进 conflicts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import CONFLICTS, JUDGMENTS, LEGACY_GOLD, PHASE2, SILVER_TRAIN, SILVER_VAL
from ner2.src.eval.metrics import label_set_difference
from ner2.src.review.conflict import analyze, apply_judgment
from ner2.src.weak.remote_label import RemoteLabeler
from ner2.src.weak.lexicon import load_legacy_lexicon


def _h(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _gold_hashes(path):
    """金标句子 hash 集（泄漏检查：训练句不得与金标重叠）。"""
    if not os.path.exists(path):
        return set()
    out = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out.add(hashlib.md5(r["text"].encode("utf-8")).hexdigest())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", default=JUDGMENTS)
    ap.add_argument("--outdir", default=PHASE2)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--val-seed-hash", action="store_true",
                    help="按句子 hash 稳定划分（默认按 qid 序号尾部）")
    args = ap.parse_args()

    if not os.path.exists(args.judgments):
        raise SystemExit(f"[error] 缺判定 {args.judgments}，先执行 P2 判定")

    # 重建规则标签（与 weak.jsonl 同源可复算；judgments 行自带 qid/text/entities(LLM)）
    lex = load_legacy_lexicon()
    labeler = RemoteLabeler(lex)
    rows = [json.loads(l) for l in open(args.judgments, encoding="utf-8")]
    gold_hashes = _gold_hashes(LEGACY_GOLD)

    train, val, conflicts = [], [], []
    vcnt = Counter()
    leak = 0
    for r in rows:
        rule = {"qid": r.get("qid"), "text": r["text"],
                "entities": labeler.tag(r["text"])}
        sample = apply_judgment(rule, r)
        if sample is None:  # drop
            conflicts.append({
                "qid": r.get("qid"), "text": r["text"],
                "verdict": "drop",
                "reason": r.get("reason", ""),
                "issues": r.get("issues", []),
                "rule_entities": rule["entities"],
            })
            vcnt["drop"] += 1
            continue
        vcnt[sample["verdict"]] += 1
        if not sample["entities"]:
            # fix 后实体被清空：等同 drop，但记录为 empty-fix
            conflicts.append({
                "qid": r.get("qid"), "text": r["text"], "verdict": "empty-fix",
                "reason": r.get("reason", ""), "issues": r.get("issues", []),
                "rule_entities": rule["entities"],
            })
            vcnt["empty-fix"] += 1
            continue
        # 金标泄漏剔除：与 gold_eval_v2 逐句 hash 重叠的训练样本不得入 silver
        if hashlib.md5(sample["text"].encode("utf-8")).hexdigest() in gold_hashes:
            conflicts.append({
                "qid": r.get("qid"), "text": r["text"], "verdict": "gold-leak",
                "reason": "与金标零泄漏：训练句与 gold_eval_v2 重叠，剔除",
                "issues": [], "rule_entities": rule["entities"],
            })
            vcnt["gold-leak"] += 1
            leak += 1
            continue
        # 泄漏防护：验证集与训练集零重叠
        key = _h(sample["text"])
        if (key % 100) / 100 < args.val_ratio:
            val.append(sample)
        else:
            train.append(sample)

    os.makedirs(args.outdir, exist_ok=True)
    for path, data in ((SILVER_TRAIN, train), (SILVER_VAL, val), (CONFLICTS, conflicts)):
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 统计
    stats = {
        "judged": len(rows),
        "verdict": dict(vcnt),
        "gold_leak_removed": leak,
        "silver_train": len(train), "silver_val": len(val),
        "conflicts": len(conflicts),
    }
    # 边界修正幅度（对 fix 样本：规则标签 vs LLM 标签差异）
    diffs = []
    for r in rows:
        rule = {"text": r["text"], "entities": labeler.tag(r["text"])}
        if r.get("verdict") == "fix":
            d = label_set_difference(rule["entities"], r.get("entities") or [])
            diffs.append(d)
    if diffs:
        stats["fix_sample_rule_only_avg"] = round(
            sum(d["rule_only"] for d in diffs) / len(diffs), 2)
        stats["fix_sample_llm_only_avg"] = round(
            sum(d["llm_only"] for d in diffs) / len(diffs), 2)
        stats["fix_sample_jaccard_avg"] = round(
            sum(d["jaccard"] for d in diffs) / len(diffs), 4)
        stats["fix_count"] = len(diffs)

    json.dump(stats, open(os.path.join(args.outdir, "silver_stats.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"-> {SILVER_TRAIN} / {SILVER_VAL} / {CONFLICTS}")


if __name__ == "__main__":
    main()
