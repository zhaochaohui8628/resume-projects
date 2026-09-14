"""P1 远程监督弱标注（纯规则，可复跑，无需 API）。

用法（项目根）：
    python ner2/scripts/weak_label.py [--lexicon data/ner/lexicon.json] [--max-sent 20000]

输入：
    data/raw/plans_internal/*.txt（6 份方案语料）
    data/raw/plans_xproj/*.txt   （外部项目方案语料，2026-09-11 起并入扩源）
    （语料只做被打标对象，不参与建词）
    词典（默认旧 data/ner/lexicon.json 种子，可用 --lexicon 指向 ner2 合并词典）

输出（ner2/data/phase1/）：
    weak.jsonl                 {"qid","text","entities":[{"type","start","end","source","term"}]}
    unlabeled_sentences.jsonl  {"text"}  全部清洗句（无标注池）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import (PHASE1, PLANS_DIR, UNLABELED, WEAK, LEGACY_LEXICON,
                                   PLANS_XPROJ_DIR)
from ner2.src.weak.lexicon import load_legacy_lexicon
from ner2.src.weak.remote_label import RemoteLabeler, is_dirty, split_long


def iter_sentences(max_sent: int, dirs):
    """多语料目录 -> 清洗样本句（清洗整行为单位，超长行按分句切）。"""
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".txt"):
                continue
            for ln in open(os.path.join(d, fn), encoding="utf-8"):
                ln = ln.strip()
                if not ln or is_dirty(ln):
                    continue
                if len(ln) <= 240:
                    yield ln
                else:
                    yield from split_long(ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default=LEGACY_LEXICON,
                    help="词典 json 路径（默认旧 data/ner/lexicon.json）")
    ap.add_argument("--max-sent", type=int, default=20000)
    ap.add_argument("--outdir", default=PHASE1)
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="语料目录（默认 internal + xproj）")
    args = ap.parse_args()

    if not os.path.exists(args.lexicon):
        raise SystemExit(f"[error] 缺词典 {args.lexicon}，先跑 build_lexicon.py")
    lex = json.load(open(args.lexicon, encoding="utf-8"))
    labeler = RemoteLabeler(lex)
    os.makedirs(args.outdir, exist_ok=True)

    dirs = args.dirs if args.dirs else [PLANS_DIR, PLANS_XPROJ_DIR]
    if not any(os.path.isdir(d) for d in dirs):
        raise SystemExit(f"[error] 语料目录都不存在: {dirs}")

    weak, unlabeled = [], []
    for i, s in enumerate(iter_sentences(args.max_sent, dirs)):
        ents = labeler.tag(s)
        unlabeled.append({"text": s})
        if ents:
            weak.append({"qid": f"w{i:05d}", "text": s, "entities": ents})

    with open(UNLABELED, "w", encoding="utf-8") as f:
        for u in unlabeled:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    with open(WEAK, "w", encoding="utf-8") as f:
        for w in weak:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")

    st = labeler.stats(weak)
    print(f"句子总数 {len(unlabeled)} | 弱标签(含实体) {st['sent']} "
          f"({st['sent']/max(1,len(unlabeled)):.0%}) | 实体 {st['ents']}")
    print("实体分布:", st["by_type"])
    print("来源分布:", st["by_source"])
    for w in weak[:3]:
        print("  样例:", w["text"][:50],
              "|", [(e["type"], w["text"][e["start"]:e["end"]], e.get("source")) for e in w["entities"][:4]])
    print(f"-> {WEAK}\n-> {UNLABELED}")


if __name__ == "__main__":
    main()
