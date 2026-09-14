"""P5 自动专家标注生成（专家A 规则层 / 专家B 弱标层）。

这两个专家是**确定性自动标注**（零 API、可复跑），与"专家C 模型精标"（人工级）
共同构成多专家投票的输入。二者共用词典但优先级/覆盖策略不同，属弱独立专家。

输出：data/phase5/expert_rule.jsonl、expert_weak.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import LEGACY_LEXICON, ROOT
from ner2.src.pipeline.rule_layer import RuleLayer
from ner2.src.weak.remote_label import RemoteLabeler

PHASE5 = os.path.join(ROOT, "ner2", "data", "phase5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=os.path.join(PHASE5, "gold_candidates.jsonl"))
    ap.add_argument("--lexicon", default=LEGACY_LEXICON)
    ap.add_argument("--outdir", default=PHASE5)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.candidates, encoding="utf-8")]
    lex = json.load(open(args.lexicon, encoding="utf-8"))
    rule = RuleLayer(lex)
    weak = RemoteLabeler(lex)

    for name, fn in (("expert_rule", lambda t: [{"type": e["type"], "start": e["start"], "end": e["end"]}
                                                for e in rule.extract(t)]),
                     ("expert_weak", lambda t: [{"type": e["type"], "start": e["start"], "end": e["end"]}
                                                for e in weak.tag(t)])):
        path = os.path.join(args.outdir, name + ".jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"cid": r["cid"], "text": r["text"],
                                    "entities": fn(r["text"])}, ensure_ascii=False) + "\n")
        print(f"-> {path}")

    ra = sum(len(rule.extract(r["text"])) for r in rows)
    wa = sum(len(weak.tag(r["text"])) for r in rows)
    print(f"专家A(规则) 实体 {ra} | 专家B(弱标) 实体 {wa} | 候选 {len(rows)} 句")


if __name__ == "__main__":
    main()
