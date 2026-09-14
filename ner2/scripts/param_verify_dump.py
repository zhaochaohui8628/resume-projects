"""参数专项 Step3：核验转储。把生成数据里**最容易出错**的部分逐条摊开供人工确认：
  · 全部 B（量名+值）/ B′（规格值）/ B″（等级值）实例的上下文
  · 全部负样本（长句优先）
  · 量名去重清单（供剔除"其实不是参数"的量名）
输出 `data/phase6/param_verify.md`。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import PHASE6  # noqa: E402


def mark(text: str, ents: list[dict]) -> str:
    t = text
    for e in sorted(ents, key=lambda x: -x["start"]):
        t = t[:e["start"]] + "〖" + t[e["start"]:e["end"]] + "〗" + t[e["end"]:]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(PHASE6, "param_specialist.jsonl"))
    ap.add_argument("--max-neg", type=int, default=90)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    out = ["# 参数专项数据 · 核验转储", "", f"- 样本 {len(rows)}"]

    byform = defaultdict(list)
    nouncnt = Counter()
    for r in rows:
        forms = r.get("forms", [])
        if len(forms) == 1 and forms[0] in ("B", "B'", "B''"):
            byform[forms[0]].append(r)
        for n in r.get("nouns", []):
            nouncnt[n] += 1
    for f in ("B", "B'", "B''"):
        out += [f"", f"## 形态 {f}（{len(byform[f])}）", "",
                "| 原文（〖〗=标注） |", "|---|"]
        for r in byform[f]:
            out.append(f"| {mark(r['text'], r['entities']).replace('|', chr(92) + '|')} |")
    out += ["", f"## 负样本（{sum(1 for r in rows if not r['entities'])}，按长度降序取前 {args.max_neg}）", "",
            "| 句 |", "|---|"]
    negs = sorted([r for r in rows if not r["entities"]], key=lambda r: -len(r["text"]))
    for r in negs[:args.max_neg]:
        out.append(f"| {r['text'].replace('|', chr(92) + '|')} |")
    out += ["", "## 命中量名清单（按命中数）", "", "| 量名 | 次数 |", "|---|---|"]
    for n, c in nouncnt.most_common():
        out.append(f"| `{n}` | {c} |")
    p = os.path.join(PHASE6, "param_verify.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"[verify] B={len(byform['B'])} B'={len(byform[chr(66)+chr(39)])} "
          f"B''={len(byform['B'+chr(39)*2])} 量名 {len(nouncnt)} → {p}")


if __name__ == "__main__":
    main()
