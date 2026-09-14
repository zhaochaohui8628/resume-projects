"""P0 词典底座：旧 lexicon 种子 + 用户增补 -> 规范化词典（独立资产，语料不参与建词）。

输出：ner2/data/lexicon/lexicon.json
规则扩充（可选 --gen）：从语料中正则抽取"参数类"候选（数值+单位），人工复核后并入。
注意：规范编号不写死词典，运行时走正则（见 remote_label.STD_RE）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import LEXICON_DIR, PLANS_DIR
from ner2.src.common.types import VALID_TYPES
from ner2.src.weak.lexicon import build_lexicon, summary

# 参数类候选：数值 + 单位（含中文量词），用于规则扩充（仅候选，人工复核后并入）
_PARAM_RE = re.compile(r"[A-Za-z0-9%±≈<>≤≥．.\-\u4e00-\u9fff]{1,24}?\d[\d\.%±≈<>≤≥]?\s*(?:m|mm|cm|km|MPa|kPa|Pa|kN|kg|t|T|V|A|kW|kVA|Ω|°|℃|%|米|毫米|厘米|度|吨|兆帕|千牛|米|米/小时)")


def gen_param_candidates(max_per_doc: int = 300) -> list[str]:
    """从语料正则抽取参数候选（去重、限长）。"""
    seen, cands = set(), []
    if not os.path.isdir(PLANS_DIR):
        return cands
    for fn in sorted(os.listdir(PLANS_DIR)):
        if not fn.endswith(".txt"):
            continue
        for ln in open(os.path.join(PLANS_DIR, fn), encoding="utf-8"):
            for m in _PARAM_RE.finditer(ln):
                w = m.group(0).strip()
                if 2 <= len(w) <= 30 and w not in seen:
                    seen.add(w)
                    cands.append(w)
        if len(cands) >= max_per_doc:
            break
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true",
                    help="从语料抽取参数候选到 user.json 待人工复核（默认不并入）")
    ap.add_argument("--merge-cands", action="store_true",
                    help="把 data/lexicon/candidates.json 中人工确认过的词并入用户词表")
    args = ap.parse_args()

    lex = build_lexicon(include_user=True)

    if args.gen:
        cands = gen_param_candidates()
        cpath = os.path.join(LEXICON_DIR, "candidates.json")
        json.dump(cands, open(cpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[gen] 参数候选 {len(cands)} 条 -> {cpath}（人工筛选后放 user.json 的\"参数\"组）")

    if args.merge_cands:
        cpath = os.path.join(LEXICON_DIR, "candidates.json")
        upath = os.path.join(LEXICON_DIR, "user.json")
        if os.path.exists(cpath):
            cands = json.load(open(cpath, encoding="utf-8"))
            user = json.load(open(upath, encoding="utf-8")) if os.path.exists(upath) else {}
            user.setdefault("参数", [])
            for w in cands:
                if w not in user["参数"]:
                    user["参数"].append(w)
            json.dump(user, open(upath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"[merge] {len(cands)} 条候选并入 user.json 参数组")

    out = os.path.join(LEXICON_DIR, "lexicon.json")
    json.dump(lex, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("词典规模:", summary(lex))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
