"""参数类专项：从**训练侧**数据反推「参数」标注口径（不看测试集，保指标诚实）。

输出 `data/phase6/param_spec_analysis.md`：
  - 每类来源的参数实体全文 + ±14 字上下文（供人工归纳边界/取值/单位口径）
  - 形态统计：是否含数字、是否含单位/符号、是否含中文量词、是否带比较符、括号形式
  - 前后邻接字符分布（判断"实体是否吞掉左侧名词"这一关键边界口径）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import (  # noqa: E402
    GOLD_CONSENSUS, GOLD_TRAIN_PORTION, PHASE6, SILVER_TRAIN, SILVER_VAL,
)

NUM_RE = re.compile(r"\d")
UNIT_RE = re.compile(
    r"(mm|cm|dm|km|m|kg|t|kN|KVA|kVA|MPa|Pa|min|s|h|d|°|℃|%|‰|"
    r"毫米|厘米|米|千米|公斤|吨|牛|帕|分钟|小时|天|度|摄氏度|"
    r"平方米|立方米|方|根|个|台|套|道|层|排|环|点|次|段|片|块|组|只|条|支)")
SYM_RE = re.compile(r"[∅ΦφΦ@×✕*×/％%‰≤≥<>±±°]")
CMP_RE = re.compile(r"(不小于|不大于|不少于|不超过|不得低于|不得小于|小于|大于|≤|≥|<|>|不少于|以内|以上|以下)")
PAREN_RE = re.compile(r"[（(].{0,8}[）)]")


def _ctx(text: str, s: int, e: int, w: int = 14) -> str:
    a, b = max(0, s - w), min(len(text), e + w)
    return text[a:s] + "【" + text[s:e] + "】" + text[e:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PHASE6, "param_spec_analysis.md"))
    args = ap.parse_args()

    srcs = [("gold_train_portion", GOLD_TRAIN_PORTION), ("silver_train", SILVER_TRAIN),
            ("silver_val", SILVER_VAL), ("gold_consensus", GOLD_CONSENSUS)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    lines = ["# 「参数」标注口径反推（仅用训练侧数据）", ""]
    allspans = []
    for name, p in srcs:
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        items = []
        for r in rows:
            for e in r.get("entities", []):
                if e["type"] == "参数":
                    items.append((r["text"], e["start"], e["end"]))
        allspans.extend([(name, t, s, e) for t, s, e in items])
        lines += [f"## {name}（{len(items)} 个参数实体 / {len(rows)} 句）", "",
                  "| # | 参数 span | 上下文（【】内为标注 span） |", "|---|---|---|"]
        for i, (t, s, e) in enumerate(items, 1):
            span = t[s:e].replace("|", "\\|")
            lines.append(f"| {i} | `{span}` | {_ctx(t, s, e).replace('|', '\\|')} |")
        lines.append("")

    # ---- 形态统计 ----
    spans = [t[s:e] for _, t, s, e in allspans]
    stat = Counter()
    for sp in spans:
        stat["total"] += 1
        stat["has_num"] += bool(NUM_RE.search(sp))
        stat["has_unit"] += bool(UNIT_RE.search(sp))
        stat["has_sym"] += bool(SYM_RE.search(sp))
        stat["has_cmp"] += bool(CMP_RE.search(sp))
        stat["has_paren"] += bool(PAREN_RE.search(sp))
        stat["no_num"] += not NUM_RE.search(sp)
        stat["pure_chinese"] += not NUM_RE.search(sp) and not UNIT_RE.search(sp)
    lenhist = Counter(len(sp) for sp in spans)
    # 边界：span 后一字符
    after = Counter()
    for _, t, s, e in allspans:
        after[t[e] if e < len(t) else "<EOS>"] += 1
    before = Counter()
    for _, t, s, e in allspans:
        before[t[s - 1] if s > 0 else "<BOS>"] += 1

    lines += ["## 形态统计", "",
              f"- 参数实体合计 **{stat['total']}**（去重后 {len(set(spans))} 种写法）",
              f"- 含数字 {stat['has_num']}｜含单位 {stat['has_unit']}｜含符号(∅Φ@×/≤) {stat['has_sym']}｜"
              f"含比较词 {stat['has_cmp']}｜含括号 {stat['has_paren']}",
              f"- **不含数字 {stat['no_num']}**（其中纯中文 {stat['pure_chinese']}）", "",
              "### span 长度分布", "",
              "| 长度 | " + " | ".join(str(k) for k in sorted(lenhist)) + " |",
              "|---|" + "---|" * len(lenhist),
              "| 个数 | " + " | ".join(str(lenhist[k]) for k in sorted(lenhist)) + " |", "",
              "### 左邻字符 top15（判断是否吞掉左侧名词）", "",
              "`" + "` `".join(f"{c}:{n}" for c, n in before.most_common(15)) + "`", "",
              "### 右邻字符 top15", "",
              "`" + "` `".join(f"{c}:{n}" for c, n in after.most_common(15)) + "`", "",
              "### 全部写法（去重，按频次）", "",
              "| 写法 | 次数 |", "|---|---|"]
    for sp, n in Counter(spans).most_common():
        lines.append(f"| `{sp}` | {n} |")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[spec] 参数实体 {stat['total']}（去重 {len(set(spans))}）"
          f"｜不含数字 {stat['no_num']}｜含符号 {stat['has_sym']}")
    print(f"[spec] → {args.out}")


if __name__ == "__main__":
    main()
