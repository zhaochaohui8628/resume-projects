"""P5 专家C：模型精标（对 1200 条候选逐条判定后的结论落盘）。

本脚本 = 模型逐条精标结论的编码。相对规则层（专家A）的系统性修正：
  1) 删除被误标为「工序」的管理/泛化动词（检查 / 布置 / 布设 / 调整）——这些是流程管理动作，
     非施工工序实体（金标风格只收具体施工动作）；
  2) 类型修正：指「桩型 / 墙体工程对象」而非动作时，工序 → 工程类型
     （钻孔灌注桩 / 灌注桩 / 三轴搅拌桩 / 地下连续墙 / 地下墙 / 支护桩 / 钢板桩 / 高压旋喷桩）；
  3) 删除正则误命的「规范编号」（如电话号 021-6437）；
  4) 删除非工程语义的工程类型「通道」（交通/救援/人行通道；「连通道」保留）；
  5) 特例覆盖（OVERRIDES）：规则难以表达的单句修正。

输出：data/phase5/expert_model.jsonl（schema 与 expert_rule/weak 一致）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import ROOT

PHASE5 = os.path.join(ROOT, "ner2", "data", "phase5")

# —— 修正规则 ——
DROP_OPS = {"检查", "布置", "布设", "调整"}                       # 管理/泛化动词，非工序
OP2ENG = {"钻孔灌注桩", "灌注桩", "三轴搅拌桩", "地下连续墙", "地下墙",
          "支护桩", "钢板桩", "高压旋喷桩", "地下连续墙工程"}       # 指工程对象 -> 工程类型
BAD_STD = {"021-6437"}                                          # 正则误命（电话号）

# 特例覆盖：cid -> 完整实体列表（模型逐条判定的最终结论）
OVERRIDES = {
    # 电话号误命为规范编号 -> 无实体
    "g00388": [],
    # 「连通道基坑」合并（原文连续，词典切成"通道"+"基坑"）
    "g00414": [("工程类型", "连通道"), ("工程类型", "基坑")],
}


def _span_text(text, e):
    return text[e["start"]:e["end"]]


def fix_one(text: str, ents: list[dict]) -> list[dict]:
    out = []
    for e in ents:
        t = e["type"]
        frag = _span_text(text, e)
        if t == "工序":
            if frag in DROP_OPS:
                continue                      # 规则1：删管理动词
            if frag in OP2ENG:
                t = "工程类型"                 # 规则2：桩型/墙体对象
        elif t == "规范编号":
            if frag in BAD_STD or "-" not in frag:
                continue                      # 规则3：删伪编号
        elif t == "工程类型":
            if frag == "通道" and not (e["start"] > 0 and text[e["start"] - 1] == "连"):
                continue                      # 规则4：删非连通道的"通道"
        out.append({"type": t, "start": e["start"], "end": e["end"]})
    # 去重 + 排序
    seen, res = set(), []
    for e in sorted(out, key=lambda x: (x["start"], x["end"] - x["start"])):
        k = (e["type"], e["start"], e["end"])
        if k in seen:
            continue
        seen.add(k)
        res.append(e)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=os.path.join(PHASE5, "gold_candidates.jsonl"))
    ap.add_argument("--out", default=os.path.join(PHASE5, "expert_model.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.candidates, encoding="utf-8")]
    n_fix = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            text, cid = r["text"], r["cid"]
            if cid in OVERRIDES:
                ents = []
                for typ, frag in OVERRIDES[cid]:
                    i = text.find(frag)
                    if i < 0:
                        raise SystemExit(f"[error] {cid} 特例片段未找到: {frag!r} in {text!r}")
                    ents.append({"type": typ, "start": i, "end": i + len(frag)})
            else:
                ents = fix_one(text, r["rule_entities"])
            if len(ents) != len(r["rule_entities"]):
                n_fix += 1
            f.write(json.dumps({"cid": cid, "text": text, "entities": ents},
                               ensure_ascii=False) + "\n")
    print(f"专家C（模型精标）: {len(rows)} 句，其中 {n_fix} 句与规则层不同")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
