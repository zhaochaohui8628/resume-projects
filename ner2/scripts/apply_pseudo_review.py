"""LLM 交叉验证清洗（P4）：由模型逐条通读挖掘伪标后固化的**复核策略**（代码可复跑）。

背景
----
第一轮只有 softmax 头被训练（BERT 主权重冻结），token 级判别力弱；高置信度（margin）只代表
"模型很自信"，**很自信地判错**很常见。实际通读 364 条后确认三类系统性噪声：
  1) 图表标题/章节标题被当正文（"附图001：…"、"表6.3 模板工程质量标准表"）；
  2) **单字碎片**（"电"、"基"、"监"、"橡"、"立"）——模型把词中间的字单独标成实体；
  3) 通用动词/名词被标成工序（"布置"、"检查"、"清理"）与无值"参数"（"厚度"、"垂直度"）。

复核策略（三条硬规则 + 一条共识规则）
------------------------------------
- R1 结构性噪声：标题行整条丢弃（正则命中 附图/图/表/附件/第X章 等）；
- R2 碎片/泛词：span 长度 < 2 丢弃；span ∈ 通用词表丢弃；
- R3 参数无值：`参数` 类 span 不含数字/∅/Φ/@/×/% 等量值符号则丢弃；
- R4 **共识**：实体必须与「词典（`build_lexicon`）或规则层」达成共识才保留
  （同 span 同类型；桩型/墙体类允许 工序→工程类型 的既定修正）。
  清后实体数 = 0 → 丢弃该句（原本就无实体的高置信句保留为负样本）。

产物：`pseudo_clean_{head}.jsonl`（src="pseudo_clean"）+ `pseudo_review_{head}.json`。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import PHASE4  # noqa: E402
from ner2.src.weak.lexicon import STD_RE, build_lexicon  # noqa: E402

CAPTION_RE = re.compile(
    r"^(附?\s*图|表|附件|照片|第[一二三四五六七八九十百\d]+[章节])"
    r"|^(图|表|附图|附件)\s*[\d一二三四五六七八九十]+"
)
GENERIC = {"布置", "布设", "检查", "调整", "清理", "施工", "操作", "组织", "搭设", "安装",
           "处理", "管理", "监护", "验收", "交底", "浇筑", "保护", "维修", "保养", "控制",
           "监测", "测量", "确认", "核对", "记录", "运输", "堆放", "拆除", "人员", "工作"}
# 桩型 / 墙体 / 结构对象：允许 工序 → 工程类型 的既定修正
OBJECT_SUFFIX = ("桩", "墙", "坑", "撑", "架", "板", "梁", "柱", "层", "构", "法", "井", "缝")
VALUE_RE = re.compile(r"[\d∅ΦφΦ@×✕xX*/%°#]|mm|MM|kN|KN|kg|KG|MPa|t\b|T\b|m\b|M\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--head", required=True, choices=["softmax", "crf"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-span-len", type=int, default=30)
    args = ap.parse_args()

    out = args.out or os.path.join(PHASE4, f"pseudo_clean_{args.head}.jsonl")
    lex = build_lexicon()
    pairs = {(w, t) for t, ws in lex.items() for w in ws}
    by_type = {t: set(ws) for t, ws in lex.items()}
    any_term = {w for ws in lex.values() for w in ws}

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    stat = Counter()
    kept = []
    for r in rows:
        stat["in"] += 1
        text = r["text"]
        if CAPTION_RE.match(text.strip()):
            stat["drop_caption"] += 1
            continue
        new_ents = []
        for e in r.get("entities", []):
            span = text[e["start"]:e["end"]]
            t = e["type"]
            if len(span) < 2:
                stat["drop_fragment"] += 1
                continue
            if len(span) > args.max_span_len:
                stat["drop_toolong"] += 1
                continue
            if span in GENERIC:
                stat["drop_generic"] += 1
                continue
            if t == "参数" and not VALUE_RE.search(span):
                stat["drop_param_no_value"] += 1
                continue
            ok = (span, t) in pairs
            if not ok and t == "规范编号" and STD_RE.search(span):
                ok = True
            if not ok and t == "工程类型" and span in any_term and \
                    span.endswith(OBJECT_SUFFIX):
                ok = True                      # 既定类型修正：桩型/墙体 → 工程类型
            if not ok and span in any_term:
                # 词典认可该词但类型不同：以词典类型为准（更高置信的封闭类目）
                for t2, ws in by_type.items():
                    if span in ws and t2 != "规范编号":
                        new_ents.append({"type": t2, "start": e["start"], "end": e["end"]})
                        stat["fix_type"] += 1
                        ok = True
                        break
            if not ok:
                stat["drop_no_consensus"] += 1
                continue
            new_ents.append({"type": t, "start": e["start"], "end": e["end"]})
        # 去重（同 span 同类型）
        seen, dedup = set(), []
        for e in new_ents:
            k = (e["type"], e["start"], e["end"])
            if k not in seen:
                seen.add(k)
                dedup.append(e)
        if not dedup:
            # 清后无实体：**丢弃**而不是当负样本——第一轮只训了头（主干冻结），
            # 其"全 O 且高置信"不构成"该句无实体"的证据（实测多条明显漏标，如"（吊具、吊索等）"）。
            stat["drop_sent_empty"] += 1
            continue
        kept.append({"text": text, "entities": dedup, "src": "pseudo_clean"})
        stat["kept_pos"] += 1

    report = {
        "in": len(rows), "out": len(kept),
        "out_pos": stat["kept_pos"], "out_neg": stat["kept_neg"],
        "out_entities": sum(len(r["entities"]) for r in kept),
        "kept_ratio": round(len(kept) / max(1, len(rows)), 4),
        "dropped": {k: v for k, v in stat.items()
                    if k.startswith("drop")},
        "by_type": dict(Counter(e["type"] for r in kept for e in r["entities"])),
        "head": args.head, "input": args.inp,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rp = os.path.join(PHASE4, f"pseudo_review_{args.head}.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("[review] " + json.dumps(report, ensure_ascii=False), flush=True)
    print(f"[review] → {out}", flush=True)


if __name__ == "__main__":
    main()
