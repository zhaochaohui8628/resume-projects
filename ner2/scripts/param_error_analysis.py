"""参数类专项：**不使用测试集**的错误分析（在 gold_train_portion + silver_val 上做）。

目的：定位当前最好模型（s2_crf_stage2）在「参数」上的具体错法，据此构造专项数据。
分类：
  exact         完全一致（类型+span）
  boundary      类型对、span 部分重叠（边界偏差：吞词/截断）
  missed        gold 有、模型完全没给（漏标）
  spurious      模型给了、gold 没有（误报）
  type_confused 模型在同一位置给了**别的类型**（类型混淆）
输出 `data/phase6/param_error_analysis.md` + `param_error_diff.jsonl`。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.bert.engine import load_ckpt, predict_entities  # noqa: E402
from ner2.src.common.paths import (  # noqa: E402
    BASE_BERT, GOLD_TRAIN_PORTION, PHASE6, SILVER_VAL,
)


def _ov(a, b) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ner2/models/s2_crf_stage2/model.pt")
    ap.add_argument("--data", nargs="*", default=[GOLD_TRAIN_PORTION, SILVER_VAL])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default=os.path.join(PHASE6, "param_error_analysis.md"))
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ck, _ = load_ckpt(args.ckpt, device=device, fallback_base=BASE_BERT)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_BERT)

    rows = []
    for p in args.data:
        rows.extend(json.loads(l) for l in open(p, encoding="utf-8") if l.strip())
    preds, _ = predict_entities(model, tok, rows, max_len=ck.get("max_len", 96),
                                batch=args.batch, device=device)

    stat = Counter()
    diffs = []
    for r, pe in zip(rows, preds):
        text = r["text"]
        gold = [e for e in r.get("entities", []) if e["type"] == "参数"]
        pred = [e for e in pe if e["type"] == "参数"]
        gold_other = [e for e in r.get("entities", []) if e["type"] != "参数"]
        used = set()
        for g in gold:
            stat["gold_param"] += 1
            hit = None
            for i, q in enumerate(pred):
                if i in used:
                    continue
                if q["start"] == g["start"] and q["end"] == g["end"]:
                    hit = ("exact", "", q)
                    used.add(i)
                    break
                if _ov(q, g):
                    hit = ("boundary", "", q)
                    used.add(i)
                    break
            if hit is None:
                # 是否被判成了别的类型
                conf = next((o for o in pe if o["type"] != "参数" and _ov(o, g)), None)
                kind = "type_confused" if conf else "missed"
                diffs.append({"text": text, "kind": kind,
                              "gold": text[g["start"]:g["end"]],
                              "gold_span": [g["start"], g["end"]],
                              "pred": text[conf["start"]:conf["end"]] if conf else None,
                              "pred_type": conf["type"] if conf else None})
            else:
                kind, _, q = hit
                stat[kind] += 1
                if kind == "boundary":
                    diffs.append({"text": text, "kind": "boundary",
                                  "gold": text[g["start"]:g["end"]],
                                  "gold_span": [g["start"], g["end"]],
                                  "pred": text[q["start"]:q["end"]],
                                  "pred_span": [q["start"], q["end"]]})
        for i, q in enumerate(pred):
            if i not in used:
                stat["spurious"] += 1
                diffs.append({"text": text, "kind": "spurious",
                              "pred": text[q["start"]:q["end"]],
                              "pred_span": [q["start"], q["end"]]})
        # gold 参数 被别的类型覆盖计数
        for o in gold_other:
            stat["gold_other_types"] += 1

    tp = stat["exact"]
    fn = stat["missed"] + stat["boundary"] + stat["type_confused"]
    fp = stat["spurious"] + stat["boundary"] + stat["type_confused"]
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0

    lines = ["# 参数类专项 · 错误分析（train 侧，未用测试集）", "",
             f"- 模型：`{args.ckpt}`（head={ck.get('head')}, val F1={ck.get('best_val_f1')}）",
             f"- 数据：{' + '.join(os.path.basename(p) for p in args.data)}（{len(rows)} 句）",
             f"- gold 参数 **{stat['gold_param']}**：完全命中 {tp}｜边界偏差 {stat['boundary']}｜"
             f"漏标 {stat['missed']}｜类型混淆 {stat['type_confused']}",
             f"- 误报（gold 无、模型给）**{stat['spurious']}**",
             f"- **参数 P={p:.4f} R={rc:.4f} F1={f1:.4f}**", "",
             "## 错误明细（按类型分组）", ""]
    for kind in ("missed", "type_confused", "boundary", "spurious"):
        items = [d for d in diffs if d["kind"] == kind]
        lines += [f"### {kind}（{len(items)}）", "",
                  "| 原文（【】=gold 参数，«»=模型给的其他类型） | 模型输出 |", "|---|---|"]
        for d in items[:120]:
            marks = []
            if d.get("gold_span"):
                marks.append((d["gold_span"][0], d["gold_span"][1], "【", "】"))
            if d.get("pred_span"):
                marks.append((d["pred_span"][0], d["pred_span"][1], "«", "»"))
            t = d["text"]
            for s, e, a, b in sorted(marks, key=lambda x: -x[0]):     # 从后往前插，保偏移有效
                t = t[:s] + a + t[s:e] + b + t[e:]
            predcell = (f"`{d['pred_type']}:{d['pred']}`" if d["kind"] == "type_confused"
                        else (f"`{d['pred']}`" if d.get("pred") else "（未给）"))
            lines.append(f"| {t.replace('|', chr(92) + '|')} | {predcell} |")
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(PHASE6, "param_error_diff.jsonl"), "w", encoding="utf-8") as f:
        for d in diffs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[err] gold 参数 {stat['gold_param']}：exact {tp} / boundary {stat['boundary']} / "
          f"missed {stat['missed']} / type_confused {stat['type_confused']} / spurious {stat['spurious']}")
    print(f"[err] 参数 P={p:.4f} R={rc:.4f} F1={f1:.4f} → {args.out}")


if __name__ == "__main__":
    main()
