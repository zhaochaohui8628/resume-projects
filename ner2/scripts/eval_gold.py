"""P5 黄金测试集评估：对若干检查点在 `gold_test`（543 句）上出严格/宽容 F1 + 分类型 + 95% CI。

口径
----
- **严格**：类型 + 字符 span 完全一致；**宽容**：类型一致 + span 有重叠。
- 语料级 micro P/R/F1（逐句 tp/fp/fn 累加）；95% CI 用**句级 bootstrap**（有放回重采样）。
- 可附带「层1 规则/词典」基线（纯规则级联，零模型）作对照。

用法
----
    $PY ner2/scripts/eval_gold.py \
        --model s1_softmax_stage1=ner2/models/s1_softmax_stage1/model.pt \
        --model s1_softmax_stage2=ner2/models/s1_softmax_stage2/model.pt \
        --model s2_crf_stage1=ner2/models/s2_crf_stage1/model.pt \
        --model s2_crf_stage2=ner2/models/s2_crf_stage2/model.pt \
        --with-rule-baseline \
        --outmd ner2/data/phase4/eval_gold_report.md \
        --outjson ner2/data/phase4/eval_gold_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.bert.data_utils import load_jsonl  # noqa: E402
from ner2.src.bert.engine import corpus_eval, load_ckpt  # noqa: E402
from ner2.src.common.paths import BASE_BERT, GOLD_TEST, PHASE4  # noqa: E402
from ner2.src.eval.metrics import evaluate as eval_entities  # noqa: E402


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def _sent_counts(preds, golds, relaxed: bool):
    """逐句 (tp,fp,fn) 列表，供 bootstrap 重采样。"""
    out = []
    for pe, g in zip(preds, golds):
        m = eval_entities(pe, g, relaxed=relaxed)
        out.append((m["tp"], m["fp"], m["fn"]))
    return out


def _agg(counts) -> tuple:
    tp = sum(c[0] for c in counts)
    fp = sum(c[1] for c in counts)
    fn = sum(c[2] for c in counts)
    return tp, fp, fn


def _bootstrap_ci(counts, n_boot: int, seed: int = 42):
    rng = random.Random(seed)
    n = len(counts)
    f1s, ps, rs = [], [], []
    for _ in range(n_boot):
        sample = [counts[rng.randrange(n)] for _ in range(n)]
        tp, fp, fn = _agg(sample)
        m = _prf(tp, fp, fn)
        f1s.append(m["f1"])
        ps.append(m["precision"])
        rs.append(m["recall"])
    f1s.sort(); ps.sort(); rs.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return {"f1_ci95": [round(f1s[lo], 4), round(f1s[hi], 4)],
            "precision_ci95": [round(ps[lo], 4), round(ps[hi], 4)],
            "recall_ci95": [round(rs[lo], 4), round(rs[hi], 4)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[],
                    help="name=ckpt_path（可重复）")
    ap.add_argument("--data", default=GOLD_TEST)
    ap.add_argument("--with-rule-baseline", action="store_true")
    ap.add_argument("--lexicon", default=None)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--device", default="")
    ap.add_argument("--outmd", default=os.path.join(PHASE4, "eval_gold_report.md"))
    ap.add_argument("--outjson", default=os.path.join(PHASE4, "eval_gold_metrics.json"))
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_jsonl(args.data)
    golds = [r.get("entities", []) for r in rows]
    texts = [r["text"] for r in rows]
    print(f"[eval] gold_test {len(rows)} 句 / {sum(len(g) for g in golds)} 实体 / device={device}",
          flush=True)

    result = {"data": args.data, "n_sent": len(rows),
              "n_gold_entities": sum(len(g) for g in golds), "models": {}}

    # ---- 规则/词典基线（纯层1）----
    if args.with_rule_baseline:
        from ner2.src.pipeline.cascade import CascadeExtractor, default_lexicon_path
        lex_path = args.lexicon or default_lexicon_path()
        ext = CascadeExtractor.from_lexicon_file(lex_path)
        preds = [ext.extract(t) for t in texts]
        entry = {"lexicon": lex_path, "ckpt": None, "head": "rule"}
        for key, relaxed in (("strict", False), ("lenient", True)):
            cnt = _sent_counts(preds, golds, relaxed)
            m = _prf(*_agg(cnt))
            m.update(_bootstrap_ci(cnt, args.bootstrap))
            entry[key] = m
        result["models"]["L1_rule_lexicon"] = entry
        print(f"[eval] L1_rule_lexicon strict F1={entry['strict']['f1']:.4f}", flush=True)

    # ---- 各检查点 ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_BERT)
    for spec in args.model:
        if "=" not in spec:
            raise SystemExit(f"[error] --model 需 name=path 形式：{spec}")
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"[eval] 跳过 {name}（缺 {path}）", flush=True)
            continue
        model, ck, info = load_ckpt(path, device=device, fallback_base=BASE_BERT)
        ev, preds, _ = corpus_eval(model, tokenizer, rows, max_len=args.max_len,
                                   batch=args.batch, device=device, return_preds=True)
        entry = {"ckpt": path, "head": ck.get("head"), "freeze": ck.get("freeze"),
                 "best_val_f1": ck.get("best_val_f1"),
                 "trainable_params": ck.get("trainable_params")}
        for key, relaxed in (("strict", False), ("lenient", True)):
            cnt = _sent_counts(preds, golds, relaxed)
            m = _prf(*_agg(cnt))
            m.update(_bootstrap_ci(cnt, args.bootstrap))
            entry[key] = m
        entry["per_type_strict"] = ev["per_type"]["strict"]
        entry["per_type_lenient"] = ev["per_type"]["lenient"]
        result["models"][name] = entry
        print(f"[eval] {name} strict F1={entry['strict']['f1']:.4f} "
              f"({entry['strict']['f1_ci95']}) | lenient F1={entry['strict']['f1']:.4f} "
              f"-> {entry['lenient']['f1']:.4f}", flush=True)

    # ---- 落盘 ----
    os.makedirs(os.path.dirname(os.path.abspath(args.outjson)), exist_ok=True)
    with open(args.outjson, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    lines = ["# P5 黄金测试集评估（ner2 BERT 两方案 × 两轮）", "",
             f"- 测试集：`{os.path.basename(args.data)}` — **{len(rows)} 句 / "
             f"{sum(len(g) for g in golds)} 实体**（LLM 交叉校验 + 多专家投票共识，零泄漏于训练）",
             f"- 口径：严格 = 类型 + 字符 span 完全一致；宽容 = 类型一致 + span 重叠",
             f"- 95% CI：句级 bootstrap {args.bootstrap} 次", "",
             "## 总体指标", "",
             "| 模型 | 严格 P | 严格 R | 严格 F1 | 严格 F1 95%CI | 宽容 P | 宽容 R | 宽容 F1 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, e in result["models"].items():
        s, ln = e["strict"], e["lenient"]
        lines.append(f"| {name} | {s['precision']:.4f} | {s['recall']:.4f} | **{s['f1']:.4f}** | "
                     f"[{s['f1_ci95'][0]:.4f}, {s['f1_ci95'][1]:.4f}] | "
                     f"{ln['precision']:.4f} | {ln['recall']:.4f} | {ln['f1']:.4f} |")
    lines += ["", "## 分类型（严格 F1）", "",
              "| 模型 | 工程类型 | 工序 | 设备 | 参数 | 规范编号 | 危大类别 |", "|---|---|---|---|---|---|---|"]
    for name, e in result["models"].items():
        pt = e.get("per_type_strict") or {}
        cells = []
        for t in ("工程类型", "工序", "设备", "参数", "规范编号", "危大类别"):
            m = pt.get(t)
            cells.append(f"{m['f1']:.3f}(n={m['tp'] + m['fn']})" if m else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    with open(args.outmd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[eval] 报告 → {args.outmd}", flush=True)
    print(f"[eval] 指标 → {args.outjson}", flush=True)


if __name__ == "__main__":
    main()
