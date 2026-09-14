"""P4 高置信度伪标挖掘：用第一轮模型对无标注池预测并按置信度筛选（第二轮数据来源之一）。

筛选口径（用户 2026-09-11 定稿）
--------------------------------
- `--head softmax`：**逐 token 相对置信度** margin = p_top1 − p_top2
    · 实体 span 内全部 token 的 margin ≥ `--sp-margin`（默认 0.25）
    · O 侧 token 的 margin 中位数 ≥ `--o-margin`（默认 0.15）
    —— 沿用旧 ner `self_train._decode_entities_margin` 的判据（20 类 softmax 绝对置信有结构性上限）
- `--head crf`：**第一轮高得分解码数据**
    · 归一化解码对数似然 `crf_score_per_token` ≥ 分位阈值（`--crf-percentile`，默认 60）
    · Viterbi 路径 == 发射 argmax 路径（解码稳定；`--no-consistency` 可关）

泄漏防护
--------
池中句子与 silver_train / silver_val / gold_train_portion / gold_test / 旧 gold_eval_v2
逐句 md5 命中即剔除。**gold_test 是测试集，绝不可进任何训练集。**

用法
----
    $PY ner2/scripts/mine_pseudo.py --ckpt ner2/models/s1_softmax_stage1/model.pt \
        --out ner2/data/phase4/pseudo_hc_softmax.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.bert.data_utils import load_jsonl  # noqa: E402
from ner2.src.bert.engine import collect_high_confidence, load_ckpt, predict_batch  # noqa: E402
from ner2.src.common.paths import (  # noqa: E402
    BASE_BERT, GOLD_TEST, GOLD_TRAIN_PORTION, PHASE1, PHASE4, SILVER_TRAIN, SILVER_VAL,
    LEGACY_GOLD,
)


def _h(t: str) -> str:
    return hashlib.md5(t.strip().encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--head", default=None, choices=["softmax", "crf"])
    ap.add_argument("--pool", default=os.path.join(PHASE1, "unlabeled_sentences.jsonl"))
    ap.add_argument("--exclude", nargs="*", default=[SILVER_TRAIN, SILVER_VAL,
                                                     GOLD_TRAIN_PORTION, GOLD_TEST, LEGACY_GOLD])
    ap.add_argument("--sp-margin", type=float, default=None, help="实体 span token margin 下限")
    ap.add_argument("--o-margin", type=float, default=None, help="O 侧 margin 中位下限")
    ap.add_argument("--crf-percentile", type=float, default=60.0,
                    help="CRF 归一化解码得分分位阈值（保留 ≥ 该分位）")
    ap.add_argument("--no-consistency", action="store_true",
                    help="CRF 不要求 Viterbi 路径 == 发射 argmax")
    ap.add_argument("--neg-ratio", type=float, default=0.05, help="保留无实体高置信句比例")
    ap.add_argument("--max-out", type=int, default=0, help="0 = 不限")
    ap.add_argument("--min-len", type=int, default=6, help="过短句子跳过")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--device", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, ck, info = load_ckpt(args.ckpt, device=device, override_head=args.head,
                                fallback_base=BASE_BERT)
    head = model.head
    sp = args.sp_margin if args.sp_margin is not None else (0.25 if head == "softmax" else 0.0)
    om = args.o_margin if args.o_margin is not None else (0.15 if head == "softmax" else 0.0)
    print(f"[mine] head={head} device={device} sp_margin={sp} o_margin={om} "
          f"crf_percentile={args.crf_percentile}", flush=True)

    banned = set()
    for p in args.exclude:
        if p and os.path.exists(p):
            n0 = len(banned)
            banned |= {_h(r["text"]) for r in load_jsonl(p)}
            print(f"[mine] 排除 {os.path.basename(p)}: +{len(banned) - n0}", flush=True)

    seen, pool = set(), []
    for r in load_jsonl(args.pool):
        t = r["text"].strip()
        if len(t) < args.min_len:
            continue
        h = _h(t)
        if h in banned or h in seen:
            continue
        seen.add(h)
        pool.append(t)
    print(f"[mine] 去重去泄漏后候选池 {len(pool)} 句", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_BERT)
    pred_rows = []
    for i in range(0, len(pool), 512):
        pred_rows.extend(predict_batch(model, tokenizer, pool[i:i + 512],
                                       max_len=args.max_len, batch=args.batch, device=device))
        if (i // 512) % 10 == 0:
            print(f"[mine] 预测 {min(i + 512, len(pool))}/{len(pool)}", flush=True)

    stats = {"head": head, "ckpt": args.ckpt, "pool": len(pool),
             "sp_margin": sp, "o_margin": om, "crf_percentile": args.crf_percentile,
             "excluded_hashes": len(banned)}
    if head == "crf":
        scores = sorted(r["crf_score_per_token"] for r in pred_rows)
        k = int(len(scores) * args.crf_percentile / 100.0)
        thr = scores[min(k, len(scores) - 1)]
        stats["crf_threshold"] = round(thr, 6)
        gated = [r for r in pred_rows if r["crf_score_per_token"] >= thr]
        stats["after_score_gate"] = len(gated)
    else:
        gated = pred_rows

    kept, cstat = collect_high_confidence(
        gated, sp_margin=sp, o_margin=om,
        require_crf_consistent=(head == "crf" and not args.no_consistency),
        keep_negatives=args.neg_ratio)
    stats.update(cstat)

    if head == "crf":
        kept.sort(key=lambda r: r.get("crf_score_per_token", 0.0), reverse=True)
    if args.max_out and len(kept) > args.max_out:
        kept = kept[:args.max_out]
    stats["out"] = len(kept)
    stats["out_entities"] = sum(len(r["entities"]) for r in kept)
    stats["by_type"] = dict(Counter(e["type"] for r in kept for e in r["entities"]))
    stats["by_len"] = {"min": min((len(r["text"]) for r in kept), default=0),
                       "max": max((len(r["text"]) for r in kept), default=0),
                       "avg": round(sum(len(r["text"]) for r in kept) / max(1, len(kept)), 1)}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            row = {"text": r["text"], "entities": r["entities"], "src": "pseudo"}
            if "crf_score_per_token" in r:
                row["crf_score_per_token"] = r["crf_score_per_token"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    sp_out = args.stats_out or os.path.join(os.path.dirname(args.out),
                                            f"mine_stats_{head}.json")
    with open(sp_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print("[mine] " + json.dumps(stats, ensure_ascii=False), flush=True)
    print(f"[mine] → {args.out}", flush=True)


if __name__ == "__main__":
    main()
