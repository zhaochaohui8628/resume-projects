"""第一轮数据的**高置信度清洗**：用第一轮模型复核第一轮训练数据，保留子集。

第二轮全量微调的数据 = 本脚本产物（第一轮高置信度数据）
                     + 无标注池挖掘的高置信度伪标（`pseudo_clean_{head}.jsonl`）
                     + `gold_train_portion` 500（LLM 交叉校验 + 多领域专家）

筛选判据（逐条可关）
--------------------
1. **一致**：模型预测的字符实体集合 == 标注集合（严格口径：类型 + span 完全一致）；
2. **实体 span**：span 内全部 token 的 margin ≥ `--sp-margin`；
3. **O 侧**：margin 中位数 ≥ `--o-margin`；
4. **CRF**：Viterbi 路径 == 发射 argmax（`--no-consistency` 可关）。

输出**保留原始标注**（已被 LLM 清洗过，模型只做"可信度背书"），额外写 `min_margin` 便于追溯。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.bert.data_utils import load_jsonl  # noqa: E402
from ner2.src.bert.decode import token_tags_to_char_entities  # noqa: E402
from ner2.src.bert.engine import load_ckpt, predict_batch  # noqa: E402
from ner2.src.common.paths import BASE_BERT, PHASE4, SILVER_TRAIN  # noqa: E402


def _key(ents):
    return {(e["type"], int(e["start"]), int(e["end"])) for e in ents}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--head", default=None, choices=["softmax", "crf"])
    ap.add_argument("--data", default=SILVER_TRAIN)
    ap.add_argument("--sp-margin", type=float, default=None)
    ap.add_argument("--o-margin", type=float, default=None)
    ap.add_argument("--no-consistency", action="store_true")
    ap.add_argument("--no-require-agree", action="store_true",
                    help="不要求模型预测与标注完全一致（仅看置信度）")
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ck, _ = load_ckpt(args.ckpt, device=device, override_head=args.head,
                             fallback_base=BASE_BERT)
    head = model.head
    sp = args.sp_margin if args.sp_margin is not None else (0.25 if head == "softmax" else 0.0)
    om = args.o_margin if args.o_margin is not None else (0.15 if head == "softmax" else 0.0)

    rows = load_jsonl(args.data)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_BERT)
    preds = []
    for i in range(0, len(rows), 512):
        preds.extend(predict_batch(model, tokenizer, [r["text"] for r in rows[i:i + 512]],
                                  max_len=args.max_len, batch=args.batch, device=device))
    stat = Counter()
    kept = []
    for r, p in zip(rows, preds):
        stat["total"] += 1
        if head == "crf" and not args.no_consistency and p.get("crf_consistent") is False:
            stat["drop_inconsistent"] += 1
            continue
        pe = token_tags_to_char_entities(p["offsets"], p["tags"])
        if not args.no_require_agree and _key(pe) != _key(r.get("entities", [])):
            stat["drop_disagree"] += 1
            continue
        m = p["margins"]
        bad = any(m[i] < sp for i, tg in enumerate(p["tags"])
                  if tg.startswith(("B-", "I-", "E-")) and i < len(m))
        if bad:
            stat["drop_span_margin"] += 1
            continue
        o_ms = sorted(m[i] for i, tg in enumerate(p["tags"]) if tg == "O")
        if o_ms and o_ms[len(o_ms) // 2] < om:
            stat["drop_o_margin"] += 1
            continue
        row = {"text": r["text"], "entities": r.get("entities", []), "src": "hc_round1",
               "min_margin": round(min(m) if m else 0.0, 4)}
        kept.append(row)
        stat["kept"] += 1

    report = {"head": head, "ckpt": args.ckpt, "data": args.data,
              "sp_margin": sp, "o_margin": om, "n_in": len(rows), "out": len(kept),
              "out_entities": sum(len(r["entities"]) for r in kept),
              "by_type": dict(Counter(e["type"] for r in kept for e in r["entities"])),
              **{k: v for k, v in stat.items()}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sp_out = args.stats_out or os.path.join(os.path.dirname(args.out), f"filter_round1_{head}.json")
    with open(sp_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("[filter_round1] " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
