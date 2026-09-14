"""ner2 全量分块实体抽取 CLI（txt / pdf → 去重实体清单 + 统计）。

用法（项目根，GPU 环境）：
    $PY ner2/scripts/extract_full.py 方案.txt [--out 实体.jsonl] [--rule-only] [--batch 64]
    $PY ner2/scripts/extract_full.py 方案.pdf [--out 实体.jsonl]
    $PY ner2/scripts/extract_full.py 方案.txt --positions --out 全量.jsonl
                                          # --positions：输出带句位置的完整实体（默认去重）

与旧 agent 三层漏斗完全不同：**完整文本全量分块送入 ner2 级联管道**（规则层 +
微调模型层，规则优先），不筛选、不收敛、不截断。10 万字符实测 ~4.7s（RTX 4060 Ti）。

**去重收尾（默认）**：10 万字方案同一实体常跨句重复出现，CLI 默认按 (type, text)
聚合输出唯一实体清单（含 count/首次句号/前 3 句佐证）；需要定位时加 --positions。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.pipeline.full_text import DEFAULT_MODEL, FullTextExtractor, dedup_entities  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="方案 .txt / .pdf 路径")
    ap.add_argument("--out", default="", help="实体输出 jsonl（默认不落盘，仅打印统计）")
    ap.add_argument("--positions", action="store_true",
                    help="输出带句位置的完整实体列表（默认：按 type+text 去重聚合）")
    ap.add_argument("--model", default=None, help="模型 ckpt（默认 s2_crf_param_v3）")
    ap.add_argument("--rule-only", action="store_true", help="仅规则/词典层（零依赖）")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=96)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"[error] 输入不存在：{args.input}")

    ext = os.path.splitext(args.input)[1].lower()
    model_path = None if args.rule_only else (args.model or DEFAULT_MODEL)
    fx = FullTextExtractor(model_path=model_path, batch=args.batch, max_len=args.max_len)

    t0 = time.time()
    if ext == ".pdf":
        ents = fx.extract_pdf(args.input)
    else:
        with open(args.input, encoding="utf-8") as f:
            ents = fx.extract_text(f.read())
    dt = time.time() - t0

    by_type = Counter(e["type"] for e in ents)
    print(f"[extract_full] {os.path.basename(args.input)} → 全量实体 {len(ents)} 个 | "
          f"耗时 {dt:.2f}s | 分类型 {dict(by_type)}")
    if fx._model is not None:
        print(f"[extract_full] 层级 规则{sum(1 for e in ents if e['layer']=='rule')} / "
              f"模型{sum(1 for e in ents if e['layer']=='model')}")

    if args.positions:
        out_rows, label = ents, "全量(含句位置)"
    else:
        out_rows = dedup_entities(ents)
        label = f"去重 {len(out_rows)}"
        dup_rate = 1 - len(out_rows) / max(1, len(ents))
        print(f"[extract_full] 去重后唯一实体 {len(out_rows)} 个 | 重复率 {dup_rate:.1%}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for e in out_rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[extract_full] → {args.out}（{label}）")
    else:
        for e in out_rows[:12]:
            if args.positions:
                print(f"  [{e['layer']}] {e['type']} = {e['text']}  @句{e['sent_idx']}")
            else:
                print(f"  [{e['type']}] {e['text']}  ×{e['count']}  @句{e['first_sent_idx']}")


if __name__ == "__main__":
    main()