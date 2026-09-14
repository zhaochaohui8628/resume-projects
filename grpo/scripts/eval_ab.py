"""A/B 对比：同一批样本上比较两个权重（如 s1_grpo vs s1_sft）的四维得分。

用法：
    python grpo/scripts/eval_ab.py --models a=path1 b=path2 [--data data/grpo/val.jsonl]
                                   [--n 6] [--max-new-tokens 256]

设计要点：
- **顺序加载**（每次只驻留一个模型）：8G 显存装不下两个 3B bf16。
- 贪心解码（do_sample=False）保证可比性。
- 打分沿用训练期同一 `reward.score_breakdown`（四维连续口径），不引入新口径。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRPO = os.path.dirname(HERE)
ROOT = os.path.dirname(GRPO)
for p in (GRPO,):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as C                          # noqa: E402
from prompts import make_system_prompt      # noqa: E402
from reward import score_breakdown          # noqa: E402


def evidence_text_of(row) -> str:
    ev = row.get("agent_scenario", {}).get("evidence") or []
    return "\n".join(f"- [{e['source']}] {e['clause_no']}: {e['text']}" for e in ev)


def load_rows(path, n):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return rows[:n]


def run_model(path, rows, max_new_tokens, tag):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    model.eval()
    out = []
    for i, row in enumerate(rows):
        msgs = [{"role": "system",
                 "content": make_system_prompt(row["task_type"], evidence_text_of(row))},
                {"role": "user", "content": row["query"]}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(model.device)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
        bd = score_breakdown(text, row["judge_meta"])
        out.append({"id": row.get("id"), "task_type": row.get("task_type"),
                    "reward": bd["total"],
                    "breakdown": {k: bd[k] for k in ("format", "cot", "basis", "answer")},
                    "text": text})
        print(f"  [{tag}] {i + 1}/{len(rows)} id={row.get('id')} reward={bd['total']:.3f} "
              f"fmt={bd['format']:.2f} cot={bd['cot']:.2f} basis={bd['basis']:.2f} "
              f"ans={bd['answer']:.2f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="形如 tag=路径，可多个（顺序执行）")
    ap.add_argument("--data", default=os.path.join(C.DATA_DIR, "val.jsonl"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(C.DATA_DIR, "eval_ab.json"))
    args = ap.parse_args()

    rows = load_rows(args.data, args.n)
    print(f"[eval_ab] {len(rows)} 条样本 | 模型 {len(args.models)} 个", flush=True)
    result = {}
    for spec in args.models:
        tag, path = spec.split("=", 1)
        print(f"[eval_ab] === {tag} ({path}) ===", flush=True)
        result[tag] = run_model(path, rows, args.max_new_tokens, tag)

    # 汇总
    print("\n[eval_ab] 汇总", flush=True)
    dims = ("format", "cot", "basis", "answer")
    print(f"{'模型':<10} {'reward':>8} " + " ".join(f"{d:>8}" for d in dims), flush=True)
    for tag, items in result.items():
        r = sum(x["reward"] for x in items) / len(items)
        ds = {d: sum(x["breakdown"][d] for x in items) / len(items) for d in dims}
        print(f"{tag:<10} {r:>8.3f} " + " ".join(f"{ds[d]:>8.3f}" for d in dims), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in result.items()}, f, ensure_ascii=False, indent=2)
    print(f"[eval_ab] 明细 -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
