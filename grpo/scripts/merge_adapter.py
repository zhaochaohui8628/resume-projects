"""把已训练的 LoRA adapter merge 进基座（幂等重跑 merge，绕开训练主流程的保存崩溃）。

用法：python grpo/scripts/merge_adapter.py --base data/models/qwen-grpo/final \
            --adapter data/models/qwen-grpo/r2_grpo_adapter \
            --output data/models/qwen-grpo/r2_grpo
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(ROOT, "grpo"),):
    if p not in sys.path:
        sys.path.insert(0, p)
from merge import _save_model_lowmem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base_m = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    print(f"[merge] 加载基座完成 {args.base}")
    m = PeftModel.from_pretrained(base_m, args.adapter)
    merged = m.merge_and_unload().eval()
    print("[merge] merge 完成，开始低内存分片保存")
    _save_model_lowmem(merged, args.output,
                       shard_gb=float(os.environ.get("GRPO_SHARD_GB", "1.0")))
    tok.save_pretrained(args.output)
    print(f"[merge] 完成 -> {args.output}")


if __name__ == "__main__":
    main()
