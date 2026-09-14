"""第二轮串跑 v2：续跑 R2-2 SFT + R2-3 GRPO。

R2-1（final -> r2_grpo 训练）已完成（120 条轨迹 + r2_grpo_adapter），
但全量 merge 在第 5/7 片 segfault（WDDM 保存路径问题，训练本身无碍）。

v2 续跑：
- R2-2 SFT：直接 `final + r2_grpo_adapter`（PEFT 叠加，绕过缺失的 r2_grpo 全量）
          -> r2_sft
- R2-3 GRPO：r2_sft -> r2_final

用法：python grpo/scripts/run_r2_v2.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.environ.get("GRPO_PY", r"C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe")
M = "data/models/qwen-grpo"
D = "data/grpo"

ENV = {
    **os.environ,
    "GRPO_SAFE_SAMPLE": "1",
    "GRPO_LOGP_CHUNK": "1",
    "GRPO_LOGP_TCHUNK": "32",
    "GRPO_INNER": "1",
    "PYTHONIOENCODING": "utf-8",
}

JOBS = [
    ("R2-2 SFT(r2_grpo -> r2_sft)", [
        PY, "-u", "grpo/scripts/train_sft.py",
        "--base-model", f"{M}/r2_grpo",
        "--trajectory", f"{D}/trajectories_r2a.jsonl",
        "--output", f"{M}/r2_sft",
        "--epochs", "5", "--batch", "1", "--lr", "1.5e-5",
    ], "grpo/scripts/r2_2_sft.log"),
    ("R2-3 GRPO(r2_sft -> r2_final)", [
        PY, "-u", "grpo/scripts/train_grpo.py", "--stage", "s2",
        "--base-model", f"{M}/r2_sft",
        "--output", f"{M}/r2_final",
        "--traj", f"{D}/trajectories_r2b.jsonl",
        "--steps", "30",
        "--query-batch", "1", "--G", "4", "--max-new-tokens", "192", "--mem-log",
    ], "grpo/scripts/r2_3_grpo.log"),
]


def main() -> int:
    t0 = time.time()
    for name, cmd, log in JOBS:
        print(f"\n{'=' * 70}\n[{time.strftime('%H:%M:%S')}] {name}\n{'=' * 70}", flush=True)
        log_path = os.path.join(ROOT, log)
        with open(log_path, "w", encoding="utf-8") as fh:
            p = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=fh,
                                 stderr=subprocess.STDOUT)
            rc = p.wait()
        print(f"[{time.strftime('%H:%M:%S')}] exit={rc} | log -> {log} "
              f"| 累计 {(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            print(f"[ABORT] {name} 失败（exit={rc}），停止串跑", flush=True)
            return rc
    print(f"\n[ALL DONE] R2 续跑完成，总耗时 {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
