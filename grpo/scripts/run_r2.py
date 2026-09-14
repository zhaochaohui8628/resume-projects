"""第二轮迭代串跑：GRPO -> SFT -> GRPO（全自动衔接，失败即停）。

起点 = 第一轮终点 `final`；产物 r2_grpo / r2_sft / r2_final。
评分器已修「中文数字 vs 阿拉伯数字」归一化缺陷（见 reward.py::_digits_norm），
本轮起 basis 奖励信号才是准确的。

第一段用 s1 风格（T=1.0 / beta=0.04，探索更强），第二段用 s2 风格
（T=0.9 / beta=0.06，收敛更稳），与首轮权重链的调度保持一致。

用法：python grpo/scripts/run_r2.py   （在项目根目录）
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

COMMON = ["--query-batch", "1", "--G", "4", "--max-new-tokens", "192", "--mem-log"]

JOBS = [
    ("R2-1 GRPO(final -> r2_grpo)", [
        PY, "-u", "grpo/scripts/train_grpo.py", "--stage", "s1",
        "--base-model", f"{M}/final",
        "--output", f"{M}/r2_grpo",
        "--traj", f"{D}/trajectories_r2a.jsonl",
        "--steps", "30", *COMMON,
    ], "grpo/scripts/r2_1_grpo.log"),
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
        "--steps", "30", *COMMON,
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
    print(f"\n[ALL DONE] 第二轮三阶段完成，总耗时 {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
