"""Phase 3 CLI：训练 CrossEncoder 精排（bge-reranker-base, BCE）。

**本机（CPU-only）不建议跑**：278M 实测约 1 样本/秒，5000 对 ×2 epoch ≈ 3 小时。
请在有 GPU 的机器上执行；device 默认自动探测（有 CUDA 用 CUDA）。

示例：
  python scripts/train_cross_encoder.py \
    --data data/phase2/ce_train.jsonl --out data/models/cross_p1 \
    --device cuda --epochs 3 --batch-size 32 --lr 1e-5 --max-len 256

显存不足：--batch-size 8 --grad-accum 4（等效 batch 32）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import data_dir  # noqa: E402
from src.train.cross_encoder.train import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(data_dir("phase2", "ce_train.jsonl")))
    ap.add_argument("--out", default=str(data_dir("models", "cross_p1")))
    ap.add_argument("--base", default=None, help="基座模型目录，默认 data/models/bge-reranker-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--device", default=None, help="cuda / cpu，默认自动探测")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 对（调试用）")
    a = ap.parse_args()
    train(data=a.data, out_dir=a.out, base_model=a.base, epochs=a.epochs,
          batch_size=a.batch_size, grad_accum=a.grad_accum, lr=a.lr,
          max_len=a.max_len, device=a.device, amp=not a.no_amp, limit=a.limit)


if __name__ == "__main__":
    main()
