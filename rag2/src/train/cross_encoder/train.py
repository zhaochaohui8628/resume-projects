"""CrossEncoder 精排训练：BCE 损失微调 bge-reranker-base（278M）。

**本机（CPU-only）不跑**：实测 278M 在 12 核 CPU 上约 1 样本/秒，
5000 对 × 2 epoch ≈ 3 小时，性价比低。请在 GPU 机器上跑：
  python scripts/train_cross_encoder.py --data data/phase2/ce_train.jsonl --device cuda
默认 device 自动探测（有 CUDA 就用 CUDA），CPU 也能跑只是慢。

超参（经验设定，集中在此处便于调）：
  lr 1e-5（reranker 微调不宜大）、batch 16（显存 >=8G 可开 32）、epochs 3、
  warmup 10% 线性、weight_decay 0.01、max_len 256（query+doc 对）、grad clip 1.0。
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.common.paths import BASE_CROSS_MODEL, data_dir  # noqa: E402


def load_jsonl(p: str | Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def pick_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(*, data: str, out_dir: str, base_model: str | None = None, epochs: int = 3,
          batch_size: int = 16, lr: float = 1e-5, max_len: int = 256,
          warmup_ratio: float = 0.1, seed: int = 42, log_every: int = 20,
          device: str | None = None, amp: bool = True,
          grad_accum: int = 1, limit: int | None = None) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = pick_device(device)
    torch.manual_seed(seed)
    random.seed(seed)
    base = base_model or BASE_CROSS_MODEL
    rows = load_jsonl(data)
    if limit:
        rows = rows[:limit]
    random.shuffle(rows)

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, math.ceil(len(rows) / (batch_size * grad_accum)))
    total = steps_per_epoch * epochs
    warmup = max(1, int(total * warmup_ratio))

    def lr_at(s: int) -> float:
        if s < warmup:
            return lr * (s + 1) / warmup
        return lr * max(0.02, 1 - (s - warmup) / max(1, total - warmup))

    use_amp = amp and dev == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if hasattr(torch, "cuda") else None
    print(f"device={dev} amp={use_amp} pairs={len(rows)} batch={batch_size} "
          f"accum={grad_accum} epochs={epochs} steps={total}")

    hist, step, t0 = [], 0, time.time()
    for ep in range(epochs):
        model.train()
        tot, nb, correct, seen = 0.0, 0, 0, 0
        opt.zero_grad()
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            enc = tok([r["query"] for r in chunk], [r["doc"] for r in chunk],
                      padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            y = torch.tensor([float(r["label"]) for r in chunk], device=dev)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(**enc).logits.squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y) / grad_accum
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (i // batch_size + 1) % grad_accum == 0:
                if use_amp:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()
                step += 1
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
            with torch.no_grad():
                correct += int(((logits > 0).float() == y).sum())
                seen += len(chunk)
            tot += float(loss) * grad_accum
            nb += 1
            if nb % log_every == 0:
                print(f"  ep{ep+1} step {nb}/{steps_per_epoch} loss {tot/nb:.4f} "
                      f"acc {correct/max(1,seen):.3f} lr {lr_at(max(1,step)):.2e} "
                      f"({time.time()-t0:.0f}s)")
        rec = {"epoch": ep + 1, "loss": tot / max(1, nb), "pair_acc": correct / max(1, seen)}
        hist.append(rec)
        print(f"epoch {ep+1}: loss {rec['loss']:.4f} pair_acc {rec['pair_acc']:.3f}")

    # 保存前释放显存/内存（Windows WDDM + safetensors 序列化容易 MemoryError）
    import gc
    opt = None
    scaler = None
    torch.cuda.empty_cache()
    gc.collect()
    model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    with open(out / "train_history.json", "w", encoding="utf-8") as f:
        json.dump({"history": hist, "pairs": len(rows), "epochs": epochs,
                   "batch_size": batch_size, "grad_accum": grad_accum, "lr": lr,
                   "max_len": max_len, "device": dev, "base": base},
                  f, ensure_ascii=False, indent=2)
    print(f"导出精排模型 -> {out}")
    return {"history": hist, "out_dir": str(out)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(data_dir("phase2", "ce_train.jsonl")))
    ap.add_argument("--out", default=str(data_dir("models", "cross_p1")))
    ap.add_argument("--base", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    train(data=a.data, out_dir=a.out, base_model=a.base, epochs=a.epochs,
          batch_size=a.batch_size, grad_accum=a.grad_accum, lr=a.lr,
          max_len=a.max_len, device=a.device, limit=a.limit)
