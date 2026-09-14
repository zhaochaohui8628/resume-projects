"""参数专项分块续训编排（抗主机内存崩溃）。

背景：本机 Windows WDDM + 主机内存紧张，全量微调训练中会**非确定性**段错误
（非代码 bug）。`engine._dump` 已原子落盘最佳权重，故崩溃最多丢「正在写的那一
块」；本脚本把 8 轮训练切成每块 ≤2 轮的子进程，崩溃后自动用现有最佳权重续训，
单次崩溃代价 ≤2 轮。

停止条件（任一满足）：
  1) 累计完成轮次 ≥ 8（LR=5e-6 的低学习率专项微调计划轮数）
  2) 全局连续 3 块（≈6 轮）best 严格 F1 无提升

用法（项目根）：
  $PY ner2/scripts/run_param_chunks.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = r"C:/Users/<用户名>/anaconda3/envs/torch_gpu/python.exe"
OUT_DIR = os.path.join(ROOT, "ner2", "models", "s2_crf_param")
TRAIN = os.path.join(ROOT, "ner2", "data", "phase6", "param_train.jsonl")
VAL = os.path.join(ROOT, "ner2", "data", "phase6", "param_val.jsonl")
ACCUM = os.path.join(OUT_DIR, "chunks_history.json")

MAX_EPOCHS = 8        # 累计轮次上限
CHUNK_EPOCHS = 2      # 每块最大轮数
MAX_STALL = 3         # 连续无提升块数上限


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    accum = {"chunks": [], "best_val_f1": -1.0, "best_epoch": 0,
             "completed_epochs": 0, "crashes": 0}
    if os.path.exists(ACCUM):
        accum = json.load(open(ACCUM, encoding="utf-8"))

    stall = 0
    chunk_no = len(accum["chunks"]) + 1
    while accum["completed_epochs"] < MAX_EPOCHS and stall < MAX_STALL:
        ckpt = os.path.join(OUT_DIR, "model.pt")
        if not os.path.exists(ckpt):
            print(f"[chunk{chunk_no}] 无检查点，终止", flush=True)
            break
        cmd = [PY, os.path.join(ROOT, "ner2", "scripts", "train_ner.py"),
               "--head", "crf", "--stage", "2",
               "--init", ckpt,
               "--train", TRAIN, "--val", VAL,
               "--lr", "5e-6", "--epochs", str(CHUNK_EPOCHS),
               "--patience", "1", "--batch", "8", "--accum", "4",
               "--out-dir", OUT_DIR]
        print(f"[chunk{chunk_no}] 续训自 {os.path.basename(ckpt)}"
              f"（已累计 {accum['completed_epochs']}/{MAX_EPOCHS} 轮）", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        crashed = r.returncode != 0
        # 读本块 history（engine 每轮已写，覆盖式，但块内轮次独立）
        hist = []
        hp = os.path.join(OUT_DIR, "history.json")
        if os.path.exists(hp):
            d = json.load(open(hp, encoding="utf-8"))
            hist = d.get("history", [])
        ep_in_block = len(hist)
        # 读检查点最佳（model.pt 内嵌 best_val_f1）
        best_here = -1.0
        try:
            import torch
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            best_here = ck.get("best_val_f1", -1.0)
            del ck
            import gc
            gc.collect()
        except Exception as e:
            print(f"[chunk{chunk_no}] 读检查点失败：{e}", flush=True)

        rec = {"chunk": chunk_no, "epochs": ep_in_block,
               "crashed": crashed, "best_val_f1": round(best_here, 4),
               "loss": hist[-1]["loss"] if hist else None}
        accum["chunks"].append(rec)
        accum["completed_epochs"] += ep_in_block
        if crashed:
            accum["crashes"] += 1
            print(f"[chunk{chunk_no}] 崩溃（rc={r.returncode}），本轮完成 {ep_in_block} 轮"
                  f"，现有权重保留", flush=True)
        if best_here > accum["best_val_f1"] + 1e-6:
            accum["best_val_f1"], accum["best_epoch"] = best_here, accum["completed_epochs"]
            stall = 0
        else:
            stall += 1
            print(f"[chunk{chunk_no}] 无提升（stall {stall}/{MAX_STALL}）", flush=True)
        json.dump(accum, open(ACCUM, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[chunk{chunk_no}] {json.dumps(rec, ensure_ascii=False)}", flush=True)
        chunk_no += 1
        if ep_in_block < CHUNK_EPOCHS and not crashed:
            print("[chunk] 块内早停（patience=1），后续块大概率也无提升，终止", flush=True)
            break

    print("[done] " + json.dumps(accum, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
