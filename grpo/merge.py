"""LoRA merge 工具：微调结束将 adapter 合并进主权重并保存（权重链核心）。

低内存约束（本机 16G RAM / 8G 显存 / Windows WDDM 实测）：
1. **不能** `merged.to("cpu")` —— 一次性 6G 常驻 RAM，直接 segfault。
2. **不能** 让 `save_pretrained` 直接吃 GPU 上的 state_dict —— 大块 GPU→CPU
   传输在 WDDM 下同样 segfault（已实测崩在 "Writing model shards 0/7"）。
3. 正解：**逐参数**搬 CPU（单个张量 ≤0.6G）→ 累积到 ~1G 分片 →
   `safe_save_file` 落盘 → 立即释放，峰值 CPU ≈ 1G。
"""
from __future__ import annotations

import json
import math
import os

import torch
from safetensors.torch import save_file as safe_save_file


def _save_model_lowmem(model, save_dir: str, shard_gb: float = 1.0) -> None:
    """逐参数搬 CPU + 分片落盘（峰值 CPU ≈ 1 个分片）。"""
    os.makedirs(save_dir, exist_ok=True)
    limit = int(shard_gb * 1024 ** 3)
    state = model.state_dict()

    # 先按张量大小规划分片，确定总分片数（不搬数据）
    sizes = {k: int(v.numel() * v.element_size()) for k, v in state.items()}
    total = sum(sizes.values())
    n_shards = max(1, math.ceil(total / limit))

    weight_map, cur, cur_sz, si = {}, {}, 0, 0

    def flush():
        nonlocal cur, cur_sz, si
        if not cur:
            return
        fn = f"model-{si + 1:05d}-of-{n_shards:05d}.safetensors"
        path = os.path.join(save_dir, fn)
        for k in cur:
            weight_map[k] = fn
        safe_save_file(cur, path, metadata={"format": "pt"})
        print(f"  分片落盘 {fn} ({cur_sz / 1024 ** 3:.2f}G, {len(cur)} 张量)", flush=True)
        del cur
        cur = {}
        cur_sz = 0
        si += 1

    cur = {}
    for k in list(state.keys()):
        t = state.pop(k).detach().to("cpu", torch.bfloat16).contiguous()
        if cur_sz + sizes[k] > limit and cur:
            flush()
            cur, cur_sz = {}, 0
        cur[k] = t
        cur_sz += sizes[k]
    flush()

    # 索引文件（Python IO 写，兼容非 ASCII 路径）
    if n_shards > 1:
        with open(os.path.join(save_dir, "model.safetensors.index.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
                      f, ensure_ascii=False, indent=2)
    # config / generation_config（transformers 内部走 Python IO，安全）
    model.config.save_pretrained(save_dir)
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.save_pretrained(save_dir)
    print(f"[merge] 保存完成 {n_shards} 个分片 -> {save_dir}", flush=True)


def merge_and_save(model, tokenizer, save_dir: str, adapter_dir: str | None = None) -> str:
    """merge_and_unload 后保存为纯基座权重，作为下一阶段的新基座。

    顺序很重要：**先存 adapter**（~230MB，绝对安全）再 merge 存全量权重。
    这样即便全量保存失败，本轮训练成果（adapter）仍在，可离线重跑 merge。
    """
    # 1) 安全网：先落 adapter
    if adapter_dir:
        os.makedirs(adapter_dir, exist_ok=True)
        try:
            model.save_pretrained(adapter_dir)
            print(f"[merge] adapter 已保存 -> {adapter_dir}", flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"[merge][warn] adapter 保存失败: {type(e).__name__}: {e}", flush=True)

    # 2) merge + 低内存分片保存
    merged = model.merge_and_unload().eval()   # eval 去 dropout，保证与重载一致
    try:
        _save_model_lowmem(
            merged, save_dir,
            shard_gb=float(os.environ.get("GRPO_SHARD_GB", "1.0")))
    except Exception as e:                                       # noqa: BLE001
        print(f"[merge][warn] 全量权重保存失败（adapter 已保存，可离线重跑）: "
              f"{type(e).__name__}: {e}", flush=True)
        return adapter_dir or save_dir

    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)
    del merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return save_dir
