"""Qwen2.5-3B 低内存加载器：绕过 transformers 5.x 的 safetensors 整体 mmap。

问题：transformers 5.x `from_pretrained` 对 safetensors 用 `safe_open(device="cpu")`
整体 mmap，在 8G 显存 + 16G 内存的 Windows 上会 `OSError 1455 页面文件太小`
（mmap 5.75G 需要连续虚拟地址空间）。

解法：用 safetensors 库**逐个张量**读 + 逐个搬到目标 device（GPU），
每读一个张量就释放 CPU 侧引用，峰值 CPU 内存只占单个大张量（如 embed 0.05G），
不再需要 mmap 整个文件。

用法：
    from load_qwen import load_qwen_lowmem
    model, tok = load_qwen_lowmem("data/models/Qwen2.5-3B-Instruct", device="cuda")
"""
from __future__ import annotations

import json
import os
from typing import Optional

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _iter_shard_tensors(shard_path: str, device: str = "cuda",
                        dtype: torch.dtype = torch.bfloat16):
    """逐张量流式读分片：CPU 取单张量 → 搬目标 device → 交出 → 立即释放。

    路径演进（本机 16G RAM / 8G 显存实测）：
    - transformers `from_pretrained`：safetensors 整体 mmap → OSError 1455 → 弃。
    - `safe_open(device="cuda")` 直读：safetensors CUDA 后端 storage bug → 弃。
    - 缓存整个分片 dict 再搬：峰值叠加 → segfault 139 → 弃。
    - **正解**：生成器逐张量产出（`get_tensor` 只映射单张量，不 mmap 整文件），
      消费完即释放，CPU 侧峰值 ≈ 单张量（embed 最大 ~0.6G）。
    """
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            try:
                out = t.to(device=device, dtype=dtype)
            finally:
                del t
            yield k, out


def load_qwen_lowmem(model_dir: str, device: str = "cuda",
                     dtype: torch.dtype = torch.bfloat16,
                     tokenizer_dir: Optional[str] = None):
    """逐张量加载 Qwen 模型到指定设备，规避整体 mmap。

    device: "cuda"（默认）或 "cpu"（纯 CPU 推理用）。
    """
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted({os.path.join(model_dir, v) for v in weight_map.values()})
    else:  # 单分片
        st = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
        shards = [os.path.join(model_dir, st[0])]

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # 锁 dtype：meta 壳的 to_empty 按 config.torch_dtype 分配，必须是 bf16（否则
    # fp32 空壳 11.5G 直接吃满 8G 显存）。
    config.torch_dtype = dtype
    if hasattr(config, "dtype"):
        config.dtype = dtype
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    # 空壳落在目标 device（bf16 → 5.75G），随后原地 copy_ 不额外增量
    model = model.to_empty(device=device)
    model = model.to(dtype)
    if getattr(config, "tie_word_embeddings", False):
        model.tie_weights()

    # 逐张量原地填充：copy_ 不新增显存（除读入的单张量）
    param_map = {n: p for n, p in model.named_parameters()}
    buffers = {n: b for n, b in model.named_buffers()}
    n_hit = n_miss = 0
    for shard in shards:
        print(f"  加载分片 {os.path.basename(shard)} ...", flush=True)
        for name, tensor in _iter_shard_tensors(shard, device=device, dtype=dtype):
            if name in param_map:
                param_map[name].data.copy_(tensor)
                n_hit += 1
            elif name in buffers:
                buffers[name].data.copy_(tensor)
            else:
                n_miss += 1
            del tensor
        if device == "cuda":
            torch.cuda.empty_cache()
    print(f"  张量命中 {n_hit} / 未匹配 {n_miss}", flush=True)
    model.eval()
    tok_dir = tokenizer_dir or model_dir
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[load_qwen_lowmem] 加载完成 -> {device} | 显存 {torch.cuda.memory_allocated()/1024**3:.2f}G", flush=True)
    return model, tok


if __name__ == "__main__":
    import sys
    model, tok = load_qwen_lowmem("data/models/Qwen2.5-3B-Instruct")
    msgs = [{"role": "system", "content": "你是施工合规助手。"},
            {"role": "user", "content": "某基坑开挖深度4米，是否属于危大工程？"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(input_ids=ids, max_new_tokens=30, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    print("生成:", tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True))
    print(f"峰值显存: {torch.cuda.max_memory_allocated()/1024**3:.2f}G / 8G")
