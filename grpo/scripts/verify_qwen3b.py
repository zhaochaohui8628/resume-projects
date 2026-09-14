"""验证 Qwen2.5-3B-Instruct 完整性（不加载，适配 16G 内存机）。

本机（16G 内存 + 小 pagefile）加载 5.75G bf16 模型会 OSError 1455（页文件太小），
属**内存限制非文件问题**。故：
  1) 分片完整性（index 与磁盘一致）
  2) safetensors 头解析（文件是合法 safetensors，权重元数据可读）
  3) 可选 `--load` 才做显存加载测试（需 32G 内存机 / 或增大 pagefile）
"""
import os
import sys
import json
import argparse

MODEL = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "models", "Qwen2.5-3B-Instruct"))


def check_files():
    required = ["config.json", "model.safetensors.index.json", "tokenizer.json",
                "tokenizer_config.json", "generation_config.json", "vocab.json", "merges.txt"]
    for f in required:
        p = os.path.join(MODEL, f)
        if not os.path.exists(p):
            print(f"[verify] 缺文件: {f}")
            return False
    ws = [f for f in sorted(os.listdir(MODEL))
          if f.startswith("model-") and f.endswith(".safetensors")]
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json"), encoding="utf-8"))
    expect = set(idx["weight_map"].values())
    if set(ws) != expect:
        print(f"[verify] 分片不完整: 期望 {sorted(expect)} 实际 {sorted(ws)}")
        return False
    total = sum(os.path.getsize(os.path.join(MODEL, w)) for w in ws) / 1024**3
    print(f"[verify] 权重分片完整: {len(ws)} 个, 合计 {total:.2f}G")
    return True


def check_safetensors():
    """解析每个 safetensors 头部，验证是合法文件且头部元数据能读。"""
    import struct
    ws = [f for f in sorted(os.listdir(MODEL))
          if f.startswith("model-") and f.endswith(".safetensors")]
    ok = True
    for w in ws:
        p = os.path.join(MODEL, w)
        with open(p, "rb") as fh:
            header = fh.read(8)
            if len(header) < 8:
                print(f"[verify] {w} 文件过短")
                ok = False
                continue
            n = struct.unpack("<Q", header)[0]
            if n <= 0 or n > 10 * 1024 * 1024:
                print(f"[verify] {w} 头长度异常: {n}")
                ok = False
                continue
            fh.seek(8)
            meta = fh.read(n)
            try:
                d = json.loads(meta)
                nt = d.get("__metadata__", {}).get("format", "?")
                keys = sum(1 for k in d if k != "__metadata__")
                print(f"[verify] {w}: safetensors 合法, format={nt}, 张量数={keys}")
            except Exception as e:
                print(f"[verify] {w} 头解析失败: {e}")
                ok = False
    return ok


def load_test():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[verify] 加载 bf16 模型（显存测试，需 32G 内存机）...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        low_cpu_mem_usage=True)
    print(f"[verify] 加载完成, device: {model.device}, 显存: {torch.cuda.memory_allocated()/1024**3:.2f}G")
    msgs = [{"role": "system", "content": "你是施工方案合规审查助手。"},
            {"role": "user", "content": "某基坑开挖深度4米，是否属于危大工程？"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(input_ids=ids, max_new_tokens=30, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"[verify] 生成测试 OK: {text[:60]!r}")
    print(f"[verify] 峰值显存: {torch.cuda.max_memory_allocated()/1024**3:.2f}G / 8G")
    print("[verify] ✅ 模型可用")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true", help="做显存加载测试（需 32G 内存机）")
    args = ap.parse_args()
    if not os.path.isdir(MODEL):
        print(f"[verify] 目录不存在: {MODEL}")
        sys.exit(1)
    if not check_files():
        sys.exit(1)
    if not check_safetensors():
        sys.exit(1)
    print("[verify] ✅ 文件完整且 safetensors 合法")
    if args.load:
        load_test()
