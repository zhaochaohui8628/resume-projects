"""SFT（阶段一数据回收，LoRA）：把初步 GRPO 轨迹过滤后作为监督语料。

过滤规则（连续奖励口径，替代旧 0.7 档重建）：
    reward >= 0.95   近满四维：目标文本 = 原样回答（四维几乎全对，直接克隆）
    0.50 <= reward < 0.95  内容部分对但格式/依据/推理有瑕疵：按 judge_meta 金据重建
                    合规 JSON（含 cot_steps）作为目标（教规范格式，学"正确的样子"）
    reward < 0.5     丢弃（答案错误，学它有害）

loss：response-only cross-entropy（user/prompt 段 mask 掉）。
结束：merge LoRA 进主权重 -> s1_sft（作为阶段二 GRPO 的基座与 ref）。

用法（GPU 机，base 应为 s1_grpo 合并权重）：
    python grpo/scripts/train_sft.py --base-model <s1_grpo 路径>
        [--trajectory data/grpo/trajectories.jsonl --epochs 5 --lr 1.5e-5]
"""
from __future__ import annotations

import argparse
import json
import os
import random
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
from reward import build_gold_response      # noqa: E402
from merge import merge_and_save            # noqa: E402


def collect_sft_data(traj_path, keep_min=C.SFT_KEEP_MIN, rebuild_min=C.SFT_REBUILD_MIN,
                     top_k=2, pretrain_ratio=0.1):
    """读轨迹 -> [(messages, target_text), ...]。按用户定稿：

    1) **每组 top_k 高分**：按 group_id 分组，每组按 reward 降序取前 top_k 条。
       reward >= keep_min 原样学（近满四维）；[rebuild_min, keep_min) 金据重建合规 JSON；
       < rebuild_min 丢弃（答案错误，学它有害）。
    2) **混入少量预训练数据**（减少对齐税）：从 rag2 语料随机抽条款，
       组装成"续写条款"通用样本，占比 pretrain_ratio。模型在对齐任务之外仍保留
       通用文本能力，避免 SFT 只学合规 JSON 导致通用能力退化。
    3) 只对 assistant 段计算 CE（encode_pair 已 mask 掉 system/user）。
    """
    if not os.path.exists(traj_path):
        print(f"[sft-data] 轨迹不存在：{traj_path}（先跑 train_grpo.py 产出）")
        return []
    groups = {}
    for line in open(traj_path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        gid = d.get("group_id", "g")          # 无 group_id（旧轨迹）归同一组，退化全选
        groups.setdefault(gid, []).append(d)
    out = []
    n_keep = n_rebuild = n_drop = 0
    n_groups = 0
    for gid, items in groups.items():
        items.sort(key=lambda x: x["reward"], reverse=True)   # 组内高分在前
        sel = items[:top_k]                                    # 取 top_k 高分
        n_groups += 1
        for d in sel:
            r = d["reward"]
            if r >= keep_min:
                target = d["response"]
                n_keep += 1
            elif r >= rebuild_min:
                rebuilt = build_gold_response(d["judge_meta"])
                if not rebuilt:
                    continue
                target = rebuilt
                n_rebuild += 1
            else:
                n_drop += 1
                continue
            if not target.strip():
                n_drop += 1
                continue
            ev = d.get("agent_scenario", {}).get("evidence") or []
            ev_text = "\n".join(f"- [{e['source']}] {e['clause_no']}: {e['text']}" for e in ev)
            msgs = [
                {"role": "system", "content": make_system_prompt(d["task_type"], ev_text)},
                {"role": "user", "content": d["query"]},
                {"role": "assistant", "content": target},
            ]
            out.append((msgs, target))
    # 混入预训练数据（通用条款续写，减少对齐税）
    n_pretrain = int(len(out) * pretrain_ratio / max(1, 1 - pretrain_ratio))
    n_pretrain_actual = 0
    if n_pretrain > 0:
        pretrain_msgs = _pretrain_samples(n_pretrain)
        out.extend(pretrain_msgs)
        n_pretrain_actual = len(pretrain_msgs)
    print(f"[sft-data] 组数 {n_groups} | top{top_k} 回收 原样 {n_keep} / 金据重建 {n_rebuild} "
          f"/ 丢弃 {n_drop} | 混入预训练 {n_pretrain_actual} 条")
    return out


def _pretrain_samples(n):
    """从 rag2 语料抽 n 条条款，组装成"续写条款"通用样本（保持通用能力）。

    样本 = system(通用助手) + user(请续写/概述条款) + assistant(条款正文)。
    条款正文来自真实规范，非合规问答格式——让模型在 SFT 后仍能生成流畅中文/通用文本。
    """
    import random
    corpus = os.path.join(ROOT, "rag2", "data", "corpus", "clauses.jsonl")
    clauses = []
    if os.path.exists(corpus):
        for line in open(corpus, encoding="utf-8"):
            try:
                d = json.loads(line)
                t = (d.get("text") or "").strip()
                if len(t) >= 30:
                    clauses.append(t)
            except Exception:
                continue
    if not clauses:
        return []
    rng = random.Random(123)
    out = []
    for _ in range(n):
        t = rng.choice(clauses)[:300]
        msgs = [
            {"role": "system", "content": "你是建筑施工安全规范领域的专业助手。"},
            {"role": "user", "content": "请概述以下规范条款的核心要求。"},
            {"role": "assistant", "content": t},
        ]
        out.append((msgs, t))
    return out


def _to_ids(out):
    """transformers 5.x 的 apply_chat_template 可能返回 tokenizers.Encoding /
    BatchEncoding / list，统一转成 list[int]。"""
    if isinstance(out, list):
        return list(out)
    if hasattr(out, "ids"):                       # tokenizers.Encoding
        return list(out.ids)
    if hasattr(out, "keys"):                      # BatchEncoding
        v = out["input_ids"]
        v = v[0] if hasattr(v, "shape") and getattr(v, "dim", lambda: 1)() > 1 else v
        return list(v)
    return list(out)


def encode_pair(tok, msgs, target, max_len):
    """全对话编码 + 计算 assistant 起点。返回 (ids, labels, astart)。"""
    ids = _to_ids(tok.apply_chat_template(msgs, tokenize=True))
    # 无 assistant 的用户前缀长度（切 assistant 起点）
    prefix = _to_ids(tok.apply_chat_template(msgs[:-1], tokenize=True))
    astart = len(prefix)
    seq = ids[:max_len]
    labels = [-100] * len(seq)
    for i in range(astart, len(seq)):
        labels[i] = seq[i]
    return seq, labels, min(astart, len(seq))


def collate(batch, tok, max_len):
    """编码一批 -> (ids, labs, attn, astart_min)。

    astart_min = batch 内最早的 assistant 起点，`logits_to_keep` 用它把 logits
    截断到「响应区 + 1」，prompt 区（本项目 p50=644 token）不再产生
    (L_prompt, 151936) 的 logits —— 8G 显存跑 3B 的必要条件。
    """
    seqs, labss, astarts = [], [], []
    for msgs, target in batch:
        s_, l_, a_ = encode_pair(tok, msgs, target, max_len)
        seqs.append(s_); labss.append(l_); astarts.append(a_)
    L = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    labs = torch.full((len(seqs), L), -100, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, (s_, l_) in enumerate(zip(seqs, labss)):
        ids[i, :len(s_)] = torch.tensor(s_, dtype=torch.long)
        labs[i, :len(l_)] = torch.tensor(l_, dtype=torch.long)
        attn[i, :len(s_)] = 1
    return ids, labs, attn, min(astarts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=None, help="默认取 s1_grpo 合并权重（须已训练阶段一）")
    ap.add_argument("--adapter", default=None, help="可选：PEFT adapter 目录（叠加在 base-model 上）")
    ap.add_argument("--trajectory", default=os.path.join(C.DATA_DIR, "trajectories.jsonl"))
    ap.add_argument("--output", default=C.WEIGHT_CHAIN["s1_sft"])
    ap.add_argument("--epochs", type=int, default=C.SFT_EPOCHS)
    ap.add_argument("--batch", type=int, default=int(os.environ.get("SFT_BATCH", "1")))
    ap.add_argument("--lr", type=float, default=C.SFT_LR)
    ap.add_argument("--max-len", type=int, default=C.MAX_SEQ_LEN)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-min", type=float, default=C.SFT_KEEP_MIN,
                    help="轨迹 reward ≥ 此值原样学")
    ap.add_argument("--rebuild-min", type=float, default=C.SFT_REBUILD_MIN,
                    help="轨迹 reward ∈ [rebuild-min, keep-min) 按金据重建合规 JSON 学")
    ap.add_argument("--top-k", type=int, default=2,
                    help="每组采样按 reward 降序取前 top-k 条用于 SFT（用户定稿）")
    ap.add_argument("--pretrain-ratio", type=float, default=0.1,
                    help="混入通用条款续写样本的比例，减少对齐税（默认 10%）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = args.base_model or C.WEIGHT_CHAIN["s1_grpo"]
    if args.dry_run:
        print(f"[dry-run] base={base} 轨迹={args.trajectory} epochs={args.epochs} "
              f"batch={args.batch} lr={args.lr} keep_min={args.keep_min} rebuild_min={args.rebuild_min} "
              f"top_k={args.top_k} pretrain_ratio={args.pretrain_ratio}")
        print(f"[dry-run] merge 输出 -> {args.output}")
        print("[dry-run] 管线验证 OK（未读取轨迹/模型）")
        return
    data = collect_sft_data(args.trajectory, args.keep_min, args.rebuild_min,
                            top_k=args.top_k, pretrain_ratio=args.pretrain_ratio)
    if not data:
        print(f"[empty] 轨迹 {args.trajectory} 无可用样本（先跑阶段一 GRPO 产出轨迹）")
        return
    if not os.path.isdir(base):
        print(f"[error] 基座不存在：{base}（请先完成阶段一 GRPO 并 merge）")
        sys.exit(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed); random.seed(args.seed)
    if args.adapter:
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        m0 = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True)
        model = PeftModel.from_pretrained(m0, args.adapter, is_trainable=True)
        print(f"[sft] 基座 {base} + adapter {args.adapter}（未 merge，直接训练）")
    else:
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        r=C.LORA_RANK, lora_alpha=C.LORA_ALPHA, lora_dropout=C.LORA_DROPOUT,
        target_modules=C.LORA_TARGETS, task_type="CAUSAL_LM"))
    # 8G 显存必需：gradient checkpointing 仅在 train 模式生效（dropout 已置 0）
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    steps_per_epoch = max(1, (len(data) + args.batch - 1) // args.batch)
    total_steps = args.epochs * steps_per_epoch
    step = 0
    for ep in range(args.epochs):
        random.shuffle(data)
        for b in range(steps_per_epoch):
            batch = data[b * args.batch:(b + 1) * args.batch]
            ids, labs, attn, astart = collate(batch, tok, args.max_len)
            ids, labs, attn = ids.to(model.device), labs.to(model.device), attn.to(model.device)
            L = ids.shape[1]
            keep = max(2, min(L - astart + 1, L))     # 只算响应区 logits
            opt.zero_grad()
            logits = model(input_ids=ids, attention_mask=attn,
                           logits_to_keep=keep).logits        # (B, keep, V)
            # logits[j] 预测 ids[L-keep+j+1]；labels 在 [L-keep+1, L) 即响应区
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labs[:, L - keep + 1:].reshape(-1), ignore_index=-100)
            del logits
            loss.backward()
            opt.step()
            step += 1
            if step % 20 == 0 or step == total_steps:
                print(f"[sft] {step}/{total_steps} loss {loss.item():.4f}", flush=True)
    adapter_dir = os.path.join(C.MODELS_ROOT, "qwen-grpo",
                                  f"{os.path.basename(args.output.rstrip(os.sep))}_adapter")
    if args.adapter:
        # 来源是未 merge 的 adapter：先 merge 到 base（final），再按主路径保存
        from peft import PeftModel
        base_m = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True)
        merged = PeftModel.from_pretrained(base_m, args.adapter).merge_and_unload()
        print(f"[merge] adapter {args.adapter} 已并入 {base}（绕过 r2_grpo 未完成 merge）")
        from merge import _save_model_lowmem
        os.makedirs(args.output, exist_ok=True)
        _save_model_lowmem(merged, args.output,
                           shard_gb=float(os.environ.get("GRPO_SHARD_GB", "1.0")))
        if tok is not None:
            tok.save_pretrained(args.output)
        del merged
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[merge] 保存完成 -> {args.output}")
    else:
        print(f"[merge] LoRA 合并进主权重 -> {args.output}")
        merge_and_save(model, tok, args.output, adapter_dir=adapter_dir)
    print(f"[done] s1_sft 完成，作为阶段二 GRPO 基座与 ref")


if __name__ == "__main__":
    main()
