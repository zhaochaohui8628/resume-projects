"""GRPO 主循环（阶段一/二通用，LoRA 合并链）。

每 step 流程：
    1. 取 QB 条 query（轮转覆盖数据集），构造 chat prompt（system 注入模拟 RAG 证据）
    2. policy.generate(T=1.0, G=8) 采样 8 条回答/query（policy = base + LoRA）
    3. 四维连续判分（reward.score_breakdown：format/cot/basis/answer 加权求和，
       平滑奖励曲线替代旧 4 档离散；组内 z-score 优势 A=(Q−V)/std 论文原版）
    4. token 级 GRPO loss（importance ratio + clip + k3 KL），组内 inner 轮原地更新
    5. 轨迹追加落盘（含 reward + 四维分解，供 SFT 回收与多维度评测）
结束：merge LoRA 进主权重 -> 作为下一阶段基座（权重链：s1_grpo -> s1_sft -> final）。

ref 策略 = 本阶段基座（不含 LoRA）：LoRA 冻结基座参数、更新只动 adapter，
peft disable_adapter 前向即 ref 输出 -> 全程单份模型显存。

用法（GPU 机）：
    python grpo/scripts/train_grpo.py --base-model <Qwen 或 s1_grpo路径> --stage s1
        [--steps 300 --query-batch 8 --G 8 --temperature 1.0 --beta 0.04 --lr 7e-6]
    --dry-run：不加载模型，仅验证数据与参数管线。
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
from reward import score_breakdown          # noqa: E402
from loss import grpo_token_loss            # noqa: E402
from merge import merge_and_save            # noqa: E402


def load_queries(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"[data] {path} -> {len(rows)} 条 query")
    return rows


def evidence_text_of(row) -> str:
    ev = row.get("agent_scenario", {}).get("evidence") or []
    return "\n".join(f"- [{e['source']}] {e['clause_no']}: {e['text']}" for e in ev)


def build_messages(row):
    return [
        {"role": "system", "content": make_system_prompt(row["task_type"], evidence_text_of(row))},
        {"role": "user", "content": row["query"]},
    ]


def load_policy(base_model, device):
    """加载基座 + 挂载 LoRA（8G 显存专用配置）。

    - `device_map="cuda:0"`：权重直接落 GPU，避免 CPU 中转（页文件不足时 OSError 1455）。
    - `dtype=` 而非 `torch_dtype=`（transformers 5.x 已弃用后者）。
    - 训练期开启 gradient checkpointing：激活显存降一个量级，代价是 ~30% 速度。
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True, low_cpu_mem_usage=True)
    lora = LoraConfig(
        r=C.LORA_RANK, lora_alpha=C.LORA_ALPHA, lora_dropout=C.LORA_DROPOUT,
        target_modules=C.LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if os.environ.get("GRPO_GRAD_CKPT", "1") == "1":
        # 8G 显存必需：激活显存 O(L·T) -> O(T)，代价约 +30% 时间
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        print("[mem] gradient checkpointing ON", flush=True)
    return model, tok


def encode_prompt(tok, msgs, device):
    """apply_chat_template → (1, plen) 张量（兼容 transformers 返回 BatchEncoding）。"""
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    return ids.to(device)


def logp_response(model, seq_ids, plen=None):
    """全序列前向 -> 响应区逐位置 logp。返回 (B,T) float32。

    显存关键（`plen` 不为 None 时启用）：只对**响应区**计算 logits
    （`logits_to_keep=T-plen+1`），prompt 区（本项目 p50=644 token，占全长 70%）
    不再产生 (T_prompt, 151936) 的 logits —— 显存直降约 3.5 倍，8G 卡可跑。

    位置换算：logits[j] 对应序列位置 p = T-keep+j，用于预测 ids[p+1]，
    故 logp_tok 的 t = T-keep+j+1，恰好覆盖 [plen, T]。
    """
    B, T = seq_ids.shape
    keep = (T - plen + 1) if plen is not None else T
    keep = max(1, min(keep, T))

    chunk = int(os.environ.get("GRPO_LOGP_CHUNK", "1"))
    tchunk = int(os.environ.get("GRPO_LOGP_TCHUNK", "0")) or keep
    logp_tok = torch.full((B, T), -1e4, dtype=torch.float32, device=seq_ids.device)
    base = T - keep                       # logits[0] 对应序列位置 base
    for s in range(0, B, chunk):
        sub = seq_ids[s:s + chunk]
        c = sub.shape[0]
        logits = model(input_ids=sub, logits_to_keep=keep).logits   # (c, keep, V)
        if tchunk >= keep:
            lp = torch.log_softmax(logits.float(), dim=-1)
            tgt = sub[:, base + 1:]                                  # (c, keep-1)
            logp_tok[s:s + c, base + 1:] = lp[:, :-1].gather(
                -1, tgt.unsqueeze(-1)).squeeze(-1)
            del lp
        else:
            for t0 in range(0, keep - 1, tchunk):
                t1 = min(t0 + tchunk, keep - 1)
                lp = torch.log_softmax(logits[:, t0:t1 + 1].float(), dim=-1)
                tgt = sub[:, base + t0 + 1: base + t1 + 1]
                logp_tok[s:s + c, base + t0 + 1: base + t1 + 1] = lp[:, :-1].gather(
                    -1, tgt.unsqueeze(-1)).squeeze(-1)
                del lp
        del logits
        if seq_ids.device.type == "cuda":
            torch.cuda.empty_cache()      # 峰值擦边（8G 卡），及时回收碎片
    return logp_tok


def sample_no_ckpt(model, tok, ids, G, temperature, max_new_tokens=None):
    """采样时临时关闭 gradient checkpointing（与 use_cache 冲突），结束恢复。"""
    was = getattr(model, "is_gradient_checkpointing", False)
    if was:
        model.gradient_checkpointing_disable()
    try:
        return sample_responses(model, tok, ids, G, temperature,
                                max_new_tokens=max_new_tokens)
    finally:
        if was:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()          # 采样结束回到 train（保持 checkpointing 生效）


def resp_mask(seq_ids, plen, eos_id):
    """响应区掩码（(B,T) bool）。

    logp_tok[t] = log p(ids[t] | ids[:t])（t>=1，由前一位置预测）。
    响应 token 区间 = [plen, end]，end = 首个 eos 位置（含 eos 停止 token，
    教模型学会收尾）；未生成 eos（跑满长度）则取 [plen, T)。注意首响应 token
    位于索引 plen，勿从 plen+1 起算（会漏掉第一个生成 token 的梯度）。
    """
    B, T = seq_ids.shape
    m = torch.zeros(B, T, dtype=torch.bool, device=seq_ids.device)
    for i, row in enumerate(seq_ids.tolist()):
        end = T
        for j in range(plen, T):
            if row[j] == eos_id:
                end = j
                break
        if end < T:                      # 正常 eos 停止：学 [plen, end]（含 eos）
            m[i, plen:end + 1] = True
        else:                            # 未触发 eos：学满 [plen, T)
            m[i, plen:T] = True
    return m


def sample_responses(model, tok, ids, G, temperature, max_new_tokens=None):
    """prompt(1,plen) 复制 G 份多样采样。返回 full(G,T) 与逐行 response 文本。

    默认一次生成 G 份（快）；`--safe-sample` 时逐条生成再拼接（8G 显存下防 OOM，
    速度略慢但峰值显存低 ~G 倍）。
    """
    import os
    max_new_tokens = max_new_tokens or C.MAX_NEW_TOKENS
    safe = os.environ.get("GRPO_SAFE_SAMPLE", "") == "1"
    if not safe:
        in_ids = ids.repeat_interleave(G, dim=0)
        model.eval()
        with torch.no_grad():
            gen = model.generate(
                input_ids=in_ids, attention_mask=torch.ones_like(in_ids),
                do_sample=True, temperature=temperature, top_p=C.TOP_P,
                max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id, use_cache=True)
        full = torch.cat([in_ids, gen[:, in_ids.shape[1]:]], dim=1)
        plen = ids.shape[1]
        texts = []
        for row in full.tolist():
            end = len(row)
            for j in range(plen, len(row)):
                if row[j] == tok.eos_token_id:
                    end = j
                    break
            texts.append(tok.decode(row[plen:end], skip_special_tokens=True).strip())
        return full, texts
    # 逐条采样（显存安全）：G 份各生成一次，再 pad 拼接
    plen = ids.shape[1]
    all_gen, all_texts = [], []
    model.eval()
    for _ in range(G):
        with torch.no_grad():
            g = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                do_sample=True, temperature=temperature, top_p=C.TOP_P,
                max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id, use_cache=True)
        all_gen.append(g[0].tolist())
        # 取响应文本
        end = len(g[0])
        for j in range(plen, len(g[0])):
            if g[0][j] == tok.eos_token_id:
                end = j
                break
        all_texts.append(tok.decode(g[0][plen:end], skip_special_tokens=True).strip())
    # pad 到等长
    L = max(len(x) for x in all_gen)
    full = torch.full((G, L), tok.pad_token_id, dtype=torch.long, device=model.device)
    for i, x in enumerate(all_gen):
        full[i, :len(x)] = torch.tensor(x, dtype=torch.long, device=model.device)
    return full, all_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=C.DEFAULT_BASE)
    ap.add_argument("--stage", default="s1", choices=["s1", "s2"])
    ap.add_argument("--data", default=os.path.join(C.DATA_DIR, "train.jsonl"))
    ap.add_argument("--traj", default=None,
                    help="轨迹输出路径；默认 s1->trajectories.jsonl, s2->trajectories_s2.jsonl")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--query-batch", type=int, default=C.QUERY_BATCH)
    ap.add_argument("--G", type=int, default=C.G)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--inner-epochs", type=int,
                    default=int(os.environ.get("GRPO_INNER", "1")))
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mem-log", action="store_true", help="打印分阶段显存峰值")
    ap.add_argument("--max-new-tokens", type=int, default=C.MAX_NEW_TOKENS)
    args = ap.parse_args()

    temperature = args.temperature or (C.TEMPERATURE if args.stage == "s1" else 0.9)
    beta = args.beta or (C.KL_BETA if args.stage == "s1" else 0.06)
    steps = args.steps or (C.GRPO_STEPS if args.stage == "s1" else C.GRPO_STEPS // 2)
    output = args.output or C.WEIGHT_CHAIN["s1_grpo" if args.stage == "s1" else "final"]
    os.makedirs(C.DATA_DIR, exist_ok=True)
    # 阶段轨迹分文件：s1/s2 不混在一起（SFT 回收与评测要按阶段取用）
    traj_path = args.traj or os.path.join(
        C.DATA_DIR, "trajectories.jsonl" if args.stage == "s1" else f"trajectories_{args.stage}.jsonl")

    rows = load_queries(args.data)
    if args.dry_run:
        print(f"[dry-run] base={args.base_model} | stage={args.stage} | steps={steps} | "
              f"QB={args.query_batch} | G={args.G} | T={temperature} | beta={beta} | "
              f"lr={args.lr or C.GRPO_LR} | inner={args.inner_epochs}")
        print(f"[dry-run] merge 输出 -> {output}\n[dry-run] 轨迹 -> {traj_path}")
        print("[dry-run] 管线验证 OK（未加载模型）")
        return

    torch.manual_seed(args.seed); random.seed(args.seed)
    model, tok = load_policy(args.base_model, args.device)
    # 必须 train()：transformers 的 gradient checkpointing 仅在 self.training 时生效，
    # 而 8G 显存跑 3B 全靠它降激活。dropout 已置 0（config.LORA_DROPOUT），
    # 故 train 模式仍是确定性前向，importance ratio 不受影响。
    model.train()
    model.config.use_cache = False                 # 训练期禁用 KV cache（省显存）
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr or C.GRPO_LR)
    plog = open(traj_path, "a", encoding="utf-8")
    memlog = args.mem_log

    def _mem(tag):
        if memlog and model.device.type == "cuda":
            print(f"    [mem] {tag}: alloc {torch.cuda.memory_allocated()/1024**3:.2f}G "
                  f"peak {torch.cuda.max_memory_allocated()/1024**3:.2f}G "
                  f"reserved {torch.cuda.memory_reserved()/1024**3:.2f}G", flush=True)

    N = len(rows); QB = min(args.query_batch, N)
    recent_scores, recent_ans = [], []
    for step in range(steps):
        start = (step * QB) % N
        batch_rows = [rows[(start + i) % N] for i in range(QB)]
        per_row = []                                # 逐 query 一组：G 条
        for row in batch_rows:
            msgs = build_messages(row)
            ids = encode_prompt(tok, msgs, model.device)
            plen = ids.shape[1]

            full_ids, texts = sample_no_ckpt(model, tok, ids, args.G, temperature,
                                             max_new_tokens=args.max_new_tokens)
            _mem(f"采样后 plen={plen} T={full_ids.shape[1]}")
            mask = resp_mask(full_ids, plen, tok.eos_token_id)

            # old（采样时 policy）/ ref（基座）各缓存一次，无梯度
            with torch.no_grad():
                old_lp = logp_response(model, full_ids, plen=plen)
            _mem("old_lp 后")
            with torch.no_grad():
                with model.disable_adapter():
                    ref_lp = logp_response(model, full_ids, plen=plen)
            _mem("ref_lp 后")

            scores, ans_dims = [], []
            grp_id = f"step{step}-q{len(per_row)}"
            for t in texts:
                bd = score_breakdown(t, row["judge_meta"])
                scores.append(bd["total"])
                recent_scores.append(bd["total"])
                ans_dims.append(bd["answer"])
                plog.write(json.dumps({
                    "group_id": grp_id,            # 同组 8 条共享，SFT 选 top2 用
                    "id": row["id"], "task_type": row["task_type"], "query": row["query"],
                    "judge_meta": row["judge_meta"], "response": t, "reward": bd["total"],
                    "reward_breakdown": {k: bd[k] for k in ("format", "cot", "basis", "answer")},
                }, ensure_ascii=False) + "\n")
            recent_ans.extend(ans_dims)
            per_row.append({
                "full_ids": full_ids, "mask": mask, "plen": plen,
                "old_lp": old_lp, "ref_lp": ref_lp,
                "scores": torch.tensor([scores], dtype=torch.float32, device=model.device),
            })
        plog.flush()

        # ---- 优势在 grpo_token_loss 内以组内 z-score 结算：A=(r−mean)/std ----
        for _ in range(args.inner_epochs):
            opt.zero_grad()
            # **逐 query 单独 backward**（梯度累加）：若先把所有 query 的 loss 相加再
            # 一次 backward，全部计算图需同时驻留显存 → 8G 卡上 backward 阶段 OOM
            # （实测 s2 第 12 步崩在 backward）。逐条 backward 后图立即释放，
            # 峰值 ≈ 单条序列的图；数学上与 Σloss/QB 一次 backward 等价。
            for pr in per_row:
                new_lp = logp_response(model, pr["full_ids"], plen=pr["plen"])  # 带梯度
                keep_used = pr["full_ids"].shape[1] - pr["plen"] + 1
                l = grpo_token_loss(
                    new_lp.unsqueeze(0), pr["old_lp"].unsqueeze(0), pr["ref_lp"].unsqueeze(0),
                    pr["mask"].unsqueeze(0).float(), pr["scores"], beta=beta) / QB
                _mem(f"new_lp 前向(keep={keep_used})")
                l.backward()
                _mem("backward 后")
                del new_lp, l
                if model.device.type == "cuda":
                    torch.cuda.empty_cache()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()

        if (step + 1) % 10 == 0 or step == 0:
            tail = recent_scores[-QB * args.G:]
            print(f"step {step+1}/{steps} | 最近组平均分 {sum(tail)/max(1,len(tail)):.3f}",
                  flush=True)

    plog.close()
    # 先存 adapter（安全网）再 merge 全量：全量保存失败也不丢本轮成果
    adapter_dir = os.path.join(C.MODELS_ROOT, "qwen-grpo",
                                  f"{os.path.basename(output.rstrip(os.sep))}_adapter")
    print(f"[merge] LoRA 合并进主权重 -> {output}", flush=True)
    merge_and_save(model, tok, output, adapter_dir=adapter_dir)
    print(f"[done] 轨迹: {traj_path}")


if __name__ == "__main__":
    main()
