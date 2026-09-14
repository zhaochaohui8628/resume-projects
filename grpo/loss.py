"""GRPO 损失（要素⑤）：重要性采样 + clip 裁剪 + k3 KL 惩罚。

损失（论文 DeepSeekMath 原版，seq/token 通用形式）：
    L = − mean[ min( ρ·A, clip(ρ, 1−ε, 1+ε)·A ) ]  +  β · mean[ k3_KL ]
其中 ρ = π_θ/π_old（importance ratio）；
优势 A 采用论文 z-score：A = (r − mean(r)) / std(r)（组内 8 条归一化，
std 为总体标准差、clamp 防全同分除零）；k3_KL 恒 ≥ 0。
min() 写法对 A 正负均正确：A>0 限高、A<0 取更负（等同对 clip 的 max 形式）。
"""
from __future__ import annotations

import torch


def _k3_kl(logp_theta, logp_ref):
    """k3 无偏 KL 估计（逐位置），恒 ≥0（数值上 clamp 防 exp 溢出）。"""
    d = (logp_ref - logp_theta).clamp(-20.0, 20.0)
    return d.exp() - d - 1.0


def _zscore(rewards):
    """组内 z-score：A = (r − mean(r)) / std(r)。std 用总体标准差(ddof=0)，
    clamp 下限防「8 条同分 → std=0 → 除零/NaN」；同分时 A=0 自然无梯度。
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-4)
    return (rewards - mean) / std


def grpo_seq_loss(logp_new, logp_old, logp_ref, rewards, beta=0.04, eps=0.2):
    """sequence-level 版本。输入形状 (B, G)（每组 G 条整句 logp）。
    rewards: (B, G) 判分。仅用于单测/快速验证，正式训练请用 token 级。
    """
    adv = _zscore(rewards)
    ratio = (logp_new - logp_old).exp()
    pg = -torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
    kl = _k3_kl(logp_new, logp_ref)
    return pg.mean() + beta * kl.mean()


def grpo_token_loss(logp_new, logp_old, logp_ref, mask, rewards,
                    beta=0.04, eps=0.2):
    """token 级版本（正式训练用）。形状：
        logp_*: (B, G, T) 仅 response 段的逐 token logp
        mask:   (B, G, T) 有效 token（0/1，已含 pad/忽略位）
        rewards:(B, G) 判分
    优势在组内 z-score（_zscore）；每样本除以自身有效 token 数消除长度偏置。
    """
    adv = _zscore(rewards).unsqueeze(-1)             # (B,G,1) 广播到 T
    ratio = (logp_new - logp_old).exp()              # (B,G,T)
    pg_t = -torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
    kl_t = _k3_kl(logp_new, logp_ref)

    denom = mask.sum(dim=-1).clamp(min=1.0)          # (B,G) 每样本 token 数
    pg = (pg_t * mask).sum(-1) / denom               # 每样本均值
    kl = (kl_t * mask).sum(-1) / denom
    return pg.mean() + beta * kl.mean()
