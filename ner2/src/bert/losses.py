"""ner2 NER 损失：类别加权 Focal Loss（= 加权 CE 的 gamma=0 特例）。

用户口径（沿用旧 ner）：实体级指标 + **focal loss CE**，O 类占 ~85% 必须降权
（否则模型学会"全猜 O"）。X / 子词 / 特殊符由调用方置 ignore(-100)，不进 loss。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ner2.src.bert.data_utils import NUM_TAGS, O_ID, TAG2ID


def make_tag_weights(o_weight: float = 0.25) -> torch.Tensor:
    """类别权重：O = o_weight（降权），实体标签 = 1.0，X = 0（已被 ignore，双保险）。"""
    w = torch.ones(NUM_TAGS, dtype=torch.float)
    w[O_ID] = float(o_weight)
    w[TAG2ID["X"]] = 0.0
    return w


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, alpha: torch.Tensor | None = None,
               gamma: float = 2.0, ignore_index: int = -100, reduction: str = "mean"):
    """多分类 Focal Loss（Lin 2017 变体）。logits (B,T,C) 或 (N,C)，labels 同前导维。"""
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    logp = F.log_softmax(flat_logits, dim=-1)
    labels_eff = flat_labels.clamp(min=0)
    one_hot = torch.zeros_like(logp).scatter_(-1, labels_eff.unsqueeze(-1), 1.0)
    pt = torch.exp((logp * one_hot).sum(-1))
    logpt = (logp * one_hot).sum(-1)
    if alpha is not None:
        a = alpha.to(flat_logits.device, dtype=logp.dtype)
        a_t = a[labels_eff]
        fl = -(a_t * ((1.0 - pt) ** gamma) * logpt)
    else:
        fl = -((1.0 - pt) ** gamma) * logpt
    mask = (flat_labels != ignore_index).to(logp.dtype)
    fl = fl * mask
    if reduction == "mean":
        return fl.sum() / mask.sum().clamp(min=1.0)
    if reduction == "sum":
        return fl.sum()
    return fl.reshape(labels.shape)


def weighted_ce(logits, labels, alpha=None, ignore_index: int = -100):
    """加权 CE（focal 的 gamma=0 语义化入口）。"""
    return focal_loss(logits, labels, alpha=alpha, gamma=0.0, ignore_index=ignore_index)
