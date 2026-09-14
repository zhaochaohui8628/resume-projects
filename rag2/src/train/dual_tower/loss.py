"""InfoNCE 损失：批内负例（in-batch）+ 显式简单负例拼接。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(query_emb: torch.Tensor, doc_emb: torch.Tensor,
             labels: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """doc_emb 的第 labels[i] 行是 query_emb 第 i 行的正样本。

    文档矩阵 = [批内正例(B 行)] ⊕ [显式负例(B*K 行)]，因此负例池为 B+B*K，
    既利用了 batch 内互负，也保证每条 query 至少见到自己那 K 个跨规范简单负例。
    """
    logits = query_emb @ doc_emb.t() / temperature
    return F.cross_entropy(logits, labels)
