"""双塔模型：query 塔 / doc 塔（各自独立参数，可 tie）。

基座 = bge-small-zh-v1.5（4 层 / hidden 512 / 24M），池化取 **CLS**（bge v1.5 官方口径）。

⚠️ 2026-09-12 曾短暂改 mean 后又回滚：黄金集实测 mean 在短文本条款检索上显著弱于 CLS
（同一权重 0.7578 vs 0.8698；重训 12ep 后 0.7854 仍追不平，且过拟合训练分布）。
**pooling = CLS 定稿，勿再改动**。若此处与导出/检索不一致会出现「训练学 A、推理是 B」
的静默错位，改 pooling 必须同时改 `src/train/dual_tower/train.py::export` 与
`src/train/distill/train.py::export`（均取 cls）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class Tower(nn.Module):
    def __init__(self, base_path: str, max_len: int = 256):
        super().__init__()
        self.max_len = max_len
        self.transformer = AutoModel.from_pretrained(base_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return F.normalize(cls, dim=-1)


class DualTower(nn.Module):
    def __init__(self, base_path: str, max_len: int = 256, tie_weights: bool = False,
                 doc_base: str | None = None, query_base: str | None = None):
        """继续训练时用 `doc_base` / `query_base` 指定已导出的检查点目录。

        已导出的 `doc_encoder/` `query_encoder/`（SentenceTransformer 格式）根目录即
        Transformer 本体，`AutoModel.from_pretrained` 可直接加载。
        **不要用"先建基座塔再 load_state_dict 覆盖"的写法**——那会在内存里同时存在
        两个 24M 模型，在 Windows 上实测会静默 segfault（无 traceback）。
        """
        super().__init__()
        self.query_tower = Tower(query_base or base_path, max_len)
        self.doc_tower = self.query_tower if tie_weights else Tower(doc_base or base_path, max_len)
        self.tie_weights = tie_weights
        self.tokenizer = AutoTokenizer.from_pretrained(base_path)

    def encode_query(self, **kw) -> torch.Tensor:
        return self.query_tower(kw["input_ids"], kw["attention_mask"])

    def encode_doc(self, **kw) -> torch.Tensor:
        return self.doc_tower(kw["input_ids"], kw["attention_mask"])


def build_dual_tower(base_path: str, max_len: int = 256, tie_weights: bool = False,
                     doc_base: str | None = None,
                     query_base: str | None = None) -> DualTower:
    return DualTower(base_path, max_len, tie_weights, doc_base=doc_base,
                     query_base=query_base)


def model_num_params(m: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))


__all__ = ["Tower", "DualTower", "build_dual_tower", "model_num_params", "AutoConfig"]
