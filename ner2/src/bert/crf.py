"""ner2 CRF 头：基于**安装库** pytorch-crf（torchcrf）的适配器（不自实现 CRF）。

职责仅限三件事，不重复实现任何动态规划：
1. 依赖加载（含旧轮子装成 `crf` 模块时的兼容）；
2. 标签/掩码规整 —— 线性链 CRF 无法 ignore 中间 token，调用方把 -100 / 子词 X 统一映射为 O；
3. 补两个训练/挖掘必需的接口：
   - `nll`：负对数似然 = `-CRF.forward(...)`。**注意 pytorch-crf 的 forward 返回的是
     对数似然 log P(tag_path|x) = gold_score − logZ**（不是损失），必须取负才是损失；
   - `path_logprob_per_token`：**归一化解码得分**（逐样本对数似然 / 有效长度），
     供第二轮「高得分解码数据」筛选使用（越大越可信）。

依赖：`pip install pytorch-crf`（本项目 GPU 环境 `anaconda3/envs/torch_gpu`）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def import_crf_class():
    """加载 pytorch-crf 的 CRF 类。"""
    try:
        from torchcrf import CRF
        return CRF
    except ImportError:  # pragma: no cover - 兼容轮子模块名
        try:
            from crf import CRF
            return CRF
        except ImportError as e:
            raise ImportError(
                "缺 pytorch-crf。请执行："
                r"C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe -m pip install pytorch-crf"
            ) from e


class CRFHead(nn.Module):
    """torchcrf.CRF 的薄封装（参数前缀 `crf.crf.*`，便于冻结策略识别）。"""

    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.crf = import_crf_class()(num_tags, batch_first=batch_first)

    def loglik(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor,
               reduction: str = "none"):
        """对数似然 log P(tag_path | x) = gold path score − logZ（库原生语义）。"""
        return self.crf(emissions, tags, mask.bool(), reduction=reduction)

    def nll(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor,
            reduction: str = "mean"):
        """负对数似然（= logZ − gold path score），训练用。"""
        return -self.loglik(emissions, tags, mask, reduction=reduction)

    @torch.no_grad()
    def path_logprob_per_token(self, emissions, tags, mask) -> torch.Tensor:
        """逐样本归一化解码得分 = log P(path|x) / 有效长度（(B,)，越大越可信）。"""
        llh = self.crf(emissions, tags, mask.bool(), reduction="none")
        lengths = mask.sum(-1).clamp(min=1)
        return llh / lengths

    @torch.no_grad()
    def decode(self, emissions, mask):
        """Viterbi 解码 → List[List[int]]（已剔除 pad 位置）。"""
        return self.crf.decode(emissions, mask.bool())

    def forward(self, emissions, tags, mask, reduction: str = "mean"):
        return self.nll(emissions, tags, mask, reduction=reduction)
