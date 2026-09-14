"""ner2 BERT-NER 模型：`softmax` 分类头 / `crf` 线性链头 + 三档冻结策略。

两套微调方案共用本模型（用户 2026-09-11 定稿）：
- 方案1 `head="softmax"`：损失 = **focal loss CE**（O 降权 0.25 + γ=2）；
  第一轮 `freeze="encoder"`（仅训 softmax 头）→ 第二轮 `freeze="none"`（全量微调）。
- 方案2 `head="crf"`：损失 = **CRF NLL**（pytorch-crf）；
  第一轮 `freeze="encoder_and_head"`（仅训 CRF 转移矩阵）→ 第二轮 `freeze="none"`（全量微调）。

参数命名（冻结策略依赖）：
    base.bert.*         BERT 编码器（主权重）
    base.classifier.*   发射层 / softmax 分类头
    crf.crf.*           CRF 转移矩阵 + 起止分数
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ner2.src.bert.crf import CRFHead
from ner2.src.bert.data_utils import NUM_TAGS, O_ID
from ner2.src.bert.losses import focal_loss, make_tag_weights

HEAD_SOFTMAX = "softmax"
HEAD_CRF = "crf"
FREEZE_MODES = ("none", "encoder", "encoder_and_head")


class BertTokenModel(nn.Module):
    def __init__(self, base, num_labels: int = NUM_TAGS, head: str = HEAD_SOFTMAX,
                 gamma: float = 2.0, o_weight: float = 0.25, crf_aux_focal: float = 0.0):
        super().__init__()
        if head not in (HEAD_SOFTMAX, HEAD_CRF):
            raise ValueError(f"head 必须是 {HEAD_SOFTMAX}/{HEAD_CRF}，得到 {head}")
        self.base = base
        self.num_labels = num_labels
        self.head = head
        self.gamma = float(gamma)
        self.o_weight = float(o_weight)
        self.crf_aux_focal = float(crf_aux_focal)
        self.crf = CRFHead(num_labels) if head == HEAD_CRF else None
        # 类别权重（O 降权、X=0）；persistent=False → 不进 state_dict
        self.register_buffer("alpha_w", make_tag_weights(o_weight), persistent=False)

    # ------------------------------------------------------------ 冻结策略
    def apply_freeze(self, mode: str = "none") -> int:
        """设置 requires_grad；返回可训参数量。"""
        if mode not in FREEZE_MODES:
            raise ValueError(f"freeze 必须是 {FREEZE_MODES}，得到 {mode}")
        n_train = 0
        for name, p in self.named_parameters():
            if mode == "none":
                p.requires_grad = True
            elif mode == "encoder":                      # 冻结主权重，仅训头
                p.requires_grad = not name.startswith("base.bert.")
            else:                                        # encoder_and_head：仅训 CRF
                p.requires_grad = name.startswith("crf.")
            if p.requires_grad:
                n_train += p.numel()
        return n_train

    def trainable_params(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(self) -> dict:
        """按 编码器 / 头 / CRF 分组（便于差异化学习率）。"""
        g = {"encoder": [], "head": [], "crf": []}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("base.bert."):
                g["encoder"].append(p)
            elif n.startswith("crf."):
                g["crf"].append(p)
            else:
                g["head"].append(p)
        return g

    # ---------------------------------------------------------------- 前向
    def emissions(self, input_ids, attention_mask):
        return self.base(input_ids=input_ids, attention_mask=attention_mask).logits

    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.emissions(input_ids, attention_mask)
        out = {"logits": logits}
        if self.head == HEAD_CRF:
            mask = attention_mask
            if labels is not None:
                lab = labels.clone()
                lab[lab < 0] = O_ID                     # -100 / pad → O（CRF 无 ignore）
                loss = self.crf.nll(logits, lab, mask)
                if self.crf_aux_focal > 0:
                    loss = loss + self.crf_aux_focal * focal_loss(
                        logits, labels, alpha=self.alpha_w, gamma=self.gamma)
                out["loss"] = loss
            out["preds"] = self.crf.decode(logits, mask)
        elif labels is not None:
            out["loss"] = focal_loss(logits, labels, alpha=self.alpha_w, gamma=self.gamma)
        return out
