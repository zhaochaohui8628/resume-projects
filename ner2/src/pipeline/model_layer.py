"""级联层 2：微调轻量实体模型层（兜底复杂上下文）。

职责：处理规则/词典层覆盖不到的"开放词表 / 复杂上下文专有名词"——
边界歧义（词典只含词干）、类型歧义（同词多类）、未登录词（新设备/新工序表述）。

本模块只定义**接口与占位实现**：P4 用 ner2 银标微调出 BertNER 后，
通过 `load_model_backend()` 接入（模型路径由调用方给出）。
接口约定：
    ModelBackend.extract(text) -> list[{"type","start","end","conf"}]
"""
from __future__ import annotations

import os


class ModelBackend:
    """微调模型后端接口。"""

    def extract(self, text: str) -> list[dict]:  # pragma: no cover - 接口
        raise NotImplementedError


class BertNERBackend(ModelBackend):
    """ner2 微调 BertNER 适配器（P4 训练产物接入点）。

    训练流程由用户给定后接入；此处只做**加载/解码胶水**，不实现训练。
    期望模型输出 BIOE 标签序列，本类负责还原 (type, start, end)。
    """

    def __init__(self, model_path: str, device: str | None = None,
                 max_len: int = 256, batch: int = 16):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[model_layer] 缺模型产物 {model_path}；P4 训练后给出 ckpt 路径再接入")
        self.model_path = model_path
        self.device = device
        self.max_len = max_len
        self.batch = batch
        self._impl = None  # 延迟加载（P4 接入 torch/transformers）

    def extract(self, text: str) -> list[dict]:
        if self._impl is None:
            raise RuntimeError(
                "[model_layer] BertNER 运行时未接入：请在 P4 后于 _load_runtime() 补齐。")
        return self._impl(text)


def load_model_backend(model_path: str | None, device: str | None = None) -> ModelBackend | None:
    """按路径加载模型后端；路径为空则返回 None（纯规则模式）。"""
    if not model_path:
        return None
    return BertNERBackend(model_path, device=device)
