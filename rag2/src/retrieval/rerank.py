"""CrossEncoder 精排器（rag2 检索后重排）。

之前精排只写在 `rag2/scripts/eval_pipeline_rerank.py` 里（评估专用、不可复用），
检索链路（HybridRetriever）本身**没有**精排，导致 agent 端 `use_rerank` 是空壳开关。

本模块把精排抽成可复用组件，供 agent（tools/rag_client.py）与 rag2 评估脚本共用：
    召回 top_k * oversample → CrossEncoder 打分 → 按 CE logit 降序 → 截断 top_k

模型：默认 `rag2/data/models/cross_v2_ep4`（领域微调 CrossEncoder，评估显著优于纯融合）。
加载失败（缺 torch/transformers/产物）→ 抛异常，由调用方决定是否降级为"不精排"。
"""
from __future__ import annotations

import os
from pathlib import Path

RAG2_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CE_DIR = RAG2_ROOT / "data" / "models" / "cross_v2_ep4"


class CrossEncoderReranker:
    """CrossEncoder 精排：cands 为 [{text: str, ...}]，返回带 rerank_score 的降序新列表。"""

    def __init__(self, model_dir: str | None = None, device: str | None = None,
                 batch_size: int = 16, max_len: int = 256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_dir = str(model_dir or DEFAULT_CE_DIR)
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"精排模型产物不存在：{self.model_dir}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.max_len = int(max_len)
        self._tok = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir).to(self.device).eval()
        self._torch = torch

    def rerank(self, query: str, cands: list[dict]) -> list[dict]:
        """按 CE 分数降序返回候选（原字段保留，新增 rerank_score）。"""
        if not cands:
            return []
        texts = [(c.get("text") or "") for c in cands]
        logits: list[float] = []
        torch = self._torch
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i:i + self.batch_size]
                enc = self._tok([query] * len(chunk), chunk, padding=True,
                                truncation=True, max_length=self.max_len,
                                return_tensors="pt").to(self.device)
                logits.extend(self._model(**enc).logits.squeeze(-1).float().cpu().tolist())
        ordered = sorted(zip(cands, logits), key=lambda x: -x[1])
        return [dict(c, rerank_score=float(s)) for c, s in ordered]


def resolve_ce_dir(cfg_model: str | None, workspace: str | None = None) -> str:
    """解析精排模型目录：配置值（绝对/相对路径，存在即用）→ 默认 cross_v2_ep4 产物。

    cfg_model 可能是旧配置的 "BAAI/bge-reranker-base"（纯模型名，非本地目录），
    此时一律回退到领域微调产物 cross_v2_ep4。
    """
    cand = (cfg_model or "").strip()
    if cand:
        p = Path(cand)
        if not p.is_absolute() and workspace:
            p = Path(workspace) / cand
        if p.is_dir():
            return str(p)
    return str(DEFAULT_CE_DIR)
