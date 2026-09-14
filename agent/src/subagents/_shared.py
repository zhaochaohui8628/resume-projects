"""subagents 共享依赖：进程级 RAG 单例（只读检索线程安全，避免每个 subagent 重复加载模型）。"""
from __future__ import annotations

_rags: dict = {}


def get_rag(use_rerank: bool | None = None):
    """进程级 RagClient 单例（按 use_rerank 分槽缓存）；构建失败返回 None（降级：不补依据）。

    ⚠️ 为什么必须单例：RagClient 持有双塔 + CrossEncoder（精排）模型，一份就有数 GB 级
    显存/内存占用。此前 qa/retrieve.dual_retrieve 每次调用都 `RagClient()` 新建，一次问答
    （controls + acceptance 两维 + ReAct 路径）会重复加载多份 → 实测触发
    `OSError: 页面文件太小 (1455)` / `MemoryError`，精排被静默降级。
    """
    key = use_rerank
    if key not in _rags:
        try:
            from tools.rag_client import RagClient
            _rags[key] = RagClient(use_rerank=use_rerank)
        except Exception:
            _rags[key] = None
    return _rags[key]