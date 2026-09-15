"""复合结构化 QA —— 条文检索器（RAG 语义路）。

用于 controls（风险管控清单）/ acceptance（验收节点）两个维度补条文：
  RAG 检索（管控/验收关键词 query）→ top-k 条文。

RAG 不可用时回落知识库/规则（不抛异常）。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.dirname(HERE)                                # agent/src


def _rag_search(query: str, top_k: int = 3):
    """语义路：RAG 检索（复用 subagents._shared 的进程级单例，失败返回 []）。

    ⚠️ 不要每次 `RagClient()` 新建——会把双塔+CE 模型重复加载，内存打爆导致静默降级。
    """
    try:
        if AGENT_SRC not in sys.path:
            sys.path.insert(0, AGENT_SRC)
        from subagents._shared import get_rag
        c = get_rag()
        if not c or not c.available():
            return []
        return c.search_with_meta(query, top_k=top_k)
    except Exception:
        return []


def _fmt_clause(h: dict) -> str:
    src = h.get("source", "?")
    cn = h.get("clause_no", "") or ""
    txt = (h.get("text", "") or "").strip()
    txt = txt[:120].replace("\n", " ")
    return f"[{src} {cn}] {txt}".strip()


def retrieve_clauses(category: str | None, query: str, top_k: int = 3) -> dict:
    """RAG 检索条文 → {items: [str], sources: [str], rag_count: int}。

    category 仅用于调用方拼 query，这里保留形参以兼容原签名。
    """
    rag_hits = _rag_search(query, top_k=top_k)
    seen, items, srcs = set(), [], []
    for h in rag_hits:
        key = f"{h.get('source', '?')}::{h.get('clause_no', '')}"
        if key in seen:
            continue
        seen.add(key)
        items.append(_fmt_clause(h))
        srcs.append(f"RAG：{h.get('source', '?')} {h.get('clause_no', '')}")
    return {"items": items, "sources": srcs, "rag_count": len(rag_hits)}
