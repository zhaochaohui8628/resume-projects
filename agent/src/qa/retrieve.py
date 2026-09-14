"""复合结构化 QA —— 双路召回器（图谱范围限定 + RAG 语义检索）。

用于 controls（风险管控清单）/ acceptance（验收节点）两个维度补条文：
  A 图谱路：HazardCategory -REGULATED_BY-> Standard 白名单（从 demo_graph_v2 读取）→ 范围限定
  B 语义路：RAG 检索（管控/验收关键词 query）→ top-k 条文

合并：按 clause_id / (source, clause_no) 去重 → 返回带来源的条文列表。
RAG 不可用时回落规则/知识库（不抛异常）。

设计原则：图谱产出的是**范围/白名单**（去哪找），不是扩展条款；禁止把扩展条款塞进上下文。
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.dirname(HERE)                                # agent/src
WS = os.path.dirname(os.path.dirname(AGENT_SRC))                 # 工作区根（agent/src/qa -> agent/src -> agent -> 工作区）
GRAPH = os.path.join(WS, "rag2", "data", "graphrag", "demo_graph_v2.json")

# 类别 key → 图谱 HazardCategory id 片段
_CAT_MAP = {
    "起重吊装": "起重吊装工程",
    "基坑工程": "深基坑工程",
    "模板支撑": "模板支撑工程",
    "脚手架": "脚手架工程",
}


def _load_graph() -> dict:
    try:
        with open(GRAPH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def graph_standard_whitelist(category: str | None) -> list[str]:
    """图谱路：HazardCategory -REGULATED_BY-> Standard 白名单（规范名列表）。"""
    if not category:
        return []
    g = _load_graph()
    if not g:
        return []
    target = _CAT_MAP.get(category)
    std_ids = {n["id"] for n in g["nodes"] if n["label"] == "Standard"}
    cats = [n["id"] for n in g["nodes"]
            if n["label"] == "HazardCategory" and (target in n["id"] if target else True)]
    names = []
    for e in g["edges"]:
        if e["type"] == "REGULATED_BY" and e["from"] in cats and e["to"] in std_ids:
            # 找到 Standard 节点的 name
            for n in g["nodes"]:
                if n["id"] == e["to"]:
                    names.append(n.get("name", n["id"]))
                    break
    return sorted(set(names))


def _rag_search(query: str, top_k: int = 3):
    """B 语义路：RAG 检索（复用 subagents._shared 的进程级单例，失败返回 []）。

    ⚠️ 以前这里每次 `RagClient()` 新建 → GraphRAG 的 controls/acceptance 两维会重复加载
    双塔+CE 模型，实测把内存打爆（页面文件太小 1455 / MemoryError）→ 精排被静默降级。
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


def dual_retrieve(category: str | None, query: str, top_k: int = 3) -> dict:
    """双路召回 → {items: [str], sources: [str], graph_whitelist: [str], rag_count: int}。"""
    whitelist = graph_standard_whitelist(category)
    rag_hits = _rag_search(query, top_k=top_k)
    # 去重（按 source::clause_no）
    seen, items, srcs = set(), [], []
    for h in rag_hits:
        key = f"{h.get('source', '?')}::{h.get('clause_no', '')}"
        if key in seen:
            continue
        seen.add(key)
        items.append(_fmt_clause(h))
        srcs.append(f"RAG：{h.get('source', '?')} {h.get('clause_no', '')}")
    return {"items": items, "sources": srcs,
            "graph_whitelist": whitelist, "rag_count": len(rag_hits)}
