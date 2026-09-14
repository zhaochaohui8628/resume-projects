"""问答 subagent v6：RAG 检索规范库（受 ctx.rag_rerank 控制）；自然语言答复由调度中心 LLM 汇总生成。

v6 变更（2026-09-11 用户定稿）：**移除 plan 片段截断/拼接**——问答链路直接
`rag.search(query)`，不做任何方案上下文收敛（与三层漏斗切割一致）。
输入：query（必填）/ ctx（rag_rerank）。
输出：SubAgentResult，含 rag_hits（全部检索明细，含 rank/score/source/clause_no）。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.dirname(HERE)
if AGENT_SRC not in sys.path:
    sys.path.insert(0, AGENT_SRC)

from ._shared import get_rag                   # noqa: E402
from .base import SubAgent, SubAgentResult    # noqa: E402


class QAAgent(SubAgent):
    name = "qa"
    title = "规范问答（RAG 条文检索）"
    description = "检索规范库条文回答一般性规范咨询（自然语言答复由 LLM 汇总生成）"
    doc = """规范问答：检索规范库条文回答一般性规范咨询；自然语言答复由汇总 LLM 生成。"""

    def run(self, query: str = "", plan: str = "", ctx: dict | None = None) -> SubAgentResult:
        if not query:
            return SubAgentResult(name=self.name, title=self.title,
                                  summary="请提供问题。",
                                  trace=[{"step": 1, "kind": "skip", "label": "无 query"}])
        ctx = ctx or {}

        # ---- 复合结构化问句：拆解 → 四维组装（不依赖 RAG，规则+知识库） ----
        try:
            from qa.decompose import is_structured_query
            if is_structured_query(query):
                from qa.assembler import assemble
                from qa.decompose import decompose
                slot = decompose(query)
                out = assemble(slot)
                gts = out.get("graph_traces") or []
                trace = [{"step": 1, "kind": "structured",
                          "route": "rule", "src": "qa/assembler.py（结构化知识库 + 规则模板）",
                          "label": f"复合结构化 QA：{slot.get('category') or '?'} / "
                                   f"{sum(1 for d in slot['dimensions'] if d['enabled'])} 维",
                          "detail": {"category": slot.get("category"),
                                     "equipment": slot.get("equipment"),
                                     "capacity": slot.get("capacity"),
                                     "is_super_scale": slot.get("is_super_scale"),
                                     "dimensions": [d["key"] for d in slot["dimensions"] if d["enabled"]]}}]
                # ---- GraphRAG 溯源：每个用到双路召回的维度单独一步 ----
                for gt in gts:
                    trace.append({
                        "step": len(trace) + 1, "kind": "graphrag",
                        "route": "graphrag",
                        "src": gt.get("graph_file", ""),
                        "label": f"GraphRAG 双路召回[{gt.get('dim')}]：图谱路 "
                                 f"{gt.get('graph_count', 0)} 本规范 → 语义路命中 "
                                 f"{gt.get('rag_count', 0)} 条",
                        "detail": {
                            "tool": "图谱路 HazardCategory -REGULATED_BY-> Standard（范围限定）"
                                    " + 语义路 RAG",
                            "graph_file": gt.get("graph_file", ""),
                            "graph_path": gt.get("graph_path", ""),
                            "graph_whitelist": gt.get("graph_whitelist", []),
                            "graph_count": gt.get("graph_count", 0),
                            "rag_count": gt.get("rag_count", 0),
                            "rag_sources": gt.get("rag_sources", []),
                            "category": gt.get("category"),
                            "use_rerank": bool(ctx.get("rag_rerank", True)),
                            "rag_backend": getattr(ctx.get("rag"), "backend", None),
                        }})
                if not gts:
                    trace.append({"step": len(trace) + 1, "kind": "graphrag_skip",
                                  "route": "rule", "src": "",
                                  "label": "本问未走 GraphRAG（纯规则/知识库组装）",
                                  "detail": {"graphrag_used": False}})
                return SubAgentResult(
                    name=self.name, title=out["title"],
                    summary=out["markdown"],
                    basis=[], meta={"structured": True, "sections": len(out["sections"])},
                    trace=trace, rag_hits=[],
                )
        except Exception as e:                                   # noqa: BLE001
            # 结构化链路异常 → 回落到普通 RAG（不静默吞掉，trace 记录）
            trace_fb = [{"step": 1, "kind": "structured_error",
                         "label": f"结构化 QA 异常，回落到 RAG：{type(e).__name__}: {e}"}]
            ctx.setdefault("_structured_error", f"{type(e).__name__}: {e}")

        rag = ctx.get("rag") or get_rag()
        if rag is None:
            return SubAgentResult(name=self.name, title=self.title,
                                  summary="RAG 不可用。",
                                  trace=[{"step": 1, "kind": "skip", "label": "RAG 不可用"}])
        use_rerank = bool(ctx.get("rag_rerank", False))
        rr_active = bool(getattr(rag, "use_rerank", use_rerank))   # 实际生效（失败会降级）
        hits = rag.search(query, top_k=5, use_rerank=use_rerank)
        rag_hits = []
        for i, h in enumerate(hits, 1):
            rag_hits.append({"rank": i,
                             "score": h.get("score", 0.0),
                             "rerank_score": h.get("rerank_score"),
                             "source": h.get("metadata", {}).get("source", "?"),
                             "clause_no": h.get("metadata", {}).get("clause_no", ""),
                             "source_path": h.get("metadata", {}).get("source_path", ""),
                             "text": (h.get("text", "") or "")[:200],
                             "query": query[:60],
                             "rerank_on": use_rerank})
        trace = [{"step": 1, "kind": "rag", "route": "rag",
                  "src": f"{getattr(rag, 'backend', '?')}"
                         f"（CE精排={'开' if rr_active else '关'}）",
                  "label": f"RAG 检索 top-5（精排={'开' if rr_active else '关'}）",
                  "detail": {"use_rerank": rr_active, "rerank_req": use_rerank,
                             "hit_count": len(hits),
                             "rag_errors": getattr(rag, "errors", None),
                             "rag_backend": getattr(rag, "backend", None),
                             "tool": "RAG 语义检索（dual_mix + RRF → CE 精排）"}}]
        return SubAgentResult(
            name=self.name, title=self.title,
            summary=f"检索到 {len(hits)} 条条文",
            basis=hits,
            meta={"hits": len(hits), "use_rerank": use_rerank},
            trace=trace,
            rag_hits=rag_hits,
        )

    def to_markdown(self, r: SubAgentResult, brief: bool = True) -> str:
        lines = [f"### {r.title}", "", f"> {r.summary}", ""]
        if not r.rag_hits:
            lines.append("（未检索到相关条文）")
        for h in r.rag_hits[:5]:
            score = h.get("rerank_score") or h.get("score")
            lines.append(f"- `#{h['rank']}` [{h['source']}] {h.get('clause_no','')} "
                         f"(score={score:.3f}) {h['text'][:160]}")
        if r.rag_hits and r.rag_hits[0].get("rerank_on"):
            lines += ["", "*（精排开启；rerank_score 优先于 score）*"]
        return "\n".join(lines)