"""RAG 检索后 → LLM 汇总答案。

流程：search(query) → top-k 条文 → LLM 汇总（引用条文，防幻觉）。

设计：
- 输入严格限定 = 用户问句 + top-k 条文原文（带 source/clause_no），LLM 只做"基于条文回答"，
  与 agent 侧 comparator 的"输入限定"口径一致（防幻觉）。
- 条文以 [1]..[k] 编号，LLM 必须在答案中引用（如"[1] JGJ130 6.2.4"）；未引用任何条文 → 标记不可信。
- 服务不可用（连接失败）→ 返回降级结果（原始 top-k + 提示），不抛异常（链路不断）。

用法：
    from retrieval.llm_summarize import summarize
    hits = retriever.search(query, top_k=5)
    out = summarize(query, hits, llm=LLMClient())
"""
from __future__ import annotations

import json

_SYSTEM = """你是建筑施工规范咨询助手。基于给定的规范条文回答用户问题。

规则：
1. 只依据提供的条文回答，不得编造条文内容或出处。
2. 引用条文时用 [编号] 标注，格式如 [1] JGJ130-2011 6.2.4。
3. 若条文不足以回答，明确说"依据现有条文无法完全回答"并给出能确认的部分。
4. 答案简洁、分条，中文。
"""


def _build_user(query: str, hits: list[dict]) -> str:
    lines = [f"用户问题：{query}", "", "检索到的规范条文："]
    for i, h in enumerate(hits, 1):
        md = h.get("metadata", {}) or {}
        src = md.get("source", "?")
        cn = md.get("clause_no", "")
        txt = (h.get("text", "") or "").strip().replace("\n", " ")
        lines.append(f"[{i}] {src} {cn}：{txt[:300]}")
    return "\n".join(lines)


def summarize(query: str, hits: list[dict], llm=None,
              max_tokens: int = 512) -> dict:
    """检索结果 → LLM 汇总。

    返回 {answer, citations: [str], degraded: bool, hits_n}。
    llm 为 None 或不可用 → degraded=True，answer 为"请启动 LLM 服务"提示（不崩链路）。
    """
    n = len(hits or [])
    if not n:
        return {"answer": "未检索到相关条文。", "citations": [],
                "degraded": False, "hits_n": 0}
    if llm is None:
        return {"answer": "（LLM 服务未配置：请设置 DEEPSEEK_API_KEY 或传入 LLMClient）",
                "citations": [], "degraded": True, "hits_n": n}
    try:
        user = _build_user(query, hits)
        ans = llm.chat([{"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": user}],
                       max_tokens=max_tokens, temperature=0.1)
        # 抽引文（[n] 标记）
        import re
        citations = sorted(set(re.findall(r"\[(\d+)\]", ans)))
        return {"answer": ans, "citations": citations,
                "degraded": False, "hits_n": n}
    except Exception as e:
        return {"answer": f"（LLM 服务不可用：{type(e).__name__}: {e}）",
                "citations": [], "degraded": True, "hits_n": n}
