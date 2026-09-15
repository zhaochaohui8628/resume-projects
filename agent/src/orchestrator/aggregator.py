"""汇总器 v6.1：多 subagent 证据 + 规则自检风险 → brief → LLM 生成最终答复。

规则自检（C1/C2/C4）由调度中心固定 tool 产出，按注入决策（HIGH 强制 / 用户提及）
以 rules 参数送入本汇总器。
"""
from __future__ import annotations

import re

SUMMARY_SYSTEM = """你是资深施工安全工程师。以下是多智能体（施工内容审查/问答）的检索证据与
**规则自检结果**（编制依据/缺项检查），请综合输出**最终答复**。
要求：
- 中文、Markdown、面向施工方工程师；
- **格式分层（硬性要求）**：用 `### 小节标题` 分节（如「总体结论」「风险清单」「整改建议」）；
  关键词与结论用 **加粗**；风险用编号列表；不要输出大段连续文字堆叠；
- **只输出面向用户的审查结论/答案本身**：严禁出现工具名、技能名、ReAct/思考过程、
  检索过程、`<details>` 等任何运行细节或 HTML 标签；
- 审查类：先给总体结论（2-3 句），再给风险清单（级别｜风险｜依据/建议），最后给整改建议要点；
- 问答类：直接给出答案与关键依据（引用规范编号/条文），不要复述工具内部结构；
- 控制在 900 字内，可引用原文但需精简。"""


def _brief(query: str, has_plan: bool, agents: list, results: list,
           rules: list | None = None, cap: int = 4800) -> str:
    parts = [f"用户输入：{query or '（仅上传方案）'}",
             f"已上传方案：{'是' if has_plan else '否'}",
             f"被调用的 subagent：{', '.join(agents)}", ""]
    # 规则自检结果（若注入）
    if rules:
        parts.append(f"【规则自检结果（{len(rules)} 项）】")
        for item in rules[:20]:
            sev = item.get("severity", "?")
            parts.append(f"  · [{sev}] {item.get('check', '')}: "
                         f"{item.get('title', '')[:60]} "
                         f"{item.get('detail', '')[:80]}")
        parts.append("")
    for r in results:
        parts.append(f"【{r.title}】")
        parts.append(f"- 摘要：{r.summary}")
        if r.risks:
            parts.append("- 风险(前12)：")
            for item in r.risks[:12]:
                basis = "；".join(item.get("basis", [])[:1])[:90] if item.get("basis") else ""
                parts.append(f"  · [{item.get('severity')}] {item.get('title','')[:60]} 依据:{basis}")
        if r.entities:
            ent_s = "、".join(f"{e.get('type')}:{e.get('text')}" for e in r.entities[:10])
            parts.append(f"- 实体(前10)：{ent_s}")
        if r.basis:
            b_s = ""
            for b in r.basis[:6]:
                if isinstance(b, dict):
                    b_s += f"\n  · {b.get('type','条文')}: {('；'.join(b.get('texts',[]))[:150]) if b.get('texts') else ''}"
                else:
                    b_s += f"\n  · {str(b)[:150]}"
            parts.append(f"- 条文依据：{b_s}")
        parts.append("")
    text = "\n".join(parts)
    return text[:cap]


def summarize(llm, query: str, has_plan: bool, agents: list, results: list,
              rules: list | None = None) -> str:
    """LLM 汇总：规则自检结果（可选） + 证据 → 最终答复。"""
    prompt = _brief(query, has_plan, agents, results, rules=rules)
    resp = llm.complete([{"role": "system", "content": SUMMARY_SYSTEM},
                         {"role": "user", "content": prompt}])
    return (resp or "").strip()


def summarize_stream(llm, query: str, has_plan: bool, agents: list, results: list,
                     rules: list | None = None, on_token=None):
    """流式 LLM 汇总：逐增量回调 on_token(piece)，返回完整文本。

    llm 不支持 stream（mock / 自定义后端）时自动退回一次性 complete，
    保证接口行为一致（调用方无需分支）。
    """
    prompt = _brief(query, has_plan, agents, results, rules=rules)
    messages = [{"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": prompt}]
    stream_fn = getattr(llm, "stream", None)
    if stream_fn is None:
        text = (llm.complete(messages) or "").strip()
        if on_token and text:
            on_token(text)
        return text
    parts: list[str] = []
    for piece in stream_fn(messages):
        parts.append(piece)
        if on_token:
            try:
                on_token(piece)
            except Exception:
                pass
    return "".join(parts).strip()


def to_markdown(agents: list, results: list) -> str:
    """subagent 明细 markdown（供折叠展示，非最终答复）。"""
    if not results:
        return "（无 subagent 输出）"
    parts = []
    for name, r in zip(agents, results):
        ag = None
        try:
            from orchestrator.dispatcher import _REGISTRY
            ag = _REGISTRY.get(name)
        except Exception:
            ag = None
        md = ag.to_markdown(r) if (ag and hasattr(ag, "to_markdown")) else \
            f"### {r.title}\n\n{r.summary}"
        parts.append(md)
    return "\n\n---\n\n".join(parts)