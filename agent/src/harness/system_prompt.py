"""系统提示词构建：项目需求定制 + 轻量技能清单 + 记忆注入。"""
from __future__ import annotations

PROJECT_BRIEF = (
    "你是「施工方案合规审查」智能助手，服务于建筑工程专项施工方案的合规自查。"
    "你能基于规范条文库与实体识别技能，协助核对方案是否存在以下问题：\n"
    "  C1 引用了已废止/失效的规范或条文；\n"
    "  C2 危险性较大分部分项工程（危大工程）缺项或缺少专家论证/监测等必备内容；\n"
    "  C4 缺少建办质〔2018〕31 号等要求的编制章节要素。\n"
    "引用规范依据时只使用检索返回的真实条文（禁止臆造规范编号或条文内容）；不确定时明确说明。"
)

ACTION_PROTOCOL = (
    "交互协议（ReAct，逐步推理）：\n"
    "1. 每步只输出一个 JSON，不得输出多余文字："
    '{"thought": "<这一步的推理>", "action": "<技能名或 answer>", "action_input": "<参数文本>"}\n'
    "2. 需要更多信息时调用技能；得到足够信息后，以 action=answer 给出结论（action_input 放最终回答）。\n"
    "3. 系统会检测死循环：重复调用同一工具、或连续几轮推理内容高度相似时会被自动拦截。"
    "被拦截后请立即改变策略——换一个技能、或改写本次输入措辞（换检索/检查角度）；"
    "若已无新信息可查，请直接 answer 给出结论，不要在同一思路上反复打转。\n"
    "4. 输出应为简体中文。"
)

MEMORY_BRIEF = (
    "你可以使用记忆：\n"
    "  相关历史任务与结论（参考但不盲从，方案各不相同）\n"
    "  项目长期约定（若有）"
)


def build_system_prompt(
    skills_brief: str,
    semantic_mem: dict = None,
    episodic_hits: list = None,
) -> str:
    """组装 system prompt：项目需求 + 记忆注入 + 技能轻量清单 + 协议。"""
    parts = [PROJECT_BRIEF, ""]
    sem = {k: v for k, v in (semantic_mem or {}).items() if v}
    if sem:
        parts.append("== 长期记忆（项目约定）==")
        parts.append("\n".join(f"- {k}: {v}" for k, v in sem.items()))
        parts.append("")
    if episodic_hits:
        parts.append("== 相似历史任务（参考）==")
        for e in episodic_hits:
            parts.append(f"- 问: {e.get('query', '')[:120]}\n  结论: {str(e.get('outcome', ''))[:160]}")
        parts.append("")
    parts.append("== 可用技能 ==")
    parts.append(skills_brief)
    parts.append("")
    parts.append("== 约束 ==\n不设工具调用次数上限；死循环会被自动拦截并引导换策略。")
    parts.append("")
    parts.append(ACTION_PROTOCOL)
    return "\n".join(parts)
