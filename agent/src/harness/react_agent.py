"""ReAct Harness 主循环 v6.1（agent/src/harness/react_agent.py）。

特性：
- 技能驱动：动作 = 注册表中的技能名 或 answer（终答）；
- 渐进式披露：技能首次被选定时才注入其完整使用说明（展开一次），此前模型只见 name+一行描述；
- **防御性控制流（v6.1）**：不设硬编码工具调用次数上限（已删除 max_tool_calls 截断）。
  防死循环完全由图状态机承担——DefensiveStateMachine 在 THINK→TOOL 转换前经
  TransitionValidator 校验：重复调用同一工具 ≥ 阈值、或 2-3 轮产出相似中间结论
  （状态向量余弦 ≥ 阈值）即触发拦截，主动引导模型切换搜索策略或换检索角度；
  纠偏次数（max_recover）耗尽仍不收敛 → 降级输出（返回已收集证据 + 明确说明未收敛）。
  max_iterations 仅作整体推理轮数的循环上界（防御性保险，非工具调用截断）；
- 记忆注入：system 含长期约定（semantic）与相似历史（episodic top-k），技能展开/调用记入
  procedural 记忆；
- LLM 后端抽象：DeepSeek / Mock 等统一 complete(messages) -> str。

用法：
    agent = ReActAgent(skills=reg, memory=mem, llm=llm)
    result = agent.run("帮我核查这段方案的基坑部分")
"""
from __future__ import annotations

import json
import re

from .memory import MemoryManager
from .skills import SkillRegistry
from .system_prompt import build_system_prompt
from .state_machine import DefensiveStateMachine, TransitionValidator


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


class ReActAgent:
    def __init__(self, skills: SkillRegistry, memory: MemoryManager = None,
                 llm=None, max_iterations: int = 16,
                 repeat_tool_threshold: int = 2,
                 similarity_threshold: float = 0.92,
                 max_recover: int = 2):
        self.skills = skills
        self.memory = memory or MemoryManager("agent/data/memory")
        self.llm = llm
        self.max_iterations = max_iterations
        # v6 防御性控制流（图状态机；不再有硬编码工具调用上限）
        self._sm = DefensiveStateMachine(
            validator=TransitionValidator(
                repeat_tool_threshold=repeat_tool_threshold,
                similarity_threshold=similarity_threshold),
            max_recover=max_recover)

    # ---------------- 对外入口 ----------------
    def run(self, user_input: str) -> dict:
        if self.llm is None:
            return {"answer": "（未配置 LLM）", "trace": [], "tool_calls": 0, "truncated": False}
        self.memory.working_set("query", user_input)
        episodic_hits = self.memory.episodic_search(user_input, k=3)
        sys_prompt = build_system_prompt(
            skills_brief=self.skills.brief(),
            semantic_mem=self.memory.semantic,
            episodic_hits=episodic_hits,
        )
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_input}]
        trace: list[dict] = []
        tool_calls = 0
        last_obs = "（尚未调用工具）"
        guard_reasons: list[dict] = []
        degraded = False
        self._sm.reset()

        for _ in range(self.max_iterations):
            resp = self.llm.complete(messages)
            messages.append({"role": "assistant", "content": resp})
            act = _extract_json(resp) or {}
            thought = str(act.get("thought", ""))
            action = str(act.get("action", "")).strip()
            action_input = str(act.get("action_input", "") or "")

            # ---------- 1) 终答 ----------
            if action in ("answer", "finish"):
                self._sm.on_action(action, thought, action_input, is_tool=False)
                trace.append({"step": len(trace) + 1, "kind": "answer",
                              "thought": thought, "content": action_input})
                self._remember(user_input, action_input, tool_calls, trace)
                return {"answer": action_input, "trace": trace,
                        "tool_calls": tool_calls, "truncated": False,
                        "degraded": False, "guard_reasons": guard_reasons}

            # ---------- 2) 图状态机转换校验（THINK→TOOL 边，唯一防死循环机制） ----------
            skill = self.skills.get(action)
            is_tool = skill is not None
            decision = self._sm.on_action(action, thought, action_input, is_tool=is_tool)

            if is_tool and not decision.ok:
                # ---- 拦截：死循环信号 ----
                guard_reasons.append(decision.to_dict())
                if self._sm.is_degrading:
                    # 纠偏耗尽 → 降级输出（保留已收集证据，明确说明未收敛）
                    degraded = True
                    fallback = self._degrade_answer(decision, thought, last_obs)
                    trace.append({"step": len(trace) + 1, "kind": "answer",
                                  "thought": thought, "content": fallback,
                                  "degraded": True,
                                  "guard_reason": decision.reason})
                    self._remember(user_input, fallback, tool_calls, trace)
                    return {"answer": fallback, "trace": trace,
                            "tool_calls": tool_calls, "truncated": True,
                            "degraded": True, "guard_reasons": guard_reasons}
                # 纠偏：注入策略切换指令，本轮不执行技能
                trace.append({"step": len(trace) + 1, "kind": "guard",
                              "thought": thought, "action": action,
                              "reason": decision.reason,
                              "detail": dict(decision.detail)})
                messages.append({"role": "user", "content": decision.suggestion})
                continue

            # ---------- 3) 调用技能 ----------
            if skill is None:
                obs = f"未知动作「{action}」。可用技能：{', '.join(self.skills.names()) or '无'}；或直接 answer。"
                trace.append({"step": len(trace) + 1, "kind": "invalid", "thought": thought, "action": action})
            else:
                # 渐进披露：首次选定该技能时只展开说明，不执行
                if not self.memory.skill_expanded(skill.name):
                    self.memory.skill_mark_expanded(skill.name)
                    obs = (f"[技能展开] 首次调用「{skill.name}」，其完整使用说明如下：\n"
                           f"{skill.doc}\n请按其说明重新输出本次 action_input（无需再次展开）。")
                    trace.append({"step": len(trace) + 1, "kind": "expand",
                                  "skill": skill.name, "thought": thought})
                else:
                    tool_calls += 1
                    self.memory.skill_note_call(skill.name)
                    try:
                        obs = skill(action_input)
                    except Exception as e:
                        obs = f"（技能 {skill.name} 执行异常：{e}）"
                    last_obs = obs
                    trace.append({"step": len(trace) + 1, "kind": "tool", "skill": skill.name,
                                  "thought": thought, "input": action_input[:120]})
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        # 达到 max_iterations（整体轮数上界）仍未 answer
        fallback = f"已达最大推理轮数（{self.max_iterations}），未能得到明确结论。最后思考：{thought}"
        return {"answer": fallback, "trace": trace,
                "tool_calls": tool_calls, "truncated": True,
                "degraded": False, "guard_reasons": guard_reasons}

    def _degrade_answer(self, decision, thought: str, last_obs: str) -> str:
        """纠偏耗尽后的降级输出：把已收集证据交给用户，明确说明未收敛原因。"""
        reason = decision.reason
        detail = decision.detail
        reason_txt = {
            "repeat_tool": f"重复调用同一工具 {detail.get('count', '?')} 次",
            "similar_thought": f"连续轮次中间结论相似（余弦 {detail.get('similarity', '?')}）",
        }.get(reason, reason)
        obs = (last_obs or "").strip()
        evidence = f"\n已收集证据：{obs[:300]}" if obs and obs != "（尚未调用工具）" else ""
        return (f"⚠️ 防御性控制流拦截：{reason_txt}，多次纠偏后仍无法收敛，"
                f"基于现有证据降级输出：\n{thought}{evidence}")

    def _remember(self, query, outcome, tool_calls, trace):
        self.memory.episodic_add(query, outcome, meta={"tool_calls": tool_calls,
                                                       "steps": len(trace)})
