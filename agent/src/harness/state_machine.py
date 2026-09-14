"""图状态机驱动的防御性控制流（零依赖）。

背景：ReAct 硬编码「≤5 步截断」治标不治本——Agent 在第 3 步陷入逻辑死胡同、
重复调同一工具或反复产出相似中间结论时，强行截断只会导致输出残缺或逻辑崩塌。
本模块把 ReAct 循环建模为**图状态机**：

  节点 = 回合状态（THINK → TOOL → OBSERVE → ANSWER / GUARD）
  边   = 节点转换，由 **TransitionValidator（转换校验器）** 把关

校验器在 THINK→TOOL 转换前检查两条死循环信号：
1) 重复工具调用：最近 window 轮内同一技能出现 ≥ repeat_tool_threshold 次 → 拦截；
2) 相似中间结论：当前轮与最近 window 轮内任一历史轮的状态向量
   （thought+action_input 的特征哈希词袋）余弦相似度 ≥ similarity_threshold → 拦截。

拦截后返回策略建议：切换搜索策略 / 换检索角度 / 降级输出（answer）。
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------- 图状态 ----------------
class State(Enum):
    THINK = "think"        # 模型思考（默认起始态）
    TOOL = "tool"          # 调用工具
    OBSERVE = "observe"    # 观察工具返回
    ANSWER = "answer"      # 终答
    GUARD = "guard"        # 被校验器拦截，注入纠偏指令
    DEGRADE = "degrade"    # 纠偏耗尽，降级输出


# 允许的转换（图邻接表）
_EDGES = {
    State.THINK: {State.TOOL, State.ANSWER, State.GUARD, State.DEGRADE},
    State.TOOL: {State.OBSERVE, State.GUARD},
    State.OBSERVE: {State.THINK, State.ANSWER, State.GUARD, State.DEGRADE},
    State.GUARD: {State.THINK, State.ANSWER, State.DEGRADE},
    State.ANSWER: set(),
    State.DEGRADE: set(),
}


# ---------------- 状态向量（特征哈希词袋 + 余弦） ----------------
def _stable_hash(text: str) -> int:
    """确定性哈希（djb2 变体，跨进程稳定，不依赖 PYTHONHASHSEED）。"""
    h = 5381
    for ch in text:
        h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
    return h


def state_vector(text: str, dim: int = 256) -> list[float]:
    """把文本映射为 L2 归一化特征向量：字符 bigram + 字符 的稳定哈希词袋。

    对中文无需分词，字符级 bigram 足以区分"检索角度"层面的相似/不相似。
    """
    s = text or ""
    v = [0.0] * dim
    for i in range(len(s)):
        gram = s[i] + (s[i + 1] if i + 1 < len(s) else "")
        v[_stable_hash(gram) % dim] += 1.0
        v[_stable_hash(s[i]) % dim] += 0.5
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-12:
        return v
    return [x / norm for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


# ---------------- 校验结果 ----------------
@dataclass
class TransitionResult:
    ok: bool = True                    # True=放行；False=拦截
    reason: str = ""                   # "" | "repeat_tool" | "similar_thought"
    detail: dict = field(default_factory=dict)
    suggestion: str = ""               # 给模型的纠偏指令（拦截时由 state_machine 生成）

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reason": self.reason,
                "detail": dict(self.detail), "suggestion": self.suggestion[:200]}


# ---------------- 转换校验器 ----------------
class TransitionValidator:
    """节点转换校验器：在 TOOL 转换前检查死循环信号。"""

    def __init__(self, repeat_tool_threshold: int = 2,
                 similarity_threshold: float = 0.92, window: int = 3):
        self.repeat_tool_threshold = max(2, repeat_tool_threshold)   # 连续/近窗重复 ≥ 阈值拦截
        self.similarity_threshold = similarity_threshold             # 中间结论余弦阈值
        self.window = max(2, window)

    def check(self, action: str, thought: str, action_input: str,
              history: list[dict]) -> TransitionResult:
        """history: [{action, vector}]（已完成转换的历史轮，不含当前轮）。"""
        # 重复工具
        recent = [h for h in history[-self.window:] if h["action"] == action]
        if len(recent) >= self.repeat_tool_threshold:
            return TransitionResult(
                ok=False, reason="repeat_tool",
                detail={"skill": action, "count": len(recent) + 1,
                        "threshold": self.repeat_tool_threshold},
                suggestion=f"你已在最近 {len(recent) + 1} 轮内重复调用技能「{action}」，且未见进展。"
                           "请停止重复：切换到一个不同技能，或改写本次输入措辞（换检索/检查角度），"
                           "若已无新信息可查，请直接 answer 给出基于现有证据的结论。")
        # 相似中间结论
        cur_vec = state_vector(f"{thought} {action_input}")
        best = 0.0
        for h in history[-self.window:]:
            if h["action"] == action and h.get("vector") is not None:
                sim = cosine(cur_vec, h["vector"])
                if sim > best:
                    best = sim
        if best >= self.similarity_threshold:
            return TransitionResult(
                ok=False, reason="similar_thought",
                detail={"similarity": round(best, 4),
                        "threshold": self.similarity_threshold},
                suggestion=f"你最近 {self.window} 轮推理与历史高度相似（余弦 {best:.3f}），"
                           "疑似在同一思路上打转。请换一个检索/检查角度重新表述问题，"
                           "或直接 answer 基于现有证据降级输出结论，不要无意义循环。")
        return TransitionResult(ok=True)


# ---------------- 图状态机 ----------------
class DefensiveStateMachine:
    """图状态机驱动的防御性控制流。

    用法（ReAct 主循环内，每次模型输出动作后）：
        result = sm.on_action(action, thought, action_input, is_tool=action in skill_names)
        if not result.ok:
            # 拦截：注入 result.suggestion，进入 GUARD，等待模型纠偏
    """

    def __init__(self, validator: Optional[TransitionValidator] = None,
                 max_recover: int = 2):
        self.validator = validator or TransitionValidator()
        self.state = State.THINK
        self.max_recover = max_recover
        self.guard_count = 0
        self.history: list[dict] = []       # 已完成转换的工具轮（状态向量）
        self.guards: list[dict] = []        # 拦截记录

    # ---------------- 转换合法性 ----------------
    def _can(self, to: State) -> bool:
        return to in _EDGES.get(self.state, set())

    def _transition(self, to: State) -> None:
        self.state = to

    # ---------------- 对外主入口 ----------------
    def on_action(self, action: str, thought: str, action_input: str,
                  is_tool: bool) -> TransitionResult:
        """模型输出一个动作后调用。

        - answer/finish → 放行（转 ANSWER）
        - 工具动作 → 校验：命中死循环信号 → 拦截（转 GUARD / DEGRADE）
        - 非工具且非法 → 放行（由上层处理 invalid）
        """
        if action in ("answer", "finish"):
            self._transition(State.ANSWER)
            return TransitionResult(ok=True)

        if not is_tool:
            # 非法动作（unknown）：不参与死循环判定，放行由上层提示
            return TransitionResult(ok=True)

        # THINK→TOOL 转换校验
        res = self.validator.check(action, thought, action_input, self.history)
        if res.ok:
            self.history.append({"action": action,
                                 "vector": state_vector(f"{thought} {action_input}")})
            self._transition(State.TOOL)
            return res

        # 拦截：进入 GUARD；纠偏次数耗尽 → DEGRADE
        self.guard_count += 1
        self.guards.append({"reason": res.reason, "detail": dict(res.detail),
                            "action": action, "guard_count": self.guard_count})
        if self.guard_count >= self.max_recover:
            res.detail["degrade"] = True
            self._transition(State.DEGRADE)
        else:
            self._transition(State.GUARD)
        return res

    def reset(self) -> None:
        """一轮任务结束后重置（每个 user_input 独立状态机）。"""
        self.state = State.THINK
        self.guard_count = 0
        self.history.clear()
        self.guards.clear()

    @property
    def is_degrading(self) -> bool:
        return self.state == State.DEGRADE

    @property
    def guard_summary(self) -> list[dict]:
        return [dict(g) for g in self.guards]
