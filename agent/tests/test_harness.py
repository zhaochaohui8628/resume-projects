"""ReAct harness 冒烟测试（零依赖 mock LLM，不联网）。

覆盖：① 正常多步 ReAct（expand→tool→…→answer）② 同一指令工具调用 ≤5 guard（截断）
③ 渐进披露只展开一次 ④ 四层记忆读写与 episodic 检索 ⑤ skill invoke（rules stub）。
运行：pytest agent/tests/test_harness.py -q
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.join(os.path.dirname(HERE), "src")  # agent/src
for p in (AGENT_SRC, os.path.dirname(AGENT_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from harness.memory import MemoryManager  # noqa: E402
from harness.react_agent import ReActAgent  # noqa: E402
from harness.skills import Skill, SkillRegistry, build_compliance_skills  # noqa: E402


class FakeLLM:
    def __init__(self, script=None):
        self.script = list(script) if script else []
        self.i = 0

    @staticmethod
    def _default():
        return json.dumps({"thought": "直接作答", "action": "answer",
                           "action_input": "（mock 默认结论）"})

    def complete(self, messages, **kwargs):
        r = self.script[self.i] if self.i < len(self.script) else self._default()
        self.i += 1
        return r


class StreamFakeLLM(FakeLLM):
    """带 stream 的假 LLM：把整段 JSON 按 chunk 长度切开逐块吐（模拟真实流式分块）。

    逐块切分正好用来验证跨块转义（`\\n` 被切在 `\\` 与 `n` 之间）这类边界。
    """

    def __init__(self, script=None, chunk=7):
        super().__init__(script)
        self.chunk = chunk

    def stream(self, messages, **kwargs):
        text = self.complete(messages)          # 复用同一游标：每轮只会走 stream 或 complete
        for i in range(0, len(text), self.chunk):
            yield text[i:i + self.chunk]


def _demo_skills(log=None) -> SkillRegistry:
    reg = SkillRegistry()
    reg.register(Skill("check_rule", "规则检查（demo）",
                       "参数：方案文本。返回风险条数。", lambda s: "命中 2 条风险"))
    reg.register(Skill("lookup_std", "规范检索（demo）",
                       "参数：查询关键词。返回条文。", lambda s: "[JGJ120-2012] 基坑支护条文"))
    return reg


def _mem(tmp):
    return MemoryManager(os.path.join(tmp, "mem"))


def test_react_normal_flow():
    with tempfile.TemporaryDirectory() as td:
        llm = FakeLLM([
            json.dumps({"thought": "先查规则", "action": "check_rule", "action_input": "基坑"}),
            json.dumps({"thought": "执行规则检查", "action": "check_rule", "action_input": "基坑"}),
            json.dumps({"thought": "再查规范", "action": "lookup_std", "action_input": "基坑 论证"}),
            json.dumps({"thought": "执行规范检索", "action": "lookup_std", "action_input": "基坑 5m 论证"}),
            json.dumps({"thought": "够了", "action": "answer", "action_input": "结论：需专家论证"}),
        ])
        agent = ReActAgent(skills=_demo_skills(), memory=_mem(td), llm=llm)
        res = agent.run("基坑 5m 深合规吗？")
        assert res["tool_calls"] == 2
        assert res["answer"] == "结论：需专家论证"
        kinds = [t["kind"] for t in res["trace"]]
        assert "expand" in kinds and "tool" in kinds and "answer" in kinds
        # 渐进披露：check_rule / lookup_std 各只展开一次
        exp = [t for t in res["trace"] if t["kind"] == "expand"]
        assert len(exp) == 2 and len({t["skill"] for t in exp}) == 2
        # episodic 已记录
        assert len(agent.memory.episodic) == 1


def test_guard_max_tool_calls():
    """无硬编码工具上限（max_tool_calls 已删除）：纯图状态机拦截死循环。

    连续重复调用同一工具时，状态机在重复第 2 次后拦截 → 纠偏 → 耗尽 → 降级。
    """
    with tempfile.TemporaryDirectory() as td:
        # 预置已展开，省去 expand（纯计数）
        mem = _mem(td)
        mem.skill_mark_expanded("check_rule")
        script = [json.dumps({"thought": f"t{i}", "action": "check_rule", "action_input": "x"})
                  for i in range(20)]
        llm = FakeLLM(script)
        agent = ReActAgent(skills=_demo_skills(), memory=mem, llm=llm)
        res = agent.run("循环测试")
        # 不再有「调用 ≤5 次强制截断」：只有状态机拦截（重复第 2 次即拦截）
        assert res["tool_calls"] <= 2            # 状态机在重复第 2 次后拦截
        assert res["truncated"] is True          # 未收敛（纠偏耗尽 → 降级）
        assert res.get("degraded") is True       # 降级输出
        assert res.get("guard_reasons"), "应有拦截记录"
        assert res["guard_reasons"][0]["reason"] == "repeat_tool"
        assert any(t["kind"] == "guard" for t in res["trace"])
        assert "防御性控制流拦截" in res["answer"]


def test_defensive_guard_then_recover_and_answer():
    """重复工具拦截后模型换策略成功收敛：guard(repeat_tool) → 换工具 → answer。"""
    with tempfile.TemporaryDirectory() as td:
        mem = _mem(td)
        mem.skill_mark_expanded("check_rule")
        mem.skill_mark_expanded("lookup_std")
        cr = lambda t: json.dumps({"thought": t, "action": "check_rule", "action_input": "基坑"})
        script = [
            cr("先跑一次规则检查"),            # 轮1 执行 tc=1
            cr("按规范流程再核查一遍"),        # 轮2 执行 tc=2（thought 与轮1 差异大，避开 similar）
            cr("继续重复调用规则检查"),        # 轮3 第3次同工具 → repeat_tool 拦截
            json.dumps({"thought": "听从纠偏，换检索角度", "action": "lookup_std",
                        "action_input": "基坑 论证"}),   # 轮4 换工具执行 tc=3
            json.dumps({"thought": "够了", "action": "answer", "action_input": "结论：需专家论证"}),
        ]
        agent = ReActAgent(skills=_demo_skills(), memory=mem, llm=FakeLLM(script),
                           similarity_threshold=0.99)
        res = agent.run("基坑 5m 深合规吗？")
        guards = [t for t in res["trace"] if t["kind"] == "guard"]
        assert guards, "应发生拦截"
        assert guards[0]["reason"] == "repeat_tool"
        assert res["answer"] == "结论：需专家论证"
        assert res["truncated"] is False
        assert res["degraded"] is False
        # 工具调用：check_rule×2（执行）+ lookup_std×1（执行）
        assert res["tool_calls"] == 3


def test_defensive_guard_similar_thoughts():
    """相似中间结论：thought 高度相似连续轮次 → 状态向量余弦拦截。"""
    with tempfile.TemporaryDirectory() as td:
        mem = _mem(td)
        mem.skill_mark_expanded("lookup_std")
        same = json.dumps({"thought": "检索基坑支护相关内容以确认合规性",
                           "action": "lookup_std", "action_input": "基坑支护 符合规定"})
        script = [same, same,  # 轮1 执行；轮2 与轮1 余弦≈1 → similar 拦截（guard）
                  json.dumps({"thought": "够了", "action": "answer", "action_input": "done"})]
        agent = ReActAgent(skills=_demo_skills(), memory=mem, llm=FakeLLM(script),
                           max_recover=2,
                           repeat_tool_threshold=5)   # 关闭 repeat 干扰，只测 similar
        res = agent.run("基坑 支护 合规")
        guards = [t for t in res["trace"] if t["kind"] == "guard"]
        assert guards
        assert guards[0]["reason"] == "similar_thought"
        assert res["answer"] == "done"      # 拦截纠偏后模型换 answer 收敛
        assert res["truncated"] is False
        assert res["tool_calls"] == 1


def test_defensive_degrades_after_recover_exhausted():
    """纠偏耗尽仍不收敛 → 降级输出（返回已收集证据，明确说明未收敛）。"""
    with tempfile.TemporaryDirectory() as td:
        mem = _mem(td)
        mem.skill_mark_expanded("check_rule")
        script = [json.dumps({"thought": f"重复执行 {i}", "action": "check_rule",
                              "action_input": "x"}) for i in range(7)]
        agent = ReActAgent(skills=_demo_skills(), memory=mem, llm=FakeLLM(script),
                           max_recover=1)   # 只给 1 次纠偏
        res = agent.run("循环测试")
        assert res["degraded"] is True
        assert res["truncated"] is True
        assert "防御性控制流拦截" in res["answer"]
        assert res["guard_reasons"], "应有拦截记录"
        assert res["tool_calls"] <= 2       # 重复工具第 2 次即被拦截，未耗尽 5 次


def test_state_vector_cosine_deterministic():
    from harness.state_machine import state_vector, cosine
    v1 = state_vector("基坑支护 深度超过5米 需要专家论证")
    v2 = state_vector("基坑支护 深度超过5米 需要专家论证")
    v3 = state_vector("塔吊基础 混凝土强度 是否符合规定")
    assert abs(cosine(v1, v2) - 1.0) < 1e-9       # 相同文本 → 余弦 1（确定性）
    assert cosine(v1, v3) < cosine(v1, v2)        # 不同语义 → 更低
    assert cosine(v1, v3) < 0.6                   # 显著不同
    # 跨调用确定性（同一输入两次调用结果一致）
    assert state_vector("重复文本") == state_vector("重复文本")


def test_progressive_expand_once():
    with tempfile.TemporaryDirectory() as td:
        mem = _mem(td)
        llm = FakeLLM([
            json.dumps({"thought": "a", "action": "lookup_std", "action_input": "q1"}),
            json.dumps({"thought": "b", "action": "lookup_std", "action_input": "q2"}),
            json.dumps({"thought": "c", "action": "answer", "action_input": "ok"}),
        ])
        agent = ReActAgent(skills=_demo_skills(), memory=mem, llm=llm)
        res = agent.run("q")
        expands = [t for t in res["trace"] if t["kind"] == "expand"]
        tools = [t for t in res["trace"] if t["kind"] == "tool"]
        assert len(expands) == 1                # 只展开一次
        assert len(tools) == 1                  # 第二次调用直接执行
        assert agent.memory.skill_calls("lookup_std") == 1


def test_unknown_action_and_invalid_json():
    with tempfile.TemporaryDirectory() as td:
        llm = FakeLLM(["{bad json",  # 非法 → 走 invalid 分支（action 为空→非技能→invalid）
                       json.dumps({"thought": "ok", "action": "answer", "action_input": "done"})])
        agent = ReActAgent(skills=_demo_skills(), memory=_mem(td), llm=llm)
        res = agent.run("t")
        assert res["answer"] == "done"
        assert any(t["kind"] == "invalid" for t in res["trace"])


def test_memory_layers():
    with tempfile.TemporaryDirectory() as td:
        m = MemoryManager(os.path.join(td, "m"))
        m.semantic_set("项目", "上海；主体为医院应急救援中心")
        m.episodic_add("基坑5m需要论证吗", "需要，见 JGJ 120-2012", {"tool_calls": 2})
        m.episodic_add("临时用电三级配电", "JGJ/T46-2024 要求三级配电两级保护")
        m.skill_mark_expanded("lookup_std")
        m.skill_note_call("lookup_std")
        hits = m.episodic_search("基坑 论证", k=1)
        assert hits and "基坑5m" in hits[0]["query"]
        assert m.semantic_get("项目") and m.skill_expanded("lookup_std")
        assert m.skill_calls("lookup_std") == 1
        # 落盘重载
        m2 = MemoryManager(os.path.join(td, "m"))
        assert m2.skill_expanded("lookup_std") and len(m2.episodic) == 2


def test_skills_invoke_rules_stub():
    calls = []

    def stub_rules(text):
        calls.append(text)
        return {"risks": [{"severity": "HIGH", "check": "C1", "title": "引用废止规范 JGJ46-2005"}]}

    reg = build_compliance_skills(rules_checker=stub_rules)  # rag/ner 未注入 → 自动跳过
    assert "compliance_check" in reg.names()
    assert "retrieve_standards" not in reg.names()
    out = reg.get("compliance_check")("某方案")
    assert calls and "JGJ46-2005" in out
    assert "[HIGH]" in out


def test_react_answer_streaming_only_final():
    """qa 路流式（2026-09-17）：只推终答正文，中间轮次的协议 JSON 一律不外泄。"""
    with tempfile.TemporaryDirectory() as td:
        answer = "结论：基坑 5m 需专家论证。\n依据：DG/TJ08-61 第 16.2.1 条。"
        llm = StreamFakeLLM([
            json.dumps({"thought": "先查规范", "action": "lookup_std", "action_input": "基坑"}),
            json.dumps({"thought": "执行检索", "action": "lookup_std", "action_input": "基坑 5m"}),
            json.dumps({"thought": "够了", "action": "answer", "action_input": answer}),
        ])
        agent = ReActAgent(skills=_demo_skills(), memory=_mem(td), llm=llm)
        got: list[str] = []
        res = agent.run("基坑 5m 深合规吗？", on_token=got.append)
        joined = "".join(got)
        assert res["answer"] == answer
        assert joined == answer          # 分块内容拼接 == 终答全文（含 \n 转义已还原）
        assert len(got) > 1              # 确实分块推送，不是一次性
        for bad in ('"thought"', '"action"', "lookup_std", "执行检索"):
            assert bad not in joined     # 中间轮次内容没外泄


def test_react_stream_fallback_reversed_key_order():
    """键顺序颠倒（action_input 在 action 之前）→ 无法边收边解 → 整段兜底推送，不丢内容。"""
    with tempfile.TemporaryDirectory() as td:
        answer = "结论：不满足要求。"
        payload = '{"thought": "t", "action_input": "%s", "action": "answer"}' % answer
        llm = StreamFakeLLM([payload], chunk=5)
        agent = ReActAgent(skills=_demo_skills(), memory=_mem(td), llm=llm)
        got: list[str] = []
        res = agent.run("q", on_token=got.append)
        assert res["answer"] == answer
        assert "".join(got) == answer
        assert len(got) == 1             # 一次性兜底


if __name__ == "__main__":
    import traceback

    tests = [test_react_normal_flow, test_guard_max_tool_calls,
             test_defensive_guard_then_recover_and_answer,
             test_defensive_guard_similar_thoughts,
             test_defensive_degrades_after_recover_exhausted,
             test_state_vector_cosine_deterministic,
             test_progressive_expand_once,
             test_unknown_action_and_invalid_json, test_memory_layers,
             test_skills_invoke_rules_stub,
             test_react_answer_streaming_only_final,
             test_react_stream_fallback_reversed_key_order]
    for fn in tests:
        try:
            fn()
            print(f"OK {fn.__name__}")
        except Exception:
            traceback.print_exc()
            print(f"FAIL {fn.__name__}")
