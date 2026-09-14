"""Mock LLM：无 API Key 时驱动全链路测试与演示。

可传入 scripted 响应序列（用于模拟 ReAct 多步）；默认返回「直接结束」动作。
"""
from __future__ import annotations

from .base import LLMBackend


class MockLLM(LLMBackend):
    def __init__(self, script: list[str] = None):
        self.script = list(script) if script else [self._default()]
        self._i = 0

    @staticmethod
    def _default():
        return '{"thought":"无可用LLM，转为确定性规则检查","action":"finish","action_input":""}'

    def complete(self, messages: list[dict], **kwargs) -> str:
        resp = self.script[self._i] if self._i < len(self.script) else self._default()
        self._i += 1
        return resp
