"""统一 LLM 接口：DeepSeek API / 本地 Ollama / Mock 共用一个接口。

Agent 与 NER 都只依赖 `complete(messages) -> str`，切换后端不改业务代码。
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def complete(self, messages: list[dict], **kwargs) -> str:
        ...


def get_llm(provider: str = "mock", **kwargs):
    if provider == "deepseek":
        from .deepseek import DeepSeekLLM

        return DeepSeekLLM(**kwargs)
    if provider == "ollama":
        from .deepseek import DeepSeekLLM  # Ollama 兼容 OpenAI 协议

        base = kwargs.pop("base_url", "http://localhost:11434/v1")
        return DeepSeekLLM(base_url=base, **kwargs)
    if provider == "mock":
        from .mock import MockLLM

        return MockLLM(**kwargs)
    raise ValueError(f"未知 LLM provider: {provider}")
