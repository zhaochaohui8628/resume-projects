"""DeepSeek（兼容 OpenAI 协议）LLM 后端。"""
from __future__ import annotations

import os

try:
    from .base import LLMBackend          # 作为 llm 包成员导入
except ImportError:
    from base import LLMBackend           # 顶层脚本直接 import deepseek 时


class DeepSeekLLM(LLMBackend):
    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        api_key_env: str = "DEEPSEEK_API_KEY",
        api_key: str = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"未设置 DeepSeek API Key。请设环境变量 {api_key_env}（你面试模拟平台桌面 config.json 里的 key 可复用）。"
            )

    def complete(self, messages: list[dict], **kwargs) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("调用 DeepSeek 需要 openai 库：pip install openai")
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict], **kwargs):
        """流式生成：逐增量 yield 文本片段（首 token 即出，降低 TTFT）。

        用法同 complete，但把 `stream=True` 交给 OpenAI SDK 并逐 chunk 产出。
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("调用 DeepSeek 需要 openai 库：pip install openai")
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        kwargs.pop("stream", None)
        resp = client.chat.completions.create(model=self.model, messages=messages,
                                              stream=True, **kwargs)
        for chunk in resp:
            if not getattr(chunk, "choices", None):
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            piece = getattr(delta, "content", None) if delta is not None else None
            if piece:
                yield piece
