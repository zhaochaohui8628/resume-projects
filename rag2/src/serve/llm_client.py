"""LLM 客户端 —— OpenAI 兼容接口（默认 DeepSeek API，可切任意 OpenAI 兼容服务）。

- 默认：DeepSeek API（https://api.deepseek.com/v1，model=deepseek-chat），
  读取环境变量 DEEPSEEK_API_KEY（缺失则报错提示）。
- 可覆盖：LLMClient(base_url="http://127.0.0.1:8000/v1", model="qwen2.5-3b")
  → 切回本地 qwen_server / SGLang，零代码改动。
- 零依赖：纯 urllib，不引入 openai 库。
- 用法：
    from serve.llm_client import LLMClient
    cli = LLMClient()                                  # DeepSeek API
    ans = cli.chat([{"role": "user", "content": "..."}], max_tokens=512)
"""
from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


class LLMClient:
    def __init__(self, base_url: str = None, model: str = None,
                 timeout: float = 120.0, api_key: str = None):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

    def _headers(self) -> dict:
        if "api.deepseek.com" in self.base_url:
            if not self.api_key:
                raise RuntimeError(
                    "DeepSeek API 需要 DEEPSEEK_API_KEY 环境变量（或传 api_key）")
            return {"Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"}
        return {"Content-Type": "application/json"}

    def chat(self, messages: list[dict], max_tokens: int = 512,
             temperature: float = 0.1) -> str:
        """单轮对话 → 返回 assistant 文本。连接失败抛异常（调用方决定降级）。"""
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM 返回空 choices: {data}")
        return choices[0]["message"]["content"]

    def available(self) -> bool:
        """健康检查（DeepSeek 用 models 接口；本地服务用 /health）。"""
        try:
            if "api.deepseek.com" in self.base_url:
                req = urllib.request.Request(
                    f"{self.base_url}/models", headers=self._headers())
            else:
                req = urllib.request.Request(
                    f"{self.base_url.replace('/v1', '')}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False
