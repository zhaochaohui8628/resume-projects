"""RAG→LLM 汇总链路测试：LLMClient / llm_summarize（fake LLM，不依赖真实服务）。

运行：pytest rag2/tests/test_llm_summarize.py -q
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAG2_SRC = os.path.join(os.path.dirname(HERE), "src")
if RAG2_SRC not in sys.path:
    sys.path.insert(0, RAG2_SRC)

from serve.llm_client import LLMClient                     # noqa: E402
from retrieval.llm_summarize import summarize, _build_user  # noqa: E402


class FakeLLM:
    """返回固定答案，带 [1] 引文。"""
    def chat(self, messages, max_tokens=512, temperature=0.1):
        return "搭设高度不宜超过7m，见[1] DG-TJ08-61 16.2.1。"


class DeadLLM:
    def chat(self, messages, max_tokens=512, temperature=0.1):
        raise ConnectionError("connection refused")


def _hits():
    return [{
        "clause_id": "DG-TJ08-61-2018_基坑工程技术标准::16.2.1",
        "uid": "u1", "score": 0.91,
        "text": "放坡开挖基坑开挖深度不宜超过7.0m。",
        "metadata": {"source": "DG-TJ08-61-2018", "clause_no": "16.2.1"},
    }]


def test_build_user_includes_citations():
    """条文编号 + 出处拼进用户消息。"""
    u = _build_user("基坑开挖深度8m行吗", _hits())
    assert "[1] DG-TJ08-61-2018 16.2.1" in u
    assert "基坑开挖深度8m行吗" in u
    print("OK test_build_user_includes_citations")


def test_summarize_with_llm():
    """LLM 汇总：答案 + 引文抽取。"""
    out = summarize("基坑开挖深度8m行吗", _hits(), llm=FakeLLM())
    assert out["degraded"] is False
    assert "7m" in out["answer"]
    assert out["citations"] == ["1"]
    print("OK test_summarize_with_llm")


def test_summarize_llm_down_degrades():
    """LLM 连接失败 → 降级（不抛异常），degraded=True。"""
    out = summarize("基坑开挖深度8m行吗", _hits(), llm=DeadLLM())
    assert out["degraded"] is True
    assert "不可用" in out["answer"]
    print("OK test_summarize_llm_down_degrades")


def test_summarize_no_hits():
    """无命中 → 直接返回提示，不调 LLM。"""
    out = summarize("xx", [], llm=FakeLLM())
    assert out["hits_n"] == 0
    assert "未检索到" in out["answer"]
    print("OK test_summarize_no_hits")


def test_llm_client_base_url_default():
    """客户端默认指向 DeepSeek API（env 无覆盖时）。"""
    os.environ.pop("LLM_BASE_URL", None)
    os.environ.pop("LLM_MODEL", None)
    c = LLMClient()
    assert c.base_url == "https://api.deepseek.com/v1"
    assert c.model == "deepseek-chat"
    print("OK test_llm_client_base_url_default")


def test_llm_client_base_url_override():
    """可切回本地 OpenAI 兼容服务（base_url 覆盖）。"""
    c = LLMClient(base_url="http://127.0.0.1:8000/v1", model="qwen2.5-3b")
    assert c.base_url == "http://127.0.0.1:8000/v1"
    assert c.model == "qwen2.5-3b"
    print("OK test_llm_client_base_url_override")


def test_llm_client_requires_api_key_for_deepseek():
    """DeepSeek 需要 api_key，缺失明确报错（不静默）。"""
    os.environ.pop("DEEPSEEK_API_KEY", None)
    c = LLMClient()  # 默认 DeepSeek
    import pytest
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        c.chat([{"role": "user", "content": "hi"}])
    print("OK test_llm_client_requires_api_key_for_deepseek")


if __name__ == "__main__":
    for f in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        f()
    print("ALL PASS")
