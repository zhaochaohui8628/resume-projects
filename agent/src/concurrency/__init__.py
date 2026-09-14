"""并发控制框架（零外部依赖）：异步事件驱动 + 令牌桶限流 + 共享内存上下文 + 依赖 DAG 裁剪。

设计目标（对应 v6 需求 1）：
- 不用「盲目多线程」：以单一 asyncio 事件循环调度多 Agent；
- LLM 推理并发限流：令牌桶（TokenBucket / RateLimitedLLM / acomplete）；
- 共享内存上下文：SharedContext 实现多 Agent 非阻塞状态同步与动态依赖裁剪；
- 动态依赖裁剪：AsyncPipeline 按依赖 DAG 执行，上游失败自动裁剪下游。

公开对象：
    TokenBucket / RateLimitedLLM / acomplete
    SharedContext
    AsyncPipeline / NodeCtx / NodeResult / SUCCESS / SKIPPED / FAILED / TIMEOUT
"""
from .token_bucket import TokenBucket, RateLimitedLLM, acomplete          # noqa: F401
from .model_slots import (ModelSlotLimiter, SlotTimeout,                  # noqa: F401
                          get_model_slots, reset_model_slots)
from .shared_context import SharedContext                                  # noqa: F401
from .pipeline import (AsyncPipeline, NodeCtx, NodeResult,                 # noqa: F401
                       SUCCESS, SKIPPED, FAILED, TIMEOUT, CANCELLED)

__all__ = [
    "TokenBucket", "RateLimitedLLM", "acomplete",
    "ModelSlotLimiter", "SlotTimeout", "get_model_slots", "reset_model_slots",
    "SharedContext",
    "AsyncPipeline", "NodeCtx", "NodeResult",
    "SUCCESS", "SKIPPED", "FAILED", "TIMEOUT", "CANCELLED",
]
