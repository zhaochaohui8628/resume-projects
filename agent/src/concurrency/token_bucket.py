"""令牌桶限流器（零依赖，线程安全 + asyncio 双模式）。

用于大模型推理并发限流：所有 LLM 调用（路由 / 汇总 / PromptNER / ReAct 循环）
统一经过令牌桶，避免盲目多线程并发把 API 打爆、触发限流或超时雪崩。

- rate      每秒补充令牌数（tokens/sec）
- capacity  桶容量（允许的突发量）
- acquire_sync()  同步阻塞获取（线程池 / 同步编排用）
- acquire()       异步获取（asyncio 编排用）
- release()       归还令牌（调用失败回滚，可选）
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional


class TokenBucket:
    """标准令牌桶。线程安全；asyncio 版本内部自旋等待（不持有锁等待）。"""

    def __init__(self, rate: float, capacity: int, initial: Optional[int] = None):
        if rate <= 0:
            raise ValueError("rate 必须 > 0")
        if capacity <= 0:
            raise ValueError("capacity 必须 > 0")
        self.rate = float(rate)
        self.capacity = int(capacity)
        self.tokens = float(initial if initial is not None else capacity)
        self._ts = time.monotonic()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    # ---------------- 内部 ----------------
    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._ts
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self._ts = now

    def _acquire_locked(self, n: float) -> bool:
        self._refill()
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

    # ---------------- 同步接口 ----------------
    def try_acquire(self, n: float = 1.0) -> bool:
        """非阻塞尝试获取 n 个令牌。"""
        with self._lock:
            return self._acquire_locked(n)

    def acquire_sync(self, n: float = 1.0, timeout: Optional[float] = None) -> bool:
        """阻塞获取（最多等 timeout 秒；None = 一直等）。"""
        deadline = time.monotonic() + timeout if timeout is not None else None
        with self._cv:
            while True:
                if self._acquire_locked(n):
                    return True
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                self._cv.wait(timeout=0.05)

    def release(self, n: float = 1.0) -> None:
        """归还令牌（调用失败时回滚，防止额度被无效消耗）。"""
        with self._lock:
            self.tokens = min(self.capacity, self.tokens + n)
            self._cv.notify_all()

    # ---------------- 异步接口 ----------------
    async def acquire(self, n: float = 1.0, timeout: Optional[float] = None) -> bool:
        """asyncio 版本：非阻塞自旋等待，不阻塞事件循环。"""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout if timeout is not None else None
        while True:
            if self.try_acquire(n):
                return True
            if deadline is not None and loop.time() >= deadline:
                return False
            await asyncio.sleep(0.02)

    # ---------------- 状态 ----------------
    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self.tokens

    def __repr__(self) -> str:
        return (f"TokenBucket(rate={self.rate}, capacity={self.capacity}, "
                f"tokens={self.available:.2f})")


class RateLimitedLLM:
    """LLM 同步包装：complete() 前经令牌桶限流（同步调用场景：路由/汇总/线程池）。

    用法：
        llm = RateLimitedLLM(deepseek, bucket=TokenBucket(rate=2.0, capacity=4))
        resp = llm.complete(messages)      # 超出速率时阻塞等待
    """

    def __init__(self, llm, bucket: TokenBucket, timeout: Optional[float] = None):
        self._llm = llm
        self.bucket = bucket
        self.timeout = timeout
        self.wait_ms: list[float] = []      # 记录每次等待耗时（评测用）

    def complete(self, messages: list[dict], **kwargs) -> str:
        t0 = time.perf_counter()
        ok = self.bucket.acquire_sync(1.0, timeout=self.timeout)
        self.wait_ms.append((time.perf_counter() - t0) * 1000)
        if not ok:
            raise TimeoutError(
                f"令牌桶限流等待超时（{self.timeout}s）：LLM 推理并发超出配额，请降低并发或提高 rate。")
        try:
            return self._llm.complete(messages, **kwargs)
        except Exception:
            self.bucket.release(1.0)        # 调用失败回滚令牌
            raise

    def stream(self, messages: list[dict], **kwargs):
        """流式透传（限流在首个 token 前 acquire 一次，与 complete 同权重）。

        ⚠️ 此前缺失本方法 → summarize_stream 的 getattr(llm,'stream') 拿不到 →
        LLM 汇总静默退回一次性生成，UI 表现为"不是流式输出"。
        回滚策略：一个 token 都没产出就失败 → 回滚令牌；已产出 → 不回滚
        （用户已消费到内容，不能重复计费）。
        """
        t0 = time.perf_counter()
        ok = self.bucket.acquire_sync(1.0, timeout=self.timeout)
        self.wait_ms.append((time.perf_counter() - t0) * 1000)
        if not ok:
            raise TimeoutError(
                f"令牌桶限流等待超时（{self.timeout}s）：LLM 推理并发超出配额，请降低并发或提高 rate。")
        got_any = False
        try:
            for piece in self._llm.stream(messages, **kwargs):
                got_any = True
                yield piece
        except Exception:
            if not got_any:
                self.bucket.release(1.0)    # 一个 token 都没产出才回滚
            raise


async def acomplete(llm, bucket: Optional[TokenBucket], messages: list[dict],
                    timeout: Optional[float] = None, **kwargs) -> str:
    """异步 LLM 调用：令牌桶限流 + to_thread 执行真实推理（asyncio 编排用）。

    返回 (resp)。bucket 为 None 时不限流。
    """
    if bucket is not None:
        ok = await bucket.acquire(1.0, timeout=timeout)
        if not ok:
            raise TimeoutError("令牌桶限流等待超时（异步）：LLM 推理并发超出配额。")
    try:
        return await asyncio.to_thread(llm.complete, messages, **kwargs)
    except Exception:
        if bucket is not None:
            bucket.release(1.0)
        raise
