"""重型模型调用槽位闸门（并发限流，保护进程内存/提交内存）。

与 `token_bucket.TokenBucket` 的区别（**语义不同，勿混用**）：
- TokenBucket = **速率**限流（令牌按时间补充，管"每秒多少次"）→ 用于 LLM API 调用；
- ModelSlotLimiter = **并发**限流（许可按占用归还，管"同时多少个在跑"）→ 用于本地重型模型
  （torch NER / CrossEncoder 精排 / 双塔编码）。

为什么需要它：Windows 上 torch + NER(BERT-CRF) + CrossEncoder 同进程并发加载时，
瞬时提交内存会超出页面文件，报 `OSError [WinError 1455] 页面文件太小`。
把重型模型调用限制在 N 个并发以内，即把峰值提交内存压在安全线以下；
超出的请求**排队等待**（可设超时），而不是让它们一起炸内存。

特性：
- 加权许可（cost）：不同模型内存代价不同（如 NER=1、CE=1、双塔=1）；
- 同步 + asyncio 双模式，均支持 `with` / `async with`；
- 超时可配（默认等 180s），超时抛 `SlotTimeout`；
- 统计：in_flight / waited / wait_ms / timeouts，便于观测与调参；
- 零第三方依赖。

用法：
    from concurrency import get_model_slots
    slots = get_model_slots()                     # 全局单例（读 config resources.*）
    with slots.slot(cost=1.0, name="ner"):
        ents = ner_client.extract(text)           # 真正的重型推理
"""
from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager, asynccontextmanager
from typing import Optional


class SlotTimeout(TimeoutError):
    """等待模型槽位超时（并发已满且在超时时间内未等到）。"""


class ModelSlotLimiter:
    """并发槽位闸门（信号量语义 + 加权 + 超时 + 统计）。"""

    def __init__(self, capacity: float = 2.0, timeout: Optional[float] = 180.0,
                 name: str = "model"):
        if capacity <= 0:
            raise ValueError("capacity 必须 > 0")
        self.capacity = float(capacity)
        self.timeout = timeout
        self.name = name
        self._in_flight = 0.0
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        # 统计
        self.waited = 0            # 发生过排队等待的次数
        self.timeouts = 0          # 等待超时次数
        self.wait_ms_total = 0.0   # 累计等待耗时
        self.acquired = 0          # 累计获得槽位次数

    # ---------------- 内部 ----------------
    def _try_take(self, cost: float) -> bool:
        if self._in_flight + cost <= self.capacity + 1e-9:
            self._in_flight += cost
            return True
        return False

    # ---------------- 同步接口 ----------------
    def try_acquire(self, cost: float = 1.0) -> bool:
        """非阻塞尝试占用 cost 个槽位。"""
        with self._lock:
            if self._try_take(cost):
                self.acquired += 1
                return True
            return False

    def acquire_sync(self, cost: float = 1.0, timeout: Optional[float] = None) -> bool:
        """阻塞占用；超时返回 False（不抛异常，交给调用方决定）。"""
        timeout = self.timeout if timeout is None else timeout
        t0 = time.perf_counter()
        deadline = time.monotonic() + timeout if timeout is not None else None
        with self._cv:
            first = True
            while True:
                if self._try_take(cost):
                    if not first:
                        self.waited += 1
                        self.wait_ms_total += (time.perf_counter() - t0) * 1000
                    self.acquired += 1
                    return True
                first = False
                if deadline is not None and time.monotonic() >= deadline:
                    self.timeouts += 1
                    self.wait_ms_total += (time.perf_counter() - t0) * 1000
                    return False
                self._cv.wait(timeout=0.05)

    def release(self, cost: float = 1.0) -> None:
        """归还槽位。"""
        with self._cv:
            self._in_flight = max(0.0, self._in_flight - cost)
            self._cv.notify_all()

    @contextmanager
    def slot(self, cost: float = 1.0, timeout: Optional[float] = None, name: str = ""):
        """同步上下文：进入前取槽位，退出必归还（异常也归还）。"""
        ok = self.acquire_sync(cost=cost, timeout=timeout)
        if not ok:
            raise SlotTimeout(
                f"模型槽位等待超时（{timeout if timeout is not None else self.timeout}s）："
                f"当前并发已满 capacity={self.capacity:g}，任务 {name or self.name} 未获得许可。")
        try:
            yield self
        finally:
            self.release(cost)

    # ---------------- 异步接口 ----------------
    async def acquire(self, cost: float = 1.0, timeout: Optional[float] = None) -> bool:
        """asyncio 版本：非阻塞自旋等待，不卡事件循环。"""
        timeout = self.timeout if timeout is None else timeout
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        deadline = loop.time() + timeout if timeout is not None else None
        first = True
        while True:
            if self.try_acquire(cost):
                if not first:
                    self.waited += 1
                    self.wait_ms_total += (time.perf_counter() - t0) * 1000
                return True
            first = False
            if deadline is not None and loop.time() >= deadline:
                self.timeouts += 1
                self.wait_ms_total += (time.perf_counter() - t0) * 1000
                return False
            await asyncio.sleep(0.02)

    @asynccontextmanager
    async def aslot(self, cost: float = 1.0, timeout: Optional[float] = None, name: str = ""):
        """异步上下文：进入前取槽位，退出必归还。"""
        ok = await self.acquire(cost=cost, timeout=timeout)
        if not ok:
            raise SlotTimeout(
                f"模型槽位等待超时（异步，{timeout if timeout is not None else self.timeout}s）："
                f"任务 {name or self.name} 未获得许可。")
        try:
            yield self
        finally:
            self.release(cost)

    # ---------------- 状态 ----------------
    @property
    def in_flight(self) -> float:
        with self._lock:
            return self._in_flight

    def stats(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "capacity": self.capacity,
                "in_flight": self._in_flight,
                "acquired": self.acquired,
                "waited": self.waited,
                "timeouts": self.timeouts,
                "wait_ms_total": round(self.wait_ms_total, 1),
            }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"ModelSlotLimiter(capacity={s['capacity']:g}, in_flight={s['in_flight']:g}, "
                f"waited={s['waited']}, timeouts={s['timeouts']})")


# ================= 全局单例（读 config，进程内共享）=================
_DEFAULT = ModelSlotLimiter(capacity=2.0, timeout=180.0, name="model")


def _from_config() -> ModelSlotLimiter:
    """从 config/config.yaml 的 resources 段构造；读不到则用默认值。"""
    import os
    try:
        import yaml  # type: ignore
    except Exception:
        return _DEFAULT
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_path = os.path.join(root, "config", "config.yaml")
    if not os.path.exists(cfg_path):
        return _DEFAULT
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        r = (cfg.get("resources") or {})
        cap = float(r.get("model_slots", 2.0))
        to = r.get("slot_timeout", 180.0)
        return ModelSlotLimiter(capacity=cap, timeout=(None if to in (None, 0) else float(to)),
                                name="model")
    except Exception:
        return _DEFAULT


def get_model_slots() -> ModelSlotLimiter:
    """获取全局模型槽位闸门（懒加载，进程内唯一）。"""
    global _DEFAULT
    if getattr(get_model_slots, "_inst", None) is None:
        get_model_slots._inst = _from_config()
    return get_model_slots._inst


def reset_model_slots(limiter: Optional[ModelSlotLimiter] = None) -> None:
    """重置单例（测试用）。"""
    get_model_slots._inst = limiter
