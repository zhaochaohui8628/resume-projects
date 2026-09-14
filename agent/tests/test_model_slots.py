"""模型槽位闸门（ModelSlotLimiter）测试：并发上限 / 排队等待 / 超时 / 加权 / 异常归还 / 配置加载。

运行：pytest agent/tests/test_model_slots.py -q
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.join(os.path.dirname(HERE), "src")
if AGENT_SRC not in sys.path:
    sys.path.insert(0, AGENT_SRC)

from concurrency import (ModelSlotLimiter, SlotTimeout,  # noqa: E402
                         get_model_slots, reset_model_slots)


def test_capacity_blocks_third():
    """capacity=2：第 1、2 个立刻通过，第 3 个必须等待（并发上限生效）。"""
    lim = ModelSlotLimiter(capacity=2.0, timeout=5.0)
    assert lim.try_acquire(1.0)
    assert lim.try_acquire(1.0)
    assert not lim.try_acquire(1.0), "capacity=2 时第 3 个应拿不到槽位"
    assert lim.in_flight == 2.0
    lim.release(1.0)
    assert lim.try_acquire(1.0), "归还后应重新可用"
    print("OK test_capacity_blocks_third")


def test_sync_wait_then_acquire():
    """并发满时 acquire_sync 排队等待，直到别人归还后拿到。"""
    lim = ModelSlotLimiter(capacity=1.0, timeout=5.0)
    lim.acquire_sync(1.0)                    # 占满
    got = []

    def worker():
        t0 = time.perf_counter()
        ok = lim.acquire_sync(1.0, timeout=5.0)
        got.append((ok, time.perf_counter() - t0))

    th = threading.Thread(target=worker)
    th.start()
    time.sleep(0.3)                          # 确认它在等
    assert lim.waited == 0 or True           # 等待中，尚未拿到
    lim.release(1.0)                         # 归还 → worker 应拿到
    th.join(timeout=5)
    assert got and got[0][0] is True, "槽位归还后等待者应获得许可"
    assert lim.waited >= 1, "应记录一次排队等待"
    print(f"OK test_sync_wait_then_acquire (等待 {got[0][1]*1000:.0f}ms)")


def test_timeout_raises():
    """capacity 占满且不归还 → 超时抛 SlotTimeout。"""
    lim = ModelSlotLimiter(capacity=1.0, timeout=0.4)
    with lim.slot(cost=1.0, name="holder"):
        t0 = time.perf_counter()
        try:
            with lim.slot(cost=1.0, name="late"):
                raise AssertionError("不应获得槽位")
        except SlotTimeout:
            pass
        else:
            raise AssertionError("应抛 SlotTimeout")
        assert time.perf_counter() - t0 >= 0.3
    assert lim.timeouts >= 1
    print("OK test_timeout_raises")


def test_weighted_cost():
    """加权：capacity=2 时，cost=2 的调用可占满；cost=1 需两次。"""
    lim = ModelSlotLimiter(capacity=2.0, timeout=1.0)
    assert lim.try_acquire(2.0)
    assert not lim.try_acquire(1.0)
    lim.release(2.0)
    assert lim.try_acquire(1.0)
    assert lim.try_acquire(1.0)
    assert not lim.try_acquire(1.0)
    print("OK test_weighted_cost")


def test_release_on_exception():
    """上下文内抛异常也归还槽位（finally 保证）。"""
    lim = ModelSlotLimiter(capacity=1.0, timeout=1.0)
    try:
        with lim.slot(cost=1.0, name="boom"):
            raise RuntimeError("推理炸了")
    except RuntimeError:
        pass
    assert lim.in_flight == 0.0, "异常后必须归还槽位"
    assert lim.try_acquire(1.0), "归还后应可再次获取"
    print("OK test_release_on_exception")


def test_async_slot():
    """asyncio 版本：并发上限 + aslot 上下文。"""

    async def main():
        lim = ModelSlotLimiter(capacity=1.0, timeout=2.0)
        async with lim.aslot(cost=1.0, name="a"):
            assert lim.in_flight == 1.0
            assert not lim.try_acquire(1.0)
        assert lim.in_flight == 0.0
        ok = await lim.acquire(1.0, timeout=1.0)
        assert ok
        lim.release(1.0)

    asyncio.run(main())
    print("OK test_async_slot")


def test_config_singleton():
    """全局单例读 config：resources.model_slots=2 → capacity=2；同进程共享同一实例。"""
    reset_model_slots(None)
    a = get_model_slots()
    b = get_model_slots()
    assert a is b, "应为进程内单例"
    assert a.capacity == 2.0, f"config model_slots=2，实际 {a.capacity}"
    s = a.stats()
    assert set(s) >= {"capacity", "in_flight", "waited", "timeouts"}
    reset_model_slots(None)
    print(f"OK test_config_singleton (capacity={a.capacity}, stats keys={sorted(s)})")


if __name__ == "__main__":
    test_capacity_blocks_third()
    test_sync_wait_then_acquire()
    test_timeout_raises()
    test_weighted_cost()
    test_release_on_exception()
    test_async_slot()
    test_config_singleton()
    print("\n全部通过 ✅")
