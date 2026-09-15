"""v6 并发控制框架测试（concurrency 包 + dispatcher 异步路径）。

覆盖：
  ① 令牌桶：try_acquire / acquire_sync / release / 异步 acquire
  ② RateLimitedLLM：限流包装 + 失败回滚令牌
  ③ SharedContext：set/publish + subscribe（非阻塞）+ wait/await_
  ④ AsyncPipeline：硬依赖动态裁剪 + 软依赖非阻塞状态同步 + 超时/重试
  ⑤ dispatcher.run_async：事件驱动编排 + 规则自检产物注入 hazard

运行：pytest agent/tests/test_concurrency.py -q
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.join(os.path.dirname(HERE), "src")
for p in (AGENT_SRC, os.path.dirname(AGENT_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from concurrency import (TokenBucket, SharedContext, AsyncPipeline,  # noqa: E402
                         RateLimitedLLM, SUCCESS, SKIPPED, FAILED, TIMEOUT)


# ================= 令牌桶 =================
class _FrozenClock:
    """把 token_bucket 依赖的 time.monotonic 冻结。

    为什么需要：`TokenBucket.available` 每次都会按 rate 连续补充令牌，
    rate=1000 时 26µs 就补 0.026 个 → 对 `available == 3` 这类精确相等断言
    会偶发失败（2026-09-15 全量回归实际观测到 3.0256）。
    冻结时钟后"成功调用消耗 1 个 / 失败回滚"的语义不变，且不受机器快慢影响。
    """

    def __init__(self, t: float = 1000000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _freeze_bucket_clock(monkeypatch):
    import concurrency.token_bucket as tb_mod
    monkeypatch.setattr(tb_mod.time, "monotonic", _FrozenClock())


def test_token_bucket_basic():
    b = TokenBucket(rate=1000, capacity=4)
    assert b.try_acquire(2)          # 突发可用
    assert not b.try_acquire(5)      # 超出容量
    b.release(2)
    assert b.try_acquire(2)
    print("OK test_token_bucket_basic")


def test_token_bucket_sync_wait():
    b = TokenBucket(rate=2.0, capacity=1)   # 每秒 2 令牌，容量 1
    t0 = time.perf_counter()
    assert b.acquire_sync(1.0, timeout=1.5)  # 立刻拿到（桶满）
    assert b.acquire_sync(1.0, timeout=1.5)  # 等约 0.5s 补充
    elapsed = time.perf_counter() - t0
    assert elapsed >= 0.4, f"应等待令牌补充，实际 {elapsed:.2f}s"
    assert not b.acquire_sync(1.0, timeout=0.1)   # 超时失败
    print(f"OK test_token_bucket_sync_wait ({elapsed:.2f}s)")


def test_token_bucket_async_acquire():
    async def main():
        b = TokenBucket(rate=2.0, capacity=2)   # 慢速补充，避免 0.05s 内回满
        assert await b.acquire(1.0, timeout=0.5)
        assert await b.acquire(1.0, timeout=0.5)
        assert not await b.acquire(1.0, timeout=0.05)   # 桶空且补充不足
    asyncio.run(main())
    print("OK test_token_bucket_async_acquire")


def test_rate_limited_llm_release_on_error(monkeypatch):
    _freeze_bucket_clock(monkeypatch)      # 冻结补充，避免浮点抖动

    class BoomLLM:
        def complete(self, messages, **kwargs):
            raise RuntimeError("api down")

    b = TokenBucket(rate=100, capacity=3)
    lim = RateLimitedLLM(BoomLLM(), bucket=b)
    try:
        lim.complete([{"role": "user", "content": "x"}])
        assert False, "应抛出异常"
    except RuntimeError:
        pass
    assert b.available == 3, "失败应回滚令牌"   # 全部归还
    print("OK test_rate_limited_llm_release_on_error")


def test_rate_limited_llm_stream_preserved(monkeypatch):
    """限流包装必须保留 stream()——否则 summarize_stream 探测不到，
    静默退回一次性 complete，UI 流式输出失效（2026-09-14 事故回归锁）。"""
    _freeze_bucket_clock(monkeypatch)      # 冻结补充，避免浮点抖动

    class StreamLLM:
        def complete(self, messages, **kwargs):
            return "整段"
        def stream(self, messages, **kwargs):
            for piece in ("流", "式", "输", "出"):
                yield piece

    b = TokenBucket(rate=1000, capacity=4)
    lim = RateLimitedLLM(StreamLLM(), bucket=b)
    assert hasattr(lim, "stream"), "限流包装必须透出 stream()"
    got = list(lim.stream([{"role": "user", "content": "x"}]))
    assert got == ["流", "式", "输", "出"], got
    assert b.available == 3, "成功调用消耗 1 个令牌"

    # 流式中途异常：已产出 token → 不回滚（用户已消费内容）
    class BoomStream:
        def stream(self, messages, **kwargs):
            yield "a"
            raise RuntimeError("mid-stream down")
    b2 = TokenBucket(rate=1000, capacity=3)
    lim2 = RateLimitedLLM(BoomStream(), bucket=b2)
    try:
        list(lim2.stream([{"role": "user", "content": "x"}]))
        assert False, "应抛出异常"
    except RuntimeError:
        pass
    assert b2.available == 2, "已产出内容不应回滚令牌"

    # 一个 token 都没产出就失败 → 回滚
    class BoomImmediate:
        def stream(self, messages, **kwargs):
            raise RuntimeError("no token")
    b3 = TokenBucket(rate=1000, capacity=3)
    lim3 = RateLimitedLLM(BoomImmediate(), bucket=b3)
    try:
        list(lim3.stream([{"role": "user", "content": "x"}]))
        assert False, "应抛出异常"
    except RuntimeError:
        pass
    assert b3.available == 3, "零产出失败应回滚令牌"

    # summarize_stream 经限流包装后仍走流式分支
    from orchestrator.aggregator import summarize_stream
    out = summarize_stream(lim, query="q", has_plan=False, agents=["qa"],
                           results=[])
    assert out == "流式输出", out
    print("OK test_rate_limited_llm_stream_preserved")


# ================= SharedContext =================
def test_shared_context_publish_subscribe():
    sc = SharedContext()
    seen = []
    sc.subscribe("detected_types", lambda k, v: seen.append(v))
    assert sc.publish("detected_types", ["基坑工程"]) is True
    assert sc.publish("detected_types", ["基坑工程"]) is False   # 未变化不通知
    assert seen == [["基坑工程"]]
    assert sc.get("detected_types") == ["基坑工程"]
    assert "detected_types" in sc
    print("OK test_shared_context_publish_subscribe")


def test_shared_context_wait():
    sc = SharedContext()

    def producer():
        time.sleep(0.3)
        sc.set("ready", True)

    import threading
    t = threading.Thread(target=producer)
    t.start()
    assert sc.wait("ready", timeout=2.0) is True
    t.join()
    assert sc.wait("nope", timeout=0.1) is None   # 超时返回 None
    print("OK test_shared_context_wait")


def test_shared_context_async_await():
    async def main():
        sc = SharedContext()
        async def setter():
            await asyncio.sleep(0.1)
            sc.set("k", "v")
        task = asyncio.ensure_future(setter())
        v = await sc.await_("k", timeout=1.0)
        await task
        assert v == "v"
    asyncio.run(main())
    print("OK test_shared_context_async_await")


# ================= AsyncPipeline =================
def test_pipeline_hard_dep_cut():
    """硬依赖失败 → 下游动态裁剪（skipped），不执行。"""
    async def main():
        executed = []
        pl = AsyncPipeline(max_concurrency=2)

        async def fail(ctx):
            raise RuntimeError("boom")
        async def downstream(ctx):
            executed.append("downstream")
            return "SHOULD_NOT_RUN"

        pl.add_node("fail", fail)
        pl.add_node("downstream", downstream, deps=["fail"])
        r = await pl.run()
        assert r["fail"].status == FAILED
        assert r["downstream"].status == SKIPPED
        assert executed == []
    asyncio.run(main())
    print("OK test_pipeline_hard_dep_cut")


def test_pipeline_soft_dep_shared_state():
    """软依赖：等上游完成（成败都继续），读取其发布产物（非阻塞状态同步）。"""
    async def main():
        pl = AsyncPipeline()

        async def a(ctx):
            await asyncio.sleep(0.05)
            ctx.publish("a.types", ["基坑工程"])
            return "A"
        async def soft(ctx):
            types = ctx.get("a.types", [])
            return f"types={types}"

        pl.add_node("a", a)
        pl.add_node("soft", soft, soft_deps=["a"])
        r = await pl.run()
        assert r["soft"].result == "types=['基坑工程']"
    asyncio.run(main())
    print("OK test_pipeline_soft_dep_shared_state")


def test_pipeline_retry_and_timeout():
    """节点重试 + 超时。"""
    async def main():
        pl = AsyncPipeline()
        tries = {"n": 0}

        async def flaky(ctx):
            tries["n"] += 1
            if tries["n"] < 3:
                raise ValueError("transient")
            return "ok"
        async def slow(ctx):
            await asyncio.sleep(5)

        pl.add_node("flaky", flaky, retries=2)
        pl.add_node("slow", slow, timeout=0.2)
        r = await pl.run()
        assert r["flaky"].status == SUCCESS and r["flaky"].attempts == 3
        assert r["slow"].status == TIMEOUT
    asyncio.run(main())
    print("OK test_pipeline_retry_and_timeout")


def test_pipeline_cycle_detection():
    """多分支 DAG 正常执行（依赖按拓扑推进，无死锁）。"""
    async def main():
        order = []
        async def x(ctx):
            order.append("x")
            return "x"
        async def y(ctx):
            order.append("y")
            return "y"
        async def z(ctx):
            order.append("z")
            return "z"

        pl2 = AsyncPipeline(max_concurrency=4)
        pl2.add_node("x", x)
        pl2.add_node("y", y, deps=["x"])
        pl2.add_node("z", z, deps=["x"])
        r = await pl2.run()
        assert r["x"].status == SUCCESS and r["y"].status == SUCCESS
        assert r["z"].status == SUCCESS
        # x 必须先于 y/z 完成
        assert order.index("x") < order.index("y")
        assert order.index("x") < order.index("z")
    asyncio.run(main())
    print("OK test_pipeline_cycle_detection")


# ================= dispatcher.run_async =================
def test_dispatcher_run_async():
    """事件驱动编排 v7.2：意图路由 → **一个** review subagent（内部三路）→ 汇总。"""
    from orchestrator.dispatcher import run_async, GlobalOpts
    plan = "基坑工程开挖深度 6m。编制依据：JGJ46-2005。\n未提及专家论证。模板支撑搭设高度 9m。"

    class FakeLLM:
        def __init__(self):
            self.i = 0

        def complete(self, messages, **kwargs):
            self.i += 1
            if self.i == 1:
                return json.dumps({"intent": "review"})
            return "（mock 汇总）"

    with tempfile.TemporaryDirectory() as td:
        opts = GlobalOpts(memory_dir=td, llm_rate=50, llm_burst=4, max_concurrency=4,
                          allow_degrade=True)
        res = asyncio.run(run_async(query="全面审查方案", plan=plan,
                                    llm=FakeLLM(), global_opts=opts))
        assert res["agents"] == ["review"]
        assert res["routing"]["intent"] == "review"
        kinds = [t["kind"] for t in res["trace"]]
        for k in ("input", "dispatch", "execute", "aggregate", "done"):
            assert k in kinds, f"trace 缺 {k}：{kinds}"
        assert res["global_opts"]["llm_rate"] == 50
        # 依据/要素路在 review 内部执行（不再是独立规则节点）
        labels = " ".join(t.get("label", "") for t in res["trace"])
        assert "依据/要素路" in labels
        names = [r.name for r in res["results"]]
        assert names == ["review"], names
    print(f"OK test_dispatcher_run_async: trace={len(res['trace'])} steps, agents={res['agents']}")