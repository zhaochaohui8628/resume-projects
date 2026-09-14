"""异步事件驱动流水线：asyncio + 依赖 DAG + 动态依赖裁剪（零依赖）。

替代「盲目多线程并行」：用单一事件循环调度节点，节点按依赖 DAG 拓扑执行；
上游节点失败/被跳过 → 下游自动裁剪（skipped），降级走兜底。

特性：
- add_node(node_id, coro, deps, soft_deps)   coro(node_ctx: NodeCtx) -> Any
- 硬依赖 deps：任一非 success → 本节点动态裁剪（skipped，不执行）
- 软依赖 soft_deps：等待依赖完成（成败都继续），配合 SharedContext
  做「非阻塞状态同步」——依赖先完成可复用其发布产物，失败也不阻断
- 并发限流：semaphore（max_concurrency）
- 单节点 timeout / retries
- SharedContext 透传，节点间非阻塞共享状态
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .shared_context import SharedContext

# 节点状态
SUCCESS = "success"
SKIPPED = "skipped"       # 依赖不满足被动态裁剪
FAILED = "failed"
TIMEOUT = "timeout"
CANCELLED = "cancelled"


@dataclass
class NodeResult:
    id: str
    status: str = SUCCESS
    result: Any = None
    error: str = ""
    duration: float = 0.0
    deps: tuple = ()
    attempts: int = 1
    ts: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "status": self.status, "duration": round(self.duration, 3),
                "error": self.error[:200], "attempts": self.attempts,
                "deps": list(self.deps), "ts": self.ts}


@dataclass
class Node:
    id: str
    coro: Callable[["NodeCtx"], Awaitable[Any]]
    deps: tuple = ()                 # 硬依赖：非 success → 本节点动态裁剪
    soft_deps: tuple = ()            # 软依赖：等待完成（成败都继续），用于非阻塞状态同步
    timeout: Optional[float] = None
    retries: int = 0
    weight: float = 1.0


@dataclass
class NodeCtx:
    """节点执行上下文：可读共享状态 / 写回共享状态 / 引用其他节点结果。"""
    node_id: str
    shared: SharedContext
    results: dict
    event_loop: Any = None

    def get(self, key, default=None):
        return self.shared.get(key, default)

    def publish(self, key, value, publisher=""):
        return self.shared.publish(key, value, publisher=publisher or self.node_id)

    def dep(self, dep_id) -> Optional[NodeResult]:
        return self.results.get(dep_id)

    def dep_status(self, dep_id) -> str:
        r = self.results.get(dep_id)
        return r.status if r else "missing"


class AsyncPipeline:
    def __init__(self, max_concurrency: int = 4, shared: Optional[SharedContext] = None,
                 default_timeout: Optional[float] = None, default_retries: int = 0):
        self.max_concurrency = max_concurrency
        self.shared = shared or SharedContext()
        self.default_timeout = default_timeout
        self.default_retries = default_retries
        self.nodes: dict[str, Node] = {}

    def add_node(self, node_id: str, coro, deps: list[str] | None = None,
                 soft_deps: list[str] | None = None,
                 timeout: Optional[float] = None, retries: int = 0,
                 weight: float = 1.0) -> "AsyncPipeline":
        for d in (deps or []) + (soft_deps or []):
            if d not in self.nodes:
                raise KeyError(f"依赖节点 {d} 尚未注册（node={node_id}）")
        self.nodes[node_id] = Node(
            id=node_id, coro=coro, deps=tuple(deps or ()),
            soft_deps=tuple(soft_deps or ()),
            timeout=self.default_timeout if timeout is None else timeout,
            retries=self.default_retries if retries == 0 else retries,
            weight=weight)
        return self

    async def run(self, timeout: Optional[float] = None) -> dict[str, NodeResult]:
        """执行全部节点，返回 {node_id: NodeResult}。节点结果有序（按 deps 保证）。"""
        results: dict[str, NodeResult] = {}
        sem = asyncio.Semaphore(self.max_concurrency)
        node_futures: dict[str, asyncio.Future] = {}

        def make_result(nid: str, status: str, **kw) -> NodeResult:
            n = self.nodes[nid]
            return NodeResult(id=nid, status=status, deps=n.deps, **kw)

        async def run_one(nid: str) -> None:
            node = self.nodes[nid]
            fut = node_futures[nid]

            async def _await_dep(d: str, hard: bool) -> Optional[NodeResult]:
                """等待单个依赖；hard=True 且失败 → 返回 None 表示裁剪。"""
                df = node_futures.get(d)
                if df is None:
                    return None
                try:
                    dr: NodeResult = await asyncio.wait_for(asyncio.shield(df),
                                                            timeout=timeout or 3600)
                except asyncio.TimeoutError:
                    return None
                except Exception:
                    return None
                if hard and dr.status != SUCCESS:
                    return None
                return dr

            # ---- 硬依赖：任一失败 → 动态裁剪 ----
            for d in node.deps:
                dr = await _await_dep(d, hard=True)
                if dr is None:
                    fut.set_result(make_result(nid, SKIPPED, error=f"依赖 {d} 不可用，动态裁剪"))
                    return

            # ---- 软依赖：等待完成（成败都继续）→ 供共享状态非阻塞同步 ----
            for d in node.soft_deps:
                await _await_dep(d, hard=False)

            # ---- 执行节点（semaphore 限流 + timeout + retries）----
            async with sem:
                attempts = 0
                while True:
                    attempts += 1
                    ctx = NodeCtx(node_id=nid, shared=self.shared,
                                  results=results, event_loop=asyncio.get_running_loop())
                    t0 = time.perf_counter()
                    try:
                        node_timeout = node.timeout or timeout
                        if node_timeout:
                            res = await asyncio.wait_for(node.coro(ctx), timeout=node_timeout)
                        else:
                            res = await node.coro(ctx)
                        fut.set_result(make_result(
                            nid, SUCCESS, result=res,
                            duration=time.perf_counter() - t0,
                            attempts=attempts,
                            ts=time.strftime("%H:%M:%S")))
                        return
                    except asyncio.TimeoutError:
                        if attempts <= node.retries:
                            continue
                        fut.set_result(make_result(
                            nid, TIMEOUT,
                            error=f"执行超时（{node.timeout}s），重试 {node.retries} 次后放弃",
                            duration=time.perf_counter() - t0, attempts=attempts))
                        return
                    except Exception as e:
                        if attempts <= node.retries:
                            continue
                        fut.set_result(make_result(
                            nid, FAILED, error=str(e)[:300],
                            duration=time.perf_counter() - t0, attempts=attempts))
                        return

        # ---- 拓扑执行：动态推进（每轮调度所有「可运行」节点）----
        # 注册 future
        for nid in self.nodes:
            node_futures[nid] = asyncio.get_running_loop().create_future()

        # 分层拓扑排序执行，保证依赖先完成
        def topo_order() -> list[str]:
            order, visited, temp = [], set(), set()
            def dfs(nid):
                if nid in temp:
                    raise RuntimeError(f"存在循环依赖：{nid}")
                if nid in visited:
                    return
                temp.add(nid)
                for d in self.nodes[nid].deps:
                    dfs(d)
                temp.discard(nid)
                visited.add(nid)
                order.append(nid)
            for nid in self.nodes:
                dfs(nid)
            return order

        tasks = [asyncio.ensure_future(run_one(nid)) for nid in topo_order()]
        await asyncio.gather(*tasks)

        for nid in self.nodes:
            results[nid] = node_futures[nid].result()
        return results

    # ---------------- 便捷 ----------------
    def statuses(self, results: dict[str, NodeResult]) -> dict[str, str]:
        return {k: v.status for k, v in results.items()}
