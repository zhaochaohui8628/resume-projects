"""共享内存上下文（零依赖，线程安全 + asyncio 双模式）。

解决多 Agent 并发下的两个问题：
1) 非阻塞状态同步：一个 subagent 产出中间结论（如 compliance 检出危大类型）后
   publish 到共享上下文，其他 subagent 通过订阅回调「感知到即可」，不必等待对方；
2) 动态依赖裁剪依据：后续节点用 has()/get(default) 判断上游产物是否可用，
   上游失败/未产出 → 下游跳过该依赖（裁剪），降级走兜底路径。

线程安全（RLock）：同时支持 ThreadPoolExecutor 与 asyncio 事件循环场景。
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Callable, Optional

Subscriber = Callable[[str, Any], None]


class SharedContext:
    """进程内共享内存上下文：set/publish + subscribe + snapshot。

    - set(key, value)：写入；值变化时在锁外触发订阅回调（fire-and-forget，不阻塞写入方）
    - publish(key, value)：语义同 set（对外统一叫 publish，便于 Agent 间协作表达）
    - subscribe(key, cb)：注册回调；set 该 key 时触发（含已存在的值，可选 replay）
    - get(key, default)：读取；key 不存在返回 default —— 动态依赖裁剪的判定依据
    - has(key)：上游产物是否已就绪
    - wait(key, timeout)：同步阻塞等待某产物就绪（线程池场景下的依赖等待）
    - snapshot()：全量快照（trace / 审计用）
    """

    def __init__(self, initial: Optional[dict] = None):
        self._data: dict[str, Any] = dict(initial or {})
        self._meta: dict[str, dict] = {}          # key -> {ts, publisher}
        self._subs: dict[str, list[Subscriber]] = defaultdict(list)
        self._cv = threading.Condition(threading.RLock())
        self._lock = threading.RLock()

    # ---------------- 写入 / 发布 ----------------
    def set(self, key: str, value: Any, publisher: str = "") -> bool:
        """写入键值；返回是否发生变化（值不同）。变化才通知订阅者。"""
        with self._cv:
            changed = self._data.get(key) != value
            self._data[key] = value
            self._meta[key] = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "publisher": publisher}
            if changed:
                self._cv.notify_all()
        if changed:
            self._notify(key, value)
        return changed

    def publish(self, key: str, value: Any, publisher: str = "") -> bool:
        """Agent 间协作发布（语义 = set，命名更语义化）。"""
        return self.set(key, value, publisher=publisher)

    # ---------------- 读取 / 查询 ----------------
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def meta(self, key: str) -> dict:
        with self._lock:
            return dict(self._meta.get(key, {}))

    # ---------------- 订阅（非阻塞状态同步） ----------------
    def subscribe(self, key: str, cb: Subscriber, replay: bool = False) -> None:
        """注册 key 变更回调。replay=True 时若 key 已存在则立即回调一次。"""
        with self._lock:
            self._subs[key].append(cb)
            if replay and key in self._data:
                value = self._data[key]
        if replay and key in self._data:
            try:
                cb(key, value)
            except Exception:
                pass

    def _notify(self, key: str, value: Any) -> None:
        """在锁外触发回调（非阻塞：不等待回调返回）。"""
        with self._lock:
            subs = list(self._subs.get(key, ()))
        for cb in subs:
            try:
                cb(key, value)
            except Exception:
                pass

    # ---------------- 同步等待（线程池场景） ----------------
    def wait(self, key: str, timeout: Optional[float] = None) -> Optional[Any]:
        """阻塞等待 key 就绪。超时返回 None。"""
        deadline = time.monotonic() + timeout if timeout is not None else None
        with self._cv:
            while key not in self._data:
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                self._cv.wait(timeout=0.05)
            return self._data[key]

    async def await_(self, key: str, timeout: Optional[float] = None) -> Optional[Any]:
        """asyncio 版本：非阻塞轮询等待 key 就绪。"""
        import asyncio
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout if timeout is not None else None
        while True:
            v = self.get(key, _MISSING)
            if v is not _MISSING:
                return v
            if deadline is not None and loop.time() >= deadline:
                return None
            await asyncio.sleep(0.02)

    # ---------------- 快照 / 清理 ----------------
    def snapshot(self) -> dict:
        with self._lock:
            return {k: (v if not isinstance(v, (dict, list)) else _deep_copy(v))
                    for k, v in self._data.items()}

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._meta.clear()
            self._subs.clear()

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data)


_MISSING = object()


def _deep_copy(v):
    try:
        import copy
        return copy.deepcopy(v)
    except Exception:
        return v
