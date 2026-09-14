"""NER 客户端 v7：对接 **ner2** 项目（级联管道 + 全量分块识别）。

v7 修复（2026-09-13，重要）：
  **原代码 `_ensure()` 里写了裸变量 `device`，而 `__init__` 从未定义它** →
  每次都抛 `NameError: name 'device' is not defined`，被 `except` 静默吞掉并降级成
  "纯规则层"，表现成"NerClient 不可用/内存不足"，实际上是**代码 bug**（与内存无关）。
  正确写法是 `self.device`（并在 `__init__` 中保存）。

同时新增**显式状态** `model_loaded`：构造后**预热一次**确认模型真的能推理，
避免"看起来加载成功、实际每次都在降级"。上层（review subagent）据此在严格模式下报错。

backend：'ner2'（默认，规则层 + s2_crf_param_v3 级联）/ 'rule'（纯规则，零依赖）。
接口保持 `extract(text)` 兼容 agent 其余模块。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.dirname(HERE)
AGENT = os.path.dirname(AGENT_SRC)
WORKSPACE = os.path.dirname(AGENT)          # 项目根（agent/ 的上级）
NER2_SRC = os.path.join(WORKSPACE, "ner2")
# v7.3.1 修复（2026-09-13）：import 写的是 `from ner2.src.pipeline...`，
# `ner2` 是**工作区根下的包**（WS/ner2），所以 sys.path 里必须是**工作区根**（ner2 的父目录），
# 而不是 ner2/ 自身。此前只加 NER2_SRC：
#   - pytest / 交互（cwd=工作区根）能跑通（cwd 兜底），单测全绿；
#   - 但服务端 `python agent/src/fastapi_app.py` 是脚本模式，sys.path[0]=agent\src、cwd 不在
#     path → `import ner2` 扫不到 → ModuleNotFoundError: No module named 'ner2'。
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from ner2.src.pipeline.full_text import DEFAULT_MODEL, FullTextExtractor  # noqa: E402
from concurrency import get_model_slots  # noqa: E402  （重型模型并发槽位限流）

def _auto_device() -> str:
    """自动选设备：ner2 内部默认 `device or "cuda"`，在 CPU 版 torch 上会抛
    `AssertionError: Torch not compiled with CUDA enabled` → 表现为"模型不可用/严格模式报错"。
    这里显式探测，无 CUDA 就走 cpu（慢但可用）。"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


_PROBE = "基坑开挖深度 8m，执行 JGJ120-2012。"


class NerClient:
    def __init__(self, backend: str = "ner2", model_path: str | None = None,
                 device: str | None = None, batch: int = 64):
        """backend='ner2' 级联（规则层 + 微调模型）；'rule' 纯规则（零依赖）。"""
        self.kind = backend
        self.model_path = model_path if model_path is not None else DEFAULT_MODEL
        self.device = device                # ← v7 修复：原来没存，_ensure 里用了裸变量
        self.batch = batch
        self._fx = None
        self._warn = ""
        self.model_loaded = False           # 模型是否真的可用（预热后确定）

    def _load(self):
        """真正加载/预热模型（**不取槽位**，供 _ensure / extract 在外层统一取槽，避免嵌套死锁）。"""
        if self._fx is None:
            if self.kind == "rule" or not self.model_path or not os.path.exists(self.model_path):
                self._fx = FullTextExtractor(model_path=None)
                self._warn = "" if self.kind == "rule" else f"[缺模型产物 {self.model_path}] "
                return self._fx
            try:
                fx = FullTextExtractor(model_path=self.model_path,
                                      device=self.device or _auto_device())
                fx.extract_text(_PROBE)     # 预热：确认模型真的能推理（不能只看构造成功）
                self._fx = fx
                self.model_loaded = True
            except Exception as e:          # noqa: BLE001
                self._warn = f"[ner2 模型加载失败：{type(e).__name__}: {e}] "
                self.model_loaded = False
                self._fx = FullTextExtractor(model_path=None)   # 降级纯规则（上层可据此拒绝降级）
        return self._fx

    def _ensure(self):
        """对外预热入口：带模型槽位限流（并发满载时排队等待）。"""
        if self._fx is not None:
            return self._fx
        with get_model_slots().slot(cost=1.0, name="ner.load"):
            return self._load()

    def extract(self, text: str, funnel=None) -> list[dict]:
        """全量分块抽取（funnel 参数保留仅为兼容旧签名，**不使用**）。

        重型模型推理统一走 `ModelSlotLimiter` 槽位闸门：同进程并发受控，
        避免多请求同时加载 torch 模型把提交内存打爆（WinError 1455）。

        返回实体：{"type","text","start","end","layer","conf","sent_idx","sent_text"}。
        """
        if not text:
            return []
        with get_model_slots().slot(cost=1.0, name="ner.infer"):
            fx = self._load()
            try:
                return fx.extract_text(text)
            except Exception as e:              # noqa: BLE001
                self._warn = f"[ner2 推理失败：{type(e).__name__}: {e}] "
                self.model_loaded = False
                return [{"error": str(e), "type": "ERROR", "text": ""}]
