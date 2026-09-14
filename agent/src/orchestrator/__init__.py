"""orchestrator 包：LLM 路由调度（dispatcher）+ LLM 汇总（aggregator）。"""
from .dispatcher import run, route, execute, _REGISTRY     # noqa: F401
from .aggregator import summarize, to_markdown               # noqa: F401