"""agent/bench 评测包：多 Agent 级联容错评测 Benchmark。

模块：
  cases           对抗性测试用例集（12 类混沌场景 + golden 标注）
  cases_real      真实施工方案用例集（11 本真实方案 → 33 用例，交叉验证标注）
  metrics         指标计算（漏报率/误报率/NER F1/RAG recall@k/耗时拆解）
  runner          端到端执行 orchestrator + 全链路指标收集
  report          Markdown 报告生成
  run_benchmark   CLI 入口
"""
__all__ = ["cases", "cases_real", "metrics", "runner", "report", "run_benchmark"]
