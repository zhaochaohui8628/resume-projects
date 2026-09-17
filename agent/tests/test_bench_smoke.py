"""Benchmark 评测框架冒烟测试（零依赖：只测指标函数 + 快速用例）。

不跑全量 12 用例（避免每用例加载 bert 拖慢）；覆盖：
  ① 审查漏报/误报指标计算
  ② NER F1 指标
  ③ RAG recall@k 指标
  ④ 耗时拆解
  ⑤ run_case 对空输入/领域外用例不崩溃
  ⑥ 全量汇总 summarize

运行：pytest agent/tests/test_bench_smoke.py -q
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(os.path.dirname(HERE), "bench")
AGENT_SRC = os.path.join(os.path.dirname(HERE), "src")
for p in (BENCH_DIR, os.path.dirname(HERE), AGENT_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from bench import metrics as M           # noqa: E402
from bench.cases import BenchCase        # noqa: E402


def test_risk_metrics():
    golden = [{"severity": "HIGH", "frag": "JGJ46-2005"},
              {"severity": "HIGH", "frag": "专家论证"}]
    pred = [{"severity": "HIGH", "title": "引用已废止/失效标准：JGJ46-2005", "check": "C1"},
            {"severity": "HIGH", "title": "【基坑工程】超规模但未提及专家论证", "check": "C2"},
            {"severity": "LOW", "title": "与 golden 无关的额外项", "check": "C4"}]
    m = M.risk_metrics(golden, pred)
    assert m["fnr"] == 0.0, m        # 2/2 命中
    assert m["hit"] == 2 and m["golden"] == 2 and m["pred"] == 3
    assert 0.0 < m["fpr"] <= 1.0
    print("OK test_risk_metrics")


def test_risk_metrics_miss():
    golden = [{"frag": "模板支撑"}]
    pred = [{"title": "无关风险"}]
    m = M.risk_metrics(golden, pred)
    assert m["fnr"] == 1.0           # 完全漏报
    assert m["hit"] == 0
    print("OK test_risk_metrics_miss")


def test_ner_metrics():
    golden = [{"type": "规范编号", "text": "JGJ46-2005"},
              {"type": "参数", "text": "6"}]
    pred = [{"type": "规范编号", "text": "JGJ46-2005"},
            {"type": "参数", "text": "6m"},
            {"type": "危大类别", "text": "基坑工程"}]
    m = M.ner_metrics(golden, pred)
    # "6" 命中 "6m"（双向包含）→ hit=2；P=2/3, R=2/2=1
    assert m["recall"] == 1.0, m
    assert m["precision"] == round(2 / 3, 4)
    assert 0.0 < m["f1"] < 1.0
    print(f"OK test_ner_metrics: {m}")


def test_rag_recall():
    golden = {"query": "q", "sources": ["JGJ120", "JGJ311"]}
    hits = [{"source": "JGJ311-2013_xx"}, {"source": "JGJ59-2011_yy"}]
    m = M.rag_recall(golden, hits)
    assert m["recall@k"] == 0.5, m   # 命中 1/2
    assert m["hit"] == 1 and m["k"] == 2
    assert m["top_sources"][0] == "JGJ311-2013_xx"
    # 无 golden → None
    assert M.rag_recall(None, hits)["recall@k"] is None
    print("OK test_rag_recall")


def test_time_metrics():
    m = M.time_metrics(1234.5, {"hazard": 800.2, "qa": 400.1})
    assert m["total_ms"] == 1234.5
    assert m["slowest"] == "hazard"
    print("OK test_time_metrics")


def test_run_case_empty_and_ood_survive():
    """空输入 / 领域外用例：不崩溃、有合法输出（快速确定性路径）。"""
    from bench.runner import run_case
    from orchestrator.dispatcher import GlobalOpts
    opts = GlobalOpts(memory_dir=None)
    # 空输入 → qa 路径（无 bert 加载，快）
    r1 = run_case(BenchCase(id="empty", category="混沌/空输入", desc="",
                            query="", plan=""), llm=None, opts=opts)
    assert not r1.crash, r1.error
    assert r1.has_output
    # 领域外 → qa 路径
    r2 = run_case(BenchCase(id="ood", category="混沌/领域外", desc="",
                            query="写首诗", plan=""), llm=None, opts=opts)
    assert not r2.crash, r2.error
    assert r2.time_m["total_ms"] >= 0
    print(f"OK test_run_case_empty_and_ood_survive: empty={r1.time_m['total_ms']}ms ood={r2.time_m['total_ms']}ms")


def test_phase_metrics():
    """阶段拆解：从 trace.ms 聚合 路由/执行/汇总，ms=None 的批次回放步不计入。"""
    trace = [
        {"step": 1, "kind": "input", "label": "用户输入", "ms": 0.5},
        {"step": 2, "kind": "dispatch", "label": "调度中心意图路由", "ms": 2.0},
        {"step": 3, "kind": "execute", "label": "并行执行 1 个 subagent", "ms": 0.3},
        {"step": 4, "kind": "subagent_step", "label": "[review] 完成", "ms": 800.0},
        {"step": 5, "kind": "subagent_step", "label": "[review] 判档路", "ms": None},   # 批次回放
        {"step": 6, "kind": "aggregate", "label": "LLM 汇总", "ms": 0.2},
        {"step": 7, "kind": "done", "label": "完成", "ms": 120.0,
         "detail": {"llm_ms": 118.5}},
    ]
    ph = M.phase_metrics(trace)
    assert ph["route_ms"] == 2.0, ph
    assert ph["execute_ms"] == 800.3, ph          # 执行 = execute + 完成步；None 步被跳过
    assert ph["summary_ms"] == 118.7, ph          # aggregate 0.2 + done.detail.llm_ms 118.5
    assert ph["other_ms"] == 0.5, ph
    assert ph["measured_ms"] == 921.5, ph
    # 空 / 全 None trace 不炸
    assert M.phase_metrics([])["measured_ms"] == 0.0
    assert M.phase_metrics([{"kind": "done", "ms": None}])["summary_ms"] == 0.0
    # 老 trace（无 llm_ms）退回 done.ms
    ph2 = M.phase_metrics([{"kind": "aggregate", "ms": 1.0},
                           {"kind": "done", "ms": 200.0}])
    assert ph2["summary_ms"] == 201.0, ph2
    print("OK test_phase_metrics")


def test_summarize_aggregation():
    from bench.runner import CaseResult
    results = [
        CaseResult(id="a", category="c", desc="",
                   crash=False, has_output=True,
                   risk_m={"golden": 2, "fnr": 0.0, "fpr": 0.5, "hit": 2, "pred": 4},
                   ner_m={"f1": 0.8, "precision": 0.8, "recall": 0.8, "golden": 2, "pred": 2},
                   rag_m={"recall@k": 0.5, "hit": 1, "golden": 2, "k": 2},
                   time_m={"total_ms": 100.0, "subagent_ms": {"hazard": 60.0, "qa": 40.0}}),
        CaseResult(id="b", category="c", desc="",
                   crash=False, has_output=True,
                   risk_m={"golden": 0, "fnr": None, "fpr": None, "hit": 0, "pred": 0},
                   ner_m={"f1": None, "precision": None, "recall": None, "golden": 0, "pred": 0},
                   rag_m={"recall@k": None, "hit": 0, "golden": 0, "k": 0},
                   time_m={"total_ms": 200.0, "subagent_ms": {"hazard": 150.0}}),
    ]
    from bench.runner import summarize
    s = summarize(results)
    assert s["total_cases"] == 2 and s["crashed"] == 0
    assert s["avg_fnr"] == 0.0
    assert s["ner_avg_f1"] == 0.8          # 只统计有值的用例
    assert s["rag_avg_recall_at_k"] == 0.5
    assert s["avg_total_ms"] == 150.0
    assert s["avg_subagent_ms"]["hazard"] == 105.0
    print(f"OK test_summarize_aggregation: {s['avg_total_ms']}ms")


if __name__ == "__main__":
    import traceback
    tests = [test_risk_metrics, test_risk_metrics_miss, test_ner_metrics,
             test_rag_recall, test_time_metrics, test_phase_metrics,
             test_run_case_empty_and_ood_survive,
             test_summarize_aggregation]
    for fn in tests:
        try:
            fn()
            print(f"OK {fn.__name__}")
        except Exception:
            traceback.print_exc()
            print(f"FAIL {fn.__name__}")
