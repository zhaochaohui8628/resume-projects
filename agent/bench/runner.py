"""Benchmark runner：对每个对抗性用例端到端跑 orchestrator，收集全链路指标。

- 崩溃鲁棒性：用例执行不抛异常 + 有合法输出
- 审查漏报/误报：调度中心规则自检 risks（res.rules_result） vs golden_risks
  （v6.1：compliance subagent 已删除，规则自检为调度中心固定 tool）
- NER F1：hazard entities vs golden_entities
- RAG recall@k：各 subagent rag_hits vs golden_rag
- 耗时拆解：总耗时 + 各 subagent 单独计时（基准分解，非并行口径）

零依赖可用：llm=None（规则确定性路径）即全量跑通；传 DeepSeek/Mock LLM
可覆盖路由/汇总/ReAct 链路。真实 RAG 可用时 golden_rag 用例生效。
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field, asdict

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.join(os.path.dirname(HERE), "src")
for p in (AGENT_SRC, os.path.dirname(AGENT_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from .cases import BenchCase, load_cases           # noqa: E402
from . import metrics as M                          # noqa: E402


@dataclass
class CaseResult:
    id: str
    category: str
    desc: str
    crash: bool = False
    error: str = ""
    has_output: bool = False
    agents: list = field(default_factory=list)
    risks: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    rag_hits: list = field(default_factory=list)
    risk_m: dict = field(default_factory=dict)
    ner_m: dict = field(default_factory=dict)
    rag_m: dict = field(default_factory=dict)
    check_m: dict = field(default_factory=dict)      # C1-C4 分层漏报
    sev_m: dict = field(default_factory=dict)        # 严重度分层漏报
    trace_m: dict = field(default_factory=dict)      # 依据可溯源率
    pipe_m: dict = field(default_factory=dict)       # 链路贯通
    time_m: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _collect_subagent_metrics(res: dict) -> tuple[list, list, list]:
    """v7 架构：risks 由 **review subagent** 产出（内部三路），不再依赖顶层 rules_result。

    ⚠️ 历史 bug：v7 删除固定规则自检 tool 后，顶层 `rules_result.risks` 恒为 0，
    若仍从 rules_result 取数会把 FNR 恒算成 100%（假性全漏报）。
    """
    risks: list = []
    entities: list = []
    rag_hits: list = []
    for r in res.get("results", []):
        for x in (getattr(r, "risks", []) or []):
            risks.append(dict(x))
        entities += [dict(x) for x in getattr(r, "entities", [])]
        for h in (getattr(r, "rag_hits", []) or []):
            rag_hits.append({"agent": r.name, **h})
    # 兜底：subagent 无 risks 时才看顶层 rules_result（旧架构兼容）
    if not risks:
        risks = [dict(r) for r in (res.get("rules_result") or {}).get("risks", []) or []]
    return risks, entities, rag_hits


def run_case(case: BenchCase, llm=None, opts=None,
             agents: list | None = None) -> CaseResult:
    """执行单个用例。agents=None 时按有无 plan 自动选 subagent（v6.1 仅 hazard/qa）。"""
    from orchestrator.dispatcher import run as orch_run, GlobalOpts

    if agents is None:
        # v7 架构：方案审查唯一 subagent = review（内部三路：判档/技术核对/依据与要素）；
        # 无 plan 的问句 → qa（RAG 问答）
        agents = ["review"] if (case.plan or "").strip() else ["qa"]
    opts = opts or GlobalOpts(memory_dir=None)
    cr = CaseResult(id=case.id, category=case.category, desc=case.desc, agents=agents)

    # ---- 端到端执行 ----
    t0 = time.perf_counter()
    try:
        res = orch_run(query=case.query, plan=case.plan, llm=llm,
                       force_agents=agents, global_opts=opts)
        total_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:                       # noqa: BLE001
        cr.crash = True
        cr.error = f"{type(e).__name__}: {e}"[:300]
        cr.time_m = M.time_metrics((time.perf_counter() - t0) * 1000, {})
        return cr

    cr.has_output = bool(res.get("final_md") or res.get("summary_line") or res.get("results"))
    cr.risks, cr.entities, cr.rag_hits = _collect_subagent_metrics(res)

    # ---- 各 subagent 单独计时（耗时拆解基准）----
    sub_ms: dict[str, float] = {}
    for name in agents:
        from orchestrator.dispatcher import _REGISTRY
        try:
            t1 = time.perf_counter()
            _REGISTRY[name].run(query=case.query, plan=case.plan,
                                ctx={"rag_rerank": opts.rag_rerank,
                                     "ner_crf": opts.ner_crf,
                                     "memory_dir": opts.memory_dir,
                                     "verbose": opts.verbose})
            sub_ms[name] = (time.perf_counter() - t1) * 1000
        except Exception as e:                   # noqa: BLE001
            sub_ms[name] = -1.0
            cr.error = (cr.error + f" | {name} 单跑异常:{type(e).__name__}" if cr.error
                        else f"{name} 单跑异常:{type(e).__name__}")
    cr.time_m = M.time_metrics(total_ms, sub_ms)
    # 阶段拆解（路由 / 执行 / 汇总）：直接从 orchestrator trace 的 ms 字段聚合
    cr.time_m["phase_ms"] = M.phase_metrics(res.get("trace"))

    # ---- 指标 ----
    cr.risk_m = M.risk_metrics(case.golden_risks, cr.risks)
    cr.ner_m = M.ner_metrics(case.golden_entities, cr.entities)
    # 多维拆解：按检查项 / 严重度分层 + 依据可溯源率 + 链路贯通
    cr.check_m = M.risk_metrics_by_check(case.golden_risks, cr.risks)
    cr.sev_m = M.risk_metrics_by_severity(case.golden_risks, cr.risks)
    cr.trace_m = M.traceability_metrics(cr.risks)
    n_params = sum(1 for e in cr.entities if e.get("type") == "参数")
    n_triples = 0
    for r in res.get("results", []):
        n_triples += len((getattr(r, "meta", {}) or {}).get("triples", []) or [])
    cr.pipe_m = M.pipeline_metrics(len(cr.entities), n_params, n_triples, len(cr.risks))

    # RAG recall@k：对 golden_rag 用例，单独用 qa 检索 golden query 的 top-5 计算
    # （避免把多 subagent 跨子查询累积的 hits 误当成单一 top-k 列表）
    if case.golden_rag:
        from subagents.qa_agent import QAAgent
        try:
            qr = QAAgent().run(query=case.golden_rag.get("query", ""), plan="",
                               ctx={"rag": None, "rag_rerank": False})
            qa_hits = [dict(h) for h in getattr(qr, "rag_hits", []) or []][:5]
        except Exception:
            qa_hits = []
        cr.rag_m = M.rag_recall(case.golden_rag, qa_hits)
    else:
        cr.rag_m = M.rag_recall(None, cr.rag_hits)
    return cr


def run_benchmark(ids: list[str] | None = None, llm=None, opts=None,
                  agents: list | None = None, include_real: bool = False) -> dict:
    """跑 benchmark（合成用例 + 可选真实方案用例）。返回 {cases, summary}。"""
    cases = load_cases(ids, include_real=include_real)
    results = [run_case(c, llm=llm, opts=opts, agents=agents) for c in cases]
    return {"cases": [r.to_dict() for r in results],
            "summary": summarize(results)}


def summarize(results: list[CaseResult]) -> dict:
    """全链路汇总指标。"""
    total = len(results)
    crash = [r for r in results if r.crash]
    no_out = [r for r in results if not r.crash and not r.has_output]

    # 鲁棒性
    survival = (total - len(crash)) / total if total else 0.0

    # 审查（漏报/误报）—— 有 golden 的用例
    risk_cases = [r for r in results if r.risk_m.get("golden")]
    fnr_list = [r.risk_m["fnr"] for r in risk_cases if r.risk_m.get("fnr") is not None]
    fpr_list = [r.risk_m["fpr"] for r in risk_cases if r.risk_m.get("fpr") is not None]
    avg_fnr = sum(fnr_list) / len(fnr_list) if fnr_list else None
    avg_fpr = sum(fpr_list) / len(fpr_list) if fpr_list else None

    # NER F1
    ner_cases = [r for r in results if r.ner_m.get("f1") is not None]
    avg_f1 = sum(r.ner_m["f1"] for r in ner_cases) / len(ner_cases) if ner_cases else None

    # RAG recall@k
    rag_cases = [r for r in results if r.rag_m.get("recall@k") is not None]
    avg_rk = sum(r.rag_m["recall@k"] for r in rag_cases) / len(rag_cases) if rag_cases else None

    # 耗时
    times = [r.time_m["total_ms"] for r in results if not r.crash]
    avg_total = sum(times) / len(times) if times else 0.0
    sub_agg: dict[str, list[float]] = {}
    for r in results:
        for k, v in r.time_m.get("subagent_ms", {}).items():
            if v is not None and v >= 0:
                sub_agg.setdefault(k, []).append(v)
    avg_sub = {k: round(sum(v) / len(v), 1) for k, v in sub_agg.items()}

    # 阶段耗时（路由 / 执行 / 汇总）—— 逐用例取自 trace.ms，只统计有 phase_ms 的用例
    ph_agg: dict[str, list[float]] = {}
    for r in results:
        ph = r.time_m.get("phase_ms") or {}
        for k in ("route_ms", "execute_ms", "summary_ms", "other_ms", "measured_ms"):
            v = ph.get(k)
            if isinstance(v, (int, float)):
                ph_agg.setdefault(k, []).append(v)
    avg_phase = {k: round(sum(v) / len(v), 1) for k, v in ph_agg.items()}

    # ---- 多维汇总 ----
    check_agg: dict[str, dict] = {}
    for c in M.CHECKS:
        g = sum(r.check_m.get(c, {}).get("golden", 0) for r in results)
        h = sum(r.check_m.get(c, {}).get("hit", 0) for r in results)
        check_agg[c] = {"name": M.CHECK_NAME[c], "golden": g, "hit": h,
                        "fnr": round((g - h) / g, 4) if g else None}
    sev_agg: dict[str, dict] = {}
    for s in M.SEVERITIES:
        g = sum(r.sev_m.get(s, {}).get("golden", 0) for r in results)
        h = sum(r.sev_m.get(s, {}).get("hit", 0) for r in results)
        sev_agg[s] = {"golden": g, "hit": h,
                      "fnr": round((g - h) / g, 4) if g else None}
    tr = [r.trace_m.get("rate") for r in results if r.trace_m.get("rate") is not None]
    trace_rate = round(sum(tr) / len(tr), 4) if tr else None
    tot_r = len(results) or 1
    pipe_rate = {
        "entity": round(sum(1 for r in results if r.pipe_m.get("has_entity")) / tot_r, 4),
        "param": round(sum(1 for r in results if r.pipe_m.get("has_param")) / tot_r, 4),
        "triple": round(sum(1 for r in results if r.pipe_m.get("has_triple")) / tot_r, 4),
        "risk": round(sum(1 for r in results if r.pipe_m.get("has_risk")) / tot_r, 4),
    }
    coverage = {c: sum(1 for r in results if r.check_m.get(c, {}).get("golden", 0) > 0)
                for c in M.CHECKS}

    return {
        "total_cases": total,
        "crashed": len(crash),
        "no_output": len(no_out),
        "survival_rate": round(survival, 4),
        "avg_fnr": round(avg_fnr, 4) if avg_fnr is not None else None,
        "avg_fpr": round(avg_fpr, 4) if avg_fpr is not None else None,
        "ner_avg_f1": round(avg_f1, 4) if avg_f1 is not None else None,
        "rag_avg_recall_at_k": round(avg_rk, 4) if avg_rk is not None else None,
        "avg_total_ms": round(avg_total, 1),
        "avg_subagent_ms": avg_sub,
        "avg_phase_ms": avg_phase,
        # —— 多维拆解（v2）——
        "by_check": check_agg,
        "by_severity": sev_agg,
        "traceability_rate": trace_rate,
        "pipeline_rate": pipe_rate,
        "check_coverage": coverage,
        "crash_ids": [r.id for r in crash],
        "no_output_ids": [r.id for r in no_out],
    }
