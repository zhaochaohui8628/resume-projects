"""调度中心 v7.2：用户输入 → 意图路由 → **方案审查 subagent（唯一）** / qa → LLM 汇总。

v7 结构变更（用户定稿，2026-09-13）：
  1) **删除"必须走的全量规则审查"**：v6.1 每次输入都固定跑一遍 C1/C2/C4 再按策略注入，
     v7 改为**问什么查什么** —— 由 `orchestrator/intent.py` 判意图，决定 review 内部走哪几路；
  2) **链路直接是 用户输入 → 调度中心**：NER 本身已是全量识别（逐句不收敛），
     不需要再用一遍全量规则扫描来"兜底"；
  3) **方案审查收敛为一个 subagent（review）**：判档 / 技术核对 / 依据与要素三路都在
     review 内部完成，产出**一份**统一风险清单；不再把规则工具拆成独立节点；
  4) 意图路由输出 `checks`（review 内部三路开关）与 `scope_terms`（问句指向章节 → 收窄范围）。

严格模式（用户要求）：`GlobalOpts.allow_degrade` 默认 **False** —— NER 模型或检索不可用时
**直接报错**，不静默降级出"看似正常"的结论（那会把漏报包装成无风险）。判档路不依赖模型与检索。

v6 并发控制保留：AsyncPipeline 异步执行 + 令牌桶 LLM 限流 + SharedContext 共享状态。
v5 兼容：AGENT_DOC_FULL（路由用）/ GlobalOpts / trace / RAG 溯源 / 记忆。
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.dirname(HERE)
if AGENT_SRC not in sys.path:
    sys.path.insert(0, AGENT_SRC)

from subagents.review_agent import ReviewAgent            # noqa: E402
from subagents.qa_agent import QAAgent                     # noqa: E402
from subagents.base import SubAgentResult                  # noqa: E402
from orchestrator.aggregator import summarize, summarize_stream, to_markdown  # noqa: E402
from orchestrator.intent import (                          # noqa: E402
    INTENT_QA, intent_route, route, _fallback,
    INTENT_REVIEW, INTENT_TECHNICAL, INTENT_HAZARD_LEVEL, INTENT_BASIS, INTENT_ELEMENTS,
)


# ---------------- 全局选项（由 UI 主控开关注入） ----------------
@dataclass
class GlobalOpts:
    """主 agent 运行时全局开关。dispatcher.run() 透传到每个 subagent 的 ctx。"""
    rag_rerank: bool = True     # v7.3.1：默认精排（可关闭；加载失败自动降级不精排）
    ner_crf: bool = True        # 兼容保留：ner2 生产模型恒为 CRF（s2_crf_param_v3，严格 F1 0.8849
                                # 优于 softmax），该开关不再切换后端，UI 已移除该选项
    memory_dir: str | None = None  # harness 记忆落盘目录；为 None 则不写入
    verbose: bool = True        # 是否在 trace 里记录每步明细（默认开）
    # ---- v6 并发控制 ----
    llm_rate: float = 0.0       # LLM 令牌桶速率（tokens/sec）；0 = 不限流
    llm_burst: int = 4          # 令牌桶容量（允许突发量）
    max_concurrency: int = 4    # 异步流水线最大并发
    subagent_timeout: float = 0.0  # 单 subagent 超时秒数；0 = 不限
    allow_degrade: bool = False    # 默认**严格**：模型/检索不可用直接报错，不静默降级出结论

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------- SubAgent 注册表（v7.2：**方案审查只有一个 subagent**） ----------------
# 三路（判档 / 技术核对 / 依据与要素）都在 review 内部，不再拆成多个 subagent。
_REGISTRY = {
    "review": ReviewAgent(),
    "qa": QAAgent(),
}


# ---------------- Agent 完整描述（给 LLM 路由看；v7.2：review / qa） ----------------
AGENT_DOC_FULL = {
    "review": """[review] **方案审查 subagent（唯一审查出口）**
  定位：对专项施工方案做合规审查，**内部三路**，输出一份统一风险清单。
  **不再拆分为多个 subagent**：判档、技术核对、依据与要素都在本 subagent 内完成。
  输入：
    plan: str  (方案全文/片段，必填)
    ctx.checks: list  (内部三路开关：hazard_level / technical / basis / elements / content；
                       缺省 = hazard_level + technical)
    ctx.rag: RagClient | None    ctx.llm: LLM | None
    ctx.scope_terms: list  (问句指向的章节/工法关键词 → 收窄审查范围)
    ctx.allow_degrade: bool  (默认 False = 严格：模型/检索不可用直接报错，不静默降级)
  输出：SubAgentResult
    summary: 三元组 X 条 / 判档 Y / 技术核对 Z → 风险 N（高危 M）｜ C1:.. C2:.. C3:..
    risks: list[{check ∈ C1/C2/C3, severity, title, detail, evidence, suggestion, source}]
    entities: list[{type,text,...}]   (ner2 级联管道实体)
    meta.checks / detected_types / n_triples / by_check / allow_degrade
  内部三路：
    a) 判档路    → 阈值表**查表**判档（危大/超规模/需论证），**不检索**
    b) 技术核对路 → 工况**多形态短查询并集**检索 → comparator 抽条文限值比对
                    → 无可用限值时交 LLM 兜底（做法类）
    c) 依据/要素路 → 规则库 C1 废止引用 / C4 九章缺失 / C2 必备内容缺失
  适用场景（全部方案审查需求都进这里）：
    - "这个基坑开挖深度算超规模危大吗，要不要专家论证"（checks=[hazard_level]）
    - "本方案施工工艺有什么不符合规范要求"（checks=[technical]）
    - "编制依据里有没有已废止的规范"（checks=[basis]）
    - "方案缺项，九章要素齐全吗"（checks=[elements, content]）
    - "全面审查方案"（checks=[全五路]）
  限制：严格模式下 NER 模型或检索不可用会**报错**（不静默降级）。""",

    "qa": """[qa] 规范问答 subagent（RAG 条文检索）
  定位：通用自然语言问答，检索规范库 top-5 条文。
  输入：
    query: str  (用户提问，必填)
    ctx.rag: RagClient
    ctx.rag_rerank: bool  (受全局开关控制)
  输出：SubAgentResult
    summary: 检索到 N 条条文
    basis: list[{rank, score, rerank_score?, source, clause_no, text, metadata}]
    meta.hits: N
  流程：
    1) 直接用 query 走 RAG 检索 top-5（受 ctx.rag_rerank 控制是否精排；不做方案上下文截断/拼接）
  适用场景：
    - 无方案上传时唯一可用 subagent
    - "JGJ46-2005 废止了吗"
    - "塔吊基础混凝土强度最低多少"（一般性条文咨询）
    - "模板支撑搭设高度超过多少需要论证"
  不适用：方案内容审查（review）——有方案一律走 review""",
}


# ---------------- 路由器 ----------------
# v7：路由已迁到 orchestrator/intent.py（问句意图 → intent/agents/checks/scope_terms）
#     dispatcher 顶部仍 import route / _fallback / intent_route，保持旧调用点兼容。

# ---------------- 执行 ----------------
def execute(agents: list, query: str, plan: str, ctx: dict | None = None) -> list:
    """并行执行被选 subagent（保序返回 SubAgentResult）。"""
    if not agents:
        return []
    from subagents._shared import get_rag
    shared_rag = (ctx or {}).get("rag") or get_rag()
    exec_ctx = dict(ctx or {})
    exec_ctx["rag"] = shared_rag

    def call(name: str):
        ag = _REGISTRY[name]
        return ag.run(query=query, plan=plan, ctx=exec_ctx)

    with ThreadPoolExecutor(max_workers=min(len(agents), 4)) as ex:
        return list(ex.map(call, agents))


# ---------------- v6 异步执行（事件驱动 + 限流 + 共享状态） ----------------
def _wrap_llm(opts: GlobalOpts, llm):
    """LLM 令牌桶限流包装：llm_rate>0 时对 complete() 做并发限流。

    路由 / 汇总 / ReAct 等所有同步 LLM 调用统一经此入口，防盲目并发打爆 API。
    """
    if llm is None or opts.llm_rate <= 0:
        return llm, None
    from concurrency import token_bucket
    bucket = token_bucket.TokenBucket(rate=opts.llm_rate, capacity=max(1, opts.llm_burst))
    limited = token_bucket.RateLimitedLLM(llm, bucket=bucket)
    return limited, bucket


async def execute_async(agents: list, query: str, plan: str,
                        opts: GlobalOpts | None = None, ctx: dict | None = None) -> list:
    """异步事件驱动并行执行被选 subagent（v6.1）。

    用 AsyncPipeline（依赖 DAG + 动态裁剪 + semaphore 并发限流）执行；
    review 是唯一审查 subagent（内部三路），不再有独立规则节点；
    SharedContext 保留作通用非阻塞状态同步通道；LLM 推理经令牌桶限流（llm_rate>0 时）。
    """
    import asyncio
    from concurrency.pipeline import AsyncPipeline
    from concurrency.shared_context import SharedContext

    if not agents:
        return []
    opts = opts or GlobalOpts()
    from subagents._shared import get_rag
    shared_rag = (ctx or {}).get("rag") or get_rag()
    exec_ctx = dict(ctx or {})
    exec_ctx["rag"] = shared_rag

    shared = SharedContext()
    order = {n: i for i, n in enumerate(agents)}

    async def run_subagent(node: "NodeCtx") -> SubAgentResult:
        name = node.node_id
        ag = _REGISTRY[name]
        local_ctx = dict(exec_ctx)
        # subagent.run 为同步实现 → to_thread 执行（事件驱动下不阻塞循环）
        if opts.subagent_timeout > 0:
            res = await asyncio.wait_for(
                asyncio.to_thread(ag.run, query=query, plan=plan, ctx=local_ctx),
                timeout=opts.subagent_timeout)
        else:
            res = await asyncio.to_thread(ag.run, query=query, plan=plan, ctx=local_ctx)
        # 发布中间产物到共享上下文（非阻塞，供审计/后续节点参考）
        if getattr(res, "meta", None) and res.meta.get("detected_types"):
            node.publish(f"{name}.detected_types", res.meta["detected_types"], publisher=name)
        if getattr(res, "rag_hits", None):
            node.publish(f"{name}.rag_hits", res.rag_hits[:5], publisher=name)
        return res

    pipeline = AsyncPipeline(max_concurrency=max(1, opts.max_concurrency),
                             shared=shared)
    for name in agents:
        pipeline.add_node(name, run_subagent)

    res_map = await pipeline.run(timeout=opts.subagent_timeout or None)

    # 保序返回（与 execute 语义一致）
    ordered = sorted(agents, key=lambda n: order[n])
    out = []
    for name in ordered:
        r = res_map.get(name)
        if r is None or r.status not in ("success",):
            from subagents.base import SubAgentResult
            out.append(SubAgentResult(
                name=name, title=f"{name} 异常",
                summary=f"执行{'' if r else ''}未产出（status={r.status if r else 'missing'}）",
                meta={"pipeline_status": r.status if r else "missing"}))
        else:
            out.append(r.result)
    return out


# ---------------- 链路 trace 结构 ----------------
def _trace_push(trace: list, kind: str, label: str, **kw) -> None:
    """trace 步骤写入。统一结构供 UI 渲染。"""
    trace.append({"step": len(trace) + 1, "kind": kind, "label": label, **kw})


# ---------------- 主入口 ----------------
def run(query: str = "", plan: str = "", llm=None,
        memory_dir: str | None = None,
        force_agents: list | None = None,
        global_opts: GlobalOpts | dict | None = None,
        on_step=None, on_token=None) -> dict:
    """编排主入口：路由→（qa 走 ReAct OTA，审查类走并行 subagent）→LLM 汇总→记忆写入。

    参数：
      on_step: 可选 callback（流式支持）。签名 on_step(snapshot: dict)，snapshot 含
               {"trace": [...], "results": [...], "partial": bool}
               每个 trace step 完成时被调用一次（含 subagent 内部步骤）。供 UI 流式渲染。
               on_step=None 时退化为同步行为（保留原接口兼容测试）。

    返回：
      query / agents / results / final_md / detail_md / summary_line /
      trace: list[dict]     ← 链路可视化（用户输入 → 调度 → subagent → LLM 汇总）
      rag_sources: list    ← RAG 溯源（所有 subagent 检索明细汇总）
    """
    # 兼容 dict 入参
    if isinstance(global_opts, dict):
        opts = GlobalOpts(**global_opts)
    else:
        opts = global_opts or GlobalOpts()
    # memory_dir 兼容旧接口
    if memory_dir and not opts.memory_dir:
        opts.memory_dir = memory_dir

    query = (query or "").strip()
    has_plan = bool(plan and plan.strip())
    trace: list[dict] = []
    results: list = []

    # v6：LLM 令牌桶限流（llm_rate>0 时路由/汇总/ReAct 全部限流）
    limited_llm, _bucket = _wrap_llm(opts, llm)

    def emit(kind: str, label: str, **kw) -> None:
        """trace 步骤写入 + 流式回调。"""
        _trace_push(trace, kind, label, **kw)
        if on_step is not None:
            try:
                on_step({"trace": list(trace), "results": list(results),
                         "partial": True, "agents": agents})
            except Exception:
                pass

    emit("input", "用户输入",
         detail={"query": query, "has_plan": has_plan, "plan_chars": len(plan or "")},
         data={"opts": opts.as_dict()})

    # 1) 意图路由（v7：用户输入 → 调度中心；**没有强制前置的全量规则审查**）
    routing = intent_route(query, has_plan, limited_llm)
    if force_agents:
        routing["agents"] = list(force_agents)
        routing["source"] = "force"
    agents = routing["agents"]
    emit("dispatch", "调度中心意图路由",
         detail={"intent": routing["intent"], "agents": agents,
                 "checks": routing["checks"],
                 "scope_terms": routing["scope_terms"],
                 "decision_source": routing["source"], "llm_rate": opts.llm_rate})

    # 2) 按需规则工具已并入 review subagent（内部三路之一），调度中心不再单独执行
    rules_result = None

    # 3) qa 单独走 ReAct OTA（harness）
    if limited_llm is not None and agents == ["qa"]:
        # qa 路径里 _run_react_qa 内部 trace 直接复用 emit
        res_qa = _run_react_qa(query, plan, limited_llm, opts, trace, on_step=on_step)
        res_qa["rules_result"] = rules_result
        res_qa["rules_injected"] = bool(rules_result)
        res_qa["routing"] = routing
        return res_qa

    # 4) 审查类：并行 subagent
    ctx = {
        "rag_rerank": opts.rag_rerank,
        "ner_crf": opts.ner_crf,
        "memory_dir": opts.memory_dir,
        "verbose": opts.verbose,
        # v7：不注入 detected_types（由 review 从 NER 三元组绑定结果自行得出）
        "checks": routing["checks"],          # review 内部三路开关
        "scope_terms": routing["scope_terms"],
        "allow_degrade": opts.allow_degrade,  # 默认严格：不静默降级
        "llm": limited_llm,                   # 供 comparator 的 LLM 兜底使用
    }
    emit("execute", f"并行执行 {len(agents)} 个 subagent",
         detail={"agents": agents, "checks": routing["checks"],
                 "scope_terms": routing["scope_terms"]})

    # subagent 流式回调：把结果生成器化（每完成一个就 emit 一次 subagent_step）
    results = _execute_streaming(agents, query, plan, ctx=ctx,
                                on_subagent_step=lambda r: _emit_subagent(trace, r, emit))

    # 4.1) 失败暴露：严格模式下**不静默** —— 失败的 subagent 集中上报，调用方/UI 不可忽略
    errors = [{"agent": r.name, "error": (r.meta or {}).get("error", r.summary)}
              for r in results if (r.meta or {}).get("failed")]
    if errors:
        emit("error", f"⚠️ {len(errors)} 个 subagent 执行失败（严格模式不降级）",
             detail={"errors": errors})

    # 4.1) 收集 RAG 溯源
    rag_sources = []
    for r in results:
        if r.rag_hits:
            for h in r.rag_hits:
                rag_sources.append({"agent": r.name, **h})

    # 5) LLM 汇总（review 的 risks 已含依据/要素结论，无需单独拼规则结果）
    emit("aggregate", "LLM 汇总")
    final_md = None
    if limited_llm is not None:
        try:
            final_md = summarize_stream(limited_llm, query=query, has_plan=has_plan,
                                        agents=agents, results=results,
                                        rules=(rules_result or {}).get("risks", []) if bool(rules_result) else None,
                                        on_token=on_token)
        except Exception as e:
            final_md = f"⚠️ LLM 汇总失败：{e}"

    detail_md = to_markdown(agents, results)
    summary_line = " ｜ ".join(f"{r.name}:{r.summary}" for r in results)

    emit("done", "完成", detail={"summary_line": summary_line})

    # 6) harness 记忆：Episodic 写入（v4 兼容）
    if opts.memory_dir:
        try:
            from harness.memory import MemoryManager
            mm = MemoryManager(opts.memory_dir)
            mm.episodic_add(query or f"方案审查({len(plan or '')}字)",
                            summary_line, meta={"agents": agents,
                                               "rag_rerank": opts.rag_rerank,
                                               "ner_crf": opts.ner_crf,
                                               "rules_injected": bool(rules_result)})
            mm.save()
        except Exception:
            pass

    return {"query": query, "agents": agents, "results": results,
            "final_md": final_md, "detail_md": detail_md,
            "summary_line": summary_line,
            "trace": trace, "rag_sources": rag_sources,
            "rules_result": rules_result, "rules_injected": bool(rules_result),
            "routing": routing, "errors": errors,
            "global_opts": opts.as_dict()}


async def run_async(query: str = "", plan: str = "", llm=None,
                    memory_dir: str | None = None,
                    force_agents: list | None = None,
                    global_opts: GlobalOpts | dict | None = None,
                    on_step=None, on_token=None) -> dict:
    """v6 全异步编排入口：asyncio 事件驱动 + 令牌桶限流 + 共享上下文依赖裁剪。

    与 run() 返回结构完全一致（query/agents/results/final_md/detail_md/summary_line/
    trace/rag_sources/global_opts），可在 asyncio 环境直接 await；
    同步环境请用 asyncio.run(run_async(...)) 或保留 run()。

    执行路径：
      input → dispatch(意图路由, 限流) → execute(AsyncPipeline 执行 review/qa)
      → aggregate(汇总, 限流) → done → 记忆写入
    """
    import asyncio
    if isinstance(global_opts, dict):
        opts = GlobalOpts(**global_opts)
    else:
        opts = global_opts or GlobalOpts()
    if memory_dir and not opts.memory_dir:
        opts.memory_dir = memory_dir

    query = (query or "").strip()
    has_plan = bool(plan and plan.strip())
    trace: list[dict] = []
    results: list = []
    limited_llm, _bucket = _wrap_llm(opts, llm)

    def emit(kind: str, label: str, **kw) -> None:
        _trace_push(trace, kind, label, **kw)
        if on_step is not None:
            try:
                on_step({"trace": list(trace), "results": list(results),
                         "partial": True, "agents": agents})
            except Exception:
                pass

    emit("input", "用户输入",
         detail={"query": query, "has_plan": has_plan, "plan_chars": len(plan or "")},
         data={"opts": opts.as_dict()})

    # 1) 意图路由（v7：用户输入 → 调度中心；无强制前置的全量规则审查）
    if limited_llm is not None:
        routing = await asyncio.to_thread(intent_route, query, has_plan, limited_llm)
    else:
        routing = intent_route(query, has_plan, None)
    if force_agents:
        routing["agents"] = list(force_agents)
        routing["source"] = "force"
    agents = routing["agents"]
    emit("dispatch", "调度中心意图路由",
         detail={"intent": routing["intent"], "agents": agents,
                 "checks": routing["checks"],
                 "scope_terms": routing["scope_terms"],
                 "decision_source": routing["source"], "llm_rate": opts.llm_rate})

    # 2) 按需规则工具已并入 review subagent（内部三路之一）
    rules_result = None

    # 3) qa 单独走 ReAct OTA（同步实现 → to_thread）
    if limited_llm is not None and agents == ["qa"]:
        res_qa = await asyncio.to_thread(
            _run_react_qa, query, plan, limited_llm, opts, trace, on_step)
        res_qa["rules_result"] = rules_result
        res_qa["rules_injected"] = bool(rules_result)
        res_qa["routing"] = routing
        return res_qa

    # 4) 审查类：异步流水线并行
    ctx = {"rag_rerank": opts.rag_rerank, "ner_crf": opts.ner_crf,
           "memory_dir": opts.memory_dir, "verbose": opts.verbose,
           "checks": routing["checks"], "scope_terms": routing["scope_terms"],
           "allow_degrade": opts.allow_degrade, "llm": limited_llm,
           "on_step": on_step}     # v7.3.2：review 每完成一环节回调 partial 快照 → 前端实时
    emit("execute", f"异步流水线并行执行 {len(agents)} 个 subagent",
         detail={"agents": agents, "max_concurrency": opts.max_concurrency,
                 "llm_rate": opts.llm_rate, "checks": routing["checks"]})
    results = await execute_async(agents, query, plan, opts=opts, ctx=ctx)
    for r in results:
        _emit_subagent(trace, r, emit)

    # 4.1) 失败暴露：严格模式下不静默
    errors = [{"agent": r.name, "error": (r.meta or {}).get("error", r.summary)}
              for r in results if (r.meta or {}).get("failed")]
    if errors:
        emit("error", f"⚠️ {len(errors)} 个 subagent 执行失败（严格模式不降级）",
             detail={"errors": errors})

    # 4.1) RAG 溯源
    rag_sources = []
    for r in results:
        if getattr(r, "rag_hits", None):
            for h in r.rag_hits:
                rag_sources.append({"agent": r.name, **h})

    # 5) LLM 汇总（流式：逐 token 回调 on_token，首 token 即出 → 降低 TTFT）
    emit("aggregate", "LLM 汇总")
    final_md = None
    if limited_llm is not None:
        try:
            final_md = await asyncio.to_thread(
                summarize_stream, limited_llm, query, has_plan, agents, results,
                (rules_result or {}).get("risks", []) if bool(rules_result) else None,
                on_token)
        except Exception as e:
            final_md = f"⚠️ LLM 汇总失败：{e}"

    detail_md = to_markdown(agents, results)
    summary_line = " ｜ ".join(f"{r.name}:{r.summary}" for r in results)
    emit("done", "完成", detail={"summary_line": summary_line})

    # 6) 记忆写入
    if opts.memory_dir:
        try:
            from harness.memory import MemoryManager
            mm = MemoryManager(opts.memory_dir)
            mm.episodic_add(query or f"方案审查({len(plan or '')}字)", summary_line,
                            meta={"agents": agents, "async_io": True,
                                  "rules_injected": bool(rules_result)})
            mm.save()
        except Exception:
            pass

    return {"query": query, "agents": agents, "results": results,
            "final_md": final_md, "detail_md": detail_md,
            "summary_line": summary_line,
            "trace": trace, "rag_sources": rag_sources,
            "rules_result": rules_result, "rules_injected": bool(rules_result),
            "routing": routing, "errors": errors,
            "global_opts": opts.as_dict()}


def _execute_streaming(agents: list, query: str, plan: str, ctx: dict,
                       on_subagent_step=None) -> list:
    """并行执行 subagent，每个 subagent 完成后立即回调 on_subagent_step(result)。

    为兼容旧 execute()（不传 on_subagent_step）走原并行路径；传了回调则用
    as_completed 流式回调，整体仍并行启动（不等所有完成才回调）。
    """
    if not agents:
        return []
    from subagents._shared import get_rag
    shared_rag = ctx.get("rag") or get_rag()
    exec_ctx = dict(ctx)
    exec_ctx["rag"] = shared_rag

    if on_subagent_step is None:
        # 退化路径：保持原 execute 行为
        def call(name: str):
            return _REGISTRY[name].run(query=query, plan=plan, ctx=exec_ctx)
        with ThreadPoolExecutor(max_workers=min(len(agents), 4)) as ex:
            return list(ex.map(call, agents))

    # 流式路径：提交后立即返回 future，as_completed 时回调
    from concurrent.futures import as_completed
    results: list = []

    def call(name: str):
        return _REGISTRY[name].run(query=query, plan=plan, ctx=exec_ctx)

    with ThreadPoolExecutor(max_workers=min(len(agents), 4)) as ex:
        futs = {ex.submit(call, name): name for name in agents}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                name = futs[fut]
                # 异常时构造伪 result，但**必须显式标记失败**（严格模式下这就是"报错"的出口，
                # 不能被当成一次正常执行的结果）
                from subagents.base import SubAgentResult
                r = SubAgentResult(name=name, title=f"{name} 执行失败",
                                   summary=f"执行异常：{e}",
                                   meta={"failed": True, "error": str(e)[:300]})
            results.append(r)
            try:
                on_subagent_step(r)
            except Exception:
                pass
    # 按 agents 顺序重排（保持原语义）
    order = {n: i for i, n in enumerate(agents)}
    results.sort(key=lambda r: order.get(r.name, 999))
    return results


# 步骤类型 → 工具路线兜底映射（subagent 未显式给 route 时按 kind 推断，保证前端一定有路线标签）
_KIND_ROUTE = {
    "ner": "ner", "scope": "ner",
    "triples": "rule", "judge": "rule", "rules": "rule", "structured": "rule",
    "technical": "rag+comparator", "technical_case": "rag+comparator",
    "rag": "rag", "rag_search": "rag", "graphrag": "graphrag",
    "llm": "llm", "aggregate": "llm", "dispatch": "llm",
}


def _infer_route(kind: str) -> str:
    return _KIND_ROUTE.get(kind, "")


def _emit_subagent(trace: list, result, emit) -> None:
    """把单个 subagent 的内部 trace 步骤作为 subagent_step 推入全局 trace 并 emit。"""
    if not getattr(result, "trace", None):
        return
    for t in result.trace:
        # route/src = 工具路线与产物溯源（规则 / NER / RAG / GraphRAG / 比对 / LLM），
        # 前端据此渲染细粒度图状态机（不再只显示"在哪个 agent"）。
        emit("subagent_step", f"[{result.name}] {t.get('label', '')}",
             detail=t.get("detail", {}),
             agent=result.name,
             route=t.get("route", "") or _infer_route(t.get("kind", "")),
             src=t.get("src", ""),
             step_kind=t.get("step_kind", ""))


# ---------------- qa ReAct OTA ----------------
def _run_react_qa(query: str, plan: str, llm, opts: GlobalOpts, trace: list,
                  on_step=None):
    from tools.ner_client import NerClient                     # noqa: E402
    from tools.rules_checker import run_checks                # noqa: E402
    from harness.memory import MemoryManager                  # noqa: E402
    from harness.react_agent import ReActAgent                # noqa: E402
    from harness.skills import build_compliance_skills        # noqa: E402

    mem_dir = opts.memory_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "memory")
    def _emit(kind, label, **kw):
        _trace_push(trace, kind, label, **kw)
        if on_step is not None:
            try:
                on_step({"trace": list(trace), "results": [], "partial": True, "agents": ["qa"]})
            except Exception:
                pass

    _emit("ota_init", "ReAct Harness 初始化",
          detail={"mem_dir": mem_dir, "rag_rerank": opts.rag_rerank})
    rag = None
    try:
        from subagents._shared import get_rag                            # noqa: E402
        rag = get_rag()                    # 进程级单例：review/qa/GraphRAG 共用一份模型实例
    except Exception as e:
        _emit("ota_warn", "RagClient 初始化失败", detail={"err": str(e)})
    rag_hit_sink: list = []          # rag skill 的结构化命中（UI 溯源用，以前是空的）
    skills = build_compliance_skills(rules_checker=run_checks,
                                     rag_client=rag,
                                     ner_client=NerClient(backend="rule") if rag else None,
                                     rag_sink=rag_hit_sink,
                                     rag_rerank=opts.rag_rerank)
    agent = ReActAgent(skills=skills, memory=MemoryManager(mem_dir), llm=llm)
    res = agent.run(query)
    answer = (res.get("answer") or "").strip() or "（ReAct 未产出终答）"

    # OTA 步骤入 trace（含 RAG 检索明细）
    # rag_sources = rag skill 实测命中（结构化，含来源/条款/分数/查询）→ 前端溯源面板
    seen_key = set()
    rag_sources = []
    for h in rag_hit_sink:
        key = (h.get("source"), h.get("clause_no"), (h.get("text") or "")[:40])
        if key in seen_key:
            continue
        seen_key.add(key)
        rag_sources.append({"agent": "qa", **h})
    for t in res.get("trace", []):
        kind = t.get("kind", "")
        if kind == "tool" and t.get("skill") == "retrieve_standards":
            _emit("rag_search", f"OTA 调用 rag：{t.get('thought', '')[:60]}",
                  route="rag",
                  src=f"{getattr(rag, 'backend', '?')}"
                      f"（CE精排={'开' if opts.rag_rerank else '关'}）",
                  detail={"skill": t["skill"], "thought": t.get("thought", ""),
                          "input": t.get("input", ""),
                          "n_hits": len(rag_sources),
                          "tool": "ReAct OTA → retrieve_standards（RAG 语义检索）"},
                  step_kind="ota")
        elif kind == "tool":
            _emit("ota_step", f"OTA 调用 {t.get('skill')}",
                  route="rule", src=f"harness/skills.py :: {t.get('skill')}",
                  detail={"skill": t.get("skill"), "thought": t.get("thought", "")},
                  step_kind="ota")
        elif kind == "expand":
            _emit("expand", f"展开技能 {t.get('skill')}",
                  detail={"skill": t.get("skill")}, step_kind="ota")
        elif kind == "answer":
            _emit("ota_answer", "OTA 终答",
                  detail={"content": t.get("content", "")[:200]}, step_kind="ota")
        elif kind == "invalid":
            _emit("invalid", f"⚠️ 无效动作「{t.get('action', '')}」：LLM 输出了未注册的技能名，已回提示引导纠偏",
                  route="llm", src="harness/react_agent.py（ReAct 动作解析）",
                  detail={"action": t.get("action", ""), "thought": t.get("thought", "")},
                  step_kind="ota")

    # ReAct 自带 trace 文本（仅作独立字段返回，供调试/导出；**不拼进用户回复**——
    # 2026-09-14 反馈：正文出现 <details>ReAct 思考轨迹</details> 原始标签 + tool_use
    # 等运行细节，用户要求"回复中只放回复"。轨迹已通过 SSE step 事件进"运行过程"折叠区）
    lines = []
    for t in res.get("trace", []):
        kind = t.get("kind", "")
        if kind == "tool":
            lines.append(f"- 调用技能 `{t.get('skill')}`｜思考：{t.get('thought', '')}")
        elif kind == "expand":
            lines.append(f"- 展开技能说明：`{t.get('skill')}`")
        elif kind == "answer":
            lines.append(f"- 终答：{t.get('content', '')}")
        elif kind == "invalid":
            lines.append(f"- ⚠️ 无效动作：{t.get('thought', '')}")
    meta = f"（工具调用 {res.get('tool_calls', 0)} 次" + \
           ("，已达上限截断）" if res.get("truncated") else "）")
    detail_md = f"{'；'.join(lines) or '（无轨迹）'}"
    final_md = answer

    _emit("done", "ReAct 完成", detail={"tool_calls": res.get("tool_calls", 0),
                                       "truncated": res.get("truncated", False)})
    return {"query": query, "agents": ["qa"],
            "results": [], "final_md": final_md,
            "detail_md": detail_md,
            "summary_line": f"ReAct 问答 {meta}，采用 harness 四层记忆",
            "trace": trace, "rag_sources": rag_sources,
            "global_opts": opts.as_dict()}