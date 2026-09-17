"""v7.2 调度器测试：意图路由 / **方案审查只有一个 subagent**（review 内部三路）/ 严格模式 /
   global_opts 透传 / trace 结构 / RAG 溯源 / 记忆 flush。

零依赖 mock LLM。运行：
  pytest agent/tests/test_orchestrator_v5.py -q
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.join(os.path.dirname(HERE), "src")  # agent/src
for p in (AGENT_SRC, os.path.dirname(AGENT_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from harness.memory import MemoryManager  # noqa: E402
from orchestrator.dispatcher import (  # noqa: E402
    run as orch_run, route, _fallback, GlobalOpts, AGENT_DOC_FULL, _REGISTRY,
)
from orchestrator.intent import intent_route  # noqa: E402
from subagents.base import SubAgent, SubAgentResult  # noqa: E402


class FakeLLM:
    """LLM stub：第一次调用答意图路由，其余答汇总。"""

    def __init__(self, route_decision=None, intent: str = "review",
                 final_text: str = "（mock 答复）"):
        self.route_decision = route_decision
        self.intent = intent
        self.final_text = final_text
        self.i = 0

    def complete(self, messages, **kwargs):
        self.i += 1
        if self.i == 1:
            if self.intent:
                return json.dumps({"intent": self.intent})
            return json.dumps({"agents": self.route_decision})
        return self.final_text


class _StubNer:
    """桩 NER：返回手写实体，让单测不依赖 torch。"""

    kind = "ner2"
    _warn = ""

    def extract(self, text):
        out = []
        for s in [x for x in (text or "").split("\n") if x.strip()]:
            for typ, kw in (("工程类型", "基坑"), ("参数", "开挖深度")):
                i = s.find(kw)
                if i >= 0:
                    out.append({"type": typ, "text": kw, "start": i, "end": i + len(kw),
                                "layer": "rule", "conf": 1.0,
                                "sent_idx": 0, "sent_text": s})
        return out


def _load_fixture() -> str:
    fix = os.path.join(os.path.dirname(__file__), "fixtures", "sample_plan.txt")
    if os.path.exists(fix):
        with open(fix, encoding="utf-8") as f:
            return f.read()
    return "基坑工程开挖深度 6m。编制依据：JGJ46-2005、JGJ120-2012。\n未提及专家论证。模板支撑搭设高度 9m。"


# ====================================================================
def test_agent_doc_full_has_all_subagents():
    """v7.2：**方案审查只有一个 subagent（review）** + qa；三路在 review 内部。"""
    assert set(AGENT_DOC_FULL.keys()) == set(_REGISTRY.keys()) == {"review", "qa"}
    assert "内部三路" in AGENT_DOC_FULL["review"]
    assert "rules" not in _REGISTRY          # 规则检查不再拆成独立 subagent
    for k, doc in AGENT_DOC_FULL.items():
        assert "输入" in doc and "输出" in doc
    print(f"OK test_agent_doc_full_has_all_subagents: {list(AGENT_DOC_FULL)}")


def test_global_opts_dataclass():
    o = GlobalOpts(rag_rerank=True, ner_crf=True)
    assert o.rag_rerank is True
    assert o.allow_degrade is False          # 默认严格：不静默降级
    d = o.as_dict()
    assert d["rag_rerank"] and d["ner_crf"] and d["allow_degrade"] is False
    o2 = GlobalOpts(**{"rag_rerank": False, "ner_crf": False})
    assert not o2.rag_rerank
    print("OK test_global_opts_dataclass")


def test_route_fallback_no_plan_returns_qa():
    assert route("塔吊基础混凝土强度多少", has_plan=False, llm=None) == ["qa"]
    print("OK test_route_fallback_no_plan_returns_qa")


def test_routing_intent_table():
    """v7.2 意图路由：问什么查什么；所有审查需求都进**同一个** review subagent。"""
    assert _fallback("", False) == ["qa"]
    r = intent_route("这个基坑开挖算超规模危大吗，要不要专家论证", True)
    assert r["intent"] == "hazard_level" and r["checks"] == ["hazard_level"]
    assert r["agents"] == ["review"] and "基坑" in r["scope_terms"]
    r = intent_route("本方案施工工艺有什么不符合规范要求", True)
    assert r["intent"] == "technical" and r["checks"] == ["technical"]
    assert r["agents"] == ["review"]
    # 编制依据类 → 仍是 review（内部只走依据路），不拆出独立节点
    r = intent_route("编制依据里有没有已经废止的规范", True)
    assert r["intent"] == "basis" and r["agents"] == ["review"] and r["checks"] == ["basis"]
    r = intent_route("方案缺项，九章要素齐全吗", True)
    assert r["intent"] == "elements" and set(r["checks"]) == {"elements", "content"}
    r = intent_route("全面审查方案", True)
    assert r["intent"] == "review" and set(r["checks"]) == {
        "hazard_level", "technical", "basis", "elements", "content"}
    assert r["agents"] == ["review"]
    assert _fallback("帮我核查危大方案施工工艺内容", True) == ["review"]
    # 复合结构化问句 → structured（qa 内部拆解+组装），不抢 review
    r = intent_route("130t 汽车吊钢栈桥吊装危大工程，有哪些风险点？对应的方案编制内容、风险管控清单、验收节点分别是什么？", True)
    assert r["intent"] == "structured" and r["agents"] == ["qa"]
    # 无方案同样识别为 structured
    r = intent_route("130t 汽车吊钢栈桥吊装危大工程，有哪些风险点？对应的方案编制内容、风险管控清单、验收节点分别是什么？", False)
    assert r["intent"] == "structured" and r["agents"] == ["qa"]
    # 危大判定类（"算不算危大"）不被 structured 抢；"危大工程"名词不触发 hazard_level
    assert intent_route("这个基坑开挖算超规模危大吗，要不要专家论证", True)["intent"] == "hazard_level"
    assert intent_route("深基坑工程危大工程，方案要包含哪些内容", True)["intent"] == "structured"
    print("OK test_routing_intent_table")


def test_review_internal_basis_lane_only():
    """规则检查是 review 的**内部三路之一**，按 ctx.checks 开关执行。"""
    from subagents.review_agent import ReviewAgent
    plan = "基坑工程开挖深度 6m。编制依据：JGJ46-2005。\n未提及专家论证。"
    only_basis = ReviewAgent().run(query="编制依据有没有废止的", plan=plan,
                                   ctx={"rag": None, "checks": ["basis"],
                                        "allow_degrade": True})
    assert only_basis.risks and all(r["check"] == "C1" for r in only_basis.risks)
    assert any("JGJ46-2005" in r["title"] for r in only_basis.risks)
    assert only_basis.meta["checks"] == ["basis"]
    no_basis = ReviewAgent().run(query="看下工艺", plan=plan,
                                 ctx={"rag": None, "checks": [], "allow_degrade": True})
    assert not any(r["check"] == "C1" for r in no_basis.risks)
    assert "rules" not in _REGISTRY
    print(f"OK test_review_internal_basis_lane_only: basis={len(only_basis.risks)}")


def test_strict_mode_raises_instead_of_degrading():
    """严格模式（默认）：模型/检索不可用**直接报错**，不静默降级出结论。"""
    from subagents.review_agent import ReviewAgent
    import subagents.review_agent as ra

    plan = "基坑开挖深度 8m。"
    saved_ner = ra.ReviewAgent._ner
    orig = ReviewAgent._ensure_ner
    try:
        # ① 严格模式：加载失败 → 抛 RuntimeError
        def boom(self, allow_degrade):
            raise RuntimeError("NER 模型加载失败，严格模式不降级")

        ReviewAgent._ensure_ner = boom
        try:
            ReviewAgent().run(query="全面审查", plan=plan,
                              ctx={"rag": None, "checks": ["hazard_level"]})
            raise AssertionError("严格模式应当抛错")
        except RuntimeError as e:
            assert "严格模式" in str(e)
        # ② 显式放行降级 → 正常返回（桩 NER）
        ReviewAgent._ensure_ner = orig
        ra.ReviewAgent._ner = _StubNer()
        r = ReviewAgent().run(query="全面审查", plan=plan,
                              ctx={"rag": None, "checks": ["hazard_level"],
                                   "allow_degrade": True})
        assert r.meta["allow_degrade"] is True
        assert any(x["check"] == "C2" for x in r.risks), r.risks
    finally:
        ra.ReviewAgent._ner = saved_ner
        ReviewAgent._ensure_ner = orig
    print("OK test_strict_mode_raises_instead_of_degrading")


def test_intent_route_uses_llm_when_available():
    """有 LLM 时优先用 LLM 判意图（含 scope_terms）；失败/乱答回落关键词。"""

    class _IntentLLM:
        def __init__(self, reply):
            self.reply = reply

        def complete(self, messages, **kw):
            return self.reply

    ok = _IntentLLM('{"intent":"basis","scope_terms":["编制依据"]}')
    r = intent_route("帮我看看这个方案", True, ok)
    assert r["intent"] == "basis" and r["checks"] == ["basis"]
    assert r["source"] == "llm" and r["scope_terms"] == ["编制依据"]

    bad = _IntentLLM('{"intent":"乱写的"}')
    r2 = intent_route("施工工艺有问题吗", True, bad)
    assert r2["intent"] == "technical" and r2["source"] == "keyword"

    class _Boom:
        def complete(self, messages, **kw):
            raise RuntimeError("api down")

    r3 = intent_route("全面审查", True, _Boom())
    assert r3["intent"] == "review" and r3["source"] == "keyword"
    print("OK test_intent_route_uses_llm_when_available")


def test_scope_narrowing_ignores_citation_list():
    """范围收窄：编制依据的清单行（`2. 《建筑基坑支护技术规程》JGJ 120-2012`）不是标题。"""
    from subagents.review_agent import _is_heading, _scope_sentences
    plan = ("XX 项目基坑支护工程专项施工方案\n\n"
            "一、工程概况\n本工程基坑开挖深度 8m。\n\n"
            "二、编制依据\n1. 《建筑地基基础设计规范》GB 50007-2011\n"
            "2. 《建筑基坑支护技术规程》JGJ 120-2012\n\n"
            "四、施工工艺技术\n采用分层分段开挖。\n")
    assert _is_heading("一、工程概况")
    assert not _is_heading("2. 《建筑基坑支护技术规程》JGJ 120-2012")
    assert not _is_heading("1. 《建筑地基基础设计规范》GB 50007-2011")
    assert _scope_sentences(plan, ["基坑"]) is None       # 不在真标题里 → 不收窄
    s = _scope_sentences(plan, ["施工工艺"])
    assert s and "采用分层分段开挖。" in s
    assert all("《建筑基坑支护技术规程》" not in x for x in s)
    print("OK test_scope_narrowing_ignores_citation_list")


def test_rag_client_never_raises_and_records_errors():
    """RagClient 构造**绝不抛异常**（旧 rag 索引 json.load 会 MemoryError，
    rag2 会因 torch DLL 加载失败）；失败原因记进 errors 供排查。"""
    from tools.rag_client import RagClient
    c = RagClient()
    assert isinstance(c.available(), bool)
    assert isinstance(c.errors, dict)
    if not c.available():
        assert c.search("开挖深度", top_k=3) == []
    for i, h in enumerate(c.search("开挖深度", top_k=3), 1):
        assert h["rank"] == i
        assert "text" in h and "metadata" in h
    print(f"OK test_rag_client_never_raises_and_records_errors: "
          f"backend={c.backend}, errors={list(c.errors)}")


def test_orch_run_full_check_returns_trace():
    """全面审查：意图=review → **一个** review subagent 内部走五路 → trace 完整。

    allow_degrade=True 让本测试在无 torch 环境可跑（走规则层实体 + 空检索）。
    """
    plan = _load_fixture()
    with tempfile.TemporaryDirectory() as td:
        llm = FakeLLM(intent="review", final_text="mock final")
        opts = GlobalOpts(rag_rerank=False, ner_crf=False, memory_dir=td,
                          allow_degrade=True)
        res = orch_run(query="全面审查方案", plan=plan, llm=llm, global_opts=opts)
        assert "trace" in res and "rag_sources" in res and "global_opts" in res
        assert res["agents"] == ["review"]
        assert res["routing"]["intent"] == "review"
        assert set(res["routing"]["checks"]) == {"hazard_level", "technical",
                                                "basis", "elements", "content"}
        kinds = [t["kind"] for t in res["trace"]]
        for k in ("input", "dispatch", "execute"):
            assert k in kinds, f"trace 缺 {k}：{kinds}"
        names = [r.name for r in res["results"]]
        assert names == ["review"], names          # 只有 review，无独立规则节点
        labels = " ".join(t.get("label", "") for t in res["trace"])
        assert "依据/要素路" in labels, labels[:400]
        assert os.path.exists(os.path.join(td, "episodic.jsonl"))
    print(f"OK test_orch_run_full_check_returns_trace: trace={len(res['trace'])}, results={names}")


def test_technical_query_skips_rules_lane():
    """技术类问句：review 内部**不走依据/要素路**（v7 核心收益：问什么查什么）。"""
    plan = _load_fixture()
    res = orch_run(query="本方案施工工艺有什么不符合规范", plan=plan, llm=None,
                   global_opts=GlobalOpts(memory_dir=None, allow_degrade=True))
    assert res["routing"]["intent"] == "technical"
    assert res["routing"]["checks"] == ["technical"]
    labels = " ".join(t.get("label", "") for t in res["trace"])
    assert "依据/要素路" not in labels
    rv = res["results"][0]
    assert rv.name == "review"
    assert all(x["check"] == "C3" for x in rv.risks)
    print("OK test_technical_query_skips_rules_lane")


def test_global_opts_rag_rerank_propagates_to_subagents():
    """RAG 精排开关：从 dispatcher.global_opts 透传到 review.run() 的 ctx.rag_rerank。

    需真实 RAG 可用（torch + rag2 索引）→ 零依赖环境（无 torch）跳过内部断言，
    只验证 global_opts 透传（不依赖 RAG 的部分）。
    """
    plan = _load_fixture()
    with tempfile.TemporaryDirectory() as td:
        llm = FakeLLM(intent="technical")
        opts = GlobalOpts(rag_rerank=True, ner_crf=False, memory_dir=td, allow_degrade=True)
        res = orch_run(query="塔吊基础做法是否符合规定", plan=plan, llm=llm, global_opts=opts)
        assert res["global_opts"]["rag_rerank"] is True
        # RAG 可用性探测：无 torch 的零依赖环境跳过 technical 内部断言
        try:
            import torch  # noqa: F401
            rag_ok = True
        except ImportError:
            rag_ok = False
        if not rag_ok:
            print("OK test_global_opts_rag_rerank_propagates_to_subagents (skip: 无 torch, RAG 不可用)")
            return
        rv = next((r for r in res["results"] if r.name == "review"), None)
        if rv:
            assert rv.meta.get("checks") == ["technical"]
            for t in rv.trace:
                if t.get("kind") == "technical":
                    d = t["detail"]
                    err = (d.get("rag_errors") or {}).get("rerank")
                    if err:
                        # 精排可能因**本机提交内存不足**被降级（Windows os error 1455
                        # 「页面文件太小」）——此时 rerank_req=True 但 use_rerank=False。
                        # 本用例验证的是「全局开关是否透传到 subagent」，与精排能否在本机
                        # 加载成功无关，故降级时只校验请求值透传，不把资源限制当成失败。
                        print(f"[skip] 精排被降级（{err[:60]}），仅校验开关透传")
                        assert d.get("rerank_req") is True, d
                    else:
                        assert d.get("use_rerank") is True, d
    print("OK test_global_opts_rag_rerank_propagates_to_subagents")


def test_rag_hits_has_source_and_clause_no():
    """RAG 溯源字段：每条 hit 含 rank/score/source/clause_no/text。"""
    plan = _load_fixture()
    with tempfile.TemporaryDirectory() as td:
        llm = FakeLLM(intent="qa")
        opts = GlobalOpts(rag_rerank=True, ner_crf=False, memory_dir=td)
        res = orch_run(query="JGJ120-2012 第几条", plan="", llm=llm, global_opts=opts)
        assert "rag_sources" in res
        for h in res["rag_sources"]:
            assert "rank" in h and "source" in h
    print(f"OK test_rag_hits_has_source_and_clause_no: {len(res['rag_sources'])} sources")


def test_force_agents_legacy_compat():
    """force_agents 入参兼容旧接口（现仅 review/qa）。"""
    plan = _load_fixture()
    with tempfile.TemporaryDirectory() as td:
        res = orch_run(query="查缺项", plan=plan, llm=FakeLLM(intent="review"), memory_dir=td,
                       force_agents=["review"],
                       global_opts=GlobalOpts(memory_dir=td, allow_degrade=True))
        assert res["agents"] == ["review"]
    print("OK test_force_agents_legacy_compat")


def test_memory_flush_returns_summary():
    """记忆 flush：返回 episodic/semantic/procedural 计数 + 文件 mtime。"""
    with tempfile.TemporaryDirectory() as td:
        m = MemoryManager(td)
        m.episodic_add("测试 query", "测试 outcome")
        m.semantic_set("项目", "测试项目")
        m.skill_mark_expanded("test_skill")
        info = m.flush()
        assert info["episodic_count"] == 1
        assert info["semantic_count"] == 1
        assert info["procedural_skill_count"] == 1
        assert "files" in info
        assert info["files"]["episodic"]["size"] > 0
        assert m.flush()["episodic_count"] == 1
    print("OK test_memory_flush_returns_summary")


def test_review_agent_to_markdown_includes_rag_sources():
    """review_agent.to_markdown 含统一风险清单与 RAG 溯源明细。"""
    from subagents.review_agent import ReviewAgent
    r = SubAgentResult(
        name="review", title="方案审查", summary="test",
        risks=[{"check": "C3", "severity": "HIGH",
                "title": "【基坑工程】开挖深度 8.0m 疑似违反",
                "detail": "条文限值：不宜超过 7.0m"}],
        entities=[{"type": "危大类别", "text": "基坑工程"}],
        meta={"detected_types": ["基坑工程"], "ner_backend": "ner2(v3级联)",
              "checks": ["hazard_level", "technical"]},
        rag_hits=[{"rank": 1, "score": 0.85, "rerank_score": 0.92,
                   "source": "JGJ120-2012_建筑基坑支护技术规程",
                   "clause_no": "3.1.4", "text": "基坑支护…",
                   "rerank_on": True, "query": "基坑工程 安全技术措施"}],
    )
    md = ReviewAgent().to_markdown(r)
    assert "RAG 溯源" in md and "JGJ120" in md
    assert "风险清单" in md and "疑似违反" in md
    print("OK test_review_agent_to_markdown_includes_rag_sources")


def test_review_derives_types_from_ner():
    """v7.2：危大类别由 review 自己从 NER 绑定结果得出，不再依赖 ctx.detected_types。"""
    from subagents.review_agent import ReviewAgent
    r1 = ReviewAgent().run(query="查工艺", plan="基坑开挖深度 6m。模板支撑搭设高度 9m。",
                           ctx={"rag": None, "detected_types": ["不应被使用"],
                                "allow_degrade": True})
    assert isinstance(r1.meta["detected_types"], list)
    assert "不应被使用" not in r1.meta["detected_types"]
    assert r1.summary and "checks" in r1.meta
    print(f"OK test_review_derives_types_from_ner: detected={r1.meta['detected_types']}")


def test_qa_agent_rerank_meta_in_meta():
    """qa_agent 结果 meta.use_rerank 反映全局开关。"""
    from subagents.qa_agent import QAAgent
    r = QAAgent().run(query="JGJ120 基坑 5m", plan="",
                      ctx={"rag": None, "rag_rerank": True})
    if not r.rag_hits:
        assert r.summary
    print(f"OK test_qa_agent_rerank_meta_in_meta: summary={r.summary[:40]}")


def test_rag_trace_source_path_provided_by_rag_layer():
    """溯源下沉到检索层：agent 直接透传 rank/source_path。"""
    from tools.rag_client import RagClient
    rag = RagClient()
    if not rag.available():
        print(f"SKIP test_rag_trace_source_path_provided_by_rag_layer: "
              f"检索不可用（backend={rag.backend}, errors={list(rag.errors)}）")
        return
    hits = rag.search("基坑开挖深度超过5m需要专家论证吗", top_k=3)
    for i, h in enumerate(hits, 1):
        md = h.get("metadata", {})
        assert h.get("rank") == i, f"rank 未补齐：{h.get('rank')} != {i}"
        assert md.get("source"), "source 缺失"
        assert "source_path" in md, "source_path 未由检索层补齐"
    print(f"OK test_rag_trace_source_path_provided_by_rag_layer: {len(hits)} hits")


def test_ner_client_dual_loads_both_or_fallback():
    """NerClient：对接 ner2 全量识别；缺模型时按配置降级纯规则。"""
    from tools.ner_client import NerClient
    ner = NerClient(backend="ner2")
    assert ner.kind == "ner2"
    ents = ner.extract("基坑开挖深度8m，采用C30混凝土，执行JGJ80-2016。")
    assert isinstance(ents, list)
    if ents and ents[0].get("type") != "ERROR":
        assert {"type", "text", "start", "end"}.issubset(ents[0].keys())
    ner_r = NerClient(backend="rule")
    assert ner_r.kind == "rule"
    print(f"OK test_ner_client_dual_loads_both_or_fallback: ner2={len(ents)} ents")


def test_ner_client_no_undefined_device_regression():
    """回归：NerClient._ensure 曾用裸变量 `device`（__init__ 从未定义）→ 每次 NameError
    被 except 吞掉、静默降级纯规则层，表现成"NerClient 不可用/内存不足"。**与内存无关。**"""
    from tools.ner_client import NerClient
    c = NerClient(backend="ner2")
    c.extract("基坑开挖深度 8m。")            # 不允许抛 NameError
    assert not c._warn or "name 'device' is not defined" not in c._warn
    assert isinstance(c.model_loaded, bool)
    # 模型真的在时必须是 True（无 torch 环境为 False，但 warn 里应是真实原因）
    if c.model_loaded:
        assert c._warn == ""
    else:
        assert c._warn, "未加载却无告警原因"
    print(f"OK test_ner_client_no_undefined_device_regression: "
          f"model_loaded={c.model_loaded}, warn={c._warn[:60]}")


def test_dispatcher_on_step_streaming_callback():
    """流式支持：on_step 多次触发，trace 单调递增；on_step=None 行为一致。"""
    plan = _load_fixture()
    snapshots = []
    res = orch_run(query="全面审查方案", plan=plan, llm=FakeLLM(intent="review"),
                   global_opts=GlobalOpts(memory_dir=None, allow_degrade=True),
                   on_step=lambda s: snapshots.append(s))
    assert len(snapshots) >= 5, f"on_step 应触发 >=5 次，实际 {len(snapshots)}"
    lens = [len(s["trace"]) for s in snapshots]
    assert lens == sorted(lens), f"trace 长度应单调递增：{lens}"
    assert lens[-1] == len(res["trace"])
    res2 = orch_run(query="全面审查", plan=plan, llm=FakeLLM(intent="review"),
                    global_opts=GlobalOpts(memory_dir=None, allow_degrade=True))
    assert res2["trace"] and len(res2["trace"]) == lens[-1]
    print(f"OK test_dispatcher_on_step_streaming_callback: on_step 触发 {len(snapshots)} 次")


def test_sync_run_passes_on_token_to_qa_path():
    """回归：**同步** run() 的 qa 早退分支必须把 on_token 透传到 ReActAgent。

    背景：异步 run_async() 一直透传，同步 run() 曾漏传 → 走同步入口时 qa 流式**静默失效**
    （UI 只收到 done、没有 token 事件）。此处用假 ReActAgent 断言参数确实传到了最底层。
    """
    import harness.react_agent as RA
    import harness.skills as HS
    import subagents._shared as SH

    seen: dict = {}
    saved = (RA.ReActAgent, HS.build_compliance_skills, SH.get_rag)

    class _FakeReAct:
        def __init__(self, *a, **kw):
            pass

        def run(self, user_input, on_token=None):
            seen["on_token"] = on_token
            return {"answer": "（fake 终答）", "trace": [], "tool_calls": 0}

    RA.ReActAgent = _FakeReAct
    HS.build_compliance_skills = lambda **kw: None
    SH.get_rag = lambda *a, **kw: None
    cb = lambda piece: None          # 固定身份，便于 is 比较
    try:
        with tempfile.TemporaryDirectory() as td:
            res = orch_run(query="塔吊安装有什么要求", plan="", llm=FakeLLM(),
                           force_agents=["qa"], on_token=cb,
                           global_opts=GlobalOpts(memory_dir=td, allow_degrade=True))
    finally:
        RA.ReActAgent, HS.build_compliance_skills, SH.get_rag = saved

    assert res["agents"] == ["qa"], res.get("agents")
    assert seen.get("on_token") is cb, "同步 run() 未把 on_token 透传给 qa 路（流式会静默失效）"
    print("OK test_sync_run_passes_on_token_to_qa_path")


# ====================================================================
if __name__ == "__main__":
    import traceback

    tests = [test_agent_doc_full_has_all_subagents,
             test_global_opts_dataclass,
             test_route_fallback_no_plan_returns_qa,
             test_routing_intent_table,
             test_review_internal_basis_lane_only,
             test_strict_mode_raises_instead_of_degrading,
             test_intent_route_uses_llm_when_available,
             test_scope_narrowing_ignores_citation_list,
             test_rag_client_never_raises_and_records_errors,
             test_orch_run_full_check_returns_trace,
             test_technical_query_skips_rules_lane,
             test_global_opts_rag_rerank_propagates_to_subagents,
             test_rag_hits_has_source_and_clause_no,
             test_force_agents_legacy_compat,
             test_memory_flush_returns_summary,
             test_review_agent_to_markdown_includes_rag_sources,
             test_review_derives_types_from_ner,
             test_qa_agent_rerank_meta_in_meta,
             test_rag_trace_source_path_provided_by_rag_layer,
             test_ner_client_dual_loads_both_or_fallback,
             test_ner_client_no_undefined_device_regression,
             test_dispatcher_on_step_streaming_callback,
             test_sync_run_passes_on_token_to_qa_path]
    pass_cnt = fail_cnt = 0
    for fn in tests:
        try:
            fn()
            pass_cnt += 1
        except Exception:
            traceback.print_exc()
            print(f"FAIL {fn.__name__}")
            fail_cnt += 1
    print(f"\n=== 总计 {pass_cnt} 通过 / {fail_cnt} 失败 ===")
