"""方案审查 subagent（统一出口）—— 判档 / 技术核对 / 依据与要素，**一个 subagent 内部三路**。

## 为什么合并（2026-09-13 用户定稿）
之前把「规则工具」拆成独立的伪 subagent 节点，还在链路图里把三路画成并列 —— 这是错的：
方案审查只有**一个** subagent，三路是它内部的分支，最终产出**一份**统一风险清单。

    NER2 全量识别 → 工况三元组 → 内部分诊
       ├ 判档路    ：阈值表查表（危大/超规模/需论证）—— 不检索
       ├ 技术核对路 ：多形态短查询并集 → comparator 比限值 → LLM 兜底（做法类）
       └ 依据/要素路：规则库 C1 废止引用 / C4 九章 / C2 必备内容
    → 统一风险清单（C1/C2/C3 混排，按严重度排序，每条带依据）

## 严格模式：不做静默降级（用户要求）
旧版在模型/检索不可用时自动降级为「纯规则层 + 空检索」，会产出一份看似正常、
实则缺了技术核对维度的结论 —— 对合规审查是危险的（漏报被包装成"无风险"）。
现在默认 **`allow_degrade=False`（严格）**：NER 模型或检索不可用 → **直接抛错**，
由调用方决定是否显式放行降级（`ctx["allow_degrade"]=True`）。
判档路本身不需要模型与检索，因此永远可用。

## 查询构造（实测结论）
- **不做类型模板查询**（`{实体} 安全技术措施 施工要求 应符合` 是样板条款的语义磁铁）；
- **不把多条实体拼成一条 query**（top3 退化为定义条/概述条/坏 chunk）；
- 技术路 = **多形态短查询并集去重**：裸量名有盲区（`开挖深度` 召回 4.1.1 无可用限值），
  而 `开挖深度 允许值 应符合` 才召回 DG-TJ08-61 16.2.1「不宜超过7.0m」→ 并集才稳；
- **数值不进 query**，只进 comparator。
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(AGENT_SRC)
ROOT = os.path.dirname(WORKSPACE)
if AGENT_SRC not in sys.path:
    sys.path.insert(0, AGENT_SRC)
if os.path.join(ROOT, "rag", "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "rag", "src"))

from ._shared import get_rag                        # noqa: E402
from .base import SubAgent, SubAgentResult         # noqa: E402

# 内部三路的开关名（由意图路由经 ctx["checks"] 传入）
LANE_JUDGE = "hazard_level"
LANE_TECH = "technical"
LANE_BASIS = "basis"          # C1 废止引用
LANE_ELEMENTS = "elements"    # C4 九章
LANE_CONTENT = "content"      # C2 必备内容
DEFAULT_CHECKS = (LANE_JUDGE, LANE_TECH)

MAX_TECHNICAL = 25            # 技术路单次最多检索的工况数（防爆量）
# 技术路查询形态：多形态 + 并集去重（单一形态各有盲区）
QUERY_FORMS = ("{m}", "{m} 允许值 应符合", "{m} 不得超过 不宜大于")
HEADING_RE = re.compile(r"^\s*(?:第?[一二三四五六七八九十]+[、.．]|\d+(?:[.．]\d+)*\s*[、.．]?)\s*\S")


def _is_heading(line: str) -> bool:
    """判定是否章节标题。

    坑（实测）：编制依据的清单项 `2. 《建筑基坑支护技术规程》JGJ 120-2012` 也以数字开头，
    会被朴素正则当成标题 —— 于是问"基坑"时"命中"的是这条清单项，范围收窄后把全部实体滤掉。
    故：含书名号 / 标准号的行一律**不是标题**。
    """
    s = (line or "").strip()
    if not s or not HEADING_RE.match(s):
        return False
    if any(c in s for c in ("《", "》", "“", "”", "(", "（")):
        return False
    return re.search(r"[A-Z]{2,}\s*\d", s) is None


def _scope_sentences(plan: str, terms: list[str]) -> set[str] | None:
    """按问句指向的章节收窄范围 → 返回该范围的行文本集合；无法定位则 None（=全篇）。"""
    if not terms or not plan:
        return None
    lines = plan.split("\n")
    heads = [(i, ln.strip()) for i, ln in enumerate(lines) if _is_heading(ln)]
    if not heads:
        return None
    picked: set[str] = set()
    for i, (idx, text) in enumerate(heads):
        if not any(t in text for t in terms):
            continue
        end = heads[i + 1][0] if i + 1 < len(heads) else len(lines)
        for ln in lines[idx:end]:
            if ln.strip():
                picked.add(ln.strip())
    return picked or None


class ReviewAgent(SubAgent):
    name = "review"
    title = "方案审查（判档 + 技术核对 + 依据/要素，统一出口）"
    description = "方案合规审查统一 subagent：危大判档、工艺参数与规范相符性、依据废止与编制要素缺失"
    doc = """方案审查（唯一审查 subagent，内部三路，统一风险清单）：
  · 判档路：阈值表查表 → 危大/超规模/是否需专家论证（不检索）
  · 技术核对路：工况短查询并集 → 条文限值比对 → LLM 兜底（做法类）
  · 依据/要素路：C1 引用废止 / C4 九章缺失 / C2 必备内容缺失
输入：plan(必填) / query / ctx.checks(内部三路开关) / ctx.scope_terms / ctx.rag / ctx.llm
      ctx.allow_degrade（默认 False = 严格：模型或检索不可用直接报错，不静默降级）
输出：SubAgentResult.risks = 统一风险清单（check ∈ C1/C2/C3，按严重度排序）
**不做**：把方案拆成多份分别审查；把所有实体拼成一条 query 去检索。"""
    _ner = None
    _ner_error = ""

    # 内存类错误 → 可操作提示（16GB 机器上"页面文件太小/提交内存耗尽"是常态，
    # 裸 MemoryError 看不出该干什么，这里直接给处置办法）
    _MEM_HINT = ("｜疑似【系统提交内存不足】（Windows 提交限额 = 物理内存 + 页面文件）："
                 "请关掉占用大的程序（VS Code / 多余浏览器窗口 / WorkBuddy 多开），"
                 "或调大页面文件（系统属性→高级→性能设置→高级→虚拟内存，建议 16–32GB），"
                 "然后重启服务。")

    @staticmethod
    def _is_mem_error(msg: str) -> bool:
        m = (msg or "").lower()
        return any(k in m for k in ("memoryerror", "not enough memory", "1455",
                                    "页面文件", "commitment", "oom"))

    @classmethod
    def _release_ner(cls):
        """释放 NER 模型：本流程 NER 只用于前半段（抽实体/三元组），
        技术核对路要加载 RAG（双塔 + CE）——两者同时驻留会顶到提交上限。
        用后即释放可省约 0.5–1GB，是 16GB 机器上能否跑通的关键。"""
        if cls._ner is not None:
            try:
                cls._ner._fx = None            # 丢掉 FullTextExtractor（含权重）
                cls._ner.model_loaded = False
            except Exception:                  # noqa: BLE001
                pass
            cls._ner = None
        try:
            import gc
            gc.collect()
        except Exception:                      # noqa: BLE001
            pass

    def _ensure_ner(self, allow_degrade: bool):
        """NER2 客户端（规则层 + s2_crf_param_v3）。**严格模式下模型不可用直接抛错**。

        注意：不能只看"构造成功"——`NerClient` 内部可能已降级为纯规则层，
        因此这里检查它的显式状态 `model_loaded`（构造后预热确定）。
        """
        if ReviewAgent._ner is not None:
            return ReviewAgent._ner
        from tools.ner_client import NerClient
        from ner2.src.pipeline.full_text import DEFAULT_MODEL
        try:
            cli = NerClient(backend="ner2", model_path=DEFAULT_MODEL)
            cli.extract("预热：基坑开挖深度 1m。")      # 触发真实加载 + 推理
        except Exception as e:                       # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:200]
            ReviewAgent._ner_error = err + (self._MEM_HINT if self._is_mem_error(err) else "")
            if not allow_degrade:
                raise RuntimeError(
                    "NER 模型加载失败，严格模式不降级（如需放行请设 ctx.allow_degrade=True）："
                    f"{ReviewAgent._ner_error}") from e
            cli = NerClient(backend="rule", model_path=None)
        if not getattr(cli, "model_loaded", False):
            msg = (getattr(cli, "_warn", "") or "").strip() or "模型未加载"
            ReviewAgent._ner_error = msg[:200] + (self._MEM_HINT if self._is_mem_error(msg) else "")
            if not allow_degrade:
                raise RuntimeError(
                    "NER 微调模型不可用（已回退纯规则层），严格模式不降级"
                    f"（如需放行请设 ctx.allow_degrade=True）：{ReviewAgent._ner_error}")
        ReviewAgent._ner = cli
        return cli

    # ---------------- 主流程 ----------------
    def run(self, query: str = "", plan: str = "", ctx: dict | None = None) -> SubAgentResult:
        if not plan:
            return SubAgentResult(name=self.name, title=self.title,
                                  summary="未提供方案文本。",
                                  trace=[{"step": 1, "kind": "skip", "label": "无 plan，跳过"}])
        ctx = ctx or {}
        checks = [c for c in (ctx.get("checks") or DEFAULT_CHECKS)]
        scope_terms = ctx.get("scope_terms") or []
        allow_degrade = bool(ctx.get("allow_degrade", False))
        llm = ctx.get("llm")
        trace: list = []
        risks: list[dict] = []
        rag_hits: list = []

        # 流式上报（v7.3.2）：每完成一个环节就 emit 一次 partial 快照，
        # 前端"会话内实时过程"不再等到最后一次性涌出（此前 review 全程不回调 = 假死）。
        on_step = ctx.get("on_step")

        def _report():
            if on_step is not None:
                try:
                    # 与 fastapi 消费端对齐：("step", snapshot) 元组（同 dispatcher.emit）
                    on_step(("step", {"trace": list(trace), "results": [],
                                      "partial": True, "agents": ["review"]}))
                except Exception:
                    pass

        # ---------- 1) NER2 全量识别 ----------
        ner = self._ensure_ner(allow_degrade)
        ents = ner.extract(plan)
        backend = "ner2(v3级联)" if ner.kind == "ner2" else ner.kind
        note = f"（{ner._warn}）" if getattr(ner, "_warn", "") else ""
        n_sent = len({e.get("sent_idx") for e in ents if e.get("sent_idx") is not None})
        trace.append({"step": len(trace) + 1, "kind": "ner",
                      "label": f"NER2 全量识别 {len(ents)} 实体（{backend}，{n_sent} 句）{note}",
                      "route": "ner",
                      "src": getattr(ner, "model_path", "") or "",
                      "detail": {"backend": backend, "n_entities": len(ents),
                                 "n_sentences": n_sent, "degraded": backend != "ner2(v3级联)",
                                 "tool": "ner2 级联（规则层 + s2_crf_param_v3 微调模型）",
                                 "model_path": getattr(ner, "model_path", "") or ""}})
        _report()     # NER 完成 → 前端即时可见

        # ---------- 2) 按问句范围收窄（收窄过头就放弃） ----------
        scoped = _scope_sentences(plan, scope_terms)
        if scoped is not None:
            kept = [e for e in ents if (e.get("sent_text") or "").strip() in scoped]
            if len(ents) >= 5 and len(kept) < max(2, int(len(ents) * 0.2)):
                trace.append({"step": len(trace) + 1, "kind": "scope",
                              "label": f"范围收窄命中过少（{len(ents)}→{len(kept)}），放弃收窄改全篇",
                              "detail": {"scope_terms": scope_terms,
                                         "kept_entities": len(kept), "reverted": True}})
            else:
                before = len(ents)
                ents = kept
                trace.append({"step": len(trace) + 1, "kind": "scope",
                              "label": f"按问句范围收窄 {'、'.join(scope_terms)} → {before}→{len(ents)} 实体",
                              "detail": {"scope_terms": scope_terms,
                                         "kept_entities": len(ents), "dropped": before - len(ents)}})
            _report()     # scope 完成 → 前端即时可见

        # ---------- 3) 工况三元组 + 内部分诊 ----------
        from tools.param_extractor import dedup, extract_triples, keep_worst, triage
        triples = dedup(extract_triples(ents))
        worst = keep_worst(triples)
        split = triage(worst)
        detected = sorted({t["category"] for t in triples if t.get("category")} |
                          {e["text"] for e in ents if e.get("type") == "危大类别"})
        trace.append({"step": len(trace) + 1, "kind": "triples",
                      "route": "rule", "src": "tools/param_extractor.py（规则）",
                      "label": f"工况三元组 {len(triples)} 条（最不利 {len(worst)}）"
                               f" → 判档候选 {len(split['hazard_level'])} / 技术候选 {len(split['technical'])}",
                      "detail": {"n_triples": len(triples),
                                 "n_judgeable": len(split["hazard_level"]),
                                 "n_technical": len(split["technical"]),
                                 "detected_types": detected,
                                 "tool": "规则：三元组绑定 + 内部分诊"}})
        _report()     # 三元组/分诊完成 → 前端即时可见

        # ---------- 4) 判档路（查表，不检索） ----------
        if LANE_JUDGE in checks:
            from tools.hazard_level import LEVEL_HAZ, LEVEL_SUPER, judge
            n_haz = 0
            for t in split["hazard_level"]:
                r = judge(t["category"], t["metric"], t["value"], t["unit"], t.get("context", ""))
                if r["level"] not in (LEVEL_HAZ, LEVEL_SUPER):
                    continue
                n_haz += 1
                is_super = r["level"] == LEVEL_SUPER
                risks.append({
                    "check": "C2", "severity": "HIGH" if is_super else "MEDIUM",
                    "title": f"【{t['category']}】{t['metric']} {t['value']}{t['unit']} → {r['level']}",
                    "detail": f"{r['threshold']}；" + ("按规定应组织专家论证。" if is_super
                                                       else "属危大工程，应编制专项施工方案。"),
                    "evidence": t.get("sent_text", ""),
                    "suggestion": "补充专家论证报告与论证结论。" if is_super else "完善专项方案审批与交底。",
                    "source": "hazard_level", "basis": r["basis"],
                })
            trace.append({"step": len(trace) + 1, "kind": "judge",
                          "route": "rule",
                          "src": "agent/data/rules/hazardous_work_types.json（阈值表）",
                          "label": f"判档路（查表）命中 {n_haz} 条危大/超规模",
                          "detail": {"n": n_haz, "n_checked": len(split["hazard_level"]),
                                     "tool": "规则：阈值表精确查表（不检索）",
                                     "table": "hazardous_work_types.json（37号令附件1/2）"}})
            _report()     # 判档路完成 → 前端即时可见

        # ---------- 5) 技术核对路（检索 + 比对；严格模式下检索不可用即报错） ----------
        n_batch = 0
        if LANE_TECH in checks and worst:
            self._release_ner()      # 先释放 NER 权重，给 RAG（双塔+CE）腾提交内存
            rag = ctx.get("rag") or get_rag()
            if not rag or not getattr(rag, "available", lambda: False)():
                if not allow_degrade:
                    errs = getattr(rag, "errors", None)
                    raise RuntimeError(
                        "检索不可用，严格模式不降级（技术核对必须有条文；"
                        "如需放行请设 ctx.allow_degrade=True）："
                        f"backend={getattr(rag, 'backend', None)} errors={errs}")
                trace.append({"step": len(trace) + 1, "kind": "technical",
                              "label": "⚠️ 检索不可用且已放行降级 → 技术核对跳过",
                              "detail": {"rag_backend": getattr(rag, "backend", None),
                                         "rag_errors": getattr(rag, "errors", None)}})
                rag = None
            else:
                from tools.comparator import VERDICT_BAD, VERDICT_OK, compare_all, llm_compare
                use_rerank = bool(ctx.get("rag_rerank", False))
                # 实际生效状态：精排加载/推理失败会降级（rag.use_rerank 被置 False），
                # 必须在 trace 里如实体现，避免"看着开了其实没开"。
                rr_active = bool(getattr(rag, "use_rerank", use_rerank))
                jkeys = {(t["category"], t["metric"], t["value"], t["unit"])
                         for t in split["hazard_level"]}
                batch = sorted(
                    worst,
                    key=lambda t: (0 if (t["category"], t["metric"], t["value"], t["unit"]) in jkeys else 1,
                                   t["category"] is None, -len(t["metric"])))[:MAX_TECHNICAL]
                n_batch = len(batch)
                n_bad = n_ok = n_na = n_llm = 0
                for t in batch:
                    # 多形态短查询 → 并集去重（召回优先；判定交给 comparator，只认显式限值措辞）
                    hits, seen = [], set()
                    for form in QUERY_FORMS:
                        q = form.format(m=t["metric"])
                        for h in rag.search(q, top_k=3, use_rerank=use_rerank):
                            md = h.get("metadata", {}) or {}
                            key = (md.get("source"), md.get("clause_no"),
                                   (h.get("text", "") or "")[:60])
                            if key in seen:
                                continue
                            seen.add(key)
                            h = dict(h)
                            h["query"] = q
                            hits.append(h)
                    for i, h in enumerate(hits, 1):
                        md = h.get("metadata", {}) or {}
                        h["rank"] = i
                        rag_hits.append({"rank": i, "score": h.get("score", 0.0),
                                         "source": md.get("source", "?"),
                                         "clause_no": md.get("clause_no", ""),
                                         "text": (h.get("text", "") or "")[:160],
                                         "query": h.get("query", ""), "rerank_on": use_rerank})
                    res = compare_all(t, hits)
                    # 逐工况出一条 trace（细粒度：可见每个工况走了哪些 query、命中几条、比对结论）
                    picked = res.get("picked") or {}
                    trace.append({"step": len(trace) + 1, "kind": "technical_case",
                                  "route": "rag+comparator",
                                  "src": f"{getattr(rag, 'backend', '?')}"
                                         f"（CE精排={'开' if rr_active else '关'}）→ comparator",
                                  "label": f"工况 [{t['category'] or '—'}] {t['metric']} "
                                           f"{t['value']}{t['unit']} → "
                                           f"{len(hits)} 条候选 → {res['verdict']}"
                                           + (f"（{picked.get('source')} {picked.get('clause_no','')}"
                                              f" 限值 {picked.get('limit')}）"
                                              if picked else ""),
                                  "detail": {"category": t.get("category"), "metric": t.get("metric"),
                                             "value": t.get("value"), "unit": t.get("unit"),
                                             "n_queries": len(QUERY_FORMS), "n_hits": len(hits),
                                             "verdict": res["verdict"],
                                             "use_rerank": rr_active,
                                             "rerank_req": use_rerank,
                                             "rag_errors": getattr(rag, "errors", None),
                                             "rag_backend": getattr(rag, "backend", None),
                                             "picked_source": picked.get("source", ""),
                                             "picked_clause": picked.get("clause_no", ""),
                                             "picked_limit": picked.get("limit", ""),
                                             "queries": list(QUERY_FORMS)}})
                    _report()     # 每条工况比对完成 → 前端即时可见（细粒度）
                    if res["verdict"] == VERDICT_BAD:
                        n_bad += 1
                        p = res["picked"]
                        risks.append({
                            "check": "C3", "severity": "HIGH",
                            "title": f"【{t['category'] or '—'}】{t['metric']} {t['value']}{t['unit']}"
                                     f" 疑似违反 {p['source']} {p['clause_no']}",
                            "detail": f"条文限值：{p['limit']}；方案值 {t['value']}{t['unit']}。",
                            "evidence": p["evidence"],
                            "suggestion": "复核并调整方案参数或补充专项说明。",
                            "source": "comparator"})
                    elif res["verdict"] == VERDICT_OK:
                        n_ok += 1
                    else:
                        n_na += 1
                        if llm is not None and t.get("sent_text"):
                            v = llm_compare(t, hits, llm)
                            if v.get("verdict") == VERDICT_BAD:
                                n_llm += 1
                                risks.append({
                                    "check": "C3", "severity": "MEDIUM",
                                    "title": f"【{t['category'] or '—'}】{t['metric']} 做法存疑（LLM 判定）",
                                    "detail": v.get("reason", "")[:200],
                                    "evidence": v.get("evidence", "")[:200],
                                    "suggestion": "人工复核该做法与条文要求的相符性。",
                                    "source": "llm_compare"})
                trace.append({"step": len(trace) + 1, "kind": "technical",
                              "route": "rag+comparator",
                              "src": f"{getattr(rag, 'backend', '?')}"
                                     f"（CE精排={'开' if rr_active else '关'}）",
                              "label": f"技术核对路：工况 {n_batch} 条 → 疑似违反 {n_bad} / 符合 {n_ok}"
                                       f" / 无法判定 {n_na}（LLM 兜底命中 {n_llm}）",
                              "detail": {"n_batch": n_batch, "n_violation": n_bad, "n_ok": n_ok,
                                         "n_na": n_na, "n_llm": n_llm, "use_rerank": rr_active,
                                         "rerank_req": use_rerank,
                                         "rag_errors": getattr(rag, "errors", None),
                                         "rag_backend": getattr(rag, "backend", None),
                                         "tool": "RAG 多形态短查询 → comparator 限值比对（LLM 兜底）"}})
                _report()     # 技术核对路完成 → 前端即时可见（每条工况已单独上报）

        # ---------- 6) 依据 / 要素路（规则库，subagent 内部完成，不再拆出独立节点） ----------
        if any(c in checks for c in (LANE_BASIS, LANE_ELEMENTS, LANE_CONTENT)):
            from tools.rules_checker import CATEGORY_KEYWORDS, RuleStore, DEFAULT_RULES_DIR, \
                check_c1, check_c2, check_c4
            store = self._store()
            got = []
            if LANE_BASIS in checks:
                got += check_c1(plan, store)
            if LANE_CONTENT in checks:
                got += check_c2(plan, store)
            if LANE_ELEMENTS in checks:
                det = [k for k, kws in CATEGORY_KEYWORDS.items() if any(kw in plan for kw in kws)]
                got += check_c4(plan, store, det)
            order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            got.sort(key=lambda r: order.get(r.severity, 3))
            risks += [r.to_dict() for r in got]
            n_high = sum(1 for r in got if r.severity == "HIGH")
            trace.append({"step": len(trace) + 1, "kind": "rules",
                          "route": "rule", "src": f"{DEFAULT_RULES_DIR}（规则库）",
                          "label": f"依据/要素路（规则库）：{len(got)} 项（高危 {n_high}）"
                                   f" 检查项={'/'.join(c for c in checks if c in (LANE_BASIS, LANE_ELEMENTS, LANE_CONTENT))}",
                          "detail": {"n": len(got), "n_high": n_high,
                                     "rules_dir": DEFAULT_RULES_DIR,
                                     "tool": "规则库：C1 废止引用 / C2 必备内容 / C4 九章要素",
                                     "titles": [r.title for r in got[:5]]}})
            _report()     # 依据/要素路完成 → 前端即时可见

        # ---------- 7) 统一风险清单（按严重度排序） ----------
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        risks.sort(key=lambda r: order.get(r.get("severity"), 3))
        n_high = sum(1 for r in risks if r["severity"] == "HIGH")
        by_check: dict[str, int] = {}
        for r in risks:
            by_check[r["check"]] = by_check.get(r["check"], 0) + 1
        return SubAgentResult(
            name=self.name, title=self.title,
            summary=f"三元组 {len(triples)} 条 / 判档 {len(split['hazard_level'])} / "
                    f"技术核对 {n_batch} → 风险 {len(risks)}（高危 {n_high}）"
                    + (f" ｜ " + " ".join(f"{k}:{v}" for k, v in sorted(by_check.items())) if risks else "")
                    + note,
            risks=risks, entities=ents, basis=[],
            meta={"detected_types": detected, "ner_backend": backend,
                  "n_entities": len(ents), "n_triples": len(triples),
                  "checks": checks, "scope_terms": scope_terms,
                  "allow_degrade": allow_degrade, "by_check": by_check,
                  "triples": triples},
            trace=trace, rag_hits=rag_hits)

    @staticmethod
    def _store():
        global _STORE
        if _STORE is None:
            from tools.rules_checker import DEFAULT_RULES_DIR, RuleStore
            _STORE = RuleStore(DEFAULT_RULES_DIR)
        return _STORE

    # ---------------- 渲染 ----------------
    def to_markdown(self, r: SubAgentResult, brief: bool = True) -> str:
        lines = [f"### {r.title}", "", f"> {r.summary}", ""]
        lines.append(f"**危大类型**：{'、'.join(r.meta.get('detected_types', []) or ['（无）'])} "
                     f"｜ NER 后端：`{r.meta.get('ner_backend', '?')}` "
                     f"｜ 内部三路：{'、'.join(r.meta.get('checks', []))}")
        if r.meta.get("scope_terms"):
            lines.append(f"｜ 范围收窄：{'、'.join(r.meta['scope_terms'])}")
        lines.append("")
        if r.risks:
            lines += ["**风险清单**（C1 废止 / C2 危大与必备内容 / C3 条文相符性）：", ""]
            for x in r.risks[:20]:
                lines.append(f"- `{x.get('check')}` `{x.get('severity')}` {x.get('title')}")
                if x.get("detail"):
                    lines.append(f"  - {x['detail']}")
                if x.get("basis"):
                    lines.append(f"  - 依据：{x['basis']}")
        else:
            lines.append("未发现风险项。")
        if r.entities:
            lines += ["", "**抽取实体（前 15）**：", ""]
            for e in r.entities[:15]:
                lines.append(f"- `{e.get('type')}` = {e.get('text')}")
        if r.rag_hits:
            lines += ["", "**RAG 溯源（前 10）**："]
            for h in r.rag_hits[:10]:
                lines.append(f"- `#{h['rank']}` [{h['source']}] {h.get('clause_no','')} "
                             f"(q=`{h.get('query','')}`) {h['text'][:100]}")
        return "\n".join(lines)


_STORE = None
