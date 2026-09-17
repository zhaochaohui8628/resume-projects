"""Benchmark 报告生成：全链路指标 Markdown 报告。"""
from __future__ import annotations

import os
import time


def _pct(v) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def _f(v) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


def render_report(data: dict, bench_dir: str) -> str:
    """data = run_benchmark() 返回值（cases + summary）。"""
    s = data["summary"]
    cases = data["cases"]
    lines = []
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# 多 Agent 级联容错评测 Benchmark 报告")
    lines.append("")
    lines.append(f"> 生成时间：{ts} ｜ 用例数：{s['total_cases']} ｜ "
                 f"崩溃：{s['crashed']} ｜ 无输出：{s['no_output']}")
    lines.append("")
    lines.append("## 一、鲁棒性总览")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 生存率（不崩溃） | {_pct(s['survival_rate'])} |")
    lines.append(f"| 崩溃用例 | {', '.join(s['crash_ids']) or '无'} |")
    lines.append(f"| 无输出用例 | {', '.join(s['no_output_ids']) or '无'} |")
    lines.append("")
    lines.append("## 二、全链路指标拆解")
    lines.append("")
    lines.append("| 指标 | 值 | 说明 |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| 审查漏报率（FNR） | {_pct(s['avg_fnr'])} | golden 风险未被检出比例 |")
    lines.append(f"| 审查误报率（FPR） | {_pct(s['avg_fpr'])} | 预测风险不在 golden 比例（规则审查为全量查表，pred 远多于 golden 属常态） |")
    lines.append(f"| NER 实体 F1 | {_f(s['ner_avg_f1'])} | 实体识别（含混沌输入） |")
    lines.append(f"| RAG recall@k | {_f(s['rag_avg_recall_at_k'])} | golden 条文命中率 |")
    lines.append(f"| 平均总耗时 | {s['avg_total_ms']} ms | 端到端编排 |")
    if s.get("avg_subagent_ms"):
        sub = " ｜ ".join(f"{k}: {v} ms" for k, v in s["avg_subagent_ms"].items())
        lines.append(f"| Subagent 平均耗时 | {sub} | 各 agent 单独分解（**单跑基准**，与总耗时不可相加） |")
    ph = s.get("avg_phase_ms") or {}
    if ph.get("measured_ms"):
        _tot = ph["measured_ms"] or 1.0

        def _pm(key: str) -> str:
            v = ph.get(key, 0.0)
            return f"{v} ms（{v / _tot * 100:.1f}%）"

        lines.append(f"| 阶段拆解 · 意图路由 | {_pm('route_ms')} | 无 Key 走关键词规则；带 --llm 时为模型往返 |")
        lines.append(f"| 阶段拆解 · 执行 | {_pm('execute_ms')} | subagent：NER 分块 + 检索精排 + 阈值/限值比对 |")
        lines.append(f"| 阶段拆解 · LLM 汇总 | {_pm('summary_ms')} | 无 Key 时跳过该路径；带 --llm deepseek 时含模型往返 |")
    lines.append("")

    # —— 多维拆解章节（v2）——
    lines.append("## 三、多维指标拆解")
    lines.append("")
    if s.get("by_check"):
        lines.append("### 3.1 按合规检查项分层（漏报率）")
        lines.append("")
        lines.append("| 检查项 | 含义 | golden 数 | 命中 | 漏报率 | 覆盖用例 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for c, v in s["by_check"].items():
            cov = (s.get("check_coverage") or {}).get(c, 0)
            lines.append(f"| {c} | {v['name']} | {v['golden']} | {v['hit']} | "
                         f"{_pct(v['fnr'])} | {cov} |")
        lines.append("")
    if s.get("by_severity"):
        lines.append("### 3.2 按严重度分层（漏报率）")
        lines.append("")
        lines.append("| 严重度 | golden 数 | 命中 | 漏报率 |")
        lines.append("| --- | --- | --- | --- |")
        for sv, v in s["by_severity"].items():
            if v["golden"]:
                lines.append(f"| {sv} | {v['golden']} | {v['hit']} | {_pct(v['fnr'])} |")
        lines.append("")
    if s.get("pipeline_rate"):
        pr = s["pipeline_rate"]
        lines.append("### 3.3 链路贯通率（无需 golden，反映链路健康度）")
        lines.append("")
        lines.append("| 环节 | 贯通率 | 说明 |")
        lines.append("| --- | --- | --- |")
        lines.append(f"| 抽出实体 | {_pct(pr['entity'])} | NER 有输出 |")
        lines.append(f"| 抽出『参数』实体 | {_pct(pr['param'])} | 三元组绑定的前提 |")
        lines.append(f"| 绑定三元组 | {_pct(pr['triple'])} | 判档/技术核对的输入 |")
        lines.append(f"| 产出风险项 | {_pct(pr['risk'])} | 端到端有审查结论 |")
        lines.append("")
    if s.get("traceability_rate") is not None:
        lines.append("### 3.4 依据可溯源率")
        lines.append("")
        lines.append(f"产出的风险项中带规范依据（basis/source/clause_no）的比例："
                     f"**{_pct(s['traceability_rate'])}**")
        lines.append("")
        lines.append("> 合规审查结论必须可溯源到条文，否则无法人工复核。")
        lines.append("")

    lines.append("## 四、逐用例明细")
    lines.append("")
    for c in cases:
        status = "❌ 崩溃" if c["crash"] else ("⚠️ 无输出" if not c["has_output"] else "✅ 正常")
        lines.append(f"### {c['id']} · {c['category']} ｜ {status}")
        lines.append("")
        lines.append(f"> {c['desc']}")
        lines.append("")
        rm, nm, rgm, tm = c["risk_m"], c["ner_m"], c["rag_m"], c["time_m"]
        lines.append("| 维度 | 指标 | 值 |")
        lines.append("| --- | --- | --- |")
        lines.append(f"| 耗时 | 总耗时 | {tm['total_ms']} ms |")
        if tm.get("subagent_ms"):
            sub = " ｜ ".join(f"{k}: {v} ms" for k, v in tm["subagent_ms"].items())
            lines.append(f"| 耗时 | Subagent 分解 | {sub} |")
        if rm.get("golden"):
            lines.append(f"| 审查 | 漏报率/误报率 | {_pct(rm['fnr'])} / {_pct(rm['fpr'])} "
                         f"（命中 {rm['hit']}/{rm['golden']}） |")
        if nm.get("f1") is not None:
            lines.append(f"| NER | F1 | {_f(nm['f1'])} "
                         f"（P={_f(nm['precision'])} R={_f(nm['recall'])}） |")
        if rgm.get("recall@k") is not None:
            lines.append(f"| RAG | recall@{rgm['k']} | {_f(rgm['recall@k'])} "
                         f"（{rgm['hit']}/{rgm['golden']}） |")
        pm = c.get("pipe_m") or {}
        if pm:
            lines.append(f"| 链路 | 实体/参数/三元组/风险 | "
                         f"{pm.get('n_entities', 0)}/{pm.get('n_params', 0)}/"
                         f"{pm.get('n_triples', 0)}/{pm.get('n_risks', 0)} |")
        tcm = c.get("trace_m") or {}
        if tcm.get("total"):
            lines.append(f"| 溯源 | 带依据比例 | {_pct(tcm.get('rate'))} "
                         f"（{tcm.get('with_basis')}/{tcm.get('total')}） |")
        if c.get("error"):
            lines.append(f"| 错误 | - | `{c['error']}` |")
        lines.append("")
    return "\n".join(lines)


def save_report(data: dict, bench_dir: str, tag: str = "") -> str:
    md = render_report(data, bench_dir)
    out_dir = os.path.join(bench_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"bench_report{'_' + tag if tag else ''}_{time.strftime('%Y%m%d_%H%M%S')}.md"
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path
