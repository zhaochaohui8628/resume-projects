"""多 Agent 级联容错评测 Benchmark（CLI 入口）。

用法（项目根目录，Windows PowerShell）：
    $PY = "C:/Users/<用户名>/anaconda3/python.exe"
    # 快速全量（零依赖 mock，确定性规则路径，默认即全量 12 用例）
    & $PY agent/bench/run_benchmark.py --quick
    # 指定用例
    & $PY agent/bench/run_benchmark.py --cases empty_input,garbage_bytes
    # 带 LLM（路由/汇总/ReAct 链路，需 DEEPSEEK_API_KEY）
    & $PY agent/bench/run_benchmark.py --llm deepseek
    # 输出报告路径
    & $PY agent/bench/run_benchmark.py --quick --out
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.dirname(HERE)
for p in (AGENT_DIR, os.path.join(AGENT_DIR, "src"),
          os.path.dirname(AGENT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    ap = argparse.ArgumentParser(description="多 Agent 级联容错评测 Benchmark")
    ap.add_argument("--quick", action="store_true", help="零依赖确定性路径（默认行为）")
    ap.add_argument("--cases", default="", help="逗号分隔的用例 id（空=全部）")
    ap.add_argument("--real", action="store_true",
                    help="追加真实方案用例集（11 本真实方案 → 33 用例）")
    ap.add_argument("--llm", default="", choices=["", "mock", "deepseek"],
                    help="LLM 后端：空/mock=零依赖；deepseek=真实路由/汇总")
    ap.add_argument("--out", action="store_true", help="保存 Markdown 报告到 bench/outputs/")
    ap.add_argument("--json", action="store_true", help="同时输出 JSON 指标")
    args = ap.parse_args()

    from bench.cases import case_ids
    from bench.runner import run_benchmark
    from bench.report import save_report
    from orchestrator.dispatcher import GlobalOpts

    ids = [i.strip() for i in args.cases.split(",") if i.strip()] or None
    if ids:
        valid = set(case_ids(include_real=args.real))
        bad = [i for i in ids if i not in valid]
        if bad:
            print(f"未知用例 id: {bad}", file=sys.stderr)
            return 2

    llm = None
    if args.llm == "mock":
        from llm.mock import MockLLM
        llm = MockLLM()
    elif args.llm == "deepseek":
        from llm.deepseek import DeepSeekLLM
        try:
            llm = DeepSeekLLM()
        except RuntimeError as e:
            print(f"⚠️ {e}；回退零依赖路径", file=sys.stderr)

    opts = GlobalOpts(memory_dir=None)
    data = run_benchmark(ids=ids, llm=llm, opts=opts, include_real=args.real)

    # 控制台摘要
    s = data["summary"]
    print("=" * 60)
    print(f"Benchmark 用例 {s['total_cases']} ｜ 崩溃 {s['crashed']} ｜ "
          f"生存率 {s['survival_rate'] * 100:.1f}%")
    print(f"  审查 FNR={s['avg_fnr']}  FPR={s['avg_fpr']}  "
          f"NER F1={s['ner_avg_f1']}  RAG recall@k={s['rag_avg_recall_at_k']}")
    print(f"  平均总耗时 {s['avg_total_ms']} ms ｜ subagent: {s['avg_subagent_ms']}")
    if s.get("avg_phase_ms"):
        pm = s["avg_phase_ms"]
        print(f"  阶段拆解：路由 {pm.get('route_ms')} ms ｜ 执行 {pm.get('execute_ms')} ms ｜ "
              f"汇总 {pm.get('summary_ms')} ms ｜ 其他 {pm.get('other_ms')} ms")
    # 多维拆解
    if s.get("by_check"):
        print("  -- 按检查项漏报率 --")
        for c, v in s["by_check"].items():
            if v["golden"]:
                print(f"     {c} {v['name']:<8s} golden={v['golden']:<3d} "
                      f"命中={v['hit']:<3d} FNR={v['fnr'] * 100:.1f}%")
    if s.get("by_severity"):
        print("  -- 按严重度漏报率 --")
        for sv, v in s["by_severity"].items():
            if v["golden"]:
                print(f"     {sv:<7s} golden={v['golden']:<3d} "
                      f"命中={v['hit']:<3d} FNR={v['fnr'] * 100:.1f}%")
    if s.get("pipeline_rate"):
        pr = s["pipeline_rate"]
        print("  -- 链路贯通率 --")
        print(f"     实体 {pr['entity'] * 100:.1f}% ｜ 参数 {pr['param'] * 100:.1f}% ｜ "
              f"三元组 {pr['triple'] * 100:.1f}% ｜ 风险 {pr['risk'] * 100:.1f}%")
    if s.get("traceability_rate") is not None:
        print(f"  -- 依据可溯源率 -- {s['traceability_rate'] * 100:.1f}%")
    for c in data["cases"]:
        flag = "❌" if c["crash"] else ("⚠️" if not c["has_output"] else "✅")
        ph = c["time_m"].get("phase_ms") or {}
        print(f"  {flag} {c['id']:<22} {c['time_m']['total_ms']:>8.1f} ms"
              f"  [路由 {ph.get('route_ms', '-')} / 执行 {ph.get('execute_ms', '-')} /"
              f" 汇总 {ph.get('summary_ms', '-')}]"
              f"  risks={len(c['risks'])}  ents={len(c['entities'])}")
    print("=" * 60)

    if args.out:
        path = save_report(data, HERE)
        print(f"报告已保存：{path}")
    if args.json:
        out = os.path.join(HERE, "outputs", "bench_metrics.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data["summary"], f, ensure_ascii=False, indent=2)
        print(f"JSON 指标已保存：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
