"""分诊式合规审查（P0 端到端验证）—— 方案 → NER → 三元组 → 查表/检索 → 比对 → 结论。

与现状链路的差别：**多了一层"分诊 + 比对"**
  判定类（是否危大/超规模/需论证）→ 阈值表查表，不检索；
  技术类 → 一条量名短查询检索 → comparator 与方案值比较 → 出 verdict。

用法（GPU 环境 torch_gpu）：
    python agent/scripts/run_triage.py agent/tests/fixtures/sample_plan.txt
    python agent/scripts/run_triage.py <方案> --no-rag     # 只跑查表，不检索（快）
    python agent/scripts/run_triage.py <方案> --rule-only  # NER 用纯规则，不加载模型
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT = os.path.dirname(HERE)
AGENT_SRC = os.path.join(AGENT, "src")
ROOT = os.path.dirname(AGENT)
for p in (AGENT_SRC, os.path.join(ROOT, "rag2"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.param_extractor import extract_triples, dedup, keep_worst, judgeable  # noqa: E402
from tools.hazard_level import judge, LEVEL_NONE, LEVEL_HAZ, LEVEL_SUPER  # noqa: E402


def _ner(plan_text: str, rule_only: bool):
    from tools.ner_client import NerClient
    import ner2.src.pipeline.full_text as ftext
    cli = NerClient(backend="ner2",
                    model_path=None if rule_only else ftext.DEFAULT_MODEL)
    return cli.extract(plan_text)


def _rag():
    try:
        from src.retrieval.hybrid import load_retriever
        mdir = os.path.join(ROOT, "rag2", "data", "models")
        doc = os.path.join(mdir, "dual_gold", "doc_encoder")
        qry = os.path.join(mdir, "dual_gold", "query_encoder")
        if os.path.isdir(doc):
            return load_retriever(tower_doc=doc, tower_query=qry,
                                  index_file="dual_gold.faiss", alpha=0.3)
        return load_retriever()
    except Exception as e:                      # 检索不可用时降级为纯查表
        print(f"[warn] RAG 不可用，仅查表：{e}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--no-rag", action="store_true", help="跳过检索，只做查表判定")
    ap.add_argument("--rule-only", action="store_true", help="NER 纯规则模式（无需 GPU）")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    with open(args.plan, encoding="utf-8") as f:
        plan = f.read()

    ents = _ner(plan, args.rule_only)
    triples = extract_triples(ents)
    uniq = dedup(triples)
    judge_set = judgeable(keep_worst(uniq))

    print(f"方案：{args.plan}")
    print(f"实体 {len(ents)} → 三元组 {len(triples)} → 去重 {len(uniq)} "
          f"→ 最不利去重 {len(keep_worst(uniq))} → 可判定 {len(judge_set)}")

    rag = None if args.no_rag else _rag()

    if not judge_set:
        print("\n未抽取到可判定的工况参数。")
        return 0

    print("\n" + "=" * 96)
    print(f"{'类别':<8}{'量名':<10}{'方案值':>10}  {'分档':<12}{'依据'}")
    print("-" * 96)
    for t in judge_set:
        r = judge(t["category"], t["metric"], t["value"], t["unit"], t.get("context", ""))
        mark = {LEVEL_SUPER: "!!", LEVEL_HAZ: "! ", LEVEL_NONE: "  "}.get(r["level"], "? ")
        print(f"{t['category'] or '—':<8}{t['metric']:<10}"
              f"{str(t['value']) + t['unit']:>10}  {mark}{r['level']:<10}{r['threshold'] or r['note']}")
        if r["need_review"]:
            print(f"{'':>20}→ 需组织专家论证（{r['basis']}）")
        if t.get("sent_text"):
            print(f"{'':>20}出处：{t['sent_text'][:56]}")

        if rag is not None:
            from tools.comparator import compare_all
            q = f"{t['metric']} 允许值 应符合"
            hits = rag.search(q, top_k=args.topk)
            cmp_res = compare_all(t, hits)
            top = cmp_res.get("picked")
            if top:
                print(f"{'':>20}检索「{q}」→ {cmp_res['verdict']}"
                      f"（{top['source']} {top['clause_no']}｜{top['limit']}）")
            else:
                print(f"{'':>20}检索「{q}」→ {cmp_res['verdict']}")
        print("-" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
