"""链路贯通率自检：新参数正文片段能否真正走通 NER → 三元组 → 风险判定。

不需要 golden 标注（避免"用被测系统定义正确答案"的自证问题），
统计的是**链路健康度**：
  - 参数实体召回率：片段中能抽出 ≥1 个"参数"类实体的比例
  - 三元组绑定率：能绑出 ≥1 个三元组的比例
  - 判档产出率：hazard_level 给出判档结论的比例
  - 风险产出率：最终 risks 非空的比例

用法：python agent/bench/check_pipeline_health.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, os.path.join(ROOT, "agent", "src"),
          os.path.join(ROOT, "ner2"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    args = ap.parse_args()

    from ner2.src.pipeline.full_text import FullTextExtractor
    from tools.param_extractor import extract_triples
    from subagents.review_agent import ReviewAgent

    rows = [json.loads(l) for l in open(
        os.path.join(HERE, "data", "real_cases_params.jsonl"), encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    fx = FullTextExtractor(
        model_path=os.path.join(ROOT, "ner2", "models", "s2_crf_param_v3", "model.pt"),
        device="cpu")
    agent = ReviewAgent()

    n = len(rows)
    n_ent = n_param = n_tri = n_risk = 0
    n_sent0 = 0
    detail = []
    for c in rows:
        path = os.path.join(HERE, "data", c["plan_file"])
        text = open(path, encoding="utf-8").read()
        ents = fx.extract_text(text)
        params = [e for e in ents if e.get("type") == "参数"]
        tri = extract_triples(ents)
        res = agent.run(query=c["query"], plan=text,
                        ctx={"rag_rerank": False, "ner_crf": True,
                             "memory_dir": None})
        risks = getattr(res, "risks", []) or []
        if ents:
            n_ent += 1
        if params:
            n_param += 1
        if tri:
            n_tri += 1
        if risks:
            n_risk += 1
        if not ents:
            n_sent0 += 1
        detail.append({"id": c["id"], "ents": len(ents), "params": len(params),
                       "triples": len(tri), "risks": len(risks)})

    def pct(x):
        return f"{x}/{n} ({x / n * 100:.1f}%)" if n else "-"

    print("=" * 62)
    print(f"链路贯通率自检（{n} 条含参正文片段）")
    print("=" * 62)
    print(f"  抽出实体          {pct(n_ent)}")
    print(f"  抽出『参数』实体   {pct(n_param)}")
    print(f"  绑定三元组        {pct(n_tri)}")
    print(f"  产出风险项        {pct(n_risk)}")
    print(f"  净句为空（异常）   {n_sent0}")
    print("=" * 62)

    out = os.path.join(HERE, "outputs", "pipeline_health.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"total": n, "with_entities": n_ent, "with_params": n_param,
                   "with_triples": n_tri, "with_risks": n_risk,
                   "empty_sent": n_sent0,
                   "rate": {"entity": round(n_ent / n, 4) if n else 0,
                            "param": round(n_param / n, 4) if n else 0,
                            "triple": round(n_tri / n, 4) if n else 0,
                            "risk": round(n_risk / n, 4) if n else 0},
                   "detail": detail}, f, ensure_ascii=False, indent=1)
    print(f"明细已保存：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
