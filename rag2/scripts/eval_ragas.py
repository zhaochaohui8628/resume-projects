"""ragas 端到端评估：检索 → LLM 生成答案 → ragas 判分（四指标）。

链路：
    query → HybridRetriever.search(top_k=3) → 条文拼上下文 → DeepSeek 生成答案
        → ragas 评估（faithfulness / answer_relevancy / context_precision / context_recall）

指标说明：
- faithfulness：答案是否忠于检索上下文（LLM 判分）
- answer_relevancy：答案相关度（LLM 从答案生成问题 + 本地 bge embedding 算相似度）
- context_precision：检索上下文与参考答案的相关度（LLM 判分）
- context_recall：检索上下文覆盖参考答案的程度（LLM 判分）
- embedding 用本地 bge-small-zh-v1.5（项目预置，无需 API）

用法（需 DEEPSEEK_API_KEY 环境变量）：
    python rag2/scripts/eval_ragas.py [--set gold|holdout20] [--limit N] [--tag 名称]

key 仅从环境变量读取，绝不落盘/打印/写日志。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # rag2/
sys.path.insert(0, str(ROOT))                # 让 from src.xxx 可用（含相对导入）
sys.path.insert(0, str(ROOT / "src"))        # 让 from serve.llm_client 可用

# ---- 1. 加载检索器 ----
from src.retrieval.hybrid import load_retriever  # noqa: E402

# ---- 2. 构建评估数据 ----
def load_queries(set_name: str) -> list[dict]:
    """读黄金评估集（query + gold clause_id）。"""
    p = ROOT / "data" / "phase7"
    f = p / ("gold_holdout20.jsonl" if set_name == "holdout20" else "gold_eval_clean.jsonl")
    return [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]


def build_ragas_dataset(queries, retriever, top_k=3):
    """query → 检索 top_k 条文（text 拼接）→ 数据集条目。"""
    rows = []
    for q in queries:
        hits = retriever.search(q["query"], top_k=top_k)
        ctx = "\n".join(f"[{i+1}] {h['clause_id']}: {h['text']}" for i, h in enumerate(hits))
        rows.append({
            "user_input": q["query"],
            "retrieved_contexts": [h["text"] for h in hits],
            "reference": retriever.corpus.text(q["clause_id"]),  # gold 条款原文
            # 预生成答案（评估前用 DeepSeek 生成，减少 ragas 内重复调用）
            "answer": None,  # 占位
            "gold_clause": q["clause_id"],
        })
    return rows


# ---- 3. 用 DeepSeek 生成答案 ----
def generate_answers(rows, llm_client, top_k=3):
    """对每个 query 用检索条文生成答案（引用条文，防幻觉）。"""
    from serve.llm_client import LLMClient
    client = llm_client or LLMClient()
    system = "你是建筑施工规范咨询助手。基于提供的规范条文回答用户问题，引用条文用[编号]标注。若条文不足说明依据现有条文无法完全回答。答案简洁中文。"
    for r in rows:
        hits_ctx = "\n".join(f"[{i+1}] {h['clause_id']}: {h['text']}" for i, h in enumerate(r["_hits"]))
        user = f"用户问题：{r['user_input']}\n\n检索到的条文：\n{hits_ctx}"
        try:
            r["answer"] = client.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                max_tokens=256, temperature=0.1)
        except Exception as e:
            r["answer"] = f"（生成失败：{type(e).__name__}）"
            r["gen_error"] = str(e)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="gold", choices=["gold", "holdout20"])
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--tag", default="", help="结果文件名后缀")
    a = ap.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("错误：需要 DEEPSEEK_API_KEY 环境变量（评估答案生成与 ragas 判分都用它）")

    # 检索器（dual_mix + RRF，最终部署配置）——模型路径基于 ROOT（rag2/）
    retriever = load_retriever(
        tower_doc=str(ROOT / "data/models/dual_mix/doc_encoder"),
        tower_query=str(ROOT / "data/models/dual_mix/query_encoder"),
        index_file="dual_mix.faiss",
        alpha=0.0, fuse_mode="rrf", rrf_k=10, cand=20)

    queries = load_queries(a.set)
    if a.limit:
        queries = queries[:a.limit]
    print(f"[eval_ragas] 测试集={a.set} 条数={len(queries)}")

    # 预检索（存 _hits 供生成答案 + ragas 上下文）
    rows = []
    for q in queries:
        hits = retriever.search(q["query"], top_k=a.top_k)
        rows.append({
            "user_input": q["query"],
            "retrieved_contexts": [h["text"] for h in hits],
            "reference": retriever.corpus.text(q["clause_id"]),
            "gold_clause": q["clause_id"],
            "_hits": hits,
        })
    print(f"[eval_ragas] 检索完成，生成答案中（{len(rows)} 条）...")

    # 生成答案
    rows = generate_answers(rows, None, a.top_k)
    for r in rows:
        r.pop("_hits", None)

    # ---- 4. ragas 评估 ----
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision, context_recall,
    )

    ds = Dataset.from_list([
        {"user_input": r["user_input"],
         "answer": r["answer"],
         "retrieved_contexts": r["retrieved_contexts"],
         "reference": r["reference"]}
        for r in rows
    ])

    # ragas 用 DeepSeek 判分（OpenAI 兼容）
    from langchain_openai import ChatOpenAI
    judge = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )

    # answer_relevancy 需要 embedding：用本地 bge-small-zh-v1.5（项目预置）
    from langchain_community.embeddings import HuggingFaceEmbeddings
    emb_model = str(ROOT.parent / "data/models/bge-small-zh-v1.5")  # 项目根 data/models/
    embeddings = HuggingFaceEmbeddings(model_name=emb_model)

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
    )
    df = result.to_pandas()

    # 输出
    tag = a.tag or a.set
    out_dir = ROOT / "data" / "eval"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"ragas_{tag}.json"
    result_data = {
        "set": a.set, "n": len(rows), "top_k": a.top_k,
        "scores": {k: float(v) for k, v in df.mean(numeric_only=True).items()},
        "per_row": df.to_dict("records"),
    }
    out_json.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval_ragas] 完成 → {out_json}")
    print(json.dumps(result_data["scores"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
