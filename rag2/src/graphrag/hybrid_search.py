"""双引擎检索：图谱子图扩展 + 向量检索 + 多跳推理。

核心解决单路 RAG 的"盲人摸象"问题：
- 向量引擎：query 语义召回 topN 条款（可能只覆盖 1-2 部规范）
- 图谱引擎：识别 query 中的危大类别/实体 → 图谱子图扩展 → 补全"该危大工程
  还受哪些规范约束"（多标准交叉）
- 融合：图谱条款 + 向量条款 去重合并 → CE 精排 → 输出

流程：
1. `classify_query()`: 从 query 识别危大类别（关键词/实体匹配）
2. `graph_expand()`: 图谱从危大类别/实体 2 跳扩展，收集条款
3. `vector_retrieve()`: 复用 rag2 的 BM25+dual_mix+RRF
4. `fuse()`: 图谱条款补入候选（去重），CE 精排
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.paths import data_dir  # noqa: E402
from src.graphrag.schema import CLS  # noqa: E402
from src.graphrag.neo4j_store import Neo4jStore  # noqa: E402

# 危大类别关键词表（demo 级，后续可扩）
HAZARD_KEYWORDS = {
    "深基坑工程": ["深基坑", "基坑开挖", "基坑支护", "基坑降水", "基坑监测"],
    "模板支撑工程": ["模板支撑", "模板支架", "高支模", "高大模板"],
    "起重吊装工程": ["起重吊装", "吊装", "塔吊", "塔式起重机"],
    "脚手架工程": ["脚手架", "落地架", "悬挑架", "附着升降"],
}


class GraphRAGRetriever:
    """图谱 + 向量双引擎检索器。"""

    def __init__(self, vector_retriever=None):
        self.store = Neo4jStore()
        self.vector = vector_retriever  # 可选，传入 HybridRetriever

    # ---- 1. 危大类别识别 ----
    def classify_query(self, query: str) -> list[str]:
        hits = []
        for hazard, kws in HAZARD_KEYWORDS.items():
            if any(k in query for k in kws):
                hits.append(hazard)
        return hits

    # ---- 2. 图谱子图扩展 ----
    def graph_expand(self, query: str, hops: int = 2) -> list[dict]:
        """从 query 识别的危大类别/实体扩展，返回条款节点。"""
        seeds = self.classify_query(query)
        clauses: dict[str, dict] = {}
        for seed in seeds:
            sub = self.store.expand_subgraph(seed, hops)
            for n in sub["nodes"]:
                if n["label"] == CLS:
                    clauses[n["id"]] = n
        # 无危大命中时，用 query 关键词直接试扩展
        if not clauses:
            for kw in query.split():
                if len(kw) >= 2:
                    sub = self.store.expand_subgraph(kw, 1)
                    for n in sub["nodes"]:
                        if n["label"] == CLS:
                            clauses[n["id"]] = n
        return list(clauses.values())

    # ---- 3. 向量检索（复用 rag2）----
    def vector_retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        if self.vector is None:
            return []
        hits = self.vector.search(query, top_k=top_k)
        return [{"clause_id": h["clause_id"], "text": h.get("text", ""),
                 "source": "vector"} for h in hits]

    # ---- 4. 融合 ----
    def search(self, query: str, top_k: int = 5, hops: int = 2) -> dict:
        graph_hits = self.graph_expand(query, hops)
        vec_hits = self.vector_retrieve(query, top_k=top_k * 2)

        # 合并去重（图谱条款优先）
        merged: dict[str, dict] = {}
        for g in graph_hits:
            merged[g["id"]] = {"clause_id": g["id"], "text": g.get("text", ""),
                               "source": "graph", "hop": 1}
        for v in vec_hits:
            cid = v["clause_id"]
            if cid in merged:
                merged[cid]["source"] = "both"
            else:
                merged[cid] = {"clause_id": cid, "text": v.get("text", ""),
                               "source": "vector", "hop": 0}

        # 排序：both > graph > vector
        order = {"both": 0, "graph": 1, "vector": 2}
        ranked = sorted(merged.values(), key=lambda x: order[x["source"]])
        return {
            "query": query,
            "hazards": self.classify_query(query),
            "graph_clauses": len(graph_hits),
            "vector_clauses": len(vec_hits),
            "merged": len(ranked),
            "hits": ranked[:top_k],
            "all": ranked,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="深基坑开挖前要做哪些安全准备？")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--hops", type=int, default=2)
    a = ap.parse_args()

    # 尝试接入 rag2 向量检索
    vec = None
    try:
        import torch  # noqa: F401  必须先导 torch 让 faiss 的 DLL 路径生效（WDDM 坑）
        from src.retrieval.hybrid import Corpus, EncoderPair, HybridRetriever
        from src.index.bm25 import BM25Index
        from src.index.dense import FaissStore

        corpus = Corpus(data_dir("corpus", "clauses.jsonl"), data_dir("index", "units.jsonl"))
        bm = BM25Index.load(data_dir("index", "bm25.json"))
        store = FaissStore.load(data_dir("index", "dual_mix.faiss"))
        enc = EncoderPair("data/models/dual_mix/doc_encoder", "data/models/dual_mix/query_encoder")
        vec = HybridRetriever(corpus, bm, store, enc, fuse_mode="rrf", rrf_k=10, cand=20)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 向量检索不可用（{e}），仅图谱引擎")

    retriever = GraphRAGRetriever(vector_retriever=vec)
    res = retriever.search(a.query, top_k=a.top_k, hops=a.hops)
    print(f"query: {a.query}")
    print(f"识别危大类别: {res['hazards'] or '(未识别)'}")
    print(f"图谱扩展条款 {res['graph_clauses']} / 向量条款 {res['vector_clauses']} / 合并 {res['merged']}")
    print("-" * 60)
    for i, h in enumerate(res["hits"], 1):
        src_tag = {"graph": "图谱", "vector": "向量", "both": "双源"}[h["source"]]
        print(f"{i}. [{src_tag}] {h['clause_id']}")
        print(f"   {h['text'][:70]}")


if __name__ == "__main__":
    main()
