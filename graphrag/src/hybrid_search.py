"""GraphRAG 双引擎检索（独立 demo 版）——图谱子图扩展 + 可选向量召回。

核心解决单路 RAG 的"盲人摸象"问题：
- 图谱引擎：识别 query 中的危大类别 → Neo4j 子图扩展 → 补全"该危大工程还受哪些
  规范约束"（多标准交叉，给出**范围/白名单**，不是塞条款）。
- 向量引擎（可选）：复用 rag2 的 BM25 ∥ 双塔 + RRF，语义命中具体条款。
- 融合：图谱范围 + 向量条款 去重合并。

两种模式（前端开关对应）：
- mode="graph"：只跑图谱路（关闭向量端）
- mode="dual" ：图谱路 + 向量路（向量端不可用时自动退化为 graph，并在结果里标注）

用法：
  python graphrag/src/hybrid_search.py --query "深基坑开挖前的安全准备" --mode dual

⚠️ 导入约定：本 demo 内部用「裸模块名」（见 neo4j_store.py 文件头说明），
`src` 这个名字留给 rag2 的 `src.common / src.index / src.retrieval`。
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import zip_longest
from pathlib import Path

HERE = Path(__file__).resolve()
GRAPH_ROOT = HERE.parents[1]                       # .../graphrag
SRC_DIR = HERE.parent                              # .../graphrag/src
WORKSPACE = GRAPH_ROOT.parent                      # 项目根
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neo4j_store import Neo4jStore, Neo4jUnavailable  # noqa: E402

# 危大类别关键词表（demo 级）
HAZARD_KEYWORDS = {
    "深基坑工程": ["深基坑", "基坑开挖", "基坑支护", "基坑降水", "基坑监测", "基坑"],
    "模板支撑工程": ["模板支撑", "模板支架", "高支模", "高大模板", "模板"],
    "起重吊装工程": ["起重吊装", "吊装", "塔吊", "塔式起重机", "汽车吊"],
    "脚手架工程": ["脚手架", "落地架", "悬挑架", "附着升降", "爬架"],
}

# 类别 key（rag2 / 前端） → 图谱 HazardCategory 名
CATEGORY_TO_HAZARD = {
    "起重吊装": "起重吊装工程",
    "基坑工程": "深基坑工程",
    "模板支撑": "模板支撑工程",
    "脚手架": "脚手架工程",
}


class GraphRAGRetriever:
    """图谱 + 向量双引擎检索器（图谱强制走 Neo4j）。"""

    def __init__(self, store: Neo4jStore | None = None, vector_retriever=None):
        self.store = store or Neo4jStore()
        self.vector = vector_retriever
        self.vector_error = ""

    # ---- 1. 危大类别识别 ----
    def classify_query(self, query: str) -> list[str]:
        return [haz for haz, kws in HAZARD_KEYWORDS.items()
                if any(k in query for k in kws)]

    # ---- 2. 图谱路：类别 → 白名单 + 子图 ----
    def graph_path(self, query: str, hops: int = 2) -> dict:
        hazards = self.classify_query(query)
        whitelist: list[str] = []
        subgraph = {"nodes": [], "edges": []}
        thresholds: list[dict] = []
        for haz in hazards:
            for name in self.store.hazard_standards(haz):
                if name not in whitelist:
                    whitelist.append(name)
            thresholds += self.store.hazard_thresholds(haz)
            sg = self.store.expand_subgraph(haz, hops)
            _merge_subgraph(subgraph, sg)
        # 未识别类别 → 用 query 关键词直接扩展
        if not hazards:
            for kw in [w for w in query.replace("？", " ").split() if len(w) >= 2][:3]:
                _merge_subgraph(subgraph, self.store.expand_subgraph(kw, 1))
        return {"hazards": hazards, "whitelist": whitelist,
                "thresholds": thresholds, "subgraph": subgraph}

    # ---- 3. 向量路：复用 rag2（可选）----
    def vector_path(self, query: str, top_k: int = 10) -> list[dict]:
        if self.vector is None:
            return []
        try:
            hits = self.vector.search(query, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            self.vector_error = f"{type(e).__name__}: {e}"
            return []
        return [{"clause_id": h.get("clause_id") or h.get("id", ""),
                 "text": h.get("text", ""), "source": h.get("source", ""),
                 "clause_no": h.get("clause_no", ""),
                 "score": h.get("score")} for h in hits]

    # ---- 4. 融合 ----
    def search(self, query: str, mode: str = "dual", top_k: int = 5,
               hops: int = 2) -> dict:
        g = self.graph_path(query, hops)
        vec_hits = self.vector_path(query, top_k * 2) if mode == "dual" else []

        merged: dict[str, dict] = {}
        for v in vec_hits:
            cid = v["clause_id"] or f"(no-id){len(merged)}"
            merged[cid] = {"clause_id": cid, "text": v.get("text", ""),
                           "source": "vector", "clause_no": v.get("clause_no", ""),
                           "score": v.get("score")}
        # 图谱路的条款节点（子图里的 Clause）
        graph_ids: list[str] = []
        for n in g["subgraph"]["nodes"]:
            if n["label"] == "Clause":
                cid = n["id"]
                if cid in merged:
                    merged[cid]["source"] = "both"
                else:
                    merged[cid] = {"clause_id": cid, "text": n.get("props", {}).get("text", ""),
                                   "source": "graph", "clause_no": "", "score": None}
                graph_ids.append(cid)

        # 两路各自的独立结果（供前端分区展示；含被两路同时命中的"双源"条款）
        graph_hits = [merged[c] for c in graph_ids]
        vector_hits: list[dict] = []
        for v in vec_hits:
            cid = v["clause_id"]
            if cid and cid in merged:
                vector_hits.append(merged[cid])
            else:
                vector_hits.append({"clause_id": cid or "", "text": v.get("text", ""),
                                    "source": "vector", "clause_no": v.get("clause_no", ""),
                                    "score": v.get("score")})

        # 合并排序：双源最前，其后图谱 / 向量**交替**出现。
        # （若单纯按来源优先级排，图谱条款会把 top_k 占满，向量结果永远看不见。）
        both = [v for v in merged.values() if v["source"] == "both"]
        g_only = [merged[c] for c in graph_ids if merged[c]["source"] == "graph"]
        v_only = [v for v in merged.values() if v["source"] == "vector"]
        ranked = both + [x for pair in zip_longest(g_only, v_only) for x in pair
                         if x is not None]
        effective_mode = "dual" if (mode == "dual" and (vec_hits or not self.vector_error)) else (
            "graph" if mode == "graph" else "dual_degraded")

        return {
            "query": query,
            "requested_mode": mode,
            "effective_mode": effective_mode,
            "hazards": g["hazards"],
            "graph_whitelist": g["whitelist"],
            "graph_thresholds": g["thresholds"],
            "graph_subgraph": g["subgraph"],          # {nodes, edges} 供前端高亮
            "graph_clauses": sum(1 for n in g["subgraph"]["nodes"] if n["label"] == "Clause"),
            "vector_clauses": len(vec_hits),
            "vector_error": self.vector_error,
            "merged": len(ranked),
            "hits": ranked[:top_k],                       # 合并结果（去重后）
            "graph_hits": graph_hits,                     # 图谱路单独结果
            "vector_hits": vector_hits,                   # 向量路单独结果
        }


def _merge_subgraph(dst: dict, src: dict) -> None:
    seen_n = {n["id"] for n in dst["nodes"]}
    for n in src.get("nodes", []):
        if n["id"] not in seen_n:
            dst["nodes"].append(n)
            seen_n.add(n["id"])
    seen_e = {(e["from"], e["type"], e["to"]) for e in dst["edges"]}
    for e in src.get("edges", []):
        k = (e["from"], e["type"], e["to"])
        if k not in seen_e:
            dst["edges"].append(e)
            seen_e.add(k)


def build_vector_retriever():
    """尝试接入 rag2 的向量检索（BM25 + dual_mix + RRF）。失败返回 None。"""
    try:
        rag2_root = WORKSPACE / "rag2"
        if str(rag2_root) not in sys.path:
            sys.path.insert(0, str(rag2_root))
        import torch  # noqa: F401,PLC0415  先导 torch，让 faiss 的 DLL 路径生效（WDDMS 坑）
        from src.common.paths import data_dir  # noqa: PLC0415
        from src.index.bm25 import BM25Index  # noqa: PLC0415
        from src.index.dense import FaissStore  # noqa: PLC0415
        from src.retrieval.hybrid import Corpus, EncoderPair, HybridRetriever  # noqa: PLC0415

        corpus = Corpus(data_dir("corpus", "clauses.jsonl"), data_dir("index", "units.jsonl"))
        bm = BM25Index.load(data_dir("index", "bm25.json"))
        store = FaissStore.load(data_dir("index", "dual_mix.faiss"))
        enc = EncoderPair(str(rag2_root / "data/models/dual_mix/doc_encoder"),
                          str(rag2_root / "data/models/dual_mix/query_encoder"))
        return HybridRetriever(corpus, bm, store, enc, fuse_mode="rrf", rrf_k=10, cand=20)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 向量检索不可用（{type(e).__name__}: {e}），仅图谱引擎", file=sys.stderr)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="深基坑开挖前要做哪些安全准备？")
    ap.add_argument("--mode", default="dual", choices=["graph", "dual"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--hops", type=int, default=2)
    a = ap.parse_args()

    vec = build_vector_retriever() if a.mode == "dual" else None
    retriever = GraphRAGRetriever(vector_retriever=vec)
    res = retriever.search(a.query, mode=a.mode, top_k=a.top_k, hops=a.hops)
    print(f"query: {a.query}  (mode={res['requested_mode']} → {res['effective_mode']})")
    print(f"识别危大类别: {res['hazards'] or '(未识别)'}")
    print(f"图谱白名单 {len(res['graph_whitelist'])} 本 / 图谱条款 {res['graph_clauses']} "
          f"/ 向量条款 {res['vector_clauses']} / 合并 {res['merged']}")
    if res["vector_error"]:
        print(f"[向量路异常] {res['vector_error']}")
    print("-" * 60)
    for i, h in enumerate(res["hits"], 1):
        tag = {"graph": "图谱", "vector": "向量", "both": "双源"}[h["source"]]
        print(f"{i}. [{tag}] {h['clause_id']}\n   {(h.get('text') or '')[:70]}")


if __name__ == "__main__":
    main()
