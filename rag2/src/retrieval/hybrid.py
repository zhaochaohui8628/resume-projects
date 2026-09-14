"""混合检索：BM25 ∥ 稠密向量 → 融合 → 父条款聚合。

## 融合两法（Phase 6 要 A/B 的就是这两条）

- `rrf`：`Σ 1/(k + rank)`，只看名次。
- `convex`：温莎截断 min-max 全局归一后加权：`α·normBM25 + (1-α)·normVec`。
  归一用**固定边界**（BM25 用标定出的 winsor_lo/hi；余弦天然落在 [-1,1]），
  因此不同 query 之间分数可比——这是 RRF 做不到、也是凸组合能保住 BM25 分差的原因。
  分母为 0 时退化为 0，不产生 NaN。

## 父条款聚合

同一 `clause_id` 的多个滑窗单元命中，只保留最高分代表，返回**整条条款正文**
（父块回扩），避免长条款只召回中间一段。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..common.limits import MAX_PARENT_LEN
from ..common.paths import data_dir
from ..corpus.recursive_split import window_around
from ..index.bm25 import BM25Index
from ..index.dense import DenseEncoder, FaissStore, load_units


def winsor_minmax(scores: list[float] | np.ndarray, lo: float, hi: float) -> np.ndarray:
    s = np.asarray(scores, dtype="float64")
    if hi <= lo:
        return np.zeros_like(s)
    return np.clip((np.clip(s, lo, hi) - lo) / (hi - lo), 0.0, 1.0)


class Corpus:
    """条款表 + 索引单元表（uid -> clause_id / 子块在父块内的字符区间）。"""

    def __init__(self, clauses_path: Path, units_path: Path):
        self.clauses: dict[str, dict] = {}
        for line in open(clauses_path, encoding="utf-8"):
            if line.strip():
                c = json.loads(line)
                self.clauses[c["id"]] = c
        self.units = load_units(units_path)
        self.uid2clause = {u["uid"]: u["clause_id"] for u in self.units}
        # 子块在父块内的位置：超长父块回扩兜底时按它取周边窗口
        self.uid2span = {u["uid"]: (int(u.get("start", 0)), int(u.get("end", len(u["text"]))))
                         for u in self.units}
        self.clause_ids = list(self.clauses.keys())

    def text(self, clause_id: str) -> str:
        return self.clauses.get(clause_id, {}).get("text", "")

    def meta(self, clause_id: str) -> dict:
        return self.clauses.get(clause_id, {}).get("metadata", {})

    def expand(self, clause_id: str, uid: str,
               limit: int = MAX_PARENT_LEN) -> tuple[str, dict]:
        """父块回扩（含超长兜底）。

        规则：父块 <= limit 直接给全文（语义完整）；超过 limit 就只给**命中子块周边**
        一段，窗口总长恰好等于 limit。不这么做的话，附录大表格那种近 10 万字的伪条款
        会把整个上下文预算吃光，而且真正命中的那 700 字反而被淹没。
        """
        full = self.text(clause_id)
        span = self.uid2span.get(uid, (0, len(full)))
        if len(full) <= limit:
            return full, {"expanded": "parent", "parent_len": len(full)}
        ws, we = window_around(span, len(full), limit)
        return full[ws:we], {"expanded": "window", "parent_len": len(full),
                             "window": [ws, we], "matched_span": list(span)}


class EncoderPair:
    """双塔：doc 塔建库、query 塔检索。未微调时同一模型。"""

    def __init__(self, doc_path: str, query_path: str | None = None,
                 max_seq_length: int = 512, device: str | None = None):
        self.doc = DenseEncoder(doc_path, max_seq_length=max_seq_length, device=device)
        self.query = (self.doc if not query_path or query_path == doc_path
                      else DenseEncoder(query_path, max_seq_length=max_seq_length,
                                        device=device))


class HybridRetriever:
    def __init__(self, corpus: Corpus, bm25: BM25Index, faiss: FaissStore,
                 encoder: EncoderPair, *, alpha: float = 0.9, fuse_mode: str = "convex",
                 rrf_k: int = 60, cand: int = 10):
        self.corpus = corpus
        self.bm25 = bm25
        self.faiss = faiss
        self.encoder = encoder
        self.alpha = alpha
        self.fuse_mode = fuse_mode
        self.rrf_k = rrf_k
        self.cand = cand

    # ---- 单路召回 ----
    def bm25_recall(self, query: str, n: int) -> list[tuple[str, float]]:
        return self.bm25.search(query, top_k=n)

    def dense_recall(self, query: str, n: int) -> list[tuple[str, float]]:
        qv = self.encoder.query.encode([query])
        idx, sc = self.faiss.search(qv, top_k=n)
        out = []
        for i, s in zip(idx[0], sc[0]):
            if i < 0:
                continue
            out.append((self.corpus.units[i]["uid"], float(s)))
        return out

    # ---- 融合 ----
    def _fuse(self, bm: list[tuple[str, float]], dn: list[tuple[str, float]],
              alpha: float | None = None,
              fuse_mode: str | None = None) -> dict[str, tuple[float, float]]:
        """返回 uid -> (融合分, 原始分兜底)。

        第二项是**必要的**：温莎截断会把一批高分候选压成并列 1.0，
        而并集来自 set（无序），只用融合分排序会让并列区顺序随机——
        实测会让 α=1.0（等价纯 BM25 排序）的 MRR 从 0.95 掉到 0.81。
        用"原始分之和"做次级排序键即可恢复被截断抹平的次序信息。
        """
        mode = fuse_mode or self.fuse_mode
        a = self.alpha if alpha is None else alpha
        if mode == "rrf":
            fused: dict[str, tuple[float, float]] = {}
            for rank, (uid, s) in enumerate(bm, 1):
                f, t = fused.get(uid, (0.0, 0.0))
                fused[uid] = (f + 1.0 / (self.rrf_k + rank), t + s)
            for rank, (uid, s) in enumerate(dn, 1):
                f, t = fused.get(uid, (0.0, 0.0))
                fused[uid] = (f + 1.0 / (self.rrf_k + rank), t + s)
            return fused
        nb = winsor_minmax([s for _u, s in bm], self.bm25.winsor_lo, self.bm25.winsor_hi)
        nd = winsor_minmax([s for _u, s in dn], -1.0, 1.0)
        fused = {}
        for (uid, s), v in zip(bm, nb):
            f, t = fused.get(uid, (0.0, 0.0))
            fused[uid] = (f + a * float(v), t + float(s))
        for (uid, s), v in zip(dn, nd):
            f, t = fused.get(uid, (0.0, 0.0))
            fused[uid] = (f + (1 - a) * float(v), t + float(s))
        return fused

    # ---- 检索 ----
    def search(self, query: str, top_k: int = 5, *, cand: int | None = None,
               alpha: float | None = None, fuse_mode: str | None = None,
               agg: bool = True) -> list[dict]:
        n = cand or self.cand
        bm = self.bm25_recall(query, n)
        dn = self.dense_recall(query, n)
        fused = self._fuse(bm, dn, alpha, fuse_mode)
        if not agg:
            hits = sorted(fused.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))[:top_k]
            return [{"uid": u, "clause_id": self.corpus.uid2clause[u], "score": s[0]}
                    for u, s in hits]
        # 父块聚合去重：同一 clause_id 的多个子块只留最高分代表。
        # 排序键用 (融合分, 原始分之和) 的**完整元组**——只用融合分会让温莎截断后的
        # 并列区按 set 的无序顺序取代表，同一父块每次返回的子块都不同（不可复现）。
        best: dict[str, tuple[str, tuple[float, float]]] = {}
        collapsed: dict[str, int] = {}
        for uid, s in fused.items():
            cid = self.corpus.uid2clause[uid]
            cur = best.get(cid)
            if cur is None:
                best[cid] = (uid, s)
            else:
                collapsed[cid] = collapsed.get(cid, 1) + 1
                if s > cur[1]:
                    best[cid] = (uid, s)
        ranked = sorted(best.items(), key=lambda kv: (-kv[1][1][0], -kv[1][1][1]))[:top_k]
        out = []
        for cid, (uid, s) in ranked:
            text, exp = self.corpus.expand(cid, uid)
            out.append({
                "clause_id": cid,
                "uid": uid,
                "score": s[0],
                "tiebreak": s[1],
                "text": text,
                "metadata": self.corpus.meta(cid),
                # 可观测性：这一条聚合掉了几个子块、是整条回扩还是窗口兜底
                "expansion": {**exp, "collapsed_children": collapsed.get(cid, 1)},
            })
        return out


def load_retriever(*, tower_doc: str | None = None, tower_query: str | None = None,
                   index_file: str = "tower_base.faiss", **kw) -> HybridRetriever:
    from ..common.paths import BASE_TOWER_MODEL

    corpus = Corpus(data_dir("corpus", "clauses.jsonl"), data_dir("index", "units.jsonl"))
    bm = BM25Index.load(data_dir("index", "bm25.json"))
    store = FaissStore.load(data_dir("index", index_file))
    enc = EncoderPair(tower_doc or BASE_TOWER_MODEL, tower_query)
    return HybridRetriever(corpus, bm, store, enc, **kw)
