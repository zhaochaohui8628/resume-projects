"""BM25 关键词检索（纯 Python，零依赖；惰性倒排）。

分数口径：`score = Σ_t idf(t) · tf·(k1+1) / (tf + k1·(1-b+b·dl/avgdl))`，
`idf(t) = ln(1 + (N-df+0.5)/(df+0.5))`（BM25+ 风格，恒正，避免高频词负分）。

温莎归一边界（winsor_lo/hi）不在这里写死，由 `scripts/calibrate_bounds.py`
用**真实 query 命中分分布**标定后写入索引文件——self-score 分布量级远高于真实命中分，
拿它当边界会把真实命中全部截断成 0（旧项目踩过这个坑）。
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from .tokenize import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, list[list[int]]] = {}
        self.avgdl: float = 0.0
        self.n: int = 0
        self.winsor_lo: float = 0.0
        self.winsor_hi: float = 1.0
        self._idf_cache: dict[str, float] = {}

    # ---- 构建 ----
    def build(self, doc_ids: list[str], texts: list[str]) -> "BM25Index":
        self.doc_ids = list(doc_ids)
        self.n = len(doc_ids)
        post = defaultdict(list)
        lens = []
        for i, txt in enumerate(texts):
            toks = tokenize(txt)
            lens.append(len(toks))
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                post[t].append([i, c])
        self.postings = dict(post)
        self.doc_len = lens
        self.avgdl = (sum(lens) / self.n) if self.n else 0.0
        self._idf_cache.clear()
        return self

    def idf(self, term: str) -> float:
        v = self._idf_cache.get(term)
        if v is None:
            df = len(self.postings.get(term, ()))
            v = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            self._idf_cache[term] = v
        return v

    # ---- 检索 ----
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        toks = tokenize(query)
        if not toks or not self.n:
            return []
        scores: dict[int, float] = defaultdict(float)
        for t in set(toks):
            post = self.postings.get(t)
            if not post:
                continue
            idf = self.idf(t)
            for i, tf in post:
                dl = self.doc_len[i] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        if not scores:
            return []
        best = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [(self.doc_ids[i], s) for i, s in best]

    # ---- 持久化 ----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "k1": self.k1, "b": self.b,
                "doc_ids": self.doc_ids, "doc_len": self.doc_len,
                "avgdl": self.avgdl, "n": self.n,
                "winsor_lo": self.winsor_lo, "winsor_hi": self.winsor_hi,
                "postings": self.postings,
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        idx = cls(k1=d.get("k1", 1.2), b=d.get("b", 0.75))
        idx.doc_ids = d["doc_ids"]
        idx.doc_len = d["doc_len"]
        idx.avgdl = d["avgdl"]
        idx.n = d["n"]
        idx.winsor_lo = d.get("winsor_lo", 0.0)
        idx.winsor_hi = d.get("winsor_hi", 1.0)
        idx.postings = {k: [list(p) for p in v] for k, v in d["postings"].items()}
        return idx
