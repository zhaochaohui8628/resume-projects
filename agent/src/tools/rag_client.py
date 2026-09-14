"""RAG 客户端 v6：优先走 **rag2**（双塔 + RRF/凸组合，v5 检索栈），旧 rag 已移除。

## v7.3.1 变更（2026-09-13）
旧 rag 项目（`rag/`、`data/vector_db`）已清理回收。本客户端**只对接 rag2**：
  - 后端 `auto`/`rag2` 均走 rag2（dual_mix → dual_gold → tower_base 优先序）；
  - 移除旧 rag 兜底路径（`retriever`/`data/vector_db/bm25.json`）；
  - **精排真正接入**（此前 `use_rerank` 是空壳：只透传参数，HybridRetriever 无精排）：
    召回 top_k×oversample → `CrossEncoderReranker`（cross_v2_ep4）打分 → 截断 top_k；
    **默认开启**（config `retrieval.rerank.enabled=true`），可在 UI 关闭；
    精排加载失败只降级为"不精排"并记录 `errors["rerank"]`，不影响主检索。

溯源字段（rank/id/source/clause_no/source_path/text）由本客户端统一补齐，
下游（review/comparator/UI）不感知后端差异。
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.dirname(HERE)
AGENT = os.path.dirname(AGENT_SRC)
WORKSPACE = os.path.dirname(AGENT)
RAG2_ROOT = os.path.join(WORKSPACE, "rag2")
if AGENT_SRC not in sys.path:
    sys.path.insert(0, AGENT_SRC)

from concurrency import get_model_slots  # noqa: E402  （重型模型并发槽位限流）


def _mem_hint(err: str) -> str:
    """内存类错误 → 可操作提示（Windows 提交限额 = 物理 + 页面文件；16GB 机器常撞顶）。"""
    m = (err or "").lower()
    if any(k in m for k in ("memoryerror", "not enough memory", "1455", "页面文件", "commitment")):
        return "（提交内存不足：关掉占内存程序（VS Code/多开窗口）或调大页面文件到 16–32GB）"
    return ""


def _load_config():
    try:
        import yaml
    except ImportError:
        return {}
    cfg_path = os.path.join(WORKSPACE, "config", "config.yaml")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class RagClient:
    def __init__(self, use_rerank: bool = None, backend: str = "auto"):
        self._cfg = _load_config()
        self._rr = (self._cfg.get("retrieval", {}) or {}).get("rerank", {}) or {}
        self._r2 = (self._cfg.get("retrieval", {}) or {}).get("rag2", {}) or {}
        # use_rerank 优先级：显式入参 > config > True（默认精排，评估显著优于纯融合）
        if use_rerank is None:
            use_rerank = bool(self._rr.get("enabled", True))
        self.use_rerank = use_rerank
        self.top_k = int((self._cfg.get("retrieval", {}) or {}).get("top_k", 5))
        # 精排前召回倍数：召回 top_k*oversample → 精排 → 截断回 top_k
        self.oversample = int(self._rr.get("oversample", 4) or 4)

        self.backend = None
        self._rag2 = None
        self._reranker = None
        self.errors: dict[str, str] = {}
        if backend in ("auto", "rag2"):
            self._build_rag2()
            if self.use_rerank:
                self._build_reranker()

    # ---------------- 精排（CrossEncoder，默认开） ----------------
    def _build_reranker(self):
        # 重型模型加载走槽位闸门（并发满载时排队，避免瞬时提交内存超限 → WinError 1455）
        with get_model_slots().slot(cost=1.0, name="ce.load"):
            try:
                if RAG2_ROOT not in sys.path:
                    sys.path.insert(0, RAG2_ROOT)
                from src.retrieval.rerank import CrossEncoderReranker, resolve_ce_dir
                model_dir = resolve_ce_dir(self._rr.get("model"), WORKSPACE)
                self._reranker = CrossEncoderReranker(
                    model_dir=model_dir,
                    batch_size=int(self._rr.get("batch_size", 16)),
                    max_len=int(self._rr.get("max_len", 256)))
            except Exception as e:                  # 缺 torch/transformers/产物 → 降级不精排
                self._reranker = None
                self.use_rerank = False
                msg = f"{type(e).__name__}: {e}"[:160]
                self.errors["rerank"] = msg + _mem_hint(msg)

    # ---------------- rag2（唯一后端） ----------------
    def _build_rag2(self):
        try:
            idx_dir = os.path.join(RAG2_ROOT, "data", "index")
            mdir = os.path.join(RAG2_ROOT, "data", "models")
            if not os.path.isdir(idx_dir):
                return
            tower, index_file = None, "tower_base.faiss"
            for tag in ("dual_mix", "dual_gold"):          # dual_mix = 用户最终裁定
                if (os.path.exists(os.path.join(idx_dir, tag + ".faiss"))
                        and os.path.isdir(os.path.join(mdir, tag, "doc_encoder"))):
                    tower, index_file = tag, tag + ".faiss"
                    break
            if not os.path.exists(os.path.join(idx_dir, index_file)):
                return
            if RAG2_ROOT not in sys.path:
                sys.path.insert(0, RAG2_ROOT)
            from src.retrieval.hybrid import load_retriever
            kw = {"index_file": index_file,
                  "fuse_mode": self._r2.get("fuse_mode", "rrf"),   # 用户裁定：RRF
                  "alpha": float(self._r2.get("alpha", 0.3))}
            if tower:
                kw.update(tower_doc=os.path.join(mdir, tower, "doc_encoder"),
                          tower_query=os.path.join(mdir, tower, "query_encoder"))
            self._rag2 = load_retriever(**kw)
            self.backend = f"rag2:{tower or 'base'}:{kw['fuse_mode']}"
        except Exception as e:                        # 缺 torch/faiss/模型/内存不足 → 不可用
            self._rag2 = None
            msg = f"{type(e).__name__}: {e}"[:160]
            self.errors["rag2"] = msg + _mem_hint(msg)

    # ---------------- 检索 ----------------
    def search(self, query: str, top_k: int = None, use_rerank: bool = None):
        """返回 [{id, text, score, rank, metadata(source, clause_no, source_path...)}]。"""
        k = top_k or self.top_k
        want_rr = self.use_rerank if use_rerank is None else use_rerank
        do_rr = bool(want_rr and self._reranker)
        if self._rag2 is not None:
            # 精排要多召回 oversample 倍候选，精排后再截断回 top_k
            fetch = max(k * self.oversample, k) if do_rr else k
            try:
                hits = self._rag2.search(query, top_k=fetch)
            except Exception:
                return []
            out = []
            for i, h in enumerate(hits, 1):
                md = dict(h.get("metadata", {}) or {})
                md.setdefault("source", "")
                md.setdefault("clause_no", "")
                md.setdefault("source_path", "")
                out.append({"id": h.get("clause_id", h.get("uid", "")),
                            "text": h.get("text", ""),
                            "score": float(h.get("score", 0.0)),
                            "rank": i, "metadata": md})
            if do_rr:
                try:
                    # 精排推理同样走槽位闸门
                    with get_model_slots().slot(cost=1.0, name="ce.infer"):
                        out = self._reranker.rerank(query, out)[:k]
                    for i, h in enumerate(out, 1):
                        h["rank"] = i
                except Exception as e:              # 精排推理失败 → 退回粗排前 k 条
                    self.errors["rerank"] = f"{type(e).__name__}: {e}"[:160]
                    out = out[:k]
            return out
        return []

    def available(self) -> bool:
        return self._rag2 is not None

    # ---- v5：UI 溯源专用（薄封装，字段已由 rag 检索层补齐） ----
    def search_with_meta(self, query: str, top_k: int = None, use_rerank: bool = None) -> list[dict]:
        """带溯源字段的检索：直接透传 rag 检索层已补好的 rank/source/source_path。"""
        hits = self.search(query, top_k=top_k, use_rerank=use_rerank)
        out = []
        for h in hits:
            md = h.get("metadata", {}) or {}
            out.append({
                "rank": h.get("rank"),
                "id": h.get("id"),
                "score": float(h.get("score", 0.0)),
                "rerank_score": h.get("rerank_score"),
                "source": md.get("source", "?"),
                "clause_no": md.get("clause_no", ""),
                "source_path": md.get("source_path", ""),
                "text": h.get("text", ""),
                "metadata": md,
            })
        return out