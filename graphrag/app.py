"""GraphRAG 独立 demo 服务（FastAPI）。

演示链路（用户定稿）：
  前端可选「单路（图谱）」或「双路（向量 + 图谱，可开关向量端）」；
  图谱一律走 Neo4j（不回退 JSON）；Neo4j 不可用时接口明确报 503 并给出启动指引。

接口：
  GET  /                前端页面（ui/index.html）
  GET  /api/health      Neo4j 连通性 + 图库统计
  GET  /api/graph       全图 {nodes, edges}（供前端可视化）
  POST /api/load        把 demo_graph_v2.json 写入 Neo4j（演示前一键建图）
  POST /api/search      {query, mode:"graph"|"dual", hops, top_k}

运行（项目根目录，需先启 Neo4j）：
  $PY graphrag/app.py
  # 或 & $PY -m uvicorn graphrag.app:app --host 127.0.0.1 --port 7870
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

GRAPH_ROOT = Path(__file__).resolve().parent          # .../graphrag
WORKSPACE = GRAPH_ROOT.parent                          # 项目根
sys.path.insert(0, str(GRAPH_ROOT))

from fastapi import FastAPI, HTTPException                  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse    # noqa: E402
from pydantic import BaseModel                              # noqa: E402

from src.neo4j_store import Neo4jStore, Neo4jUnavailable     # noqa: E402
from src.hybrid_search import GraphRAGRetriever, build_vector_retriever  # noqa: E402

PORT = int(os.environ.get("GRAPHRAG_UI_PORT", "7870"))
UI_HTML = GRAPH_ROOT / "ui" / "index.html"

app = FastAPI(title="GraphRAG 规范知识图谱 Demo", version="1.0")

_lock = threading.Lock()
_store: Neo4jStore | None = None
_retriever: GraphRAGRetriever | None = None


def get_store() -> Neo4jStore:
    """进程级 Neo4j 单例；不可用抛 HTTPException(503)。"""
    global _store
    with _lock:
        if _store is None:
            try:
                _store = Neo4jStore()
            except Neo4jUnavailable as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
        return _store


def get_retriever(mode: str) -> GraphRAGRetriever:
    """单例检索器；dual 模式懒加载向量端（重，仅首次）。"""
    global _retriever
    with _lock:
        if _retriever is None:
            _retriever = GraphRAGRetriever(store=get_store(), vector_retriever=None)
        if mode == "dual" and _retriever.vector is None and not _retriever.vector_error:
            _retriever.vector = build_vector_retriever()
        return _retriever


class SearchReq(BaseModel):
    query: str
    mode: str = "dual"      # graph | dual
    hops: int = 2
    top_k: int = 5


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not UI_HTML.exists():
        raise HTTPException(500, f"前端文件缺失：{UI_HTML}")
    return HTMLResponse(UI_HTML.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> JSONResponse:
    try:
        store = get_store()
        return JSONResponse({"ok": True, "uri": store.uri, "stats": store.stats()})
    except HTTPException as e:
        return JSONResponse({"ok": False, "error": e.detail}, status_code=503)


@app.get("/api/graph")
def graph() -> JSONResponse:
    store = get_store()
    return JSONResponse(store.fetch_graph())


@app.post("/api/load")
def load() -> JSONResponse:
    store = get_store()
    try:
        result = store.load_graph(clear=True)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"写入失败：{type(e).__name__}: {e}") from e
    return JSONResponse({"ok": True, **result})


@app.post("/api/search")
def search(req: SearchReq) -> JSONResponse:
    mode = req.mode if req.mode in ("graph", "dual") else "graph"
    try:
        r = get_retriever(mode)
    except HTTPException:
        raise
    res: dict[str, Any] = r.search(req.query, mode=mode, top_k=req.top_k, hops=req.hops)
    return JSONResponse(res)


def main() -> None:
    import uvicorn  # noqa: PLC0415
    print(f"GraphRAG demo 前端： http://127.0.0.1:{PORT}/   （Neo4j Browser: http://localhost:7474）")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
