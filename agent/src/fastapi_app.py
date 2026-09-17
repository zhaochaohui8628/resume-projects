"""FastAPI 部署层（v7.2）——替代旧 http.server web_app。

链路：用户输入 → 调度中心（意图路由，仅 review/qa 两 subagent）→ subagent 并发
      → LLM 汇总输出。

特性：
  1) **任务队列**：后台 asyncio.Task 执行编排，`/api/tasks` 可查状态/取消，
     避免请求线程阻塞（旧版 threading + queue 轮询）；
  2) **异步并发 IO**：dispatcher.run_async 全程 asyncio（AsyncPipeline 事件驱动），
     非阻塞 SSE 推送；
  3) **图状态机余弦拦截**：ReAct/防御性控制流在 harness 内（react_agent + state_machine），
     FastAPI 层无需改动，trace 原样推送；
  4) SSE 流式：`/api/stream`（GET）逐步推送 step/done/error/ping 事件；
  5) 方案上传 / 记忆 flush / 健康检查接口保留。

运行（PowerShell，项目根目录）：
  & $PY agent/src/fastapi_app.py            # uvicorn 内嵌启动
  或
  & $PY -m uvicorn agent.src.fastapi_app:app --host 127.0.0.1 --port 7860
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import threading
import time
import uuid
from typing import Any, AsyncGenerator

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = HERE
if AGENT_SRC not in sys.path:
    sys.path.insert(0, AGENT_SRC)
WORKSPACE = os.path.dirname(AGENT_SRC)
ROOT = os.path.dirname(WORKSPACE)

from llm.deepseek import DeepSeekLLM                          # noqa: E402
from orchestrator.dispatcher import run_async as orch_run_async  # noqa: E402
from orchestrator.dispatcher import GlobalOpts                # noqa: E402
from harness.memory import MemoryManager                      # noqa: E402

from fastapi import FastAPI, HTTPException, UploadFile, File, Form   # noqa: E402
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

UI_DIR = os.path.join(HERE, "ui")
INDEX_HTML = os.path.join(UI_DIR, "index.html")
MEMORY_DIR = os.path.join(WORKSPACE, "data", "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

PORT = int(os.environ.get("AGENT_UI_PORT", "7860"))

app = FastAPI(title="施工方案合规审查 Agent", version="7.2")

# 单会话内存态（本地工具，单用户）
PLAN: dict[str, Any] = {"text": "", "name": "", "chars": 0}


# ---------------- 方案解析（复用原逻辑） ----------------
def _parse_plan_bytes(filename: str, raw: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".txt":
        return raw.decode("utf-8", errors="ignore")
    import tempfile
    suffix = ext if ext in (".docx", ".pdf") else ".txt"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        if ext == ".docx":
            from ingestion.docx_parser import parse_docx
            return parse_docx(tmp)
        if ext == ".pdf":
            from ingestion.pdf_parser import parse_pdf
            return parse_pdf(tmp)
        return raw.decode("utf-8", errors="ignore")
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


# ---------------- 任务队列 ----------------
class Task:
    __slots__ = ("id", "status", "created", "updated", "error", "result", "_cancel")

    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.status = "pending"      # pending / running / done / error / cancelled
        self.created = time.time()
        self.updated = time.time()
        self.error = ""
        self.result: dict | None = None
        self._cancel = False

    def cancel(self):
        self._cancel = True


_TASKS: dict[str, Task] = {}


# ---------------- SSE 事件格式 ----------------
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _run_task(task: Task, query: str, api_key: str,
                    rag_rerank: bool, ner_crf: bool) -> None:
    """后台任务：跑 dispatcher.run_async（全异步编排），逐步推送 trace。"""
    task.status = "running"
    task.updated = time.time()
    snap_q: "asyncio.Queue" = asyncio.Queue()

    def on_step(snapshot):
        # review 的 _report 在工作线程被调用（asyncio.to_thread）→ put_nowait 非线程安全，
        # 必须 call_soon_threadsafe 回主循环（与 on_token 一致）。
        # 兼容两种调用：dispatcher.emit 传 dict（老路径）；review._report 传 ("step", dict) 元组。
        if isinstance(snapshot, tuple):
            kind, payload = snapshot
        else:
            kind, payload = "step", snapshot
        try:
            loop.call_soon_threadsafe(snap_q.put_nowait, (kind, payload))
        except Exception:
            pass

    llm = _build_llm(api_key)
    opts = GlobalOpts.from_config(rag_rerank=rag_rerank, ner_crf=ner_crf,
                                  memory_dir=MEMORY_DIR)

    async def worker():
        try:
            task.result = await orch_run_async(query=query, plan=PLAN.get("text", ""),
                                               llm=llm, global_opts=opts, on_step=on_step)
        except Exception as e:  # noqa: BLE001
            task.error = f"{type(e).__name__}: {e}"
        finally:
            try:
                snap_q.put_nowait(None)
            except Exception:
                pass

    t = asyncio.ensure_future(worker())
    while True:
        if task._cancel:
            t.cancel()
            task.status = "cancelled"
            task.updated = time.time()
            return
        try:
            snap = await asyncio.wait_for(snap_q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            if t.done():
                break
            continue
        if snap is None:
            break
    await t
    if task.error:
        task.status = "error"
    else:
        task.status = "done"
    task.updated = time.time()


# ---------------- 请求模型 ----------------
class UploadBody(BaseModel):
    filename: str = "plan.txt"
    b64: str = ""


class TaskQuery(BaseModel):
    query: str = ""
    api_key: str = ""
    rag_rerank: bool = True     # v7.3.1：默认精排（可关闭）
    ner_crf: bool = True        # 兼容保留：ner2 生产模型恒为 CRF（s2_crf_param_v3），不提供切换


def _build_llm(api_key: str = ""):
    """LLM 构造：优先入参 key，其次环境变量 DEEPSEEK_API_KEY（服务端配置，前端免填）。"""
    key = (api_key or "").strip() or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return None
    return DeepSeekLLM(api_key=key)


# ---------------- 路由 ----------------
@app.get("/", response_class=HTMLResponse)
async def index():
    if not os.path.exists(INDEX_HTML):
        return HTMLResponse("<h3>index.html not found</h3>", status_code=404)
    with open(INDEX_HTML, "r", encoding="utf-8") as f:
        # no-cache：UI 迭代频繁，避免浏览器缓存旧页面（表现为"改了没生效"）
        return HTMLResponse(f.read(), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/api/health")
async def health():
    return {"ok": True, "plan_chars": PLAN.get("chars", 0),
            "plan_name": PLAN.get("name", ""), "tasks": len(_TASKS)}


@app.post("/api/upload")
async def upload(body: UploadBody):
    raw = base64.b64decode(body.b64) if body.b64 else b""
    if not raw:
        raise HTTPException(status_code=400, detail="空文件")
    text = _parse_plan_bytes(body.filename, raw)
    PLAN["text"] = text
    PLAN["name"] = body.filename
    PLAN["chars"] = len(text)
    return {"ok": True, "name": body.filename, "chars": len(text)}


@app.post("/api/plan/clear")
async def plan_clear():
    """清除服务端驻留的方案文本。

    ⚠️ PLAN 是全局状态：不清除会跨页面/跨任务污染——用户以为没上传方案，
    实际服务端还留着上一次的文本，LLM 路由与汇总会按它回答（实测踩坑：
    测试上传的基坑样例方案残留，导致无关提问输出基坑相关内容）。
    """
    PLAN["text"] = ""
    PLAN["name"] = ""
    PLAN["chars"] = 0
    return {"ok": True, "plan_chars": 0}


# ============================================================================
# 任务队列准入控制（2026-09-16 依据实测定档，读 config/config.yaml `concurrency`）
# 实测：并发 1/2/3 峰值内存 3630/3439/3615 MB，单任务 1.33s / 0.97s。
# 目的：防止无限并发提交拖垮内存（历史上出现 WinError 1455 页面文件不足）。
# ============================================================================
def _load_concurrency_cfg() -> tuple[int, int]:
    """返回 (task_concurrency, queue_max)；读不到配置时用实测保守值 (3, 8)。"""
    try:
        import yaml  # type: ignore
        # 用模块级 ROOT（项目根）。原先自己 dirname 两次，只到 agent/ 少一层，
        # 导致 agent/config/config.yaml 不存在 → 配置静默失效、永远回落到默认值。
        cfg_path = os.path.join(ROOT, "config", "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                c = ((yaml.safe_load(f) or {}).get("concurrency") or {})
            return (int(c.get("task_concurrency", 3)), int(c.get("queue_max", 8)))
    except Exception:
        pass
    return (3, 8)


TASK_CONCURRENCY, QUEUE_MAX = _load_concurrency_cfg()
# 信号量：同时执行的编排任务数（模型槽位闸门另行兜底重型模型并发）
_TASK_SLOT = threading.BoundedSemaphore(TASK_CONCURRENCY)
# 队列计数：正在执行 + 等待信号量的任务总数
_QUEUE_N = 0
_QUEUE_LOCK = threading.Lock()


@app.post("/api/tasks", response_model=dict)
async def create_task(body: TaskQuery) -> dict:
    """创建编排后台任务，返回 task_id。

    准入控制：队列满（queue_max）直接 429；否则入队，由信号量限制同时执行数
    （task_concurrency）。任务结束/取消时释放信号量与队列计数。
    """
    global _QUEUE_N
    with _QUEUE_LOCK:
        if _QUEUE_N >= QUEUE_MAX:
            raise HTTPException(
                status_code=429,
                detail=(f"系统繁忙：任务队列已满（{_QUEUE_N}/{QUEUE_MAX}），"
                        f"当前并发上限 {TASK_CONCURRENCY}。请稍后重试或先取消旧任务。"))
        _QUEUE_N += 1
    task = Task()
    _TASKS[task.id] = task

    async def _guarded():
        global _QUEUE_N
        try:
            # 信号量在工作线程获取（避免阻塞事件循环）
            await asyncio.to_thread(_TASK_SLOT.acquire)
            try:
                await _run_task(task, body.query, body.api_key,
                                body.rag_rerank, body.ner_crf)
            finally:
                _TASK_SLOT.release()
        finally:
            with _QUEUE_LOCK:
                _QUEUE_N -= 1

    asyncio.ensure_future(_guarded())
    return {"ok": True, "task_id": task.id, "status": task.status,
            "queue": {"waiting": _QUEUE_N, "queue_max": QUEUE_MAX,
                      "concurrency": TASK_CONCURRENCY}}


@app.get("/api/tasks/{task_id}")
async def task_status(task_id: str):
    task = _TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return {"task_id": task_id, "status": task.status,
            "error": task.error, "result": task.result}


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    task = _TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    task.cancel()
    return {"ok": True, "task_id": task_id, "status": task.status}


@app.get("/api/stream")
async def stream(query: str = "", api_key: str = "", rag_rerank: bool = True,
                 ner_crf: bool = True) -> StreamingResponse:
    """SSE 流式：逐步推送编排 trace（兼容旧 /api/stream 用法）。"""
    async def gen() -> AsyncGenerator[str, None]:
        yield _sse("ping", {"ts": time.time()})
        llm = _build_llm(api_key)
        if llm is None:
            yield _sse("error", {"message": "未配置 DeepSeek API Key（请在前端填写，或服务端设置环境变量 DEEPSEEK_API_KEY 后重启）。"})
            return
        opts = GlobalOpts.from_config(rag_rerank=rag_rerank, ner_crf=ner_crf,
                                  memory_dir=MEMORY_DIR)
        snap_q: "asyncio.Queue" = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_step(snapshot: dict):
            try:
                snap_q.put_nowait(("step", snapshot))
            except Exception:
                pass

        def on_token(piece: str):
            """LLM 汇总流式增量（在工作线程里被调用 → 必须 call_soon_threadsafe 回主循环）。"""
            try:
                loop.call_soon_threadsafe(snap_q.put_nowait, ("token", piece))
            except Exception:
                pass

        async def runner():
            try:
                return await orch_run_async(query=query, plan=PLAN.get("text", ""),
                                            llm=llm, global_opts=opts,
                                            on_step=on_step, on_token=on_token)
            except Exception as e:  # noqa: BLE001
                return {"error": f"{type(e).__name__}: {e}"}

        fut = asyncio.ensure_future(runner())
        last_heartbeat = time.time()
        while True:
            try:
                kind, snap = await asyncio.wait_for(snap_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if fut.done():
                    break
                if time.time() - last_heartbeat > 10:
                    last_heartbeat = time.time()
                    yield _sse("ping", {"ts": time.time()})
                continue
            if kind == "token":
                yield _sse("token", {"piece": snap})       # 首 token 即推 → 流式呈现
                continue
            # snap 统一为 dict：on_step/on_token 已打包成 (kind, payload) 元组，此处取 payload
            if isinstance(snap, tuple):
                snap = snap[1] if len(snap) > 1 else {}
            if snap is None:
                break
            sources = []
            for r in (snap.get("results") or []):
                for h in (getattr(r, "rag_hits", None) or []):
                    sources.append({"agent": getattr(r, "name", "?"), **h})
            yield _sse("step", {"trace": snap.get("trace", []),
                                "agents": snap.get("agents", []),
                                "sources": sources})
        res = await fut
        if res.get("error"):
            yield _sse("error", {"message": res["error"]})
            return
        all_sources = []
        for r in (res.get("results") or []):
            for h in (getattr(r, "rag_hits", None) or []):
                all_sources.append({"agent": getattr(r, "name", "?"), **h})
        all_sources.extend(res.get("rag_sources") or [])
        yield _sse("done", {"final_md": res.get("final_md"),
                            "detail_md": (res.get("detail_md") or "")[:6000],
                            "agents": res.get("agents", []),
                            "summary_line": res.get("summary_line", ""),
                            "trace": res.get("trace", []),      # 完整 trace（含 done 步骤）
                            "sources": all_sources})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache, no-store",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/flush")
async def flush():
    try:
        mm = MemoryManager(MEMORY_DIR)
        return {"ok": True, **mm.flush()}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- 启动 ----------------
def main():
    import uvicorn
    url = f"http://127.0.0.1:{PORT}/"
    print(f"* 施工方案合规审查 Agent · FastAPI：{url}")
    print("* 停止服务：Ctrl+C")
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
