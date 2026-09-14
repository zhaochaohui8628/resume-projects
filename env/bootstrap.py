"""项目自举脚本：环境快照 / 迁移校验 / 依赖安装。

设计目标——**整个项目文件夹拷到新机器后，一条命令就能确认"环境齐不齐、数据坏没坏"**，
不需要人工逐项对照文档，也不依赖任何写死的解释器路径。

用法（在项目根目录，即同时含 rag2/ 与 data/ 的那一层执行）：

    python env/bootstrap.py snapshot          # 旧机器：冻结环境与产物指纹 -> env/environment.json
    python env/bootstrap.py check             # 新机器：校验环境/依赖/数据/模型 + 跑冒烟检索
    python env/bootstrap.py check --install   # 缺依赖时自动 pip install
    python env/bootstrap.py install           # 只装依赖

退出码：0 = 全通过；1 = 有致命问题（缺关键数据/依赖）；2 = 可用但有告警。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RAG2 = PROJECT_ROOT / "rag2"
ENV_JSON = HERE / "environment.json"
FROZEN = HERE / "requirements-frozen.txt"
REQ_TXT = RAG2 / "requirements.txt"

# 关键产物指纹：小文件算 sha256（防拷贝截断/改坏），大文件只记字节数（省时间）
HASH_FILES = [
    "rag2/data/corpus/clauses.jsonl",
    "rag2/data/phase1/train.jsonl",
    "rag2/data/phase1/val.jsonl",
    "rag2/data/index/bm25_calib.json",
    "rag2/data/corpus/corpus_stats.json",
]
SIZE_FILES = [
    "rag2/data/index/bm25.json",
    "rag2/data/index/units.jsonl",
    "rag2/data/index/tower_base.faiss",       # 基座塔（未微调 bge）兜底索引
    "rag2/data/index/dual_mix.faiss",         # ★ 现行检索后端（用户裁定）
    "rag2/data/index/dual_gold.faiss",        # 备选后端（纯黄金集重训）
    "rag2/data/models/dual_mix/doc_encoder/model.safetensors",
    "rag2/data/models/dual_mix/query_encoder/model.safetensors",
    "rag2/data/models/cross_v2_ep4/model.safetensors",   # CrossEncoder 精排产物
    "data/models/bge-small-zh-v1.5/model.safetensors",
    "data/models/qwen-grpo/r2_final/config.json",        # GRPO 最终权重（唯一保留档）
]
REQUIRED = [
    "rag2/src/common/paths.py",
    "rag2/scripts/build_corpus.py",
    "rag2/scripts/eval_retrieval.py",
    "rag2/data/corpus/clauses.jsonl",
    "rag2/data/index/bm25.json",
    "rag2/data/index/units.jsonl",
    "rag2/data/index/tower_base.faiss",
    "rag2/data/phase1/train.jsonl",
    "rag2/data/phase1/val.jsonl",
    "data/models/bge-small-zh-v1.5/config.json",
]
# 检索 / 评估 / 训练链路必需——缺任一项整条链路都跑不起来，算致命。
CORE_PKGS = ["numpy", "torch", "transformers", "sentence_transformers", "faiss"]
# 只在「从 PDF 重建语料」时才需要——语料已建好时缺它不影响检索与训练，算告警。
OPTIONAL_PKGS = ["fitz"]
KEY_PKGS = CORE_PKGS + OPTIONAL_PKGS
SMOKE_CMD = [
    "{py}", "scripts/eval_retrieval.py", "--tag", "smoke", "--mode", "bm25",
]
SMOKE_CMD_DUAL = [
    # 用**线上配置**冒烟：dual_mix 塔 + RRF 融合（与 agent/src/tools/rag_client.py 默认一致）
    "{py}", "scripts/eval_retrieval.py", "--tag", "smoke_dual", "--mode", "rrf",
    "--index", "dual_mix.faiss",
    "--doc-tower", "data/models/dual_mix/doc_encoder",
    "--query-tower", "data/models/dual_mix/query_encoder",
]
# ⚠ 语义为「下界阈值」：≥ 即通过，低于才告警。
#   理由——这套数是「模型/环境有没有搬坏」的探测器，不是精度回归断言；
#   语料/塔版本微调会让精确值小幅漂移，用下界可避免天天误报，
#   同时"搬坏了"（大幅下降）依然会被抓住。改动语料或索引后请重跑 snapshot 并复核本下界。
EXPECT_SMOKE = {"hit@k": 0.80, "mrr": 0.75}
# 2026-09-13 实测（phase1/val, n=43，单线程 BLAS）：dual_mix+RRF 0.930/0.814、BM25 0.860/0.806、
# dual_mix+convex 0.884/0.816、dual_gold+convex 0.791/0.724。
# ⚠ 本集是 P1 冷启动分布，不是简历口径——真口径看黄金集（RESUME_NUMBERS.md §1）。
#   dual_gold 在冷启动集上天然偏低（它按黄金集口径重训），低不代表塔坏了，故此处只设"未退化"下界。
EXPECT_SMOKE_DUAL = {"hit@k": 0.90, "mrr": 0.78}


# ---------------------------------------------------------------- helpers
def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def fingerprint() -> tuple[dict, dict]:
    hashes, sizes = {}, {}
    for r in HASH_FILES:
        p = PROJECT_ROOT / r
        if p.exists():
            hashes[r] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
        else:
            hashes[r] = {"missing": True}
    for r in SIZE_FILES:
        p = PROJECT_ROOT / r
        sizes[r] = p.stat().st_size if p.exists() else None
    return hashes, sizes


def pkg_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in KEY_PKGS:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception as e:  # noqa: BLE001
            out[name] = f"MISSING({type(e).__name__})"
    return out


def hardware_gpu() -> str | None:
    """探测物理显卡型号（不依赖 torch / PATH 上的 nvidia-smi）。

    注意：nvml 在部分环境下初始化失败（如容器/受限会话），所以只把结果当"提示"用，
    真正判断能否训练一律以 torch.cuda.is_available() 为准。
    """
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController).Name -join ' | '"],
            capture_output=True, text=True, timeout=20)
        name = (out.stdout or "").strip()
        return name or None
    except Exception:  # noqa: BLE001
        return None


def gpu_info() -> dict:
    info = {"cuda_available": None, "threads": None}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_build"] = getattr(torch.version, "cuda", None)
        info["threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        # 注意：这里必须是 None 而不是字符串。曾经的 "unknown(torch missing)" 是真值，
        # 会让第 3 节把"没装 torch"误报成"本机有 GPU"。
        info["cuda_available"] = None
        info["error"] = "torch missing"
    return info


def model_dirs() -> dict:
    cands = {
        "base_tower(bge-small-zh-v1.5)": PROJECT_ROOT / "data" / "models" / "bge-small-zh-v1.5",
        "dual_mix/doc_encoder": RAG2 / "data" / "models" / "dual_mix" / "doc_encoder",
        "dual_mix/query_encoder": RAG2 / "data" / "models" / "dual_mix" / "query_encoder",
        "dual_gold/doc_encoder": RAG2 / "data" / "models" / "dual_gold" / "doc_encoder",
        "cross_v2_ep4(CrossEncoder)": RAG2 / "data" / "models" / "cross_v2_ep4",
    }
    out = {}
    for k, p in cands.items():
        files = sorted(f.name for f in p.glob("*")) if p.is_dir() else []
        out[k] = {"path": rel(p), "exists": p.is_dir(), "n_files": len(files),
                  "has_weights": any(f.endswith((".safetensors", ".bin", ".pt")) for f in files)}
    return out


def corpus_stats() -> dict:
    cl = RAG2 / "data" / "corpus" / "clauses.jsonl"
    st: dict = {"clauses": None, "units": None, "sources": None}
    if cl.exists():
        n, srcs = 0, set()
        with open(cl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
                    try:
                        srcs.add(json.loads(line)["metadata"]["source"])
                    except Exception:  # noqa: BLE001
                        pass
        st["clauses"], st["sources"] = n, len(srcs)
    u = RAG2 / "data" / "index" / "units.jsonl"
    if u.exists():
        st["units"] = sum(1 for _ in open(u, encoding="utf-8"))
    return st


def py_exe() -> str:
    """优先用当前解释器；若项目内有 .venv 就用它。"""
    for c in (RAG2.parent / ".venv" / "Scripts" / "python.exe",
              RAG2.parent / ".venv" / "bin" / "python"):
        if c.exists():
            return str(c)
    return sys.executable


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# ---------------------------------------------------------------- commands
def cmd_snapshot(_a) -> int:
    hashes, sizes = fingerprint()
    stats = corpus_stats()
    data = {
        "schema": 1,
        "captured_on": {"platform": platform.platform(), "machine": platform.node(),
                        "python": sys.version.replace("\n", " ")},
        "project_root_relative_contract": "rag2/ 与 data/ 必须是同级兄弟目录",
        "python": {"executable": sys.executable, "version": platform.python_version(),
                   "implementation": platform.python_implementation()},
        "packages": pkg_versions(),
        "gpu": gpu_info(),
        "models": model_dirs(),
        "corpus_stats": stats,
        "artifact_sha256": hashes,
        "artifact_bytes": sizes,
        "env_files": {"requirements": "rag2/requirements.txt",
                      "frozen": "env/requirements-frozen.txt"},
    }
    ENV_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[snapshot] 写入 {rel(ENV_JSON)}")
    print(f"  python {data['python']['version']} | cuda={data['gpu'].get('cuda_available')}")

    pkgs = data["packages"]
    print("  关键依赖: " + ", ".join(f"{k}={v}" for k, v in pkgs.items()))
    rc, out = run([sys.executable, "-m", "pip", "freeze"], PROJECT_ROOT)
    if rc == 0:
        FROZEN.write_text(out, encoding="utf-8")
        print(f"[snapshot] 写入 {rel(FROZEN)}（{len(out.splitlines())} 个包）")
    else:
        print("  ⚠ pip freeze 失败，跳过" + out[:200])
    missing = [r for r in REQUIRED if not (PROJECT_ROOT / r).exists()]
    if missing:
        print("  ⚠ 关键文件缺失（拷贝时务必补上）:")
        for m in missing:
            print("     -", m)
        return 1
    print("  ✓ 关键文件齐全")
    return 0


def cmd_check(a) -> int:
    if not ENV_JSON.exists():
        print("✗ 找不到 env/environment.json（快照缺失）→ 先拷全，或在旧机跑 snapshot")
        return 1
    snap = json.loads(ENV_JSON.read_text(encoding="utf-8"))
    fatal, warn = [], []

    print("=" * 68)
    print("1) 目录契约")
    print("=" * 68)
    for r in REQUIRED:
        p = PROJECT_ROOT / r
        ok = p.exists()
        print(("  ✓ " if ok else "  ✗ ") + r)
        if not ok:
            fatal.append(f"缺文件 {r}")
    if not (PROJECT_ROOT / "data").is_dir():
        fatal.append("找不到 data/ 目录（rag2/ 必须与 data/ 同级）")

    print("=" * 68)
    print("2) Python 与依赖")
    print("=" * 68)
    print(f"  当前解释器: {sys.executable}")
    print(f"  版本: {platform.python_version()}（快照时 {snap['python']['version']}）")
    if platform.python_version_tuple()[:2] != tuple(snap["python"]["version"].split(".")[:2]):
        warn.append("Python 版本与快照不一致（不致命，但可复现性下降）")
    pkgs = pkg_versions()
    missing_pkgs, missing_opt = [], []
    for k, v in pkgs.items():
        miss = str(v).startswith("MISSING")
        mark = "✗" if miss else "✓"
        print(f"  {mark} {k}: {v}    (快照: {snap['packages'].get(k, '?')})")
        if miss:
            (missing_opt if k in OPTIONAL_PKGS else missing_pkgs).append(k)
    if missing_opt:
        warn.append(f"缺可选依赖: {missing_opt}——只影响「从 PDF 重建语料」，"
                    f"检索/评估/训练不受影响")
    if missing_pkgs:
        fatal.append(f"缺依赖: {missing_pkgs}")
        if a.install:
            print("  → 执行 pip install -r rag2/requirements.txt")
            rc, out = run([sys.executable, "-m", "pip", "install", "-r", str(REQ_TXT)], PROJECT_ROOT)
            print("    " + ("成功" if rc == 0 else "失败") + f"（exit {rc}）")
            if rc == 0:
                fatal = [f for f in fatal if not f.startswith("缺依赖")]
                print("  ✓ 依赖已装，请重跑 check")

    print("=" * 68)
    print("3) GPU")
    print("=" * 68)
    g = gpu_info()
    print(f"  torch={g.get('torch')} cuda_available={g.get('cuda_available')} "
          f"cuda_build={g.get('cuda_build')} device={g.get('device_name', '-')}")
    hw = hardware_gpu()
    if hw:
        print(f"  物理显卡: {hw}")
    if g.get("cuda_available") is None:
        warn.append("无法判断 GPU：torch 未安装（先装依赖再重跑本项）")
    elif g["cuda_available"] is False:
        if hw and "nvidia" in hw.lower():
            warn.append("本机有 NVIDIA 显卡但 torch 是 CPU 版：P3/P4 想在本机训，"
                        "需换装 CUDA 版 torch（见文件末尾提示）")
        else:
            warn.append("无 CUDA：P3 精排训练会很慢（约 1 样本/秒），请在 GPU 机器上跑")
    if snap["gpu"].get("cuda_available") is not True and g.get("cuda_available") is True:
        print(f"  ↑ 本机 GPU 可用（快照来自 CPU 机），P3/P4 可以在这里跑：{g.get('device_name')}")

    print("=" * 68)
    print("4) 模型目录")
    print("=" * 68)
    md = model_dirs()
    for k, v in md.items():
        mark = "✓" if (v["exists"] and v["has_weights"]) else ("~" if v["exists"] else "✗")
        print(f"  {mark} {k}  {v['path']}  files={v['n_files']} weights={v['has_weights']}")
        if not v["exists"]:
            if k.startswith("base_tower") or k.startswith("dual_mix"):
                fatal.append(f"缺模型 {k}")
            else:
                warn.append(f"缺 {k}（检索仍可用；补齐后精排/备选塔才生效）")

    print("=" * 68)
    print("5) 数据指纹（防拷贝截断）")
    print("=" * 68)
    hashes, sizes = fingerprint()
    for r, v in hashes.items():
        exp = snap["artifact_sha256"].get(r, {})
        if v.get("missing"):
            print(f"  ✗ {r} 不存在")
            continue
        same = (v.get("sha256") == exp.get("sha256"))
        print(f"  {'✓' if same else '!'} {r}  {v['bytes']}B"
              + ("" if same else f"  (快照 {exp.get('bytes')}B)"))
        if not same and exp.get("sha256"):
            warn.append(f"{r} 内容与快照不一致（可能被改过或拷贝不完整）")
    for r, b in sizes.items():
        exp = snap["artifact_bytes"].get(r)
        if b is None:
            print(f"  ~ {r} 不存在（可重建）")
        else:
            note = "" if exp in (None, b) else f"  ⚠ 快照 {exp}B"
            print(f"  ✓ {r}  {b:,}B{note}")

    print("=" * 68)
    print("6) 语料统计")
    print("=" * 68)
    st = corpus_stats()
    exp = snap["corpus_stats"]
    print(f"  条款 {st['clauses']}（快照 {exp['clauses']}）/ "
          f"规范 {st['sources']}（{exp['sources']}）/ 索引单元 {st['units']}（{exp['units']}）")
    for k in ("clauses", "units", "sources"):
        if st[k] is not None and exp[k] is not None and st[k] != exp[k]:
            warn.append(f"语料统计不一致：{k} {st[k]} vs 快照 {exp[k]}")

    print("=" * 68)
    print("7) 冒烟检索（最能证明模型与环境都没搬坏）")
    print("=" * 68)
    if not missing_pkgs and st["units"]:
        py = py_exe()
        for label, tmpl, exp_m, mode in (
            ("BM25-only", SMOKE_CMD, EXPECT_SMOKE, "bm25"),
            ("双塔 dual_mix + RRF（线上配置）", SMOKE_CMD_DUAL, EXPECT_SMOKE_DUAL, "rrf"),
        ):
            cmd = [c.format(py=py) for c in tmpl]
            rc, out = run(cmd, RAG2)
            line = next((l for l in out.splitlines() if l.strip().startswith("{")), "")
            if not line:
                hint = ""
                if any(k in out for k in ("Memory allocation", "bad_alloc",
                                          "页面文件太小", "1455", "OpenBLAS",
                                          "Unable to allocate", "MemoryError")):
                    hint = ("（**内存/提交内存不足**导致的偶发失败，不是数据损坏——"
                            "释放内存或调大页面文件后重跑即可；本机实测该路径可跑出指标）")
                print(f"  ✗ {label}: 未拿到指标{hint}\n     {out.strip()[:300]}")
                warn.append(f"冒烟检索 {label} 未产出指标{hint}")
                continue
            m = json.loads(line)
            ok = (m["hit@k"] >= exp_m["hit@k"] - 5e-3
                  and m["mrr"] >= exp_m["mrr"] - 5e-3)
            print(f"  {'✓' if ok else '!'} {label}: hit@5={m['hit@k']:.3f} mrr={m['mrr']:.3f} "
                  f"（下界 {exp_m['hit@k']:.3f}/{exp_m['mrr']:.3f}）")
            if not ok:
                warn.append(f"冒烟检索 {label} 低于下界阈值（模式 {mode}）——"
                            f"多半是模型/索引搬坏或版本不符")
    else:
        print("  - 跳过（依赖或索引缺失）")

    print("=" * 68)
    if fatal:
        print(f"结论：✗ 不可用（{len(fatal)} 个致命问题）")
        for f in fatal:
            print("  ✗", f)
    elif warn:
        print(f"结论：~ 可用，但有 {len(warn)} 条告警")
        for w in warn:
            print("  !", w)
    else:
        print("结论：✓ 全部通过，环境与快照一致，可直接开工")
    print("=" * 68)
    print("下一步：读 rag2/docs/PROGRESS.md 第 5 节「下一会话第一步」继续。")
    return 1 if fatal else (2 if warn else 0)


def cmd_install(_a) -> int:
    print(f"pip install -r {rel(REQ_TXT)}")
    rc, out = run([sys.executable, "-m", "pip", "install", "-r", str(REQ_TXT)], PROJECT_ROOT)
    print(out[-2000:])
    if rc == 0:
        print("\n提示：torch 若要 GPU 版，请按目标机 CUDA 版本单独安装：")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu124")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="施工方案合规审查 · 环境自举/校验（依赖/数据/模型）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("snapshot", help="旧机器：生成 env/environment.json 与 frozen 依赖")
    p1.set_defaults(func=cmd_snapshot)
    p2 = sub.add_parser("check", help="新机器：校验环境/数据/模型并跑冒烟检索")
    p2.add_argument("--install", action="store_true", help="缺依赖时自动安装")
    p2.set_defaults(func=cmd_check)
    p3 = sub.add_parser("install", help="安装 rag2 依赖")
    p3.set_defaults(func=cmd_install)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
