"""项目清理（2026-09-13）：一次性执行，跑完即可删除本文件。

用法（项目根目录）：
    python scripts/_cleanup_20260913.py --dry-run   # 只列清单
    python scripts/_cleanup_20260913.py --apply     # 真删

删除范围（用户已确认）：
  1) 缓存/日志/临时脚本/备份文件
  2) 旧路线残留（data/vector_db、rag2 dual_p1 与旧 faiss、scripts/eval_real.py）
     —— 但保留 neo4j-community-4.4.8/
  3) GRPO 中间权重 5 档（仅保留 r2_final 与全部 adapter）
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _size(p: str) -> int:
    if os.path.isfile(p):
        return os.path.getsize(p)
    s = 0
    for dp, _, fn in os.walk(p):
        for f in fn:
            try:
                s += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return s


def _count(p: str) -> int:
    if os.path.isfile(p):
        return 1
    return sum(len(fn) for _, _, fn in os.walk(p))


# ---------------- 1) 缓存 / 日志 / 临时 / 备份 ----------------
JUNK_DIRS = [
    ".pytest_cache",
    "rag2/.pytest_cache",
    "agent/src/harness/__pycache__",
    "agent/src/llm/__pycache__",
    "agent/src/orchestrator/__pycache__",
    "agent/src/qa/__pycache__",
    "agent/src/subagents/__pycache__",
    "agent/src/tools/__pycache__",
    "agent/src/report/__pycache__",
    "agent/src/concurrency/__pycache__",
    "agent/src/agent/__pycache__",
    "scripts/_env_fix_trash",
    "scripts/_stdlib_backup",
]

JUNK_FILES = [
    "agent/_ui.log",
    "agent/_ui.err.log",
    "neo4j-community-4.4.8/neo4j.stdout.log",
    "neo4j-community-4.4.8/logs/debug.log",
    "data/train/eval_natural.jsonl.bak_doclevel",
    "data/train/natural_queries.jsonl.bak_noannot",
    "data/vector_db/bm25.json.bak_preparent",
]

JUNK_GLOBS = [
    ("grpo/scripts", ".log"),
    ("scripts", "_"),          # 下划线前缀 = 一次性环境诊断脚本（保留本清理脚本）
]


# ---------------- 2) 旧路线残留 ----------------
OBSOLETE = [
    "data/vector_db",
    "scripts/eval_real.py",
    "rag2/data/models/dual_p1",
]

# 保留：dual_mix / dual_gold / tower_base（当前检索后端 + 兜底）
KEEP_FAISS = {"dual_mix", "dual_gold", "tower_base"}


def _obsolete_faiss() -> list[str]:
    idx = os.path.join(ROOT, "rag2", "data", "index")
    out = []
    if not os.path.isdir(idx):
        return out
    for f in sorted(os.listdir(idx)):
        stem = f.split(".")[0]
        # dual_distill*.faiss / *_bak_cls.faiss / dual_p1*.faiss + 其配套 .meta.json/.dim
        if f.endswith((".faiss", ".meta.json", ".dim")) and stem not in KEEP_FAISS:
            out.append(os.path.relpath(os.path.join(idx, f), ROOT))
    return out


# ---------------- 3) GRPO 中间权重 ----------------
GRPO_KEEP = {"r2_final"}
GRPO_KEEP_SUFFIX = ("adapter",)          # 所有 *_adapter 目录保留


def _grpo_drop() -> list[str]:
    p = os.path.join(ROOT, "data", "models", "qwen-grpo")
    if not os.path.isdir(p):
        return []
    out = []
    for name in sorted(os.listdir(p)):
        if name in GRPO_KEEP or name.endswith(GRPO_KEEP_SUFFIX):
            continue
        out.append(os.path.relpath(os.path.join(p, name), ROOT))
    return out


def collect() -> list[str]:
    items: list[str] = []
    for rel in JUNK_DIRS:
        if os.path.exists(os.path.join(ROOT, rel)):
            items.append(rel)
    for rel in JUNK_FILES:
        if os.path.exists(os.path.join(ROOT, rel)):
            items.append(rel)
    for sub, pat in JUNK_GLOBS:
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f == os.path.basename(__file__):
                continue
            if sub == "grpo/scripts" and f.endswith(pat):
                items.append(f"{sub}/{f}")
            elif sub == "scripts" and f.startswith(pat) and f.endswith(".py"):
                items.append(f"{sub}/{f}")
    items += OBSOLETE
    items += _obsolete_faiss()
    items += _grpo_drop()
    # 去重 + 剔除已被父目录覆盖的子项
    seen, final = set(), []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        final.append(it)
    final = [it for it in final
             if not any(it != o and it.startswith(o.rstrip("/") + "/") for o in final)]
    return sorted(final)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真删（默认只列清单）")
    args = ap.parse_args()

    targets = collect()
    total_bytes = total_files = 0
    print(f"{'[APPLY]' if args.apply else '[DRY-RUN]'} 待处理 {len(targets)} 项\n")
    for rel in targets:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        sz, n = _size(p), _count(p)
        total_bytes += sz
        total_files += n
        tag = "DIR " if os.path.isdir(p) else "FILE"
        print(f"  {tag} {rel:<52} {n:>6} f  {sz / 1048576:>9.1f} MB")

    print(f"\n合计 {total_files} 个文件 / {total_bytes / 1048576 / 1024:.2f} GB")
    if not args.apply:
        print("（未做任何修改；加 --apply 执行）")
        return 0

    removed = failed = 0
    for rel in targets:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            removed += 1
        except OSError as e:
            failed += 1
            print(f"  !! 删除失败 {rel}: {e}")
    print(f"\n完成：删除 {removed} 项，失败 {failed} 项。")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
