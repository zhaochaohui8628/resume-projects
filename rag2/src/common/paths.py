"""路径与配置解析：所有路径都相对项目根（含 data/raw 语料的那个目录）。"""
from __future__ import annotations

import os
from pathlib import Path

# rag2/src/common/paths.py -> rag2/src/common -> rag2/src -> rag2 -> 项目根
RAG2_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = RAG2_ROOT.parent


def raw_standards_dir() -> Path:
    """规范 PDF 源目录（原始素材，不复制、不修改）。"""
    env = os.environ.get("RAG2_RAW_DIR")
    if env:
        return Path(env)
    return PROJECT_ROOT / "data" / "raw" / "standards"


def raw_extra_dirs() -> list[Path]:
    """补充规范目录（上海地标等）。"""
    base = PROJECT_ROOT / "data" / "raw"
    return [base / "guidelines"]


def data_dir(*parts: str) -> Path:
    p = RAG2_ROOT / "data"
    for x in parts:
        p = p / x
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def models_dir(*parts: str) -> Path:
    p = PROJECT_ROOT / "data" / "models"
    for x in parts:
        p = p / x
    return p


# 本地预置模型（避免联网）
BASE_TOWER_MODEL = str(models_dir("bge-small-zh-v1.5"))
BASE_CROSS_MODEL = str(models_dir("bge-reranker-base"))
