"""PDF 读取：逐页抽取文本，保留页边界（供页眉页脚统计）。"""
from __future__ import annotations

import re
from pathlib import Path


def read_pages(path: str | Path) -> list[str]:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("需要 PyMuPDF：pip install pymupdf") from e
    doc = fitz.open(str(path))
    pages = [p.get_text("text") for p in doc]
    doc.close()
    return pages


def strip_page_boilerplate(pages: list[str], *, min_pages: int = 6) -> list[str]:
    """按页统计重复的首/末两行，跨页出现 >= 30% 判为页眉/页脚，逐页剥离。"""
    if len(pages) < min_pages:
        return pages
    from collections import Counter

    edge = Counter()
    for p in pages:
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        if not lines:
            continue
        for ln in lines[:2] + lines[-2:]:
            if 0 < len(ln) < 40:
                edge[ln] += 1
    thresh = max(3, int(len(pages) * 0.3))
    boiler = {ln for ln, c in edge.items() if c >= thresh}
    out = []
    for p in pages:
        out.append("\n".join(ln for ln in p.split("\n") if ln.strip() not in boiler))
    return out


_BLANK_RE = re.compile(r"\s")


def drop_toc_pages(pages: list[str], *, front_ratio: float = 0.25) -> list[str]:
    """整页丢弃**前置区**的目录页（目录一定在正文之前，故只在前 front_ratio 内动手）。

    判据偏保守，避免把"短行多"的正文本（如 GB550xx 把条款号单独排一行的版式）误删：
    必须命中点线/省略号，或短行占比 > 0.85 且条款号行 >= 6 且中位行长 < 15。
    """
    limit = max(4, int(len(pages) * front_ratio))
    out = []
    for i, p in enumerate(pages):
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        if i >= limit or len(lines) < 8:
            out.append(p)
            continue
        has_leader = bool(re.search(r"\.{3,}|…{2,}", p))
        short = sum(1 for ln in lines if len(ln) < 20)
        numish = sum(1 for ln in lines
                     if re.match(r"^\d{1,2}(?:\.\d{1,2}){1,3}", ln) or re.search(r"\.{3,}", ln))
        median_len = sorted(len(ln) for ln in lines)[len(lines) // 2]
        if has_leader or (short / len(lines) > 0.85 and numish >= 6 and median_len < 15):
            continue
        out.append(p)
    return out


def text_density(pages: list[str]) -> float:
    """有效字符数（去空白），用于扫描件检测。"""
    return float(sum(len(_BLANK_RE.sub("", p)) for p in pages))
