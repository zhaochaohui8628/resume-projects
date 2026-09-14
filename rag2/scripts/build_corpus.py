"""构建语料库：81 本规范 PDF -> 条款级父块 JSONL。

## 三级切分策略

优先按**条款**切分（规范的本质结构就是章-条），这是信息量最高的一层。
但 PDF 文本层的质量参差，所以要按下面三级退让，每一级都记进 `chunk_mode`：

| 级别 | 触发条件 | 做法 | chunk_mode |
|---|---|---|---|
| 1 | 正常能切出 >= MIN_CLAUSES 条 | 按条款号递增链切分（`split_clauses`） | `clause` |
| 2 | 级别 1 条款数不足 | 先把被排版拆散的条款号拼回一行（`7` / `．2` / `．2` -> `7.2.2`）再切 | `clause_repaired` |
| 3 | 级别 2 仍不足 | 认定该规范**无条款结构**，整本按固定大小 700/100 递归切分 | `fixed` |

级别 3 是为 `GB50550-2010 建筑结构加固工程施工质量验收规范`（17.1 万字符）这类准备的：
它的条款号在 PDF 里逐字符成行，旧实现一个都认不出来，`find_body_start` 返回 0，
**整本规范被静默丢弃**——语料里完全没有这本，而它是加固验收的主力规范。

## 产物

  rag2/data/corpus/clauses.jsonl       每行一个父块（条款 / 无条款的定长块）
  rag2/data/corpus/corpus_stats.json   统计与质量报告
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import data_dir, raw_extra_dirs, raw_standards_dir  # noqa: E402
from src.corpus.clause_split import (  # noqa: E402
    _iter_lines,
    make_children,
    reassemble_number_lines,
    split_clauses,
    split_fixed,
)
from src.corpus.pdf_reader import (  # noqa: E402
    read_pages,
    strip_page_boilerplate,
    text_density,
)

MIN_CLAUSES = 20     # 少于这么多条款就认为"这一级没切出来"
CHILD_SIZE = 700     # 过长条款的子块（与需求给定口径一致）
CHILD_OVERLAP = 100
MIN_BLOCK_LEN = 24   # 比这更短的块丢弃（页码碎片、孤立表头）


def source_id(pdf: Path) -> str:
    """GB55037-2022_建筑防火通用规范.pdf -> GB55037-2022_建筑防火通用规范"""
    return pdf.stem


def to_record(clause, mode: str) -> dict:
    return {
        "id": clause.clause_id,
        "text": clause.text,
        "metadata": {
            "source": clause.source,
            "clause_no": clause.clause_no,
            "chapter": clause.chapter,
            "part": clause.part,
            "appendix": clause.appendix,
            "chunk_mode": mode,
            "text_len": len(clause.text),
            # 只记子块数量，不存子块文本：子块由索引层用同一套切分器确定性生成，
            # 存进语料会让 JSONL 体积翻倍（父块已经含全文）。
            "n_children": len(clause.children),
        },
    }


def _attach(clauses, mode: str) -> list[dict]:
    """过滤过短块 + 同号去重（保留最长的一条，通常是拼得更完整的）。"""
    dedup: dict[str, dict] = {}
    for c in clauses:
        if len(c.text) < MIN_BLOCK_LEN:
            continue
        c.children = make_children(c.text, CHILD_SIZE, CHILD_OVERLAP)
        rec = to_record(c, mode)
        old = dedup.get(rec["id"])
        if old is None or len(rec["text"]) > len(old["text"]):
            dedup[rec["id"]] = rec
    return list(dedup.values())


def parse_standard(pages: list[str], src: str) -> tuple[list[dict], str]:
    """三级切分。返回 (父块记录列表, 实际使用的级别)。"""
    base_lines = _iter_lines(pages)

    # 级别 1：正常切分
    kept = [c for c in split_clauses(pages, src, lines=base_lines) if c.part != "条文说明"]
    if len(kept) >= MIN_CLAUSES:
        return _attach(kept, "clause"), "clause"

    # 级别 2：条款号跨行重组后再切
    repaired = [c for c in split_clauses(pages, src, lines=reassemble_number_lines(base_lines))
                if c.part != "条文说明"]
    if len(repaired) >= MIN_CLAUSES:
        return _attach(repaired, "clause_repaired"), "clause_repaired"

    # 级别 3：整本固定大小切分
    fixed = split_fixed(pages, src, size=CHILD_SIZE, overlap=CHILD_OVERLAP)
    return _attach(fixed, "fixed"), "fixed"


def build(pdf_paths: list[Path], *, verbose: bool = True):
    records: list[dict] = []
    report: list[dict] = []
    for pdf in sorted(pdf_paths):
        src = source_id(pdf)
        try:
            pages = read_pages(pdf)
        except Exception as e:  # noqa: BLE001
            report.append({"source": src, "status": "read_error", "error": str(e)})
            continue
        density = text_density(pages)
        if density < 500:
            report.append({"source": src, "status": "scanned_no_text", "chars": density})
            continue
        pages = strip_page_boilerplate(pages)
        kept, mode = parse_standard(pages, src)
        records.extend(kept)
        lens = sorted(r["metadata"]["text_len"] for r in kept)
        report.append({
            "source": src, "status": "ok", "chunk_mode": mode,
            "pages": len(pages), "chars": int(density),
            "clauses_kept": len(kept),
            "median_len": lens[len(lens) // 2] if lens else 0,
            "max_len": lens[-1] if lens else 0,
            "n_children": sum(r["metadata"]["n_children"] for r in kept),
        })
        if verbose:
            print(f"[{mode:15s}] {src:56s} blocks={len(kept):5d} chars={int(density):8d}")
    return records, report


def main() -> None:
    dirs = [raw_standards_dir()] + raw_extra_dirs()
    pdfs: list[Path] = []
    for d in dirs:
        if d.exists():
            pdfs.extend(sorted(p for p in d.glob("*.pdf")))
    print(f"发现 PDF {len(pdfs)} 个")
    records, report = build(pdfs)
    out = data_dir("corpus", "clauses.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = sorted(r["metadata"]["text_len"] for r in records)
    n = len(lens)

    def q(p: float) -> int:
        return lens[min(n - 1, int(n * p))]

    stats = {
        "pdf_total": len(pdfs),
        "docs_ok": sum(1 for r in report if r["status"] == "ok"),
        "blocks": n,
        "by_chunk_mode": dict(Counter(r["metadata"]["chunk_mode"] for r in records)),
        "by_part": dict(Counter(r["metadata"]["part"] for r in records)),
        "with_children": sum(1 for r in records if r["metadata"]["n_children"]),
        "n_children_total": sum(r["metadata"]["n_children"] for r in records),
        "len_p50": q(0.5), "len_p90": q(0.9), "len_p95": q(0.95),
        "len_p99": q(0.99), "len_max": lens[-1] if lens else 0,
        "top_sources": Counter(r["metadata"]["source"] for r in records).most_common(12),
        "report": report,
    }
    with open(data_dir("corpus", "corpus_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n父块总数 {n} -> {out}")
    print("切分级别:", stats["by_chunk_mode"])
    print("按部分:", stats["by_part"])
    print(f"长度 p50/p90/p95/p99/max: {stats['len_p50']}/{stats['len_p90']}/"
          f"{stats['len_p95']}/{stats['len_p99']}/{stats['len_max']}")
    print(f"需切子块的父块 {stats['with_children']} 个，子块合计 {stats['n_children_total']}")


if __name__ == "__main__":
    main()
