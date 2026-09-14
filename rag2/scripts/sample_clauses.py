"""挑选"值得写问句"的条款并切批输出，供逐批撰写 query。

第一版按"来源轮转取第 i 条"取样，结果每本规范的第一条都是 `1.0.1 为…制定本规范`
这类总则/适用范围条款——写不出有检索价值的问句。改为**按信息量打分**：

加分：出现阈值判据（不应小于/不得超过/宜控制在…）、出现数值+单位（50m / 20% / 30kN）
      长度 60~320（自成一体）
减分：总则与适用范围套话（"制定本规范"/"适用"）、"必须执行本规范"、
      "由相关责任主体判定"、术语条（含长英文词）、纯 1.0.x / 2.0.x 条款号

再按来源轮转，保证 77 本规范都有覆盖。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import data_dir  # noqa: E402

BATCH = 110
MAX_CLAUSES = 1100

_THRESHOLD_RE = re.compile(
    r"不应(?:小于|大于|少于|超过|低于|高于)|不得(?:小于|大于|少于|超过|低于|高于)"
    r"|不宜(?:小于|大于|少于|超过|低于|高于)|应(?:控制|小于|大于|按).{0,4}在"
    r"|至少|最多|不应低于|不应少于")
_UNIT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mm|cm|dm|km|m2|m3|kN|kPa|MPa|N·m|kg|t|%|℃|min|h|d|s"
    r"|倍|层|根|人|个|次|遍|道|级|度|米|厘米|毫米|千帕|兆帕|吨|小时|分钟)")
_BOILER = ("制定本规范", "制定本标准", "制定本规程", "必须执行本规范", "由相关责任主体判定",
           "除应符合本规范外", "除应执行本", "尚应符合国家现行有关标准")
_ENWORD_RE = re.compile(r"[A-Za-z]{5,}")


def score_clause(text: str, clause_no: str) -> int:
    s = 0
    if _THRESHOLD_RE.search(text):
        s += 3
    if _UNIT_RE.search(text):
        s += 2
    if 60 <= len(text) <= 320:
        s += 1
    if _ENWORD_RE.search(text):
        s -= 1
    if any(b in text for b in _BOILER):
        s -= 6
    if re.match(r"^[12]\.0\.\d", clause_no) or re.match(r"^\d\.0\.1$", clause_no):
        s -= 2
    return s


def main() -> None:
    rows = [json.loads(l) for l in open(data_dir("corpus", "clauses.jsonl"), encoding="utf-8")]
    pool: dict[str, list[tuple[int, dict]]] = {}
    seen: set[str] = set()
    for r in rows:
        m = r["metadata"]
        t = r["text"]
        if m["part"] != "正文" or not (40 <= len(t) <= 520):
            continue
        if not re.search(r"\d", t):
            continue
        if sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff") / len(t) < 0.5:
            continue
        key = re.sub(r"\s", "", t)[:60]
        if key in seen:
            continue
        seen.add(key)
        sc = score_clause(t, m["clause_no"])
        if sc < 3:
            continue
        pool.setdefault(m["source"], []).append((sc, r))

    order = sorted(pool, key=lambda s: -len(pool[s]))
    for s in order:
        pool[s].sort(key=lambda x: -x[0])
    print(f"可用条款 {sum(len(v) for v in pool.values())}（{len(pool)} 本规范）")

    picked: list[dict] = []
    i = 0
    while len(picked) < MAX_CLAUSES and any(i < len(pool[s]) for s in order):
        for s in order:
            if i < len(pool[s]):
                picked.append(pool[s][i][1])
        i += 1
    print(f"选取 {len(picked)} 条")

    out_dir = data_dir("authoring")
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(picked), BATCH):
        with open(out_dir / f"batch_{k // BATCH:02d}.txt", "w", encoding="utf-8") as f:
            for r in picked[k:k + BATCH]:
                f.write(f"### {r['id']}\n{r['text']}\n\n")
    print(f"输出 {(len(picked) + BATCH - 1) // BATCH} 批 -> {out_dir}")


if __name__ == "__main__":
    main()
