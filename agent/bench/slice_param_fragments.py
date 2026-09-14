"""从 11 本真实方案全文重新切出「含施工参数正文」的评测片段。

背景：原 real_fragments2 取的是方案首部 3KB → 多为封面/目录页，
经 ner2 的 is_dirty 过滤后大量用例保留 0 句，NER 抽不到"参数"实体
→ 三元组为空 → 判档/技术路无输入 → risks=[] → FNR 恒为 1.0。

本脚本改为：**以含参句为中心切段**，保证每个片段都有可被 NER 抽取的参数正文。

判据（含参句）：量名 + 数值（+ 可选单位）
产物：agent/bench/data/real_fragments3/<方案>_p<序号>.txt
      agent/bench/data/real_cases_params.jsonl（新增用例，id=real_p_<方案>_<序号>）
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NER2 = os.path.join(ROOT, "ner2")
for _p in (ROOT, NER2):          # ner2.src.* 需要项目根在 path
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src.weak.remote_label import is_dirty  # noqa: E402

PLAN_DIRS = (os.path.join(ROOT, "data", "raw", "plans_internal"),
             os.path.join(ROOT, "data", "raw", "plans_xproj"))
OUT_FRAG = os.path.join(ROOT, "agent", "bench", "data", "real_fragments3")
OUT_JSONL = os.path.join(ROOT, "agent", "bench", "data", "real_cases_params.jsonl")

# 量名 + 数值（+ 可选单位）
PARAM = re.compile(
    r"(开挖深度|开挖|深度|搭设高度|高度|跨度|荷载|起重量|起重|厚度|间距|步距|纵距|横距|"
    r"宽度|长度|直径|面积|层数|坡率|标高|承载力|距离|半径|速度|风力|温度|压力|重量|"
    r"截面|配筋|混凝土强度|强度等级)"
    r"[^。；\n]{0,12}?\d+(?:\.\d+)?\s*(?:m|米|mm|cm|kN|kPa|MPa|t|吨|层|级|%|°|#|号)?"
)
CTX = 12          # 每个含参句前后各带多少行上下文
PER_FRAG = 6      # 每个片段包含多少个含参句中心
MIN_CHARS = 600   # 片段最小字符数（太小则合并）


def clean_lines(text: str) -> list[str]:
    """按 ner2 同口径清洗，返回"能进 NER"的净行。"""
    out = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or is_dirty(line):
            continue
        out.append(line)
    return out


def build_fragments(fname: str, text: str) -> list[tuple[str, int]]:
    """返回 [(片段文本, 含参句数)]。"""
    lines = [l.strip() for l in text.split("\n")]
    idx = [i for i, l in enumerate(lines) if l and not is_dirty(l) and PARAM.search(l)]
    if not idx:
        return []
    frags = []
    for s in range(0, len(idx), PER_FRAG):
        group = idx[s:s + PER_FRAG]
        lo = max(0, group[0] - CTX)
        hi = min(len(lines), group[-1] + CTX + 1)
        body = "\n".join(l for l in lines[lo:hi] if l.strip())
        if len(body) >= 200:
            frags.append((body, len(group)))
    # 合并过短片段
    merged: list[tuple[str, int]] = []
    for body, n in frags:
        if merged and len(merged[-1][0]) < MIN_CHARS:
            merged[-1] = (merged[-1][0] + "\n" + body, merged[-1][1] + n)
        else:
            merged.append((body, n))
    return merged


def main() -> int:
    os.makedirs(OUT_FRAG, exist_ok=True)
    rows, total = [], 0
    for d in PLAN_DIRS:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".txt"):
                continue
            stem = os.path.splitext(f)[0]
            text = open(os.path.join(d, f), encoding="utf-8", errors="ignore").read()
            frags = build_fragments(f, text)
            for i, (body, n) in enumerate(frags, 1):
                fid = f"{stem}_p{i}"
                fp = os.path.join(OUT_FRAG, fid + ".txt")
                with open(fp, "w", encoding="utf-8") as fh:
                    fh.write(body)
                rows.append({
                    "id": f"real_p_{fid}",
                    "source": f,
                    "source_note": f"{stem} 含参正文切片 #{i}（{n} 个参数句）",
                    "category": "真实方案/参数正文",
                    "query": "全面审查该方案，重点核查危大工程判定、技术参数合规性与编制要素完整性",
                    "plan_file": f"real_fragments3/{fid}.txt",
                    "desc": f"{stem} 参数正文片段（{n} 个含参句）",
                    "golden_risks": [],      # 需人工/规则标注，见 REVIEW_GUIDE.md
                    "golden_entities": [],
                    "expect": {"survive": True, "has_output": True},
                    "annotation": "auto_slice_v2",
                })
                total += 1
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"生成片段 {total} 个 -> {OUT_FRAG}")
    print(f"生成用例 {len(rows)} 条 -> {OUT_JSONL}")
    # 自检：每个片段过 is_dirty 后是否还有句子 + 含参
    bad = 0
    for r in rows:
        p = os.path.join(ROOT, "agent", "bench", "data", r["plan_file"])
        t = open(p, encoding="utf-8").read()
        cl = clean_lines(t)
        if not cl or not any(PARAM.search(l) for l in cl):
            bad += 1
    print(f"自检：净句为空或不含参的片段 {bad}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
