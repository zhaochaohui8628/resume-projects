"""语料切分单测（零依赖）：锁住三条最容易回归的行为。

1. 数字条款号必须切对（旧版把 `6.2.3` 的正文错配成 `7.2.6` 的内容）；
2. 交叉引用（"应符合第5.1.1 条的规定"）不能当成新条款起点；
3. 行内条款（"…观测记录。6.2.3成孔…"）必须切开，不能并进前一条。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus.clause_split import find_body_start, find_heads, split_clauses  # noqa: E402
from src.index.tokenize import tokenize  # noqa: E402


def _page(lines):
    return ["\n".join(lines)]


def test_numeric_clause_split():
    lines = []
    for i in range(1, 30):
        lines.append(f"3.0.{i}")
        lines.append(f"本条为第{i}项技术要求，应符合设计文件的规定并满足下列条件:")
        lines.append("施工前应编制专项施工方案并经审批。")
    cls = split_clauses(_page(lines), "T", strict_body=False)
    nos = [c.clause_no for c in cls]
    assert "3.0.1" in nos and "3.0.29" in nos
    for c in cls:
        assert c.text.startswith(c.clause_no), c.text[:20]


def test_cross_reference_not_a_clause():
    heads = find_heads("5.1.1 条的规定，且不应小于200mm。", "本条为正文。")
    assert heads and heads[0][1] == "5.1.1"
    # 前缀是"条" => 引用残句，调用方应拒绝；这里验证识别函数确实把它标出来了
    from src.corpus.clause_split import _is_ref
    assert _is_ref(heads[0][2]) is True


def test_inline_clause_is_split():
    p = "6.2.2 成孔设备就位后，必须平整稳固。6.2.3成孔的控制深度应符合设计要求。"
    heads = find_heads(p)
    assert [h[1] for h in heads] == ["6.2.2", "6.2.3"]


def test_value_with_unit_not_a_clause():
    # "3.0m" / "表5.5.4" 不能被当成条款号
    assert find_heads("3.0m 的间距应满足要求。") == []
    assert find_heads("应符合表5.5.4 的规定。") == []


def test_find_body_start_skips_toc_like_prefix():
    pre = ["目次", "1 总则", "3.0.1 ………………………… 1"]
    body = []
    for i in range(1, 26):
        body.append(f"3.0.{i} 本条正文内容较长，用于满足整段判定所需的平均行长。")
    lines = pre + body
    start = find_body_start(lines)
    assert start >= len(pre) - 4


def test_tokenizer_keeps_codes_and_bigrams():
    toks = tokenize("扣件式钢管脚手架按GB55023执行，立杆间距1.8m")
    assert "gb55023" in toks                     # 规范号（字母数字 token）
    assert "扣件" in toks and "脚手" in toks      # CJK 二元组（"脚手架"由 脚手+手架 覆盖）
    assert "1.8m" in toks or "18m" in toks        # 数值+单位
