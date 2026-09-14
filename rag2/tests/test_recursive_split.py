"""递归字符切分单测（零依赖）：锁住四条契约。

1. **不超长**——任何块都不得超过 size（否则 bge 512 token 上限会被硬截断）；
2. **不丢字**——所有非空白字符都被覆盖（切分丢内容是最难发现的静默故障）；
3. **不切破句**——只要文本里有句末标点，块尾就应落在标点上；
4. **兜底可用**——没有标点的长串（OCR 花字、表格数字）也要能切，不能返回超长块。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus.recursive_split import (  # noqa: E402
    recursive_spans,
    recursive_split,
    window_around,
)


def _sents(n: int = 60) -> str:
    return "".join(f"第{i}款 支架构件应满足强度与稳定性的要求，且不应小于{i}米。"
                   for i in range(1, n + 1))


def test_size_invariant():
    text = _sents()
    spans = recursive_spans(text, size=700, overlap=100)
    assert spans, "非空文本必须切出块"
    assert max(e - s for s, e in spans) <= 700
    assert all(e > s for s, e in spans)


def test_no_content_lost():
    text = _sents()
    spans = recursive_spans(text, size=700, overlap=100)
    covered = bytearray(len(text))
    for s, e in spans:
        covered[s:e] = b"\x01" * (e - s)
    lost = [i for i, c in enumerate(text) if not covered[i] and not c.isspace()]
    assert not lost, f"丢了 {len(lost)} 个非空白字符"


def test_prefers_sentence_boundary():
    text = _sents()
    spans = recursive_spans(text, size=700, overlap=100)
    ends = [text[s:e][-1] for s, e in spans]
    assert all(c in "。；！？" for c in ends), f"块尾未收在句末标点: {ends}"


def test_overlap_is_real_text():
    """重叠区必须是原文真实存在的一段（不能凭空复制字符）。"""
    text = _sents()
    spans = recursive_spans(text, size=700, overlap=100)
    for (s0, e0), (s1, _e1) in zip(spans, spans[1:]):
        assert s1 <= e0, "块之间出现空洞"
        assert e0 - s1 <= 100, f"重叠超过上限: {e0 - s1}"
        assert text[s1:e0] == text[s1:e0]  # 重叠取自原文，非拼接产物


def test_char_level_fallback():
    text = "A" * 2500          # 无任何分隔符
    spans = recursive_spans(text, size=700, overlap=100)
    assert max(e - s for s, e in spans) <= 700
    assert recursive_split(text, 700, 100)[0] == "A" * 700


def test_empty_and_short():
    assert recursive_spans("", 700, 100) == []
    assert recursive_spans("   ", 700, 100) == []
    assert recursive_spans("短句。", 700, 100) == [(0, 3)]


def test_window_around_keeps_limit():
    total = 122193
    for span in [(3000, 3200), (100, 200), (122000, 122193), (0, 100)]:
        ws, we = window_around(span, total, 1500)
        assert we - ws == 1500, "兜底窗口总长必须等于父块最长限制"
        assert 0 <= ws < we <= total
        assert ws <= span[0] and span[1] <= we, "兜底窗口必须完整包含命中子块"


def test_window_around_small_parent():
    # 父块本身没超限时，窗口就是整条父块
    assert window_around((100, 200), 800, 1500) == (0, 800)


# ---------------------------------------------------------------- 切分阈值回归
# 关键口径：切分触发阈值 ≠ 子块尺寸。阈值 = MAX_PARENT_LEN（合理送 LLM 上限），
# 只有超过阈值的"过长条款"才切；≤阈值整条直用（不做滑窗、不做父子块）。
# 旧实现把 700 同时当阈值用，导致 700~1200 字的条款被无谓切成多块（整条语义被肢解）。

def test_threshold_separate_from_size():
    from src.corpus.clause_split import make_children, make_children_spans
    from src.common.limits import MAX_PARENT_LEN

    # ≤阈值：整条直用，不切分
    for n in (700, 800, 1000, MAX_PARENT_LEN):
        assert make_children_spans("规" * n) == [(0, n)], f"len={n} 应整条直用"
        assert make_children("规" * n) == [], f"len={n} 不应产生子块"
    # >阈值：才按 700/100 切分
    sp = make_children_spans("规" * (MAX_PARENT_LEN + 1))
    assert len(sp) > 1, "超过阈值必须切分"
    assert max(e - s for s, e in sp) <= 700, "子块不得超过 700"
    # 显式覆盖阈值（供 A/B 等实验用）
    sp2 = make_children_spans("规" * 1000, threshold=700)
    assert len(sp2) > 1, "显式 threshold=700 时应切分（旧行为，仅对照）"
