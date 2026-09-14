"""P1 远程监督标注单测。"""
from __future__ import annotations

from ner2.src.weak.lexicon import merge_lexicons
from ner2.src.weak.remote_label import RemoteLabeler, is_dirty, normalize_text


def _lex():
    return {
        "工程类型": ["基坑工程", "止水帷幕"],
        "工序": ["土方开挖", "混凝土浇筑", "开挖"],
        "设备": ["塔吊", "箱变"],
        "参数": ["桩长", "深度24.5m"],
        "危大类别": ["拆除"],
        "规范编号": [],
    }


def test_is_dirty_rejects_noise():
    assert is_dirty("───┬───")
    assert is_dirty("a")
    assert is_dirty("x" * 300)
    assert not is_dirty("基坑土方开挖应分层进行。")


def test_normalize_removes_spaces():
    assert normalize_text("基坑 工程") == "基坑工程"


def test_tag_longest_match_wins():
    l = RemoteLabeler(_lex())
    ents = l.tag("本工程采用基坑工程与土方开挖工艺。")
    assert ("工程类型", "基坑工程") in [(e["type"], e["term"]) for e in ents]
    # 土方开挖(长) 优先于 开挖(短)
    terms = [e["term"] for e in ents]
    assert "土方开挖" in terms and "开挖" not in terms


def test_tag_no_overlap_grid():
    l = RemoteLabeler(_lex())
    ents = l.tag("拆除土方开挖。")
    # 拆除(危大) 与 土方开挖(工序) 不重叠
    spans = sorted((e["start"], e["end"]) for e in ents)
    for i in range(1, len(spans)):
        assert spans[i][0] >= spans[i - 1][1]


def test_tag_standard_regex():
    l = RemoteLabeler(_lex())
    ents = l.tag("按JGJ80-2016执行。")
    hit = [e for e in ents if e["type"] == "规范编号"]
    assert hit and hit[0]["source"] == "regex"


def test_merge_lexicons_dedup():
    a = {"工序": ["开挖", "浇筑"]}
    b = {"工序": ["浇筑", "绑扎"]}
    m = merge_lexicons(a, b)
    assert m["工序"] == ["开挖", "浇筑", "绑扎"]
