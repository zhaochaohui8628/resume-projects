"""级联管道单测（规则层 + 合并逻辑；模型层仅测接口与占位行为）。"""
from __future__ import annotations

import pytest

from ner2.src.pipeline.cascade import CascadeExtractor
from ner2.src.pipeline.model_layer import load_model_backend
from ner2.src.pipeline.rule_layer import RuleLayer


def _lex():
    return {
        "工程类型": ["基坑工程", "止水帷幕"],
        "工序": ["土方开挖", "混凝土浇筑"],
        "设备": ["塔吊", "箱变"],
        "参数": ["桩长"],
        "危大类别": ["拆除"],
        "规范编号": [],
    }


def test_rule_layer_standard_no():
    r = RuleLayer(_lex())
    ents = r.extract("按JGJ80-2016执行。")
    hit = [e for e in ents if e["type"] == "规范编号"]
    assert hit and hit[0]["layer"] == "rule" and hit[0]["conf"] == 1.0


def test_rule_layer_category_keyword():
    r = RuleLayer(_lex())
    ents = r.extract("本工程塔吊安装属于起重吊装作业。")
    cats = [e for e in ents if e["type"] == "危大类别"]
    assert cats, "应命中危大类别关键词"


def test_rule_layer_lexicon_longest():
    r = RuleLayer(_lex())
    ents = r.extract("基坑工程采用土方开挖。")
    terms = [e["term"] for e in ents]
    assert "基坑工程" in terms and "土方开挖" in terms


def test_cascade_rule_only():
    c = CascadeExtractor(_lex())
    ents = c.extract("塔吊基础采用JGJ80-2016。")
    assert ents and all(e["layer"] == "rule" for e in ents)


def test_cascade_merge_rule_priority():
    """模型层补充的实体若与规则层重叠应被丢弃；不重叠则保留。"""
    rule = [{"type": "设备", "start": 0, "end": 2, "layer": "rule", "conf": 1.0}]
    model = [
        {"type": "设备", "start": 0, "end": 2},   # 重叠 -> 丢弃
        {"type": "工序", "start": 5, "end": 9},    # 空缺 -> 补充
        {"type": "非法", "start": 10, "end": 12},  # 非法类型 -> 丢弃
    ]
    merged = CascadeExtractor._merge(rule, model)
    assert len(merged) == 2
    assert merged[1]["layer"] == "model" and merged[1]["start"] == 5


def test_load_model_backend_none():
    assert load_model_backend(None) is None


def test_load_model_backend_missing_ckpt():
    with pytest.raises(FileNotFoundError):
        load_model_backend("__not_exist__.pt")
