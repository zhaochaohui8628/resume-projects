"""P2/P3 校验与冲突判定单测。"""
from __future__ import annotations

from ner2.src.review.parser import normalize_judgment, parse_llm_json, sanitize_entities
from ner2.src.review.conflict import analyze, apply_judgment
from ner2.src.eval.metrics import evaluate, label_set_difference


def test_parse_llm_json_code_block():
    raw = '```json\n{"verdict": "fix", "entities": [{"type": "设备", "start": 0, "end": 2}]}\n```'
    obj = parse_llm_json(raw)
    assert obj["verdict"] == "fix"


def test_parse_llm_json_garbage():
    assert parse_llm_json("不好意思，报错了") is None
    assert parse_llm_json("") is None


def test_sanitize_entities_filters_bad_type():
    text = "塔吊安装完成"
    ents = sanitize_entities(text, [
        {"type": "施工工艺", "start": 0, "end": 2},   # 非法类型 -> 丢弃
        {"type": "设备", "start": -5, "end": 99},      # 越界 -> clamp 为整句
    ])
    assert len(ents) == 1 and ents[0]["type"] == "设备"
    assert ents[0]["start"] == 0 and ents[0]["end"] == len(text)


def test_normalize_judgment_bad_verdict_downgrades():
    j = normalize_judgment("x", {"verdict": "banana"}, "w0")
    assert j["verdict"] == "keep"


def test_conflict_analyze_keep():
    rule = [{"type": "设备", "start": 0, "end": 2}]
    llm = [{"type": "设备", "start": 0, "end": 2}]
    assert analyze(rule, llm)["verdict"] == "keep"


def test_conflict_analyze_fix_type():
    rule = [{"type": "工序", "start": 3, "end": 8}]
    llm = [{"type": "工程类型", "start": 3, "end": 8}]
    a = analyze(rule, llm)
    assert a["verdict"] == "fix" and a["conflicts"][0]["type"] == "type"


def test_conflict_analyze_removed():
    rule = [{"type": "工序", "start": 0, "end": 2}]
    a = analyze(rule, [])
    assert a["verdict"] == "drop" and a["conflicts"][0]["type"] == "removed"


def test_apply_judgment_drop():
    r = {"text": "x", "entities": []}
    assert apply_judgment(r, {"verdict": "drop"}) is None


def test_metrics_exact_and_relaxed():
    preds = [{"type": "设备", "start": 0, "end": 2}]
    golds = [{"type": "设备", "start": 0, "end": 2}]
    assert evaluate(preds, golds, relaxed=False)["f1"] == 1.0
    golds2 = [{"type": "设备", "start": 1, "end": 3}]
    assert evaluate(preds, golds2, relaxed=True)["f1"] == 1.0
    assert evaluate(preds, golds2, relaxed=False)["f1"] == 0.0


def test_label_set_difference():
    rule = [{"type": "设备", "start": 0, "end": 2}]
    llm = [{"type": "设备", "start": 0, "end": 2}, {"type": "设备", "start": 5, "end": 7}]
    d = label_set_difference(rule, llm)
    assert d["shared"] == 1 and d["llm_only"] == 1 and d["rule_only"] == 0
