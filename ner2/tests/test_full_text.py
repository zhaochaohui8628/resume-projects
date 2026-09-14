"""FullTextExtractor 全量分块识别单测。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.pipeline.full_text import FullTextExtractor, dedup_entities  # noqa: E402


def _fx_rule() -> FullTextExtractor:
    return FullTextExtractor(model_path=None)  # 纯规则（零依赖、无 GPU）


def test_split_keeps_param_dense_sentence():
    """含数字/字母多的参数密集句**必须保留**（曾因中文占比阈值被误判 dirty 丢弃）。"""
    t = "基坑开挖深度8m，采用C30混凝土，塔吊基础执行JGJ80-2016。"
    sents = _fx_rule().split_sentences(t)
    assert sents, "参数密集句被误过滤"
    assert sents[0]["text"].startswith("基坑开挖深度")


def test_split_skips_dirty_lines():
    """伪影/非中文行仍被过滤（与弱标同口径）。"""
    fx = _fx_rule()
    assert fx.split_sentences("────────\n┌────┐\n正常的中文句子。")[0]["text"] == "正常的中文句子。"
    assert fx.split_sentences("hello world this is english text") == []


def test_split_long_line_filtered_like_weak_label():
    """超长行（>240）与弱标同口径被过滤（is_dirty 含长度判定，模型未见该分布）。"""
    fx = _fx_rule()
    long = "第一句内容。" + "长" * 300
    assert fx.split_sentences(long) == []
    # 正常行仍保留
    assert fx.split_sentences("第一句内容。第二句内容。")[0]["text"].startswith("第一句")


def test_extract_rule_cascade_fulltext():
    """纯规则全量抽取：实体带 type/text/layer/sent_idx 且 text 与原文一致。"""
    t = "基坑开挖深度8m，塔吊基础执行JGJ80-2016。"
    ents = _fx_rule().extract_text(t)
    assert ents, "应抽到实体"
    for e in ents:
        assert {"type", "text", "start", "end", "layer", "sent_idx"}.issubset(e.keys())
        assert e["start"] >= 0 and e["end"] > e["start"]
        assert e["text"] == e["sent_text"][e["start"]:e["end"]], "text 与偏移不一致"


def test_dedup_entities_aggregates_by_type_text():
    """去重收尾：同一 (type,text) 跨句合并为一条，count 累计，保留首次句号与前几句佐证。"""
    ents = [
        {"type": "参数", "text": "开挖深度", "sent_idx": 0, "sent_text": "开挖深度8m。", "layer": "rule"},
        {"type": "参数", "text": "开挖深度", "sent_idx": 5, "sent_text": "本基坑开挖深度为6m。", "layer": "rule"},
        {"type": "参数", "text": "开挖深度", "sent_idx": 0, "sent_text": "开挖深度8m。", "layer": "rule"},  # 同句重复也应计数
        {"type": "设备", "text": "塔吊", "sent_idx": 2, "sent_text": "采用塔吊吊装。", "layer": "model"},
        {"type": "危大类别", "text": "塔吊", "sent_idx": 3, "sent_text": "本工程塔吊作业。", "layer": "rule"},
    ]
    out = dedup_entities(ents)
    assert len(out) == 3, f"应聚合为 3 条唯一实体（含同文本不同类型），实际 {out}"
    by = {(r["type"], r["text"]): r for r in out}
    assert by[("参数", "开挖深度")]["count"] == 3
    assert by[("参数", "开挖深度")]["first_sent_idx"] == 0
    assert len(by[("参数", "开挖深度")]["sents"]) == 2          # 去重后的不同句子
    assert by[("设备", "塔吊")]["count"] == 1 and by[("危大类别", "塔吊")]["count"] == 1
    # 排序：count 高者在前
    assert out[0] == by[("参数", "开挖深度")]
