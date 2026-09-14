"""P0 分诊链路单测：三元组抽取 / 阈值查表 / 条文比对（纯逻辑，零依赖，无需 GPU）。

构造手写的 ner2 实体（{type,text,start,end,sent_idx,sent_text}），不加载模型。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT = os.path.dirname(HERE)
AGENT_SRC = os.path.join(AGENT, "src")
if AGENT_SRC not in sys.path:
    sys.path.insert(0, AGENT_SRC)

from tools.param_extractor import (extract_triples, dedup, keep_worst, judgeable,
                                   dim_ok, triage)  # noqa: E402
from tools.hazard_level import judge, LEVEL_NONE, LEVEL_HAZ, LEVEL_SUPER, normalize  # noqa: E402
from tools.comparator import compare, extract_limits, VERDICT_BAD, VERDICT_OK, VERDICT_NA  # noqa: E402


def _ent(t, text, sent, sid=0, offset=None):
    start = sent.find(text) if offset is None else offset
    return {"type": t, "text": text, "start": start, "end": start + len(text),
            "sent_idx": sid, "sent_text": sent}


# ------------------------------------------------------------ 三元组绑定
def test_bind_value_after_metric():
    s = "本工程基坑开挖深度 8m，采用地下连续墙支护。"
    ents = [_ent("工程类型", "基坑", s), _ent("参数", "开挖深度", s)]
    tr = extract_triples(ents)
    assert len(tr) == 1
    assert tr[0]["value"] == 8.0 and tr[0]["unit"] == "m"
    assert tr[0]["category"] == "基坑工程"
    assert tr[0]["subject"] == "基坑"


def test_bind_rejects_wrong_dimension():
    """量纲不相容：起重量不能配 m（实测「起重量=51m」就是这种误绑）。"""
    s = "塔吊工作半径为51m，最大起重量约7T。"
    ents = [_ent("参数", "起重量", s), _ent("设备", "塔吊", s)]
    tr = extract_triples(ents)
    # 51m 在量名之前且无连接词 → 不绑；7T 在量名之后但距离 >10 → 也不绑
    assert tr == []
    assert not dim_ok("起重量", "m")
    assert dim_ok("起重量", "t")


def test_bind_value_with_linker_before_metric():
    s = "当搭设高度为10m时，应按高支模要求组织专家论证。"
    ents = [_ent("参数", "搭设高度", s)]
    tr = extract_triples(ents)
    assert len(tr) == 1 and tr[0]["value"] == 10.0 and tr[0]["unit"] == "m"
    assert tr[0]["category"] == "模板支撑"       # 高支模 → 模板支撑


def test_bind_negative_elevation_takes_abs():
    """实测误判修复：'开挖深度-22.25m' 的负号是标高口径，深度应取绝对值。"""
    s = "开启条件为开挖落深4.25m，既开挖深度-22.25m 时启用。"
    ents = [_ent("参数", "开挖深度", s, offset=s.find("开挖深度", 12))]
    tr = extract_triples(ents)
    assert len(tr) == 1 and tr[0]["value"] == 22.25
    assert judge("基坑工程", tr[0]["metric"], tr[0]["value"], tr[0]["unit"])["level"] == LEVEL_SUPER


def test_triage_splits_two_lanes():
    """分诊不是筛选：阈值表维度走判档，其余走技术核对（全量都在）。"""
    s = "基坑开挖深度 8m；桩身混凝土强度 35MPa，保护层厚度 50mm。"
    ents = [_ent("参数", "开挖深度", s), _ent("参数", "混凝土强度", s),
            _ent("参数", "保护层厚度", s)]
    tr = dedup(extract_triples(ents))
    lanes = triage(tr)
    hl = {t["metric"] for t in lanes["hazard_level"]}
    te = {t["metric"] for t in lanes["technical"]}
    assert "开挖深度" in hl
    assert {"混凝土强度", "保护层厚度"} <= te
    assert hl | te == {t["metric"] for t in tr}          # 全量三元组不丢


def test_spec_value_has_no_triple():
    """无量纲规格值（C35/P8/∅800@600）绑不出三元组 → 只能走**实体级检索**。

    这是刻意保留的边界：PARAM_SPEC 把规格值当独立参数实体，但它没有可比较的量纲，
    不能进 comparator；正确做法是拿实体本身当查询词去检索条文。
    """
    s = "桩身混凝土设计强度等级 C35，抗渗等级 P8。"
    ents = [_ent("参数", "C35", s), _ent("参数", "P8", s)]
    assert extract_triples(ents) == []


def test_keep_worst_and_judgeable():
    s = "基坑开挖深度约17.95m；医技平台基坑开挖深度约15.15m。"
    ents = [_ent("参数", "开挖深度", s), _ent("参数", "开挖深度", s, offset=s.find("开挖深度", 12))]
    tr = keep_worst(dedup(extract_triples(ents)))
    assert len(tr) == 1 and tr[0]["value"] == 17.95      # 最不利工况
    # 混凝土强度不在阈值表 → 被筛掉
    s2 = "桩身混凝土强度等级C30，保护层厚度50mm。"
    ents2 = [_ent("参数", "混凝土强度", s2), _ent("参数", "保护层厚度", s2)]
    assert judgeable(extract_triples(ents2)) == []


# ------------------------------------------------------------ 阈值查表
def test_judge_foundation_pit():
    assert judge("基坑工程", "开挖深度", 8, "m")["level"] == LEVEL_SUPER
    assert judge("基坑工程", "开挖深度", 8, "m")["need_review"] is True
    assert judge("基坑工程", "开挖深度", 4, "m")["level"] == LEVEL_HAZ
    assert judge("基坑工程", "开挖深度", 2, "m")["level"] == LEVEL_NONE
    assert "住建部令第37号" in judge("基坑工程", "开挖深度", 8, "m")["basis"]


def test_judge_formwork_vs_scaffold():
    """同一量名「搭设高度」在不同类别阈值不同 —— 类别必须消歧。"""
    assert judge("模板支撑", "搭设高度", 10, "m")["level"] == LEVEL_SUPER      # ≥8
    assert judge("脚手架", "搭设高度", 10, "m", "落地式钢管脚手架")["level"] == LEVEL_NONE   # <24
    assert judge("脚手架", "搭设高度", 30, "m", "悬挑式脚手架")["level"] == LEVEL_SUPER      # ≥20


def test_judge_unit_conversion():
    assert normalize(1000, "mm") == (1.0, "m")
    assert normalize(3, "t") == (30.0, "kN")
    assert judge("基坑工程", "开挖深度", 8000, "mm")["level"] == LEVEL_SUPER   # 8m


def test_judge_unknown_category():
    assert judge(None, "开挖深度", 8, "m")["level"] == "无法判定"


# ------------------------------------------------------------ 条文比对
def test_compare_upper_limit():
    t = {"metric": "开挖深度", "value": 8.0, "unit": "m"}
    r = compare(t, "采用放坡开挖的基坑开挖深度不宜超过7.0m。")
    assert r["verdict"] == VERDICT_BAD and "不宜超过 7.0m" in r["limit"]
    assert compare({"metric": "开挖深度", "value": 5.0, "unit": "m"},
                   "采用放坡开挖的基坑开挖深度不宜超过7.0m。")["verdict"] == VERDICT_OK


def test_compare_ocr_noise():
    """语料 OCR 把 7.0m 识别成「7. Om」，清洗后仍须命中。"""
    t = {"metric": "开挖深度", "value": 8.0, "unit": "m"}
    assert extract_limits("开挖深度不宜超过7. Om 。") != []
    assert compare(t, "开挖深度不宜超过7. Om 。")["verdict"] == VERDICT_BAD


def test_compare_lower_limit_and_na():
    assert compare({"metric": "壁厚", "value": 3.0, "unit": "mm"},
                   "井管的壁厚不应小于4mm。")["verdict"] == VERDICT_BAD
    assert compare({"metric": "开挖深度", "value": 8.0, "unit": "m"},
                   "应编制专项方案并组织验收。")["verdict"] == VERDICT_NA


def test_compare_ignores_other_dimension():
    """量纲不同的限值不可比（8m 不能和 20kN 比）。"""
    assert compare({"metric": "开挖深度", "value": 8.0, "unit": "m"},
                   "起重量不宜超过20kN。")["verdict"] == VERDICT_NA


# ------------------------------------------------------------ LLM 兜底比对
class _StubLLM:
    """极简 LLM stub：返回预设文本，并记录收到的 prompt。"""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, messages, **kw):
        self.prompts.append("\n".join(m.get("content", "") for m in messages))
        return self.reply


def test_llm_compare_parses_verdict():
    """做法类（无可用限值）走 LLM 兜底：不符合 → 疑似违反，并回带引文。"""
    from tools.comparator import llm_compare
    llm = _StubLLM('{"verdict":"不符合","reason":"未按条文要求设置剪刀撑",'
                   '"evidence":"应沿架体高度连续设置剪刀撑"}')
    clauses = [{"text": "应沿架体高度连续设置剪刀撑。",
                "metadata": {"source": "JGJ130-2011", "clause_no": "6.4.3"}}]
    r = llm_compare({"metric": "剪刀撑", "value": 0, "unit": "",
                     "sent_text": "架体未设置剪刀撑。"}, clauses, llm)
    assert r["verdict"] == VERDICT_BAD
    assert "剪刀撑" in r["evidence"] and r["source"] == "JGJ130-2011"
    # 输入被严格限定：prompt 里必须同时含工况句与条文原文（防幻觉的前提）
    p = llm.prompts[0]
    assert "架体未设置剪刀撑" in p and "应沿架体高度连续设置剪刀撑" in p


def test_llm_compare_degrades_safely():
    """无 LLM / 无条文 / 模型乱答 → 一律「无法判定」，绝不臆造结论。"""
    from tools.comparator import llm_compare
    t = {"metric": "剪刀撑", "value": 0, "unit": "", "sent_text": "未设置。"}
    c = [{"text": "条文", "metadata": {}}]
    assert llm_compare(t, c, None)["verdict"] == VERDICT_NA
    assert llm_compare(t, [], _StubLLM('{"verdict":"符合"}'))["verdict"] == VERDICT_NA
    assert llm_compare(t, c, _StubLLM("不是 JSON"))["verdict"] == VERDICT_NA
    # 模型自造结论"无问题"但没给引文 → 仍按无法判定处理（不采信无引文结论）
    assert llm_compare(t, c, _StubLLM('{"verdict":"无法判定"}'))["verdict"] == VERDICT_NA


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
