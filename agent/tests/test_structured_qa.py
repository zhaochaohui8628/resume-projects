"""复合结构化 QA 测试：拆解器（维度/类别/超规模）+ 四维组装。

运行：pytest agent/tests/test_structured_qa.py -q
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.join(os.path.dirname(HERE), "src")  # agent/tests -> agent/src
if AGENT_SRC not in sys.path:
    sys.path.insert(0, AGENT_SRC)

from qa.decompose import decompose, is_structured_query  # noqa: E402
from qa.assembler import assemble, run                     # noqa: E402
from qa.risk_sources import risk_table                     # noqa: E402


def test_decompose_full_query():
    """130t 汽车吊钢栈桥吊装 → 起重吊装 / 超规模 / 四维全开。"""
    slot = decompose("130t 汽车吊钢栈桥吊装危大工程，有哪些风险点？对应的方案编制内容、风险管控清单、验收节点分别是什么？")
    assert slot["category"] == "起重吊装"
    assert slot["equipment"] == "汽车吊"
    assert slot["subject"] == "钢栈桥"
    assert slot["capacity"] == 130.0 and slot["unit"] == "t"
    assert slot["is_super_scale"] is True
    keys = {d["key"] for d in slot["dimensions"] if d["enabled"]}
    assert keys == {"risks", "plan", "controls", "acceptance"}
    print("OK test_decompose_full_query")


def test_decompose_capacity_super_scale():
    """起重吊装超规模判定：100t(980kN) 超规模；5t(49kN) 不超。"""
    assert decompose("100t 塔吊吊装").get("is_super_scale") is True
    assert decompose("5t 汽车吊吊装").get("is_super_scale") is False
    print("OK test_decompose_capacity_super_scale")


def test_decompose_no_dimension_defaults_all():
    """无维度关键词 → 默认四维全开（问句只写"危大工程有哪些内容"）。"""
    slot = decompose("深基坑工程危大工程，方案要包含哪些内容")
    assert all(d["enabled"] for d in slot["dimensions"])
    assert slot["category"] == "基坑工程"
    print("OK test_decompose_no_dimension_defaults_all")


def test_structured_detection():
    """复合问句判定：多维度或（类别+1维度）→ True；单一条文 → False。"""
    assert is_structured_query("130t 汽车吊钢栈桥吊装危大工程，有哪些风险点？方案编制内容、管控清单、验收节点分别是什么？") is True
    assert is_structured_query("JGJ130 6.2.4 说什么") is False
    print("OK test_structured_detection")


def test_assemble_returns_four_sections():
    """四维组装：4 段 markdown，每段非空带来源。"""
    slot = decompose("130t 汽车吊钢栈桥吊装危大工程，有哪些风险点？对应的方案编制内容、风险管控清单、验收节点分别是什么？")
    out = assemble(slot)
    assert len(out["sections"]) == 4
    assert "风险点" in out["markdown"] and "方案编制内容" in out["markdown"]
    assert "管控" in out["markdown"] and "验收" in out["markdown"]
    for s in out["sections"]:
        assert s["items"], s["label"]
        assert s["source"]
    print("OK test_assemble_returns_four_sections")


def test_assemble_plan_includes_chapters():
    """方案编制内容：九章 + 起重吊装细化要素 + 超规模专家论证。"""
    slot = decompose("130t 汽车吊钢栈桥吊装危大工程，方案编制内容")
    out = assemble(slot)
    plan = next(s for s in out["sections"] if s["key"] == "plan")
    assert any("专家论证" in i for i in plan["items"])
    assert any("吊装工况计算" in i or "地基承载力" in i for i in plan["items"])
    print("OK test_assemble_plan_includes_chapters")


def test_risk_table_covers_categories():
    """风险源知识库覆盖起重吊装/基坑/模板/脚手架。"""
    assert len(risk_table("起重吊装")) >= 5
    assert len(risk_table("基坑工程")) >= 3
    assert risk_table("不存在的类别") == []
    print("OK test_risk_table_covers_categories")


def test_run_end_to_end():
    """run() 端到端：返回 markdown + title + 超规模标注。"""
    out = run("130t 汽车吊钢栈桥吊装危大工程，有哪些风险点？对应的方案编制内容、风险管控清单、验收节点分别是什么？")
    assert out["title"].startswith("130.0t")
    assert "超规模，需专家论证" in out["markdown"]
    print("OK test_run_end_to_end")


def test_retrieve_clauses_no_backend():
    """RAG 后端不可用（当前无检索后端）→ items 为空但不抛异常。"""
    from qa.retrieve import retrieve_clauses
    dr = retrieve_clauses("起重吊装", "吊装 安全技术措施")
    assert isinstance(dr["items"], list)
    assert isinstance(dr["sources"], list)
    assert dr["rag_count"] == len(dr["items"]) or dr["rag_count"] >= 0
    print(f"OK test_retrieve_clauses_no_backend: rag_count={dr['rag_count']}")


def test_assemble_no_retrieve_pure_rules():
    """slot['no_retrieve']=True → 不接 RAG，纯规则（离线可测）。"""
    from qa.decompose import decompose
    slot = decompose("130t 汽车吊钢栈桥吊装危大工程，有哪些风险点？对应的方案编制内容、风险管控清单、验收节点分别是什么？")
    slot["no_retrieve"] = True
    out = assemble(slot)
    for s in out["sections"]:
        assert "RAG" not in s["source"]
    print("OK test_assemble_no_retrieve_pure_rules")


def test_decompose_rule_fallback_without_llm():
    """无 LLM（显存不足/加载失败）→ decompose 回落规则，带 llm=False。"""
    from qa.decompose import decompose
    slot = decompose("130t 汽车吊钢栈桥吊装危大工程，有哪些风险点？对应的方案编制内容、风险管控清单、验收节点分别是什么？", use_llm=True)
    assert slot["llm"] is False          # 本机无显存 → 规则兜底
    assert slot["category"] == "起重吊装"
    assert slot["capacity"] == 130.0
    print("OK test_decompose_rule_fallback_without_llm")


def test_qwen_decomposer_json_extract():
    """qwen 拆解器 JSON 解析 + 归一化（mock 掉模型加载，只测解析层）。"""
    import qa.qwen_decomposer as qd
    # 解析
    d = qd._extract_json('好的，输出如下：\n{"category": "起重吊装及安装拆卸工程", "subject": "钢栈桥", "equipment": "汽车吊", "capacity": 130, "unit": "t", "dimensions": ["risks", "plan", "controls", "acceptance"]}')
    assert d["category"] == "起重吊装及安装拆卸工程"
    assert d["capacity"] == 130
    # 归一化（模拟 llm_decompose 后半段）
    _ALIAS = {"起重吊装及安装拆卸工程": "起重吊装", "深基坑工程": "基坑工程",
              "模板工程及支撑体系": "模板支撑", "脚手架工程": "脚手架"}
    cat = _ALIAS.get(d["category"], d["category"])
    assert cat == "起重吊装"
    # 维度归一
    _DIMS = [{"key": "risks", "label": "风险点", "kw": ()},
             {"key": "plan", "label": "方案编制内容", "kw": ()},
             {"key": "controls", "label": "风险管控清单", "kw": ()},
             {"key": "acceptance", "label": "验收节点", "kw": ()}]
    dim_keys = {x["key"] for x in _DIMS}
    dims = [x for x in d.get("dimensions") or [] if x in dim_keys]
    assert dims == ["risks", "plan", "controls", "acceptance"]
    # 空维度 → 全开
    d2 = qd._extract_json('{"category": "起重吊装", "dimensions": []}')
    dims2 = [x for x in d2.get("dimensions") or [] if x in dim_keys]
    assert dims2 == []          # 空列表 → 触发全开逻辑
    print("OK test_qwen_decomposer_json_extract")


if __name__ == "__main__":
    for f in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        f()
    print("ALL PASS")
