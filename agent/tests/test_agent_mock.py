import os
import sys

AGENT_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(AGENT_SRC))
from tools.rules_checker import run_checks
from tools.rag_client import RagClient
from tools.ner_client import NerClient
from agent.compliance_pipeline import ComplianceAgent
from llm.mock import MockLLM

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "sample_plan.txt")


def test_run_deterministic():
    """旧主链路（ComplianceAgent）仍可用。

    注意：2026-09-13 删除了 rules_checker 里硬编码的 SCALE_PATTERNS（基坑5m/模板8m/脚手架50m），
    因此原「超规模但未提及专家论证」那条 HIGH 不再由 C2 产出（少 1 项 → 11）——
    危大/超规模判档已迁到 `tools/hazard_level.py`，由 review subagent（内部判档路）负责，
    见 agent/tests/test_triage.py 与 test_orchestrator_v5.py。
    """
    ner = NerClient(backend="rule")
    rag = RagClient()  # 尚未入库时为降级空检索
    agent = ComplianceAgent(run_checks, ner_client=ner, rag_client=rag, llm=None)
    rep = agent.run(open(FIX, encoding="utf-8").read())
    assert rep["risk_count"] >= 11, rep["risk_count"]
    assert "基坑工程" in rep["detected_types"]
    types = {e["type"] for e in rep["entities"]}
    assert "危大类别" in types
    # C2 不再产出阈值类风险（已迁出）
    assert not any(r["check"] == "C2" and "超规模" in r["title"] for r in rep["risks"])
    print(f"OK(agent 主链路): {rep['risk_count']} 项风险；实体类型 {types}")


def test_react_with_mock():
    llm = MockLLM(script=['{"thought":"先看危大类型","action":"finish","action_input":""}'])
    agent = ComplianceAgent(run_checks, llm=llm)
    trace = agent.react("某基坑工程专项方案", [])
    assert trace and trace[-1]["action"] == "finish"
    print(f"OK(agent ReAct): {len(trace)} 步，末动作={trace[-1]['action']}")


if __name__ == "__main__":
    test_run_deterministic()
    test_react_with_mock()
