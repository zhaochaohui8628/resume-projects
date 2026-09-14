"""CLI：对一份方案做合规自查并导出报告。

用法：
  python agent/scripts/run_check.py <方案.txt>                # 规则引擎 + ner2 纯规则（零依赖）
  python agent/scripts/run_check.py <方案.txt> --ner2         # ner2 级联（规则层 + s2_crf_param_v3）全量识别
  python agent/scripts/run_check.py <方案.txt> --deepseek     # DeepSeek 供 LLM 环节（ReAct/自然语言描述）

注意：**已舍弃三层漏斗**（旧 --funnel 选项移除）——NER 一律对完整方案全量分块识别。
输出：agent/outputs/report.md（+ report.xlsx，若装了 openpyxl）
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.join(os.path.dirname(HERE), "src")
ROOT = os.path.dirname(os.path.dirname(HERE))          # 项目根（agent/ 的上级）
for p in (AGENT_SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.rules_checker import run_checks  # noqa: E402
from tools.ner_client import NerClient  # noqa: E402
from tools.rag_client import RagClient  # noqa: E402
from agent.compliance_pipeline import ComplianceAgent  # noqa: E402
from report.exporter import export  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        print("用法：python agent/scripts/run_check.py <方案.txt> [--ner2] [--deepseek]")
        return
    with open(args[0], "r", encoding="utf-8") as f:
        text = f.read()

    # NER 后端：默认 ner2 级联（规则层 + s2_crf_param_v3）全量识别
    if "--ner2" in flags or True:
        ner = NerClient(backend="ner2")  # 级联：规则层 + s2_crf_param_v3（缺模型自动降级纯规则）

    rag = RagClient()
    agent = ComplianceAgent(run_checks, ner_client=ner, rag_client=rag)
    rep = agent.run(text)
    outs = export(rep, os.path.join(os.path.dirname(AGENT_SRC), "outputs"))
    print(rep["summary"])
    print(f"[NER] 全量识别 {len(rep.get('entities', []))} 个实体")
    print("报告已导出：", outs)


if __name__ == "__main__":
    main()
