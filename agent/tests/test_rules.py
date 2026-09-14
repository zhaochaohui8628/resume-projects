import os
import sys
import json

AGENT_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(AGENT_SRC))
from tools.rules_checker import run_checks

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "sample_plan.txt")


def test_run():
    text = open(FIX, encoding="utf-8").read()
    res = run_checks(text)
    titles = [r["title"] for r in res["risks"]]

    assert any("JGJ 46-2005" in t for t in titles), "应检出 JGJ46-2005 废止"
    assert any("建质〔2009〕87号" in t for t in titles), "应检出 建质87号 废止"
    assert any("第 2.0.1 条" in t for t in titles), "应检出 JGJ33-2012 第2.0.1条 废止"
    assert any("专家论证" in t for t in titles), "应检出超规模缺专家论证"
    assert any("验收要求" in t for t in titles), "应检出缺验收要求章节"
    assert any("监测方案" in t for t in titles), "应检出基坑缺监测方案"

    print(f"OK(agent rules): 检出 {res['risk_count']} 项风险，识别危大类型 {res['detected_types']}")


if __name__ == "__main__":
    test_run()
