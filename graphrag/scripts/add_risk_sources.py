"""把 risk_sources 风险源知识并入图谱（RiskSource 节点 + HAS_RISK 关系）。

数据源 = agent/src/qa/risk_sources.py 的 RISK_SOURCES（**同源**，图谱与规则兜底
永不打架；后续 Neo4j 全量图谱也由本脚本生成）。

产出：合并进 graphrag/data/demo_graph_v2.json
  - RiskSource 节点：{id: "risk:{类别}:{序号}", label: "RiskSource",
                       category, source(风险源), hazard(后果), control(管控)}
  - HazardCategory -HAS_RISK-> RiskSource（类别名映射见 _CAT_MAP）

运行：python graphrag/scripts/add_risk_sources.py
（数据源 agent/src/qa/risk_sources.py 为可选：找不到则跳过，不影响图谱 demo。）
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # .../graphrag
WS = os.path.dirname(ROOT)                                     # 工作区根
AGENT_SRC = os.path.join(WS, "agent", "src")
if AGENT_SRC not in sys.path:
    sys.path.insert(0, AGENT_SRC)

try:
    from qa.risk_sources import RISK_SOURCES  # noqa: E402
except Exception as _e:  # noqa: BLE001
    RISK_SOURCES = {}
    print(f"[warn] 风险源知识库不可用（{_e}），本脚本跳过（不影响图谱 demo）")

GRAPH = os.path.join(ROOT, "data", "demo_graph_v2.json")

# risk_sources.py 的类别 key → 图谱 HazardCategory 节点 id（含 id 片段即可匹配）
_CAT_MAP = {
    "起重吊装": "起重吊装工程",
    "基坑工程": "深基坑工程",
    "模板支撑": "模板支撑工程",
    "脚手架": "脚手架工程",
}


def main() -> None:
    g = json.load(open(GRAPH, encoding="utf-8"))
    nodes, edges = g["nodes"], g["edges"]
    node_ids = {n["id"] for n in nodes}
    edge_keys = {(e["from"], e["to"], e["type"]) for e in edges}
    # 找到 HazardCategory 节点 id
    cat_ids = {n["id"] for n in nodes if n["label"] == "HazardCategory"}

    n_new = n_skip = 0
    for cat_key, risks in RISK_SOURCES.items():
        target = _CAT_MAP.get(cat_key)
        if target is None:
            print(f"  [skip] 类别 {cat_key} 无图谱映射，跳过")
            continue
        # 匹配 HazardCategory 节点（id 含 target 片段）
        matched = [cid for cid in cat_ids if target in cid]
        if not matched:
            print(f"  [skip] 图谱无 HazardCategory「{target}」，跳过 {cat_key}")
            continue
        for i, r in enumerate(risks, 1):
            rid = f"risk:{cat_key}:{i:02d}"
            if rid in node_ids:
                n_skip += 1
                continue
            nodes.append({"id": rid, "label": "RiskSource", "name": r["source"],
                          "category": cat_key, "source": r["source"],
                          "hazard": r["hazard"], "control": r["control"]})
            node_ids.add(rid)
            for cid in matched:
                if (cid, rid, "HAS_RISK") not in edge_keys:
                    edges.append({"from": cid, "to": rid, "type": "HAS_RISK"})
                    edge_keys.add((cid, rid, "HAS_RISK"))
            n_new += 1

    g["nodes"], g["edges"] = nodes, edges
    g.setdefault("meta", {})["risk_sources_added"] = True
    g["meta"]["node_labels"] = sorted({n["label"] for n in nodes})
    g["meta"]["rel_types"] = sorted({e["type"] for e in edges})
    with open(GRAPH, "w", encoding="utf-8") as f:
        json.dump(g, f, ensure_ascii=False, indent=1)

    from collections import Counter
    print(f"RiskSource 新增 {n_new}（已存在跳过 {n_skip}）")
    print("节点:", dict(Counter(n["label"] for n in nodes)))
    print("关系:", dict(Counter(e["type"] for e in edges)))


if __name__ == "__main__":
    main()
