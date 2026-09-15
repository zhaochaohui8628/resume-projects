"""目标图谱 Demo v2 构建脚本（GraphRAG demo 多跳演示用）。

在 demo_graph.json（仅 Standard/Clause/HazardCategory/Entity）基础上，
新增 demo 多跳演示所需的 4 类节点 + 5 类关系，全部素材来自仓库现有文件：
  - Metric（量名）      ← ner2/data/ner/param_lexicon.json（102 词）
  - Threshold（阈值）   ← agent/data/rules/hazardous_work_types.json + grpo 判定口径
  - Obligation（义务）  ← 37 号令/条文"应…"义务，demo 手工精编
  - Term（术语别名）    ← ner2 词典 + 规范 2.1 术语章

运行：python graphrag/scripts/build_demo_graph_v2.py
输出：graphrag/data/demo_graph_v2.json
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # .../graphrag
OUT = os.path.join(ROOT, "data", "demo_graph_v2.json")


def node(id_: str, label: str, **props) -> dict:
    return {"id": id_, "label": label, "name": id_, **props}


def edge(frm: str, to: str, typ: str, **props) -> dict:
    return {"from": frm, "to": to, "type": typ, **props}


nodes: list[dict] = []
edges: list[dict] = []

# ---------------- 1) Term 术语别名（query 口语词归一演示） ----------------
TERMS = [
    # (术语, 指向节点, 指向类型, 规范用语说明)
    ("高支模", "模板支撑工程", "HazardCategory", "规范叫法：混凝土模板支撑工程"),
    ("深基坑", "深基坑工程", "HazardCategory", "规范叫法：深基坑工程"),
    ("排架", "模板支架", "Entity", "规范叫法：模板支撑架/支撑结构"),
    ("满堂架", "模板支架", "Entity", "规范叫法：满堂支撑架"),
    ("塔吊", "起重吊装工程", "HazardCategory", "规范叫法：塔式起重机"),
    ("爬架", "脚手架工程", "HazardCategory", "规范叫法：附着式升降脚手架"),
]
for term, target, tlabel, note in TERMS:
    tid = f"term:{term}"
    nodes.append(node(tid, "Term", note=note))
    edges.append(edge(tid, target, "ALIAS_OF", target_label=tlabel))

# ---------------- 2) Metric 量名（量名与类别的绑定） ----------------
# (量名, 单位, 所属危大类别, 素材来源)
METRICS = [
    ("搭设高度", "m", "模板支撑工程", "hazardous_work_types.json"),
    ("开挖深度", "m", "深基坑工程", "hazardous_work_types.json"),
    ("施工总荷载", "kN/m2", "模板支撑工程", "hazardous_work_types.json"),
    ("集中线荷载", "kN/m", "模板支撑工程", "hazardous_work_types.json"),
    ("单件起吊重量", "kN", "起重吊装工程", "hazardous_work_types.json"),
    ("搭设高度", "m", "脚手架工程", "hazardous_work_types.json"),
    ("开挖深度", "m", "降水工程", "hazardous_work_types.json"),
]
for metric, unit, cat, src in METRICS:
    mid = f"metric:{cat}:{metric}"
    nodes.append(node(mid, "Metric", unit=unit, source=src))
    edges.append(edge(cat, mid, "HAS_METRIC"))

# ---------------- 3) Threshold 阈值节点（危大线/超规模线） ----------------
# (类别, 量名, 档位, 值, 单位, 来源口径)
THRESHOLDS = [
    ("模板支撑工程", "搭设高度", "危大", 5, "m", "37号令附件1 二(二)"),
    ("模板支撑工程", "搭设高度", "超规模", 8, "m", "37号令附件2 二(二)"),
    ("模板支撑工程", "施工总荷载", "危大", 10, "kN/m2", "37号令附件1 二(二)"),
    ("模板支撑工程", "施工总荷载", "超规模", 15, "kN/m2", "37号令附件2 二(二)"),
    ("深基坑工程", "开挖深度", "危大", 3, "m", "37号令附件1 三(一)"),
    ("深基坑工程", "开挖深度", "超规模", 5, "m", "37号令附件2 三(一)"),
    ("脚手架工程", "搭设高度", "危大", 24, "m", "37号令附件1 四(一)"),
    ("脚手架工程", "搭设高度", "超规模", 50, "m", "37号令附件2 四(一)"),
    ("起重吊装工程", "单件起吊重量", "危大", 10, "kN", "37号令附件1 五(一)"),
    ("起重吊装工程", "单件起吊重量", "超规模", 100, "kN", "37号令附件2 五(一)"),
]
for cat, metric, level, value, unit, src in THRESHOLDS:
    tid = f"threshold:{cat}:{metric}:{level}:{value}{unit}"
    nodes.append(node(tid, "Threshold", category=cat, metric=metric,
                      level=level, value=value, unit=unit, source=src))
    edges.append(edge(f"metric:{cat}:{metric}", tid, "HAS_THRESHOLD"))
    edges.append(edge(tid, cat, "FOR_CATEGORY"))

# ---------------- 4) Obligation 义务节点（阈值触发的后果） ----------------
OBLIGATIONS = [
    ("obligation:应编制专项施工方案", "危大工程必须编制专项施工方案（37号令第10条）", "危大"),
    ("obligation:应组织专家论证", "超规模危大工程专项方案必须专家论证（37号令第12条）", "超规模"),
    ("obligation:应进行基坑监测", "深基坑工程应实施第三方监测（JGJ311-2013 第8章）", "危大"),
    ("obligation:应进行分阶段验收", "脚手架搭设完成后应分阶段验收（JGJ130-2011 第8章）", "危大"),
    ("obligation:方案应经审批", "专项方案应经施工单位技术负责人、总监理工程师签字（37号令第13条）", "危大"),
]
for oid, desc, _lvl in OBLIGATIONS:
    nodes.append(node(oid, "Obligation", description=desc))

# 阈值 → 义务（TRIGGERS）：危大线触发编制/审批/监测，超规模线追加论证
for th in THRESHOLDS:
    cat, metric, level = th[0], th[1], th[2]
    tid = f"threshold:{cat}:{metric}:{level}:{th[3]}{th[4]}"
    if level == "危大":
        edges.append(edge(tid, "obligation:应编制专项施工方案", "TRIGGERS"))
        edges.append(edge(tid, "obligation:方案应经审批", "TRIGGERS"))
        if cat == "深基坑工程":
            edges.append(edge(tid, "obligation:应进行基坑监测", "TRIGGERS"))
    else:
        edges.append(edge(tid, "obligation:应组织专家论证", "TRIGGERS"))

# ---------------- 5) SUPERSEDES 废止/替代链（"还能不能用"演示） ----------------
SUPERSEDES = [
    ("GB55023-2022_施工脚手架通用规范", "JGJ130-2011_建筑施工扣件式钢管脚手架安全技术规范",
     "脚手架强条以 GB55023 为准（demo 示例，全量由 abolished_clauses.json 驱动）"),
    ("GB55032-2022_建筑与市政工程施工质量控制通用规范", "JGJ46-2005_施工现场临时用电安全技术规范",
     "JGJ46-2005 废止，由 GB55024 系列替代（demo 示例）"),
]
for frm, to, note in SUPERSEDES:
    nodes.append(node(frm, "Standard", full_name=frm, level="强制性国标", demo_note=note))
    edges.append(edge(frm, to, "SUPERSEDES", note=note))

# ---------------- 6) REFERENCES_CLAUSE 条文/附录跳转 ----------------
# 语料实测：「附录X」319 条、「第 x.y.z 条」445 条可抽（附录 clause_no 错标需先修）
REFERENCES_CLAUSE = [
    ("JGJ130-2011_建筑施工扣件式钢管脚手架安全技术规范::6.2.4",
     "JGJ130-2011_建筑施工扣件式钢管脚手架安全技术规范::附录B",
     "架高超7m 连墙件设置依据附录B（demo 示例）"),
]
for frm, to, note in REFERENCES_CLAUSE:
    nodes.append(node(frm, "Clause", text="架高超7m时，连墙件应按两步三跨或两步两跨设置，并宜采用梅花形布置。"))
    nodes.append(node(to, "Clause", text="（附录B：连墙件布置表）", is_appendix=True))
    edges.append(edge(frm, to, "REFERENCES_CLAUSE", note=note))

# ---------------- 7) 复用既有 demo 的骨架（做 2 跳示例的锚点） ----------------
BASE = json.load(open(os.path.join(ROOT, "data", "demo_graph.json"),
                      encoding="utf-8"))
# 只并入 Standard/HazardCategory/Entity/Clause 骨架，避免与新增节点 id 冲突
base_ids = {n["id"] for n in nodes}
for n in BASE["nodes"]:
    if n["id"] not in base_ids:
        nodes.append(n)
        base_ids.add(n["id"])
for e in BASE["edges"]:
    if (e["from"], e["to"], e["type"]) not in {(x["from"], x["to"], x["type"]) for x in edges}:
        edges.append(e)

g = {"nodes": nodes, "edges": edges, "meta": {
    "purpose": "GraphRAG demo v2 图谱（术语归一 + 阈值链 + 废止链 + 条文跳转）",
    "node_labels": sorted({n["label"] for n in nodes}),
    "rel_types": sorted({e["type"] for e in edges}),
}}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(g, f, ensure_ascii=False, indent=1)

from collections import Counter  # noqa: E402
print("输出:", OUT)
print("节点:", dict(Counter(n["label"] for n in nodes)))
print("关系:", dict(Counter(e["type"] for e in edges)))
print("三元组示例:")
for e in edges:
    if e["type"] in ("ALIAS_OF", "HAS_THRESHOLD", "TRIGGERS", "SUPERSEDES",
                     "REFERENCES_CLAUSE", "HAS_METRIC"):
        print(f"   ({e['from']}) -[{e['type']}]-> ({e['to']})")
