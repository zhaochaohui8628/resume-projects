"""构建 GraphRAG demo 图谱（危大工程多标准交叉判定示例）。

用真实语料 clause_id 构建一个**小而完整**的图谱，覆盖"深基坑工程""模板支撑工程"
两个危大类别的多标准交叉关系，验证双引擎检索链路。

场景：问"深基坑开挖前要做哪些准备？"时，单路 RAG 可能只命中基坑规范，
但实际涉及：JGJ120（支护规程）+ JGJ311（深基坑安全）+ DG-TJ08-61（上海地标）+
DG-TJ08-2077（危大工程管理）+ GB55023（脚手架通用）等多部规范。
图谱通过 HazardCategory -> REGULATED_BY -> Standard -> Clause 的路径，
把"盲人摸象"变成"全图视角"。

输出：
- data/graphrag/demo_graph.json  图谱序列化（Neo4j 导入格式，也供 networkx 调试）
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import data_dir  # noqa: E402
from src.graphrag.schema import *  # noqa: E402,F403


# ---- 1. 规范节点（真实存在的 81 部规范中选取）----
STANDARDS = {
    "JGJ311-2013_建筑深基坑工程施工安全技术规范": {"full_name": "建筑深基坑工程施工安全技术规范", "level": "行业标准"},
    "JGJ120-2012_建筑基坑支护技术规程": {"full_name": "建筑基坑支护技术规程", "level": "行业标准"},
    "DG-TJ08-61-2018_基坑工程技术标准": {"full_name": "基坑工程技术标准（上海市）", "level": "地方标准"},
    "DG-TJ08-2077-2021_危险性较大的分部分项工程安全管理标准": {"full_name": "危险性较大的分部分项工程安全管理标准（上海市）", "level": "地方标准"},
    "GB51210-2016_建筑施工脚手架安全技术统一标准": {"full_name": "建筑施工脚手架安全技术统一标准", "level": "国家标准"},
    "GB55023-2022_施工脚手架通用规范": {"full_name": "施工脚手架通用规范", "level": "强制性国标"},
    "GB51004-2015_建筑地基基础工程施工规范": {"full_name": "建筑地基基础工程施工规范", "level": "国家标准"},
    "JGJ130-2011_建筑施工扣件式钢管脚手架安全技术规范": {"full_name": "建筑施工扣件式钢管脚手架安全技术规范", "level": "行业标准"},
}

# ---- 2. 危大类别节点 ----
HAZARDS = {
    "深基坑工程": "开挖深度≥3m 或未超过3m但地质条件复杂的基坑",
    "模板支撑工程": "搭设高度≥5m、跨度≥10m、施工总荷载≥10kN/m²等条件的模板支架",
    "起重吊装工程": "采用非常规起重设备且单件起吊重量≥10kN 的吊装",
    "脚手架工程": "搭设高度≥24m 的落地式脚手架等",
}

# ---- 3. 实体节点（关键工序/参数）----
ENTITIES = {
    "基坑开挖": "基坑土方开挖工序",
    "支护结构": "基坑围护/支撑结构",
    "降排水": "基坑降水与排水",
    "监测": "基坑变形监测",
    "模板支架": "模板支撑架体",
    "连墙件": "脚手架与结构连接件",
}

# ---- 4. 条款节点（真实 clause_id，从语料验证）----
# 每个条款: (clause_id, 所属规范, 文本摘要)
CLAUSES = [
    ("JGJ311-2013_建筑深基坑工程施工安全技术规范::7.1.2", "JGJ311-2013_建筑深基坑工程施工安全技术规范",
     "降排水施工方案应包含各种泵的扬程、功率，排水管路尺寸、材料、路线，水箱位置、尺寸，电力配置等。"),
    ("JGJ120-2012_建筑基坑支护技术规程::3.3.1", "JGJ120-2012_建筑基坑支护技术规程",
     "支护结构选型时，应综合考虑下列因素:1基坑深度;2土的性状及地下水条件;3基坑周边环境对基坑变形的承受能力及支护结构失效的后果;..."),
    ("DG-TJ08-61-2018_基坑工程技术标准::2.1.2", "DG-TJ08-61-2018_基坑工程技术标准",
     "基坑工程为挖除建(构)筑物地下结构处的土方，保证主体地下结构的安全施工及保护基坑周边环境而采取的围护、支撑、降水、加固、挖土与回填等工程措施的总称。"),
    ("DG-TJ08-2077-2021_危险性较大的分部分项工程安全管理标准::8.4.7", "DG-TJ08-2077-2021_危险性较大的分部分项工程安全管理标准",
     "监理单位发现施工单位未按照专项施工方案施工的，应要求其进行整改；情节严重的，应要求其暂停施工，并及时报告建设单位；..."),
    ("GB51210-2016_建筑施工脚手架安全技术统一标准::3.2.3", "GB51210-2016_建筑施工脚手架安全技术统一标准",
     "脚手架结构重要性系数，应按表3.2.3 的规定取值。承载能力极限状态设计结构重要性系数安全等级I级1.1、II级1.0。"),
    ("GB55023-2022_施工脚手架通用规范::5.2.1", "GB55023-2022_施工脚手架通用规范",
     "脚手架应按顺序搭设...落地作业脚手架、悬挑脚手架的搭设应与主体结构工程施工同步，一次搭设高度不应超过最上层连墙件2步，且自由高度不应大于4m。"),
    ("GB51004-2015_建筑地基基础工程施工规范::5.3.2", "GB51004-2015_建筑地基基础工程施工规范",
     "钢筋混凝土条形基础施工应符合下列规定...混凝土宜分段分层连续浇筑，每层厚度宜为300mm~500mm。"),
    ("JGJ130-2011_建筑施工扣件式钢管脚手架安全技术规范::6.2.4", "JGJ130-2011_建筑施工扣件式钢管脚手架安全技术规范",
     "架高超7m 时，连墙件应按两步三跨或两步两跨设置，并宜采用梅花形布置。"),
]


def build_demo_graph() -> dict:
    """构建 demo 图谱（dict 形式：nodes + edges）。"""
    nodes = []
    edges = []

    # 规范节点
    for sid, meta in STANDARDS.items():
        nodes.append({"id": sid, "label": STD, "name": sid, **meta})

    # 危大类别
    for hid, desc in HAZARDS.items():
        nodes.append({"id": hid, "label": HAZ, "name": hid, "description": desc})

    # 实体
    for eid, desc in ENTITIES.items():
        nodes.append({"id": eid, "label": ENT, "name": eid, "description": desc})

    # 条款节点 + BELONGS_TO
    clause_std = {}
    for cid, sid, _ in CLAUSES:
        nodes.append({"id": cid, "label": CLS, "name": cid, "text": _})
        clause_std[cid] = sid
        edges.append({"from": cid, "to": sid, "type": BELONGS_TO})

    # 危大类别 REGULATED_BY 规范（多标准交叉的核心）
    regulated = {
        "深基坑工程": ["JGJ311-2013_建筑深基坑工程施工安全技术规范",
                       "JGJ120-2012_建筑基坑支护技术规程",
                       "DG-TJ08-61-2018_基坑工程技术标准",
                       "DG-TJ08-2077-2021_危险性较大的分部分项工程安全管理标准"],
        "模板支撑工程": ["GB51210-2016_建筑施工脚手架安全技术统一标准",
                         "GB55023-2022_施工脚手架通用规范",
                         "JGJ130-2011_建筑施工扣件式钢管脚手架安全技术规范",
                         "DG-TJ08-2077-2021_危险性较大的分部分项工程安全管理标准"],
        "起重吊装工程": ["DG-TJ08-2077-2021_危险性较大的分部分项工程安全管理标准"],
        "脚手架工程": ["GB51210-2016_建筑施工脚手架安全技术统一标准",
                       "GB55023-2022_施工脚手架通用规范",
                       "JGJ130-2011_建筑施工扣件式钢管脚手架安全技术规范"],
    }
    for hid, stds in regulated.items():
        for sid in stds:
            edges.append({"from": hid, "to": sid, "type": REGULATED_BY})

    # 条款 COVERS 危大类别
    covers = {
        "JGJ311-2013_建筑深基坑工程施工安全技术规范::7.1.2": "深基坑工程",
        "JGJ120-2012_建筑基坑支护技术规程::3.3.1": "深基坑工程",
        "DG-TJ08-61-2018_基坑工程技术标准::2.1.2": "深基坑工程",
        "DG-TJ08-2077-2021_危险性较大的分部分项工程安全管理标准::8.4.7": "模板支撑工程",
        "GB51210-2016_建筑施工脚手架安全技术统一标准::3.2.3": "模板支撑工程",
        "GB55023-2022_施工脚手架通用规范::5.2.1": "模板支撑工程",
        "JGJ130-2011_建筑施工扣件式钢管脚手架安全技术规范::6.2.4": "脚手架工程",
        "GB51004-2015_建筑地基基础工程施工规范::5.3.2": "深基坑工程",
    }
    for cid, hid in covers.items():
        edges.append({"from": cid, "to": hid, "type": COVERS})

    # 条款 MENTIONS 实体
    mentions = {
        "JGJ311-2013_建筑深基坑工程施工安全技术规范::7.1.2": ["降排水"],
        "JGJ120-2012_建筑基坑支护技术规程::3.3.1": ["支护结构", "基坑开挖"],
        "DG-TJ08-61-2018_基坑工程技术标准::2.1.2": ["支护结构", "降排水", "基坑开挖"],
        "GB55023-2022_施工脚手架通用规范::5.2.1": ["模板支架", "连墙件"],
        "JGJ130-2011_建筑施工扣件式钢管脚手架安全技术规范::6.2.4": ["连墙件"],
        "GB51210-2016_建筑施工脚手架安全技术统一标准::3.2.3": ["模板支架"],
    }
    for cid, ents in mentions.items():
        for e in ents:
            edges.append({"from": cid, "to": e, "type": MENTIONS})

    # 规范层级/上位法：强制性国标 > 国标 > 行业 > 地方
    hierarchy = [
        ("GB55023-2022_施工脚手架通用规范", "GB51210-2016_建筑施工脚手架安全技术统一标准", HIERARCHY),
        ("GB51210-2016_建筑施工脚手架安全技术统一标准", "JGJ130-2011_建筑施工扣件式钢管脚手架安全技术规范", HIERARCHY),
        ("JGJ120-2012_建筑基坑支护技术规程", "DG-TJ08-61-2018_基坑工程技术标准", HIERARCHY),
        ("JGJ311-2013_建筑深基坑工程施工安全技术规范", "DG-TJ08-61-2018_基坑工程技术标准", HIERARCHY),
    ]
    for a, b, rel in hierarchy:
        edges.append({"from": a, "to": b, "type": rel})

    # 条款 REFERENCES 规范（如"应符合GB50007"）
    references = [
        ("JGJ120-2012_建筑基坑支护技术规程::3.3.1", "GB51004-2015_建筑地基基础工程施工规范", REFERENCES),
        ("JGJ311-2013_建筑深基坑工程施工安全技术规范::7.1.2", "GB51004-2015_建筑地基基础工程施工规范", REFERENCES),
    ]
    for a, b, rel in references:
        edges.append({"from": a, "to": b, "type": rel})

    graph = {"nodes": nodes, "edges": edges}
    return graph


def main() -> None:
    graph = build_demo_graph()
    out = data_dir("graphrag") / "demo_graph.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    n_by_label = {}
    for n in graph["nodes"]:
        n_by_label[n["label"]] = n_by_label.get(n["label"], 0) + 1
    e_by_type = {}
    for e in graph["edges"]:
        e_by_type[e["type"]] = e_by_type.get(e["type"], 0) + 1
    print(f"demo 图谱 -> {out}")
    print(f"节点: {n_by_label}")
    print(f"关系: {e_by_type}")


if __name__ == "__main__":
    main()
