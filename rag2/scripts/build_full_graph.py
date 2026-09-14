"""构建 GraphRAG 全量图谱：81 部规范 + 13661 条款 + 引用/危大/上位法关系。

节点：
- Standard（81）：{id, name, code, level}
- Clause（13661）：{id, name, source, clause_no, chapter, text(截断)}
- HazardCategory（12）：危大工程类别
- Entity（~30）：关键工序/设备/参数

关系：
- Clause -[:BELONGS_TO]-> Standard
- Clause -[:REFERENCES]-> Standard       （条款正文引用其他规范，增强编号匹配）
- HazardCategory -[:REGULATED_BY]-> Standard （由条款关键词统计反推 + 核心映射）
- Clause -[:COVERS]-> HazardCategory      （条款关键词命中危大类别）
- Standard -[:HIERARCHY]-> Standard       （强制性国标 GB550xx -> 相关行业/国标）
- Clause -[:MENTIONS]-> Entity            （条款命中关键实体）

用法：
  python scripts/build_full_graph.py --build            # 生成 data/graphrag/full_graph.json
  python scripts/build_full_graph.py --load             # 写入 Neo4j（需服务在跑）
  python scripts/build_full_graph.py --build --load
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import data_dir  # noqa: E402
from src.graphrag.schema import (BELONGS_TO, CLS, COVERS, ENT, HAZ, HIERARCHY,  # noqa: E402
                                 MENTIONS, REFERENCES, REGULATED_BY, STD)

# ---------- 规范层级 ----------
def std_level(sid: str) -> str:
    code = sid.split("_")[0]
    if re.match(r"^GB\s?55", code):
        return "强制性国标"
    if re.match(r"^GB", code):
        return "国家标准"
    if re.match(r"^JGJ", code):
        return "行业标准"
    if re.match(r"^DG", code):
        return "地方标准"
    return "其他"


# ---------- 编号解析与匹配 ----------
def parse_code(code: str) -> tuple[str, tuple[int, ...]]:
    """GB55037-2022 -> (GB,(55037)); DG-TJ08-61-2018 -> (DGTJ,(8,61)); JGJ-T46-2024 -> (JGJ,(46))"""
    code = code.split("_")[0]
    code = re.sub(r"-\d{4}$", "", code)          # 去末尾年份
    letters = "".join(re.findall(r"[A-Za-z]+", code)).upper()
    let = letters.replace("T", "")               # JGJT->JGJ, GBT->GB
    nums = tuple(int(x) for x in re.findall(r"\d+", code))
    return let, nums


REF_PAT = re.compile(r"((?:GB|JGJ|DGJ|DG-TJ|CJJ)[/\-\s]?T?\s?\d{2,6}(?:[-—]\d{4})?)", re.IGNORECASE)


# ---------- 危大工程类别（住建部 37 号令 + 上海地标） ----------
HAZARDS = {
    "基坑工程": ["基坑", "支护结构", "围护", "土方开挖", "放坡"],
    "模板工程及支撑体系": ["模板", "支架", "支撑架", "支撑结构"],
    "起重吊装及安装拆卸工程": ["起重", "吊装", "塔式起重机", "塔吊", "施工升降机", "物料提升机", "龙门架"],
    "脚手架工程": ["脚手架", "悬挑架", "附着式升降", "落地架"],
    "拆除工程": ["拆除", "爆破"],
    "暗挖工程": ["暗挖", "盾构", "顶管", "隧道", "冻结法"],
    "建筑幕墙安装工程": ["幕墙"],
    "钢结构安装工程": ["钢结构安装", "网架", "索膜", "钢构件安装"],
    "人工挖孔桩工程": ["人工挖孔", "挖孔桩"],
    "装配式构件安装工程": ["预制构件", "装配式", "预制墙板"],
    "降水工程": ["降水", "井点", "疏干", "集水明排", "回灌"],
    "桩基工程": ["灌注桩", "预制桩", "钢管桩", "桩基", "沉桩"],
}

# ---------- 关键实体 ----------
ENTITIES = {
    "基坑开挖": ["基坑开挖", "土方开挖"],
    "支护结构": ["支护结构", "围护墙", "支撑体系", "锚杆", "土钉"],
    "降排水": ["降水", "排水", "井点", "疏干", "回灌"],
    "监测": ["监测", "观测", "测点"],
    "模板支架": ["模板支架", "支撑架", "立杆", "水平杆"],
    "脚手架": ["脚手架", "连墙件", "剪刀撑"],
    "起重设备": ["塔式起重机", "施工升降机", "物料提升机", "汽车起重机"],
    "混凝土": ["混凝土", "浇筑", "养护"],
    "钢筋": ["钢筋", "预应力筋", "箍筋"],
    "钢结构": ["钢结构", "焊缝", "螺栓", "高强螺栓"],
    "安全防护": ["安全帽", "安全带", "安全网", "临边防护", "洞口"],
    "临时用电": ["临时用电", "配电箱", "接地", "漏电"],
    "验收": ["验收", "检验批", "主控项目", "一般项目"],
    "绿色施工": ["扬尘", "噪声", "污水", "废弃物"],
}

# ---------- 上位法（强制性通用规范 -> 相关标准），初始映射可人工校准 ----------
HIERARCHY_MAP = {
    "GB55023-2022_施工脚手架通用规范": [
        "GB51210-2016_建筑施工脚手架安全技术统一标准",
        "JGJ130-2011_建筑施工扣件式钢管脚手架安全技术规范",
        "JGJ166-2016_建筑施工碗扣式钢管脚手架安全技术规范",
        "JGJ202-2010_建筑施工工具式脚手架安全技术规范",
        "JGJ164-2008_建筑施工木脚手架安全技术规范",
        "JGJ254-2011_建筑施工竹脚手架安全技术规范",
        "JGJ-T231-2021_建筑施工承插型盘扣式钢管脚手架安全技术标准",
        "JGJ300-2013_建筑施工临时支撑结构技术规范",
    ],
    "GB55003-2021_建筑与市政地基基础通用规范": [
        "GB51004-2015_建筑地基基础工程施工规范",
        "GB50202-2018_建筑地基基础工程施工质量验收标准",
        "JGJ79-2012_建筑地基处理技术规范",
        "JGJ94-2008_建筑桩基技术规范",
        "JGJ120-2012_建筑基坑支护技术规程",
        "DGJ08-11-2018_地基基础设计标准",
        "DG-TJ08-61-2018_基坑工程技术标准",
        "GB50497-2019_建筑基坑工程监测技术标准",
        "JGJ311-2013_建筑深基坑工程施工安全技术规范",
    ],
    "GB55008-2021_混凝土结构通用规范": [
        "GB50666-2011_混凝土结构工程施工规范",
        "GB50204-2015_混凝土结构工程施工质量验收规范",
    ],
    "GB55006-2021_钢结构通用规范": ["GB50205-2020_钢结构工程施工质量验收标准"],
    "GB55034-2022_建筑与市政施工现场安全卫生与职业健康通用规范": [
        "JGJ33-2012_建筑机械使用安全技术规程",
        "JGJ160-2016_施工现场机械设备检查技术规范",
        "JGJ59-2011_建筑施工安全检查标准",
        "JGJ80-2016_建筑施工高处作业安全技术规范",
        "JGJ147-2016_建筑拆除工程安全技术规范",
        "JGJ276-2012_建筑施工起重吊装工程安全技术规范",
        "JGJ196-2010_建筑施工塔式起重机安装、使用、拆卸安全技术规程",
        "JGJ215-2010_建筑施工升降机安装、使用、拆卸安全技术规程",
        "JGJ88-2010_龙门架及井架物料提升机安全技术规范",
        "GB50870-2013_建筑施工安全技术统一规范",
        "DG-TJ08-903-2022_现场施工安全生产管理标准",
    ],
    "GB55021-2021_既有建筑鉴定与加固通用规范": [
        "GB50550-2010_建筑结构加固工程施工质量验收规范",
        "GB55022-2021_既有建筑维护与改造通用规范",
    ],
    "GB55037-2022_建筑防火通用规范": [
        "GB55036-2022_消防设施通用规范",
        "GB25506-2010_消防控制室通用技术要求",
    ],
    "GB55032-2022_建筑与市政工程施工质量控制通用规范": [
        "GB50300-2013_建筑工程施工质量验收统一标准",
        "GB50202-2018_建筑地基基础工程施工质量验收标准",
        "GB50204-2015_混凝土结构工程施工质量验收规范",
        "GB50205-2020_钢结构工程施工质量验收标准",
        "GB50550-2010_建筑结构加固工程施工质量验收规范",
    ],
    "GB55018-2021_工程测量通用规范": [
        "JGJ8-2016_建筑变形测量规范",
        "DG-TJ08-2001-2016_基坑工程施工监测规程",
    ],
    "GB55017-2021_工程勘察通用规范": ["DGJ08-11-2018_地基基础设计标准"],
    "GB55009-2021_燃气工程项目规范": ["GB29550-2013_民用建筑燃气安全技术条件"],
    "GB55002-2021_建筑与市政工程抗震通用规范": [
        "DGJ08-11-2018_地基基础设计标准",
        "GB51004-2015_建筑地基基础工程施工规范",
    ],
}


def build() -> dict:
    clauses = [json.loads(l) for l in open(data_dir("corpus", "clauses.jsonl"), encoding="utf-8") if l.strip()]
    std_ids = sorted({c["metadata"]["source"] for c in clauses})
    std_set = set(std_ids)

    # 规范编号 -> 规范 id 索引
    key2std: dict[tuple, list[str]] = defaultdict(list)
    for sid in std_ids:
        key2std[parse_code(sid)].append(sid)

    def match_ref(ref: str) -> list[str]:
        let, nums = parse_code(ref)
        if not nums:
            return []
        hits = []
        for (klet, knums), sids in key2std.items():
            if klet == let and len(nums) <= len(knums) and knums[: len(nums)] == nums:
                hits.extend(sids)
        return hits

    nodes, edges = [], []
    # 规范节点
    for sid in std_ids:
        nodes.append({"id": sid, "label": STD, "name": sid,
                      "code": sid.split("_")[0], "level": std_level(sid)})
    # 危大类别
    for hid, _kws in HAZARDS.items():
        nodes.append({"id": hid, "label": HAZ, "name": hid})
    # 实体
    for eid in ENTITIES:
        nodes.append({"id": eid, "label": ENT, "name": eid})

    # 条款节点 + 三类关系
    haz_std = defaultdict(Counter)      # hazard -> Counter(standard)
    n_ref_edge = 0
    for c in clauses:
        cid, sid = c["id"], c["metadata"]["source"]
        txt = c.get("text", "")
        nodes.append({
            "id": cid, "label": CLS, "name": cid, "source": sid,
            "clause_no": c["metadata"].get("clause_no", ""),
            "chapter": c["metadata"].get("chapter", ""),
            "text": txt[:400],
        })
        edges.append({"from": cid, "to": sid, "type": BELONGS_TO})
        # 引用
        for ref in set(REF_PAT.findall(txt)):
            for target in match_ref(ref):
                if target != sid:
                    edges.append({"from": cid, "to": target, "type": REFERENCES})
                    n_ref_edge += 1
        # 危大类别
        for hid, kws in HAZARDS.items():
            if any(k in txt for k in kws):
                edges.append({"from": cid, "to": hid, "type": COVERS})
                haz_std[hid][sid] += 1
        # 实体
        for eid, kws in ENTITIES.items():
            if any(k in txt for k in kws):
                edges.append({"from": cid, "to": eid, "type": MENTIONS})

    # 危大类别 REGULATED_BY 规范（条款数 >= 5 即认为该规范监管该危大类别）
    n_reg = 0
    for hid, cnt in haz_std.items():
        for sid, n in cnt.items():
            if n >= 5:
                edges.append({"from": hid, "to": sid, "type": REGULATED_BY})
                n_reg += 1

    # 上位法
    n_hier = 0
    for sup, subs in HIERARCHY_MAP.items():
        if sup not in std_set:
            continue
        for sub in subs:
            if sub in std_set:
                edges.append({"from": sup, "to": sub, "type": HIERARCHY})
                n_hier += 1

    g = {"nodes": nodes, "edges": edges,
         "stats": {"n_std": len(std_ids), "n_clause": len(clauses),
                   "n_ref_edge": n_ref_edge, "n_regulated": n_reg, "n_hierarchy": n_hier}}
    return g


def load_neo4j(g: dict, uri="bolt://127.0.0.1:7687", user="neo4j", pwd="neo4j123456") -> None:
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(uri, auth=(user, pwd))
    drv.verify_connectivity()
    node_batch, edge_batch = 500, 500
    with drv.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        # 索引
        for lab in ["Standard", "Clause", "HazardCategory", "Entity"]:
            s.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{lab}) ON (n.id)")
        # 节点
        nodes = g["nodes"]
        for i in range(0, len(nodes), node_batch):
            chunk = nodes[i:i + node_batch]
            by_label = defaultdict(list)
            for n in chunk:
                by_label[n["label"]].append({k: v for k, v in n.items() if k != "label"})
            for lab, items in by_label.items():
                s.run(f"UNWIND $items AS n CREATE (x:{lab}) SET x = n", items=items)
            print(f"  节点 {min(i+node_batch, len(nodes))}/{len(nodes)}", flush=True)
        # 关系（按类型分组）
        edges = g["edges"]
        by_type = defaultdict(list)
        for e in edges:
            by_type[e["type"]].append({"a": e["from"], "b": e["to"]})
        for rtype, items in by_type.items():
            for i in range(0, len(items), edge_batch):
                chunk = items[i:i + edge_batch]
                s.run(
                    f"UNWIND $items AS e MATCH (a {{id:e.a}}), (b {{id:e.b}}) "
                    f"CREATE (a)-[:{rtype}]->(b)",
                    items=chunk,
                )
            print(f"  关系 {rtype}: {len(items)}", flush=True)
    drv.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--load", action="store_true")
    a = ap.parse_args()

    out = data_dir("graphrag") / "full_graph.json"
    if a.build or not a.load:
        g = build()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(g, f, ensure_ascii=False)
        n_lab = Counter(n["label"] for n in g["nodes"])
        n_rel = Counter(e["type"] for e in g["edges"])
        print(f"全量图谱 -> {out}")
        print(f"节点 {dict(n_lab)} 共 {len(g['nodes'])}")
        print(f"关系 {dict(n_rel)} 共 {len(g['edges'])}")
    if a.load:
        g = json.load(open(out, encoding="utf-8"))
        print("写入 Neo4j ...")
        load_neo4j(g)
        print("写入完成")


if __name__ == "__main__":
    main()
