"""Neo4j 存储层（GraphRAG 独立 demo 版）。

设计（用户定稿）：**演示链路强制走 Neo4j**——连接失败直接抛 `Neo4jUnavailable`，
不静默回退内存/JSON。JSON 仅作为「建图素材」，由 scripts/load_neo4j.py 一次性写入 Neo4j。

用法：
  # 写入 demo 图谱（需 Neo4j 在跑）
  python -m src.neo4j_store --load
  # 子图扩展（Cypher 多跳）
  python -m src.neo4j_store --query "深基坑工程" --hops 2
  # 打印图库统计
  python -m src.neo4j_store --stats
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
GRAPH_ROOT = HERE.parents[1]                       # .../graphrag
if str(GRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPH_ROOT))

from src.schema import ALL_NODE_LABELS, ALL_REL_TYPES  # noqa: E402

DATA_DIR = GRAPH_ROOT / "data"

# 连接参数（demo 默认，可用环境变量覆盖）
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD", "neo4j123456")


class Neo4jUnavailable(RuntimeError):
    """Neo4j 驱动缺失 / 服务不可达。demo 链路要求显式失败，不降级。"""


class Neo4jStore:
    """Neo4j 图库（demo）。构造即校验连通性，失败抛异常。"""

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER,
                 password: str = NEO4J_PASS,
                 graph_file: str = "demo_graph_v2.json"):
        self.uri, self.user, self.password = uri, user, password
        self.graph_file = DATA_DIR / graph_file
        self.driver = None
        self._connect()

    # ---------------------------------------------------------------- 连接
    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase  # noqa: PLC0415
        except Exception as e:  # noqa: BLE001
            raise Neo4jUnavailable(
                f"未安装 neo4j Python 驱动（pip install neo4j）：{e}") from e
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
        except Exception as e:  # noqa: BLE001
            self.driver = None
            raise Neo4jUnavailable(
                f"Neo4j 不可达（{self.uri}）：{type(e).__name__}: {e}\n"
                f"请先启动 Neo4j：python neo4j-community-4.4.8/start_neo4j.py --background"
            ) from e

    def available(self) -> bool:
        return self.driver is not None

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    # ---------------------------------------------------------------- 写入
    def load_graph(self, clear: bool = True) -> dict:
        """把 data/demo_graph_v2.json 写入 Neo4j（默认先清空）。返回统计。"""
        if not self.graph_file.exists():
            raise FileNotFoundError(f"建图素材不存在：{self.graph_file}")
        g = json.loads(self.graph_file.read_text(encoding="utf-8"))
        with self.driver.session() as s:
            if clear:
                s.run("MATCH (n) DETACH DELETE n")
            for n in g["nodes"]:
                lbl = n["label"]
                if lbl not in ALL_NODE_LABELS:
                    raise ValueError(f"非法节点类型：{lbl}（见 src/schema.py）")
                props = {k: v for k, v in n.items() if k != "label"}
                s.run(f"MERGE (n:{lbl} {{id:$id}}) SET n += $props",
                      id=n["id"], props=props)
            written = 0
            for e in g["edges"]:
                if e["type"] not in ALL_REL_TYPES:
                    raise ValueError(f"非法关系类型：{e['type']}（见 src/schema.py）")
                s.run(
                    f"MATCH (a {{id:$a}}), (b {{id:$b}}) MERGE (a)-[:{e['type']}]->(b)",
                    a=e["from"], b=e["to"],
                )
                written += 1
        return {"nodes": len(g["nodes"]), "edges": written,
                "labels": dict(Counter(n["label"] for n in g["nodes"])),
                "rel_types": dict(Counter(e["type"] for e in g["edges"]))}

    # ---------------------------------------------------------------- 统计
    def stats(self) -> dict:
        with self.driver.session() as s:
            nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            by_label = {r["label"]: r["c"] for r in s.run(
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS c "
                "ORDER BY c DESC").data()}
            by_rel = {r["t"]: r["c"] for r in s.run(
                "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c "
                "ORDER BY c DESC").data()}
        return {"nodes": nodes, "edges": rels,
                "labels": by_label, "rel_types": by_rel}

    # ---------------------------------------------------------------- 全图（可视化用）
    def fetch_graph(self, limit: int = 3000) -> dict:
        """返回全图 {nodes, edges}（供前端可视化）。"""
        with self.driver.session() as s:
            nrec = s.run(
                "MATCH (n) RETURN n.id AS id, n.name AS name, labels(n) AS labels, "
                "properties(n) AS props LIMIT $lim", lim=limit).data()
            erec = s.run(
                "MATCH (a)-[r]->(b) RETURN a.id AS a, b.id AS b, type(r) AS t "
                "LIMIT $lim", lim=limit * 3).data()
        nodes = []
        for r in nrec:
            props = dict(r.get("props") or {})
            props.pop("id", None)
            props.pop("name", None)
            nodes.append({"id": r["id"], "name": r["name"],
                          "label": (r["labels"] or ["Entity"])[0], "props": props})
        edges = [{"from": r["a"], "to": r["b"], "type": r["t"]} for r in erec]
        return {"nodes": nodes, "edges": edges}

    # ---------------------------------------------------------------- 子图扩展
    def expand_subgraph(self, seed: str, hops: int = 2, limit: int = 400) -> dict:
        """从 seed 节点出发做 n 跳扩展（Cypher），返回 {nodes, edges}。"""
        hops = max(1, min(int(hops), 4))
        query = (
            "MATCH (n) WHERE n.id CONTAINS $seed OR n.name CONTAINS $seed "
            f"MATCH p=(n)-[*1..{hops}]-(m) RETURN p LIMIT {int(limit)}"
        )
        with self.driver.session() as s:
            recs = s.run(query, seed=seed).data()
        nodes: dict[str, dict] = {}
        edges: dict[tuple, dict] = {}
        for r in recs:
            path = r["path"]
            for node in _path_nodes(path):
                nid = _node_id(node)
                if nid and nid not in nodes:
                    nodes[nid] = {
                        "id": nid,
                        "name": _node_prop(node, "name") or nid,
                        "label": _node_label(node),
                        "props": {k: v for k, v in dict(node).items()
                                  if k not in ("id", "name")},
                    }
            for a_id, b_id, rtype in _path_edges(path):
                if a_id and b_id:
                    edges.setdefault((a_id, rtype, b_id),
                                     {"from": a_id, "to": b_id, "type": rtype})
        return {"seed": seed, "hops": hops,
                "nodes": list(nodes.values()), "edges": list(edges.values())}

    def hazard_standards(self, category: str) -> list[str]:
        """图谱路：HazardCategory -REGULATED_BY-> Standard 白名单（范围限定）。"""
        with self.driver.session() as s:
            recs = s.run(
                "MATCH (h:HazardCategory)-[:REGULATED_BY]->(st:Standard) "
                "WHERE h.id CONTAINS $cat OR h.name CONTAINS $cat "
                "RETURN DISTINCT coalesce(st.name, st.id) AS name ORDER BY name",
                cat=category).data()
        return [r["name"] for r in recs]

    def hazard_thresholds(self, category: str) -> list[dict]:
        """阈值链：HazardCategory -HAS_METRIC-> Metric -HAS_THRESHOLD-> Threshold。"""
        with self.driver.session() as s:
            recs = s.run(
                "MATCH (h:HazardCategory)-[:HAS_METRIC]->(m:Metric)"
                "-[:HAS_THRESHOLD]->(t:Threshold) "
                "WHERE h.id CONTAINS $cat OR h.name CONTAINS $cat "
                "RETURN m.name AS metric, m.unit AS unit, t.level AS level, "
                "t.value AS value, t.source AS source ORDER BY metric, value",
                cat=category).data()
        return recs


# ---------------------------------------------------------------- path 解析（兼容 neo4j 4.x/5.x 驱动）
def _path_nodes(path):
    if hasattr(path, "nodes"):
        return list(path.nodes)
    if isinstance(path, dict):
        return list(path.get("nodes", []))
    if isinstance(path, list):
        return [x for x in path if isinstance(x, (dict,)) or hasattr(x, "get")]
    return []


def _path_edges(path):
    """产出 (start_id, end_id, rel_type)。"""
    out = []
    rels = []
    if hasattr(path, "relationships"):
        rels = list(path.relationships)
    elif isinstance(path, dict):
        rels = list(path.get("relationships", []))
    for rel in rels:
        rtype = getattr(rel, "type", None) or (
            rel.get("type") if isinstance(rel, dict) else None) or ""
        ends = getattr(rel, "nodes", None)
        if ends and len(ends) == 2:
            a, b = ends
        else:
            a = getattr(rel, "start_node", None)
            b = getattr(rel, "end_node", None)
        out.append((_node_id(a), _node_id(b), rtype))
    # 兼容旧版扁平 list（node, rel_type_str, node, ...）
    if not rels and isinstance(path, list):
        i = 0
        while i + 2 < len(path):
            if isinstance(path[i], str) or hasattr(path[i], "type"):
                a, rt, b = path[i - 1], path[i], path[i + 1]
                if not isinstance(rt, str):
                    rt = getattr(rt, "type", "")
                out.append((_node_id(a), _node_id(b), rt))
                i += 2
            else:
                i += 1
    return out


def _node_id(node):
    if node is None:
        return None
    if isinstance(node, dict):
        return node.get("id")
    try:
        return node.get("id")
    except Exception:  # noqa: BLE001
        return None


def _node_label(node):
    try:
        labels = list(node.labels) if hasattr(node, "labels") else (
            node.get("labels") if isinstance(node, dict) else [])
        return labels[0] if labels else "Entity"
    except Exception:  # noqa: BLE001
        return "Entity"


def _node_prop(node, key):
    try:
        return node.get(key)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true", help="写入 demo 图谱")
    ap.add_argument("--query", default=None, help="子图扩展 seed")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--stats", action="store_true", help="打印图库统计")
    a = ap.parse_args()

    try:
        store = Neo4jStore()
    except Neo4jUnavailable as e:
        print(f"[x] {e}", file=sys.stderr)
        print("    手动启动： python neo4j-community-4.4.8/start_neo4j.py --background", file=sys.stderr)
        print("    一键启动： .\\graphrag\\start_demo.ps1", file=sys.stderr)
        return 2

    print(f"[neo4j] 已连接 {store.uri}")
    if a.load:
        print("[neo4j] 写入：", store.load_graph())
    if a.stats:
        print("[neo4j] 统计：", store.stats())
    if a.query:
        sub = store.expand_subgraph(a.query, a.hops)
        labels = Counter(n["label"] for n in sub["nodes"])
        print(f"子图扩展 [{a.query}] hops={a.hops}: "
              f"{len(sub['nodes'])} 节点 {dict(labels)} / {len(sub['edges'])} 关系")
        for n in sub["nodes"]:
            print(f"  [{n['label'][:4]}] {n['name'][:60]}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
