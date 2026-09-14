"""Neo4j 存储层：写入 demo 图谱 + 图谱子图扩展查询。

设计：Neo4j 服务器可用时走 Cypher；不可用（未启动/未装）时降级到内存图
（networkx 或纯 dict BFS），保证 demo 链路不依赖服务器也能跑通。

用法：
  # 写入（需 Neo4j 运行）
  python -m src.graphrag.neo4j_store --load
  # 查询（图谱子图扩展）
  python -m src.graphrag.neo4j_store --query "深基坑工程"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.paths import data_dir  # noqa: E402
from src.graphrag.schema import *  # noqa: E402,F403

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "neo4j123456"  # demo 用，生产改环境变量


class Neo4jStore:
    """Neo4j 存储。连接失败时 backend='memory' 降级。"""

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER,
                 password: str = NEO4J_PASS):
        self.uri, self.user, self.password = uri, user, password
        self.backend = self._connect()

    def _connect(self) -> str:
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            return "neo4j"
        except Exception as e:  # noqa: BLE001
            print(f"[Neo4jStore] Neo4j 不可用（{type(e).__name__}: {e}），降级为 memory 后端")
            self.driver = None
            self._mem_graph = self._load_demo_memory()
            return "memory"

    # ---------- memory 后端 ----------
    def _load_demo_memory(self):
        """加载 demo_graph.json 为内存邻接表（dict: id -> {label, edges}）。"""
        p = data_dir("graphrag") / "demo_graph.json"
        g = json.load(open(p, encoding="utf-8"))
        nodes = {n["id"]: dict(n) for n in g["nodes"]}
        adj = {nid: [] for nid in nodes}
        for e in g["edges"]:
            adj[e["from"]].append((e["to"], e["type"]))
            adj[e["to"]].append((e["from"], e["type"]))  # 无向扩展
        return {"nodes": nodes, "adj": adj}

    # ---------- 公共：写入 ----------
    def load_demo(self) -> int:
        """把 demo_graph.json 写入图库，返回关系数。"""
        g = json.load(open(data_dir("graphrag") / "demo_graph.json", encoding="utf-8"))
        if self.backend == "neo4j":
            return self._load_neo4j(g)
        # memory 后端已加载，无需操作
        n = sum(len(v) for v in self._mem_graph["adj"]) // 2
        print(f"[memory] demo 图谱已加载（{len(self._mem_graph['nodes'])} 节点 / {n} 关系）")
        return n

    def _load_neo4j(self, g: dict) -> int:
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")  # 清空
            # 节点
            for n in g["nodes"]:
                label = n["label"]
                props = {k: v for k, v in n.items() if k not in ("label",)}
                s.run(
                    f"CREATE (n:{label} {{id:$id, name:$name}}) SET n += $props",
                    id=n["id"], name=n["name"], props=props,
                )
            # 关系
            for e in g["edges"]:
                s.run(
                    f"MATCH (a {{id:$a}}), (b {{id:$b}}) "
                    f"CREATE (a)-[:{e['type']}]->(b)",
                    a=e["from"], b=e["to"],
                )
        print(f"[neo4j] demo 图谱写入完成（{len(g['nodes'])} 节点 / {len(g['edges'])} 关系）")
        return len(g["edges"])

    # ---------- 公共：图谱子图扩展 ----------
    def _label_of(self, nid: str) -> str:
        """从 demo 图谱映射节点 label（Neo4j 标签不在属性里）。"""
        if not hasattr(self, "_label_map"):
            self._label_map = {}
            p = data_dir("graphrag") / "demo_graph.json"
            if p.exists():
                g = json.load(open(p, encoding="utf-8"))
                for n in g["nodes"]:
                    self._label_map[n["id"]] = n["label"]
        return self._label_map.get(nid, "Entity")

    def expand_subgraph(self, seed: str, hops: int = 2) -> list[dict]:
        """从 seed 节点出发做 BFS 扩展，返回子图（节点+边）。

        用于双引擎检索：query 先识别实体/危大类别 → 图谱扩展出相关规范/条款 →
        与向量检索结果融合。
        """
        if self.backend == "neo4j":
            return self._expand_neo4j(seed, hops)
        return self._expand_memory(seed, hops)

    def _expand_memory(self, seed: str, hops: int) -> list[dict]:
        mg = self._mem_graph
        if seed not in mg["nodes"]:
            # 尝试部分匹配（如"深基坑"匹配"深基坑工程"）
            for nid in mg["nodes"]:
                if seed in nid or nid in seed:
                    seed = nid
                    break
            else:
                return []
        # BFS
        visited = {seed}
        frontier = [seed]
        for _ in range(hops):
            nxt = []
            for nid in frontier:
                for (to, rtype) in mg["adj"].get(nid, []):
                    if to not in visited:
                        visited.add(to)
                        nxt.append(to)
            frontier = nxt
        nodes = [mg["nodes"][nid] for nid in visited]
        edges = []
        for a in visited:
            for (b, rtype) in mg["adj"].get(a, []):
                if b in visited:
                    edges.append({"from": a, "to": b, "type": rtype})
        return {"nodes": nodes, "edges": edges}

    def _expand_neo4j(self, seed: str, hops: int) -> list[dict]:
        with self.driver.session() as s:
            rec = s.run(
                "MATCH (n) WHERE n.id CONTAINS $seed OR n.name CONTAINS $seed "
                "MATCH path = (n)-[*1..%d]-(m) "
                "RETURN path LIMIT 200" % hops,
                seed=seed,
            ).data()
        nodes, edges, seen_n, seen_e = [], [], set(), set()
        for r in rec:
            path = r["path"]
            # neo4j 4.4 驱动 path = 扁平 list: [node, rel_type_str, node, ...]
            if isinstance(path, list):
                items = path
            elif isinstance(path, dict):
                items = []
                for i, n in enumerate(path.get("nodes", [])):
                    items.append(n)
                    if i < len(path.get("relationships", [])):
                        rel = path["relationships"][i]
                        items.append(rel.get("type", "") if isinstance(rel, dict) else rel.type)
            else:
                items = []
                for i, n in enumerate(path.nodes):
                    items.append(n)
                    if i < len(path.relationships):
                        items.append(path.relationships[i].type)
            # 解析扁平列表
            i = 0
            while i < len(items):
                node = items[i]
                if isinstance(node, dict) and "id" in node:
                    nid = node["id"]
                    if nid not in seen_n:
                        seen_n.add(nid)
                        # 补 label：从 demo 图谱映射（Neo4j 节点 label 是标签不是属性）
                        nd = dict(node)
                        if "label" not in nd:
                            nd["label"] = self._label_of(nid)
                        nodes.append(nd)
                    if i + 2 < len(items) and isinstance(items[i + 2], dict):
                        rt = items[i + 1] if isinstance(items[i + 1], str) else ""
                        bnode = items[i + 2]
                        bid = bnode.get("id")
                        key = (nid, rt, bid)
                        if key not in seen_e:
                            seen_e.add(key)
                            edges.append({"from": nid, "to": bid, "type": rt})
                        i += 2
                        continue
                i += 1
        return {"nodes": nodes, "edges": edges}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true", help="写入 demo 图谱")
    ap.add_argument("--query", default=None, help="子图扩展 seed")
    ap.add_argument("--hops", type=int, default=2)
    a = ap.parse_args()

    store = Neo4jStore()
    if a.load:
        store.load_demo()
    if a.query:
        sub = store.expand_subgraph(a.query, a.hops)
        labels = {}
        for n in sub["nodes"]:
            labels[n["label"]] = labels.get(n["label"], 0) + 1
        print(f"子图扩展 [{a.query}] hops={a.hops}: {len(sub['nodes'])} 节点 "
              f"({labels}) / {len(sub['edges'])} 关系")
        for n in sub["nodes"]:
            print(f"  [{n['label'][:4]}] {n['name'][:60]}")


if __name__ == "__main__":
    main()
