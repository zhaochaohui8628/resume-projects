"""把 demo 图谱写入 Neo4j（演示前一次性执行；Neo4j 必须在跑）。

用法（项目根目录）：
  $PY graphrag/scripts/load_neo4j.py                 # 写入 demo_graph_v2.json（先清空）
  $PY graphrag/scripts/load_neo4j.py --no-clear      # 不清空，增量 MERGE
  $PY graphrag/scripts/load_neo4j.py --stats         # 只打印图库统计

连接参数可用环境变量覆盖：NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
默认 bolt://localhost:7687 · neo4j / neo4j123456
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

GRAPH_ROOT = Path(__file__).resolve().parents[1]        # .../graphrag
sys.path.insert(0, str(GRAPH_ROOT))

from src.neo4j_store import Neo4jStore, Neo4jUnavailable  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-clear", action="store_true", help="不清空旧图（增量 MERGE）")
    ap.add_argument("--stats", action="store_true", help="只打印统计，不写入")
    a = ap.parse_args()

    try:
        store = Neo4jStore()
    except Neo4jUnavailable as e:
        print(f"[x] {e}", file=sys.stderr)
        return 2

    print(f"[ok] 已连接 Neo4j：{store.uri}")
    if not a.stats:
        r = store.load_graph(clear=not a.no_clear)
        print(f"[ok] 写入完成：{r['nodes']} 节点 / {r['edges']} 关系")
        print(f"     节点类型：{r['labels']}")
        print(f"     关系类型：{r['rel_types']}")
    s = store.stats()
    print(f"[ok] 图库现有：{s['nodes']} 节点 / {s['edges']} 关系")
    store.close()
    print("\n下一步：浏览器打开 http://localhost:7474 查看图谱（Bolt 连接 bolt://localhost:7687）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
