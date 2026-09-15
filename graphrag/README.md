# GraphRAG 规范知识图谱 · 独立 Demo

> 本目录是**独立演示模块**，不属于 rag2 / agent 的落地产物。rag2（检索）与 agent（编排）中**不包含任何 GraphRAG 代码或依赖**——本 demo 单向依赖 rag2 的向量检索（仅双路模式用到，可关闭）。

## 这是什么

把施工规范知识组织成**图谱**（规范 / 条款 / 危大类别 / 实体 / 量名 / 阈值 / 义务 / 术语），
演示 **GraphRAG 双引擎检索**：图谱子图扩展（多标准交叉，解决单路 RAG"盲人摸象"）+ 可选向量召回。

- 存储后端：**Neo4j**（演示链路强制走 Neo4j，不回退 JSON）
- 前端可选：**单路（图谱）** / **双路（向量 + 图谱，向量端可开关）**

## 图谱规模（demo v2）

| 项 | 值 |
|---|---|
| 节点 | 56（8 类：Standard / Clause / HazardCategory / Entity / Metric / Threshold / Obligation / Term） |
| 关系 | 96（13 类：BELONGS_TO / REFERENCES / REGULATED_BY / COVERS / MENTIONS / HIERARCHY / SUPERSEDES / ALIAS_OF / HAS_METRIC / HAS_THRESHOLD / FOR_CATEGORY / TRIGGERS / REFERENCES_CLAUSE） |
| 建图素材 | `data/demo_graph_v2.json`（三元组清单见 `data/demo_graph_v2_triples.md`） |

Schema 定义见 `src/schema.py`。

## 演示操作（面试流程）

### 0. 一键启动（推荐）

```powershell
cd C:\Users\<用户名>\Desktop\施工方案合规审查
.\graphrag\start_demo.ps1
```

它会按顺序做完：**起 Neo4j（后台）→ 等 bolt 就绪 → 图库为空则写入 demo 图谱 → 前台起 demo 前端**，
并打印前端地址、Neo4j Browser 地址和演示用的 Cypher。

常用开关：

| 命令 | 作用 |
|---|---|
| `.\graphrag\start_demo.ps1` | 一键启动（前台跑前端，Ctrl+C 只停前端） |
| `.\graphrag\start_demo.ps1 -Background` | 前端也放后台 + 自动开浏览器 + 健康检查 |
| `.\graphrag\start_demo.ps1 -Reload` | 强制把 demo 图谱重新写进 Neo4j |
| `.\graphrag\start_demo.ps1 -Neo4jOnly` | 只起 Neo4j 并等就绪 |
| `.\graphrag\start_demo.ps1 -Status` | 看端口 / 图库 / 依赖状态 |
| `.\graphrag\start_demo.ps1 -Stop` | 停 demo 前端 + Neo4j |
| `.\graphrag\start_demo.ps1 -Install` | 缺依赖时自动 pip install |

可选参数：`-Py <python.exe>`、`-Port 7870`、`-Timeout 150`。

> ★ **前置：JDK 11**（Neo4j 4.4.8 要 11，不是 17+）。
> 脚本找不到 Java 会直接给出提示；可显式指定：`$env:NEO4J_JAVA = "C:\path\to\jdk-11\bin\java.exe"`。

### 0'. 手动分步（等价，排查用）

```powershell
$PY = "C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe"   # 带 CUDA 的解释器
& $PY -m pip install -r requirements.txt        # 至少要有 fastapi / uvicorn / neo4j / torch(双路才需要)
```

`0'` 之后的 1–3 步就是上面脚本做的事，手动分步时按下面执行。

### 1. 启动 Neo4j

```powershell
cd C:\Users\<用户名>\Desktop\施工方案合规审查
& $PY neo4j-community-4.4.8/start_neo4j.py --background
# 首次登录 http://localhost:7474 会要求改初始密码；本 demo 默认用 neo4j / neo4j123456
# （如改过密码： $env:NEO4J_PASSWORD="你的密码"）
```

### 2. 把 demo 图谱写入 Neo4j（一次性）

```powershell
& $PY graphrag/scripts/load_neo4j.py
# [ok] 写入完成：56 节点 / 96 关系
# [ok] 图库现有：56 节点 / 96 关系
```

### 3. 启动 demo 服务

```powershell
& $PY graphrag/app.py
# 前端 http://127.0.0.1:7870/   Neo4j Browser http://localhost:7474
```

### 4. 面试演示脚本

1. **面试官：演示 GraphRAG 的 demo 效果**
   → 打开 `http://127.0.0.1:7870/`，前端选「**单路 · 图谱**」，输入
   `深基坑开挖前要做哪些安全准备？`，点检索：
   展示 *识别危大类别 → 图谱范围白名单（REGULATED_BY 跨规范）→ 阈值链 → 命中条款*。
2. **演示切换** → 选「**双路 · 向量+图谱**」，勾选/取消「启用向量端」对比：
   勾选 = 图谱 + 向量合并去重（命中的条款标注 图谱/向量/双源）；取消 = 退化为单路。
3. **面试官：看看 Neo4j 的图谱**
   → 切到 `http://localhost:7474`，登录后粘贴下面 Cypher（见下），展示节点-关系图。
4. 右侧画布同步展示图谱，命中子图会**高亮**（联动）。

### 5. Neo4j Browser 复制即用 Cypher

```cypher
// 全图（56 节点，建议先跑这个看整体）
MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 300

// 深基坑工程的多标准交叉（图谱路核心）
MATCH p=(h:HazardCategory)-[:REGULATED_BY]->(st:Standard) WHERE h.name CONTAINS '深基坑' RETURN p

// 阈值链：危大类别 → 量名 → 阈值 → 义务
MATCH p=(h:HazardCategory)-[:HAS_METRIC]->(:Metric)-[:HAS_THRESHOLD]->(:Threshold)-[:TRIGGERS]->(:Obligation)
RETURN p LIMIT 100

// 术语归一（口语 → 规范词）
MATCH p=(t:Term)-[:ALIAS_OF]->(x) RETURN p
```

## 命令行（不走前端时）

```powershell
# 图谱子图扩展（Cypher 多跳）
& $PY graphrag/src/neo4j_store.py --query "深基坑工程" --hops 2

# 双引擎检索（graph / dual）
& $PY graphrag/src/hybrid_search.py --query "深基坑开挖前的安全准备" --mode dual
& $PY graphrag/src/hybrid_search.py --query "深基坑开挖前的安全准备" --mode graph
```

## 目录

```
graphrag/
├── start_demo.ps1             一键启动（Neo4j + 建图 + 前端；-Status / -Stop / -Reload …）
├── app.py                     FastAPI 服务（/ 前端 · /api/search · /api/graph · /api/load · /api/health）
├── ui/index.html              前端：单路/双路开关 + 结果 + 图谱可视化
├── src/
│   ├── schema.py              节点/关系类型定义
│   ├── neo4j_store.py         Neo4j 存储（强制走 Neo4j；load / expand_subgraph / fetch_graph / stats）
│   └── hybrid_search.py       双引擎检索（graph / dual）
├── scripts/
│   ├── load_neo4j.py          把 demo_graph_v2.json 写入 Neo4j
│   ├── build_demo_graph.py    生成旧骨架 demo_graph.json
│   ├── build_demo_graph_v2.py 生成 demo_graph_v2.json（当前演示用）
│   ├── build_full_graph.py    全量图谱（81 规范，读 rag2 语料，只读引用）
│   └── add_risk_sources.py    并入风险源节点（可选）
├── data/
│   ├── demo_graph_v2.json     建图素材（56 节点 / 96 关系）
│   ├── demo_graph_v2_triples.md
│   ├── demo_graph.json        旧骨架（26 / 44）
│   └── graph_viz.html         离线交互式图谱（无 Neo4j 时的备用展示）
└── docs/NEO4J_GUIDE.md        Neo4j 部署与查询手册
```

## 诚实边界

- 本 demo 是**演示级**图谱（56 节点），非全量（全量建图见 `build_full_graph.py`，需清洗 OCR 噪声）。
- 图谱产出的**是范围/白名单**（"去哪找"），不直接塞条款；条款由向量路/检索提供。
- `data/graph_viz.html` 是**离线备份展示**（同一份 v2 数据，客户端渲染），仅在无 Neo4j 环境时兜底，
  正式演示优先走 Neo4j Browser。
