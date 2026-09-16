# GraphRAG 规范知识图谱 · 独立 Demo

> 本目录是**独立演示模块**，不属于 rag2 / agent 的落地产物。rag2（检索）与 agent（编排）中**不包含任何 GraphRAG 代码或依赖**——本 demo 单向依赖 rag2 的向量检索（仅双路模式用到，可关闭）。

## 这是什么

把施工规范知识组织成**图谱**（规范 / 条款 / 危大类别 / 实体 / 量名 / 阈值 / 义务 / 术语），
演示 **GraphRAG 检索**：图谱子图扩展给出"这问题该看哪几部规范"的范围（多标准交叉，解决单路 RAG"盲人摸象"），
可选叠加向量召回（BM25 ∥ 双塔 RRF）拿到具体条款。

- 存储后端：**Neo4j**（演示链路强制走 Neo4j，不回退 JSON）
- 前端可选：**单路（图谱）** / **双路（向量 + 图谱，向量端可开关）**
- 图谱规模：**56 节点（8 类）/ 96 条边**（建图素材口径，Neo4j 实入库 94 条—见"诚实边界"）
- Schema 定义：`src/schema.py`；建图素材：`data/demo_graph_v2.json`

---

## 演示操作

### 0. 一键启动（推荐）

```powershell
cd <REPO_ROOT>
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

> ★ **环境分工（两套现成 conda 环境，无需新建）**
>
> | 角色 | 环境 | 说明 |
> |---|---|---|
> | Python 端（demo 服务 / 建图 / 检索） | `torch_gpu` | 需 `neo4j / fastapi / uvicorn / pydantic`；双路模式另需 `torch / faiss / transformers / sentence-transformers`（已随 rag2 装好） |
> | Neo4j 服务本体（Java 进程） | `neo4j-java11` | OpenJDK 11 —— Neo4j 4.4.8 要 **11**，不是 17+ |
>
> 缺 Java 时脚本会提示；可显式指定：`$env:NEO4J_JAVA = "C:\path\to\jdk-11\bin\java.exe"`。
> 依赖清单见 `graphrag\requirements.txt`。

### 0'. 手动分步（等价，排查用）

```powershell
$PY = "C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe"   # 带 CUDA 的解释器（Python 端统一用它）
& $PY -m pip install -r graphrag\requirements.txt   # neo4j / fastapi / uvicorn / pydantic
```

**1. 启动 Neo4j**

```powershell
cd <REPO_ROOT>
& $PY neo4j-community-4.4.8/start_neo4j.py --background
# 首次登录 http://localhost:7474 会要求改初始密码；本 demo 默认用 neo4j / neo4j123456
# （如改过密码： $env:NEO4J_PASSWORD="你的密码"）
```

**2. 把 demo 图谱写入 Neo4j（一次性）**

```powershell
& $PY graphrag/scripts/load_neo4j.py
# [ok] 写入完成：56 节点 / 96 关系      ← 脚本自报（按 json 条数计）
# [ok] 图库现有：56 节点 / 94 关系      ← Neo4j 实查（2 条悬空边不写入，见"诚实边界"）
```

**3. 启动 demo 服务**

```powershell
& $PY graphrag/app.py
# 前端 http://127.0.0.1:7870/   Neo4j Browser http://localhost:7474
```

---

## 图谱 Schema：实体类型 · 关系 · 属性

> 权威定义在 `src/schema.py`（`ALL_NODE_LABELS` / `ALL_REL_TYPES`），数据在 `data/demo_graph_v2.json`。
> 下表按**实际建图素材**逐项核对：节点 8 类 / 56 个，关系 13 类 / 96 条（Neo4j 实入库 94 条）。

### 1. 节点类型（8 类 / 56 个）

| # | 标签（Neo4j Label） | 中文 | 数量 | 主键 `id` 形态 | 属性（properties） |
|---|---|---|---|---|---|
| 1 | `Standard` | 规范 | 9 | `GB55023-2022_施工脚手架通用规范` | `id` `name` `full_name` `level` `demo_note` |
| 2 | `Clause` | 条款 | 9 | `规范id::条款号`，如 `JGJ130-2011_…::6.2.4` | `id` `name` `text` `is_appendix`(可选) |
| 3 | `HazardCategory` | 危大工程类别 | 4 | `深基坑工程` | `id` `name` `description` |
| 4 | `Entity` | 关键实体（工序/构件） | 6 | `基坑开挖` | `id` `name` `description` |
| 5 | `Metric` | 量名（含单位） | 7 | `metric:<类别>:<量名>` | `id` `name` `unit` `source` |
| 6 | `Threshold` | 阈值（危大线/超规模线） | 10 | `threshold:<类别>:<量名>:<level>:<值>` | `id` `name` `category` `metric` `level` `value` `unit` `source` |
| 7 | `Obligation` | 义务（阈值触发的后果） | 5 | `obligation:应编制专项施工方案` | `id` `name` `description` |
| 8 | `Term` | 术语别名（口语 → 规范词） | 6 | `term:高支模` | `id` `name` `note` |

**属性字典（逐字段）**

| 属性 | 所属节点 | 类型 | 取值 / 示例 |
|---|---|---|---|
| `id` | 全部 | string | 图谱内唯一主键（`MERGE` 依据） |
| `name` | 全部 | string | 默认与 `id` 相同；Neo4j Browser 节点显示名 |
| `full_name` | Standard | string | `GB55023-2022_施工脚手架通用规范` |
| `level` | Standard | enum | `强制性国标` / `国家标准` / `行业标准` / `地方标准` |
| `demo_note` | Standard | string | 仅 demo 说明该规范为何入选（如"脚手架强条以 GB55023 为准"） |
| `text` | Clause | string | 条文正文（部分为节选，含 `…`） |
| `is_appendix` | Clause | bool | `true` = 该"条款"实为附录（如 `附录B`） |
| `description` | HazardCategory / Entity / Obligation | string | 一句话释义 |
| `unit` | Metric / Threshold | string | `m` / `kN/m2` / `kN/m` / `kN` |
| `source` | Metric / Threshold | string | 出处，如 `37号令附件1 二(二)`、`hazardous_work_types.json` |
| `metric` | Threshold | string | 量名（与所属 Metric 的 `name` 对齐） |
| `category` | Threshold | string | 归属危大类别 |
| `level` | Threshold | enum | `危大`（附件1，需专项方案） / `超规模`（附件2，需专家论证） |
| `value` | Threshold | number | 数值，如 `5`、`10`、`24`、`100` |
| `note` | Term | string | 规范叫法，如 `规范叫法：混凝土模板支撑工程` |

> ⚠️ **schema 定义了但本 demo 未启用**：节点 `RiskSource`（风险源，见 `scripts/add_risk_sources.py` 可选并入）；
> 关系 `CONFLICTS_WITH`（规范冲突，需人工判定）、`HAS_RISK`（类别 → 风险源）——均未出现在 v2 建图素材里，前端不会展示。

### 2. 关系类型（13 类 / 96 条）

| # | 关系类型 | 方向（from → to） | 条数 | 语义 | 演示价值 |
|---|---|---|---|---|---|
| 1 | `REGULATED_BY` | HazardCategory → Standard | 12 | 该危大工程受哪些规范监管 | ★ **多标准交叉的核心边** |
| 2 | `TRIGGERS` | Threshold → Obligation | 16 | 阈值触发什么义务 | ★ **判定结论的落点** |
| 3 | `HAS_THRESHOLD` | Metric → Threshold | 10 | 量名对应的危大／超规模阈值 | 阈值链第 2 跳 |
| 4 | `FOR_CATEGORY` | Threshold → HazardCategory | 10 | 阈值归属类别（反向） | 反查 |
| 5 | `MENTIONS` | Clause → Entity | 10 | 条款涉及某实体（工序/构件） | 实体级检索入口 |
| 6 | `BELONGS_TO` | Clause → Standard | 8 | 条款归属规范 | 溯源"出自哪本" |
| 7 | `COVERS` | Clause → HazardCategory | 8 | 该条款覆盖哪个危大类别 | 反向定位条款 |
| 8 | `HAS_METRIC` | HazardCategory → Metric | 7 | 类别有哪些判定量名 | 阈值链第 1 跳（实入库 6，1 条悬空边） |
| 9 | `ALIAS_OF` | Term → HazardCategory ｜ Entity | 6 | 口语词归一 | ★ **实体链接的桥梁** |
| 10 | `HIERARCHY` | Standard → Standard | 4 | 上位法层级（强条 > 国标 > 行标 > 地标） | 效力优先级 |
| 11 | `SUPERSEDES` | Standard → Standard | 2 | 废止 / 替代 | 检查引用是否过期（实入库 1，1 条悬空边） |
| 12 | `REFERENCES` | Clause → Standard | 2 | 条款引用其他规范 | 跨规范跳转 |
| 13 | `REFERENCES_CLAUSE` | Clause → Clause | 1 | 条款引用具体条文/附录 | 条文级跳转 |

**关系属性**：绝大多数关系只带端点（`from` / `to`）；仅 3 种带额外属性：
`ALIAS_OF.target_label`（目标是 HazardCategory 还是 Entity）、`SUPERSEDES.note`、`REFERENCES_CLAUSE.note`。

### 3. 节点实例清单（56 个）

```
Standard (9)
  GB55023-2022_施工脚手架通用规范                          强制性国标
  GB55032-2022_建筑与市政工程施工质量控制通用规范             强制性国标
  GB51210-2016_建筑施工脚手架安全技术统一标准                国家标准
  GB51004-2015_建筑地基基础工程施工规范                     国家标准
  JGJ311-2013_建筑深基坑工程施工安全技术规范                  行业标准
  JGJ120-2012_建筑基坑支护技术规程                          行业标准
  JGJ130-2011_建筑施工扣件式钢管脚手架安全技术规范             行业标准
  DG-TJ08-61-2018_基坑工程技术标准                         地方标准
  DG-TJ08-2077-2021_危险性较大的分部分项工程安全管理标准       地方标准

HazardCategory (4)
  深基坑工程      开挖深度≥3m 或未超 3m 但地质条件复杂的基坑
  模板支撑工程    搭设高度≥5m、跨度≥10m、施工总荷载≥10kN/m² 等条件的模板支架
  起重吊装工程    采用非常规起重设备且单件起吊重量≥10kN 的吊装
  脚手架工程      搭设高度≥24m 的落地式脚手架等

Clause (9)   —— 规范id::条款号
  JGJ311-2013_…::7.1.2        降排水施工方案应包含各种泵的扬程、功率，排水管路尺寸、材料、路线…
  JGJ120-2012_…::3.3.1        支护结构选型时应综合考虑：基坑深度；土的性状及地下水条件…
  DG-TJ08-61-2018_…::2.1.2    基坑工程为挖除建(构)筑物地下结构处土方…采取的围护、支撑、降水等工程措施
  DG-TJ08-2077-2021_…::8.4.7  监理单位发现施工单位未按专项方案施工的，应要求整改…
  GB51210-2016_…::3.2.3       脚手架结构重要性系数取值（安全等级 I 级 1.1、II 级 1.0）
  GB55023-2022_…::5.2.1       脚手架应按顺序搭设…一次搭设高度不应超过最上层连墙件 2 步，自由高度≤4m
  GB51004-2015_…::5.3.2       钢筋混凝土条形基础…混凝土宜分段分层连续浇筑，每层厚度 300~500mm
  JGJ130-2011_…::6.2.4        架高超 7m 时，连墙件应按两步三跨或两步两跨设置，宜梅花形布置
  JGJ130-2011_…::附录B         （附录B：连墙件布置表）   ← is_appendix=true

Entity (6)
  基坑开挖 / 支护结构 / 降排水 / 监测 / 模板支架 / 连墙件

Metric (7)   —— 全部 source=hazardous_work_types.json
  metric:深基坑工程:开挖深度        m
  metric:模板支撑工程:搭设高度       m
  metric:模板支撑工程:施工总荷载     kN/m2
  metric:模板支撑工程:集中线荷载     kN/m
  metric:起重吊装工程:单件起吊重量    kN
  metric:脚手架工程:搭设高度        m
  metric:降水工程:开挖深度          m     ← 其类别"降水工程"未建为节点（悬空边来源）

Threshold (10)   —— level：危大（附件1）/ 超规模（附件2）
  深基坑工程   · 开挖深度     : 危大 3m      / 超规模 5m
  模板支撑工程 · 搭设高度     : 危大 5m      / 超规模 8m
  模板支撑工程 · 施工总荷载   : 危大 10kN/m2  / 超规模 15kN/m2
  脚手架工程   · 搭设高度     : 危大 24m     / 超规模 50m
  起重吊装工程 · 单件起吊重量 : 危大 10kN    / 超规模 100kN

Obligation (5)
  应编制专项施工方案（37号令第10条）
  方案应经审批（施工单位技术负责人、总监理工程师签字，37号令第13条）
  应组织专家论证（超规模危大工程，37号令第12条）
  应进行基坑监测（深基坑第三方监测，JGJ311-2013 第8章）
  应进行分阶段验收（脚手架搭设完成，JGJ130-2011 第8章）

Term (6)
  term:高支模 → 混凝土模板支撑工程      term:深基坑 → 深基坑工程
  term:排架   → 模板支撑架/支撑结构     term:满堂架 → 满堂支撑架
  term:塔吊   → 塔式起重机             term:爬架   → 附着式升降脚手架
```

### 4. 关系实例清单（96 条）

```
# ALIAS_OF（术语归一，6）—— 实体链接入口
term:高支模 -[ALIAS_OF]-> 模板支撑工程        term:深基坑 -[ALIAS_OF]-> 深基坑工程
term:排架   -[ALIAS_OF]-> 模板支架            term:满堂架 -[ALIAS_OF]-> 模板支架
term:塔吊   -[ALIAS_OF]-> 起重吊装工程        term:爬架   -[ALIAS_OF]-> 脚手架工程

# REGULATED_BY（危大类别 → 监管规范，12）—— 多标准交叉核心
深基坑工程   → JGJ311-2013 / JGJ120-2012 / DG-TJ08-61-2018 / DG-TJ08-2077-2021
模板支撑工程 → GB51210-2016 / GB55023-2022 / JGJ130-2011 / DG-TJ08-2077-2021
起重吊装工程 → DG-TJ08-2077-2021
脚手架工程   → GB51210-2016 / GB55023-2022 / JGJ130-2011

# COVERS（条款 → 危大类别，8）
JGJ311-2013::7.1.2 / JGJ120-2012::3.3.1 / DG-TJ08-61-2018::2.1.2 / GB51004-2015::5.3.2 → 深基坑工程
DG-TJ08-2077-2021::8.4.7 / GB51210-2016::3.2.3 / GB55023-2022::5.2.1              → 模板支撑工程
JGJ130-2011::6.2.4                                                              → 脚手架工程

# BELONGS_TO（条款 → 所属规范，8）
JGJ311-2013::7.1.2        → JGJ311-2013_建筑深基坑工程施工安全技术规范
JGJ120-2012::3.3.1        → JGJ120-2012_建筑基坑支护技术规程
DG-TJ08-61-2018::2.1.2    → DG-TJ08-61-2018_基坑工程技术标准
DG-TJ08-2077-2021::8.4.7  → DG-TJ08-2077-2021_危险性较大的分部分项工程安全管理标准
GB51210-2016::3.2.3       → GB51210-2016_建筑施工脚手架安全技术统一标准
GB55023-2022::5.2.1       → GB55023-2022_施工脚手架通用规范
GB51004-2015::5.3.2       → GB51004-2015_建筑地基基础工程施工规范
JGJ130-2011::6.2.4        → JGJ130-2011_建筑施工扣件式钢管脚手架安全技术规范
（9 个条款中，附录B 那条无 BELONGS_TO）

# MENTIONS（条款 → 实体，10）
JGJ311-2013::7.1.2        → 降排水
JGJ120-2012::3.3.1        → 支护结构, 基坑开挖
DG-TJ08-61-2018::2.1.2    → 支护结构, 降排水, 基坑开挖
GB55023-2022::5.2.1       → 模板支架, 连墙件
JGJ130-2011::6.2.4        → 连墙件
GB51210-2016::3.2.3       → 模板支架

# HAS_METRIC（类别 → 量名，7）
模板支撑工程 → 搭设高度 / 施工总荷载 / 集中线荷载
深基坑工程   → 开挖深度
起重吊装工程 → 单件起吊重量
脚手架工程   → 搭设高度
降水工程     → 开挖深度      ← ⚠️ 悬空边：降水工程未建节点（Neo4j 不写入）

# HAS_THRESHOLD（量名 → 阈值，10）+ FOR_CATEGORY（阈值 → 类别，10）
与上文 Threshold 清单一一对应：每条量名下挂"危大""超规模"两个阈值节点，
每个阈值节点再反向指回其所属类别（FOR_CATEGORY）。

# TRIGGERS（阈值 → 义务，16）
危大档（5 个：基坑3m/模板5m/模板10kN·m⁻²/脚手架24m/吊装10kN）
    → 应编制专项施工方案 + 方案应经审批                    共 10 条
    其中 深基坑 3m 额外 → 应进行基坑监测                   1 条
超规模档（5 个：基坑5m/模板8m/模板15kN·m⁻²/脚手架50m/吊装100kN）
    → 应组织专家论证                                      共 5 条

# HIERARCHY（上位法层级，4）
GB55023-2022 → GB51210-2016 → JGJ130-2011
JGJ120-2012  → DG-TJ08-61-2018
JGJ311-2013  → DG-TJ08-61-2018

# SUPERSEDES（废止 / 替代，2 —— 入库 1）
GB55023-2022 → JGJ130-2011        （脚手架强条以 GB55023 为准）
GB55032-2022 → JGJ46-2005         ← ⚠️ 悬空边：JGJ46 未建节点（Neo4j 不写入）

# REFERENCES（条款 → 其他规范，2）
JGJ120-2012::3.3.1 → GB51004-2015          JGJ311-2013::7.1.2 → GB51004-2015

# REFERENCES_CLAUSE（条款 → 条文/附录，1）
JGJ130-2011::6.2.4 → JGJ130-2011::附录B
```

### 5. 建图素材字段格式（JSON ↔ Neo4j）

```jsonc
// data/demo_graph_v2.json
{
  "nodes": [ { "id": "...", "label": "Standard", "name": "...", ...其余属性 } ],
  "edges": [ { "from": "<node id>", "to": "<node id>", "type": "REGULATED_BY" } ],
  "meta":  { "purpose": "...", "node_labels": [...], "rel_types": [...] }
}
```

写入规则（`src/neo4j_store.py::load_graph`）：

- 节点：`MERGE (n:<label> {id:$id}) SET n += $props`（`label` 以外的字段全部成为 Neo4j 属性）
- 关系：`MATCH (a {id:$a}), (b {id:$b}) MERGE (a)-[:<type>]->(b)`
  → **端点不存在时 `MATCH` 匹配不到，该边被静默跳过**（下面 2 条悬空边不入库的原因）
- 校验：`label` 必须 ∈ `ALL_NODE_LABELS`、`type` 必须 ∈ `ALL_REL_TYPES`，否则抛错

---

## Neo4j 图谱演示操作

### 1. 打开 Browser 并登录

| 项 | 值 |
|---|---|
| 地址 | `http://localhost:7474` |
| 账号 / 密码 | `neo4j` / `neo4j123456`（可用 `$env:NEO4J_PASSWORD` 覆盖） |
| Bolt | `bolt://localhost:7687` |
| 数据位置 | `neo4j-community-4.4.8\data\`（重启后图谱仍在） |

登录后在顶部命令框粘贴 Cypher 回车，结果以**节点圆圈 + 关系箭头**渲染；拖拽可布局、滚轮缩放、
点节点 / 关系看右侧属性面板（字段即上文"节点类型 / 关系 / 属性"三张表）。

### 2. 看全图（由粗到细）

```cypher
// ① 节点类型统计 —— 先让面试官看到"有 8 类实体"
MATCH (n) UNWIND labels(n) AS label RETURN label AS 节点类型, count(*) AS 数量 ORDER BY 数量 DESC

// ② 关系类型统计 —— 13 类
MATCH ()-[r]->() RETURN type(r) AS 关系类型, count(*) AS 数量 ORDER BY 数量 DESC

// ③ 全图（56 节点，建议最后跑，避免画面过密）
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 300
```

### 3. 六个演示查询（每个都能讲一个技术点）

```cypher
// ① 多标准交叉：深基坑工程受哪几部规范监管（REGULATED_BY）
MATCH p=(h:HazardCategory)-[:REGULATED_BY]->(st:Standard)
WHERE h.name CONTAINS '深基坑' RETURN p

// ② 阈值链：类别 → 量名 → 阈值 → 义务（判定类问答的完整闭环）
MATCH p=(h:HazardCategory)-[:HAS_METRIC]->(:Metric)-[:HAS_THRESHOLD]->(:Threshold)-[:TRIGGERS]->(:Obligation)
RETURN p LIMIT 100

// ③ 只看深基坑这一条链（对 ② 收窄）
MATCH p=(h:HazardCategory {name:'深基坑工程'})-[:HAS_METRIC]->(:Metric)
      -[:HAS_THRESHOLD]->(t:Threshold)-[:TRIGGERS]->(o:Obligation)
RETURN p

// ④ 术语归一：口语词怎么对上规范词（实体链接的桥梁）
MATCH p=(t:Term)-[:ALIAS_OF]->(x) RETURN p

// ⑤ 上位法层级（效力优先级）
MATCH p=(a:Standard)-[:HIERARCHY]->(b:Standard) RETURN p

// ⑥ 废止链 + 条文跳转（引用 / 替代检查）
MATCH p=(a:Standard)-[:SUPERSEDES]->(b) RETURN p
UNION
MATCH p=(c:Clause)-[:REFERENCES_CLAUSE]->(:Clause) RETURN p
```

### 4. 前端联动演示（`http://127.0.0.1:7870/`）

左侧结果区**分四段**展示，双路模式尤其要按这个顺序讲：

| 段 | 内容 | 说明 |
|---|---|---|
| 前置 | 识别危大类别 · 图谱范围白名单 · 阈值链 | 图谱"框范围"的三步 |
| **① 合并结果** | 图谱 + 向量**去重合并**后的排序 | 排序规则：**双源 → 图谱/向量交替**（此前的纯来源优先级排序会让图谱条款占满 top_k，向量结果永远看不见） |
| **② 图谱路结果** | 子图内 `Clause` 节点（跨规范） | 每个条款带 `图谱` 标签 |
| **③ 向量路结果** | BM25 ∥ 双塔 + RRF 的召回，按分数降序 | 每个条款带 `向量` 标签 + **向量分**（RRF 融合分） |

演示脚本：

1. **单路 · 图谱** → 输入 `深基坑开挖前要做哪些安全准备？` → 检索
   → 展示链路：*识别危大类别 → 图谱范围白名单（`REGULATED_BY` 给出 4 部监管规范）→ 阈值链 → ① 合并结果*；
   此时 ③ 向量路显示"单路模式：向量端未启用"。
2. **双路 · 向量+图谱** → 勾选「启用向量端」→ 检索
   → ① 合并结果里能同时看到 `图谱` 与 `向量` 两种标签（**首条为图谱，第二条即向量**）；
   → 下翻 ② / ③ 两段对比：图谱路给的是"该危大工程相关的跨规范条款"，向量路给的是"语义最近的具体条文"
   （例：向量路能捞到 `GB50202-2018::8.1.3 基坑降排水效果检验` 这类图谱没连上的条款）。
   → 取消「启用向量端」= 退化为单路，可现场对比 ① 的条目变化。
3. **看图**：右侧画布同步展示图谱，命中子图**高亮**；要看原始图时切 `http://localhost:7474`。
4. 收尾话术：*"图谱给范围（该看哪几部规范），向量给条款（具体条文）——两路合并后既有范围约束、
   又不会漏掉图谱没连上的条文，所以不会像单路检索那样只捞到一部规范的局部答案。"*

> ⏱ 双路**首次**点检索会等约 18~30 秒（服务端首次加载 torch + 双塔模型，日志里能看到 `Loading weights`），
> **第二次起 0.1 秒**（检索器与向量端进程内缓存）。演示前先点一次预热即可。

### 5. 命令行方式（等价，不走前端）

```powershell
# 图谱子图扩展（真 Cypher 多跳）
& $PY graphrag/src/neo4j_store.py --query "深基坑工程" --hops 2

# 双引擎检索
& $PY graphrag/src/hybrid_search.py --query "深基坑开挖前的安全准备" --mode graph
& $PY graphrag/src/hybrid_search.py --query "深基坑开挖前的安全准备" --mode dual

# 只统计不写入 / 增量写入
& $PY graphrag/scripts/load_neo4j.py --stats
& $PY graphrag/scripts/load_neo4j.py --no-clear
```

HTTP 接口（`graphrag/app.py`）：`GET /`（前端）· `GET /api/health`（Neo4j 连通 + 图库统计）·
`GET /api/graph`（全图，供可视化）· `POST /api/load`（把 demo_graph_v2.json 写入 Neo4j）·
`POST /api/search`（请求 `{query, mode:"graph"|"dual", hops, top_k}`）。

`POST /api/search` 返回的关键字段：

| 字段 | 含义 |
|---|---|
| `hits` | **① 合并结果**（去重，按「双源 → 图谱/向量交替」排序，取 `top_k` 条） |
| `graph_hits` | **② 图谱路**全部命中条款（子图内 `Clause` 节点，含被两路同时命中的） |
| `vector_hits` | **③ 向量路**全部召回条款（按 RRF 分数降序，含 `score`） |
| `graph_clauses` / `vector_clauses` / `merged` | 三路计数 |
| `hazards` / `graph_whitelist` / `graph_thresholds` | 识别出的危大类别、范围白名单、阈值链 |
| `graph_subgraph` | 子图 `{nodes, edges}`，供前端高亮 |
| `effective_mode` / `vector_error` | 实际生效模式（`graph` / `dual` / `dual_degraded`）与向量端异常信息 |

### 6. 停止

```powershell
.\graphrag\start_demo.ps1 -Stop
# 或只停某个端口：
$pid = (Get-NetTCPConnection -LocalPort 7687 -State Listen).OwningProcess; Stop-Process -Id $pid -Force
```

---

## 目录

```
graphrag/
├── start_demo.ps1             一键启动（Neo4j + 建图 + 前端；-Status / -Stop / -Reload …）
├── app.py                     FastAPI 服务（/ 前端 · /api/search · /api/graph · /api/load · /api/health）
├── requirements.txt           Python 端依赖清单 + 环境分工说明
├── ui/index.html              前端：单路/双路开关 + 结果 + 图谱可视化
├── src/
│   ├── schema.py              节点/关系类型定义（8 节点 + 15 关系，其中 13 关系在 demo 启用）
│   ├── neo4j_store.py         Neo4j 存储（强制走 Neo4j；load / expand_subgraph / fetch_graph / stats）
│   └── hybrid_search.py       双引擎检索（graph / dual）
├── scripts/
│   ├── load_neo4j.py          把 demo_graph_v2.json 写入 Neo4j
│   ├── build_demo_graph.py    生成旧骨架 demo_graph.json
│   ├── build_demo_graph_v2.py 生成 demo_graph_v2.json（当前演示用）
│   ├── build_full_graph.py    全量图谱（81 规范，读 rag2 语料，只读引用）
│   └── add_risk_sources.py    并入风险源节点（可选）
├── data/
│   ├── demo_graph_v2.json     建图素材（56 节点 / 96 边）
│   ├── demo_graph_v2_triples.md
│   ├── demo_graph.json        旧骨架（26 / 44）
│   └── graph_viz.html         离线交互式图谱（无 Neo4j 时的备用展示）
└── docs/NEO4J_GUIDE.md        Neo4j 部署与查询手册
```

> ★ **导入约定（改代码前必读）**：本目录内部一律用**裸模块名**导入
> （`from neo4j_store import ...` / `from schema import ...`），启动时把 `graphrag/src` 加入 `sys.path`。
> **不要写成 `from src.xxx`** —— rag2 也有一个 `src` 包，两套同名包会互相抢占，
> 双路模式会直接报 `No module named 'src.common'`（`src` 这个名字留给 rag2 的
> `src.common / src.index / src.retrieval`）。
> 同理，命令行入口用**文件路径**方式：`python graphrag/src/neo4j_store.py --query ...`，
> 不要用 `python -m src.neo4j_store`。

## 诚实边界

- 本 demo 是**演示级**图谱（56 节点），非全量；全量建图见 `build_full_graph.py`（81 规范，引用抽取需清洗 OCR 噪声）。
- **2 条悬空边（本机实测）**：建图素材里 `降水工程 -[HAS_METRIC]-> metric:降水工程:开挖深度` 与
  `GB55032-2022 -[SUPERSEDES]-> JGJ46-2005` 的端点**没有对应节点**，`load_graph` 的
  `MATCH (a),(b) MERGE` 匹配不到 → 不写入 Neo4j。故脚本自报 96 条、**Neo4j 实查 94 条**
  （`HAS_METRIC` 7→6、`SUPERSEDES` 2→1）。修复方向：补齐"降水工程"节点与 JGJ46 规范节点，
  或删掉这两条边；同时把 `load_graph` 的 `written` 计数器改为按实际匹配结果统计。
- 图谱产出的**是范围 / 白名单**（"去哪找"），不直接塞条款；条款由向量路 / 检索提供。
- `data/graph_viz.html` 是**离线备份展示**（同一份 v2 数据，客户端渲染），仅在无 Neo4j 环境时兜底，
  正式演示优先走 Neo4j Browser。
