# Neo4j 使用手册（GraphRAG 独立 Demo）

> 面向本机环境：Neo4j Community 4.4.8（绿色版，无需安装，位于仓库根 `neo4j-community-4.4.8/`）+ Java 11。
> 本手册属独立 demo `graphrag/`，与 rag2 / agent 落地链路无关。

---

## 一、前置条件

| 项 | 值 | 备注 |
|---|---|---|
| Neo4j 主目录 | `<项目根>/neo4j-community-4.4.8` | 绿色版，随仓库 |
| Java | **需 JDK 11**（Neo4j 4.4 要求，非 17+） | `start_neo4j.py` 会自动探测；找不到可用 `$env:NEO4J_JAVA` 显式指定 |
| Python 驱动 | `pip install neo4j` | 未装则 demo 会明确报错（不回退 JSON） |
| 连接地址 | `bolt://localhost:7687` / `http://localhost:7474` | |
| 账号 / 密码 | `neo4j` / `neo4j123456` | 可用 `NEO4J_USER` / `NEO4J_PASSWORD` 覆盖 |

> ⚠️ **换机器必读**：`start_neo4j.py` 早期版本把 Java 路径硬编码为旧机 conda 环境。
> 现已改为自动探测（`NEO4J_JAVA` > `JAVA_HOME` > PATH > conda 环境 > 常见安装目录）。
> 若新机没有 JDK 11，请先安装，否则 Neo4j 起不来。

---

## 二、启动 Neo4j

### 方式 A：项目自带脚本（推荐）

```powershell
cd C:\Users\<用户名>\Desktop\施工方案合规审查
& $PY neo4j-community-4.4.8/start_neo4j.py --background
# 或前台跑（Ctrl+C 停止）：  & $PY neo4j-community-4.4.8/start_neo4j.py
```

### 方式 B：直接用 Java 启动（等价，便于理解）

```powershell
cd C:\Users\<用户名>\Desktop\施工方案合规审查\neo4j-community-4.4.8
& "$env:NEO4J_JAVA" -Xms512m -Xmx1g -Dfile.encoding=UTF-8 `
  -cp "lib\*;plugins" org.neo4j.server.CommunityEntryPoint `
  --home-dir "$PWD" --config-dir "$PWD\conf"
```

看到日志出现 **`Started.`** 即启动成功（首次约 20–40 秒）。

### 方式 C：官方脚本（本机不可用，仅记录）

```powershell
bin\neo4j.bat console      # ❌ 报「已添加了具有相同键的项」
```

> 原因：Neo4j 4.4 的 PowerShell 包装脚本（`neo4j.ps1` 的 `Get-Args`）与新版 PowerShell
> 不兼容，构建参数哈希表时键重复。**用方式 A/B 绕过。**

### 验证

```powershell
Test-NetConnection 127.0.0.1 -Port 7687 | Select-Object TcpTestSucceeded   # 应 True
```

---

## 三、把 demo 图谱写入 Neo4j（一次性）

```powershell
cd C:\Users\<用户名>\Desktop\施工方案合规审查
& $PY graphrag/scripts/load_neo4j.py
# [ok] 已连接 Neo4j：bolt://localhost:7687
# [ok] 写入完成：56 节点 / 96 关系
```

---

## 四、★ 图形化看图谱（Neo4j Browser）

1. 浏览器打开 `http://localhost:7474`
2. 登录：`neo4j` / `neo4j123456`（地址保持 `bolt://localhost:7687`）
3. 顶部命令框粘贴 Cypher，`Ctrl+Enter` 执行；结果以**节点圆圈 + 关系箭头**绘出，点选可看属性。

### 复制即用 Cypher

```cypher
// ① 全貌（限 300，避免拥挤）
MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 300

// ② 【核心】危大工程 → 受哪些规范监管（多标准交叉）
MATCH (h:HazardCategory)-[r:REGULATED_BY]->(s:Standard) RETURN h,r,s

// ③ 深基坑工程的 2 跳关联子图
MATCH p = (h:HazardCategory {name:'深基坑工程'})-[*1..2]-(x) RETURN p LIMIT 100

// ④ 阈值链：类别 → 量名 → 阈值 → 义务
MATCH p=(:HazardCategory)-[:HAS_METRIC]->(:Metric)-[:HAS_THRESHOLD]->(:Threshold)-[:TRIGGERS]->(:Obligation)
RETURN p LIMIT 100

// ⑤ 术语归一（口语 → 规范词）
MATCH p=(t:Term)-[:ALIAS_OF]->(x) RETURN p

// ⑥ 各类型节点数量
MATCH (n) RETURN labels(n)[0] AS 节点类型, count(*) AS 数量
```

> 节点太挤时：每次只查一种关系，或用 `LIMIT` 控制。

---

## 五、不用写 Cypher（前端 / 命令行）

**前端（推荐，带单路/双路开关 + 图谱高亮）：**

```powershell
& $PY graphrag/app.py       # http://127.0.0.1:7870/
```

**命令行：**

```powershell
cd C:\Users\<用户名>\Desktop\施工方案合规审查

# 子图扩展（Cypher 多跳）
& $PY graphrag/src/neo4j_store.py --query "深基坑工程" --hops 2

# 双引擎检索（graph / dual）
& $PY graphrag/src/hybrid_search.py --query "深基坑开挖前要做哪些安全准备？" --mode dual
& $PY graphrag/src/hybrid_search.py --query "深基坑开挖前要做哪些安全准备？" --mode graph
```

---

## 六、停止 / 重启

```powershell
$pid = (Get-NetTCPConnection -LocalPort 7687 -State Listen).OwningProcess
Stop-Process -Id $pid -Force
```

> 数据持久化在 `neo4j-community-4.4.8\data\`，重启后图谱还在。

---

## 七、扩大到 81 部规范（实测估算）

| 阶段 | 规模 | 耗时 |
|---|---|---|
| 图谱构建（`build_full_graph.py`） | 13661 条款 + 81 规范 → ~3.4 万条关系 | 约 0.5 秒 |
| 写入 Neo4j | ~1.4 万节点 + 3.4 万关系 | 约 30–60 秒 |
| **合计** | | **约 1 分钟** |

> 风险：引用抽取基于正则，`GB 50010` 等**不在 81 部内的外部规范**会被过滤，需人工校准。

---

## 八、常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `已添加了具有相同键的项` | 官方 PS 脚本与新 PS 不兼容 | 用方式 A/B（java 直启） |
| `未找到 Java 11` | 新机没装 JDK 11 | 安装 JDK 11，或设 `$env:NEO4J_JAVA` |
| 连不上 `localhost:7687` | 服务没起 / 被终端关闭杀死 | 确认日志有 `Started.`；用独立窗口 |
| demo 报 `Neo4jUnavailable` | 服务没起 / 没装 `neo4j` 驱动 | 启动服务 + `pip install neo4j`（demo 设计为显式失败，不降级） |
| `CredentialsExpired` | 首次登录必须改密码 | 改为 `neo4j123456` |
| 查询结果为空 | 图谱没写入 | 跑 `graphrag/scripts/load_neo4j.py` |
| 图谱节点乱码 | 编码问题 | 启动已带 `-Dfile.encoding=UTF-8` |
