# Neo4j 使用手册（GraphRAG 规范知识图谱）

> 面向本机环境：Neo4j Community 4.4.8（绿色版，无需安装）+ conda 隔离的 Java 11。

---

## 一、前置条件（已验证就绪）

| 项 | 路径 / 值 | 状态 |
|---|---|---|
| Neo4j 主目录 | `<REPO_ROOT>\neo4j-community-4.4.8` | ✅ 绿色版 |
| Java 11 | `C:\Users\<用户名>\anaconda3\envs\neo4j-java11\Library\bin\java.exe` | ✅ Zulu 11.0.30 |
| 连接地址 | `bolt://127.0.0.1:7687`（数据库）/ `http://localhost:7474`（网页界面） | ✅ |
| 账号 / 密码 | `neo4j` / `neo4j123456` | ✅ 已设置 |

> 说明：Neo4j 4.4 需要 Java 11。系统原本没有 Java，已用 conda 装在独立环境
> `neo4j-java11` 里，**不污染系统**，也不影响 `torch_gpu` 环境。

---

## 二、启动 Neo4j（三选一）

### 方式 A：项目自带脚本（最省事，推荐）

```powershell
cd "<REPO_ROOT>\neo4j-community-4.4.8"
& "C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe" start_neo4j.py
```

- 窗口保持打开 = 服务运行；按 `Ctrl+C` 停止。
- 想后台跑：加 `--background`（但**随当前终端关闭会被杀**，长期运行不建议）。

### 方式 B：直接用 Java 启动（等价，便于理解）

```powershell
cd "<REPO_ROOT>\neo4j-community-4.4.8"
& "C:\Users\<用户名>\anaconda3\envs\neo4j-java11\Library\bin\java.exe" `
  -Xms512m -Xmx1g -Dfile.encoding=UTF-8 `
  -cp "lib\*;plugins" `
  org.neo4j.server.CommunityEntryPoint `
  --home-dir "$PWD" --config-dir "$PWD\conf"
```

看到日志出现 **`Started.`** 即启动成功（首次约 20-40 秒）。

### 方式 C：官方脚本（**本机不可用，仅记录**）

```powershell
bin\neo4j.bat console      # ❌ 会报「已添加了具有相同键的项」
```

> **为什么失败**：Neo4j 4.4 的 PowerShell 包装脚本（`neo4j.ps1` 里 `Get-Args`）
> 与新版 PowerShell 不兼容，构建参数哈希表时键重复。**用方式 A/B 绕过**。

### 验证是否启动成功

```powershell
# 看端口（应返回 True）
Test-NetConnection 127.0.0.1 -Port 7687 | Select-Object TcpTestSucceeded
```

---

## 三、★ 怎么看"软件内部演示"（图形化看图谱）

Neo4j 自带一个网页版图形界面 **Neo4j Browser**，这就是"软件内部演示"：

1. **浏览器打开** → `http://localhost:7474`
2. **登录** → 用户名 `neo4j`，密码 `neo4j123456`（连接地址保持 `bolt://localhost:7687`）
3. 顶部有一条**命令行输入框**，粘贴下面的 Cypher 语句，按 `Ctrl+Enter` 执行
4. 结果会以 **节点（圆圈）+ 关系（箭头）** 的图形直接画出来，点击节点/箭头可看属性

### 复制即用的查询语句（能直接看到我们的图谱）

```cypher
// ① 看全貌：所有节点与关系（限 300 个，避免卡顿）
MATCH (n) RETURN n LIMIT 300
```

```cypher
// ② 【核心】危大工程 → 受哪些规范监管（多标准交叉）
MATCH (h:HazardCategory)-[r:REGULATED_BY]->(s:Standard)
RETURN h, r, s
```

```cypher
// ③ 深基坑工程的完整关联子图（2 跳）
MATCH p = (h:HazardCategory {name:'深基坑工程'})-[*1..2]-(x)
RETURN p LIMIT 100
```

```cypher
// ④ 规范之间的上位法层级（强制性国标 → 行业标准）
MATCH p = (a:Standard)-[:HIERARCHY]->(b:Standard)
RETURN p
```

```cypher
// ⑤ 条款引用了哪些其他规范
MATCH (c:Clause)-[r:REFERENCES]->(s:Standard)
RETURN c, r, s LIMIT 50
```

```cypher
// ⑥ 各类型节点数量统计（表格形式）
MATCH (n) RETURN labels(n)[0] AS 节点类型, count(*) AS 数量
```

> 提示：图形界面里如果节点太挤，可以点节点→右键→"Dismiss"隐藏，或用
> `LIMIT` 控制数量。想看清结构建议每次只查一种关系。

---

## 四、不用写 Cypher 的封装接口（命令行直接问）

```powershell
cd "<REPO_ROOT>\rag2"

# 双引擎检索：图谱子图扩展 + 向量检索融合
& "C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe" -m src.graphrag.hybrid_search `
  --query "深基坑开挖前要做哪些安全准备？" --top-k 8

# 只做图谱子图扩展（看某危大类别关联到哪些规范/条款）
& "C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe" -m src.graphrag.neo4j_store `
  --query "深基坑" --hops 2

# 重新写入图谱（先 --build 生成，再 --load 写库）
& "C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe" scripts\build_demo_graph.py
& "C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe" -m src.graphrag.neo4j_store --load
```

---

## 五、停止 / 重启 Neo4j

```powershell
# 停止：找到占用 7687 端口的 java 进程并结束
$pid = (Get-NetTCPConnection -LocalPort 7687 -State Listen).OwningProcess
Stop-Process -Id $pid -Force
```

```powershell
# 重启：再跑一次方式 A 或 B
```

> 数据是持久化的（存在 `neo4j-community-4.4.8\data\`），重启后图谱还在。

---

## 六、扩大到 81 部规范的耗时（实测估算）

| 阶段 | 规模 | 耗时 |
|---|---|---|
| 图谱构建（Python） | 13661 条款 + 81 规范 → ~3.4 万条关系 | **约 0.5 秒**（实测 26496 条/秒） |
| 写入 Neo4j（批量 UNWIND） | 约 1.4 万节点 + 3.4 万关系 | **约 30-60 秒** |
| **合计** | | **约 1 分钟** |

> 前提：`scripts/build_full_graph.py` 已写好（含编号智能匹配、12 类危大工程、
> 上位法映射）。风险点：引用抽取基于正则，`GB 50010`（混凝土设计规范）等
> **不在 81 部内的外部规范**会被过滤，需人工校准。

---

## 七、常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `已添加了具有相同键的项` | 官方 PS 脚本与新 PS 不兼容 | 用方式 A/B（java 直启） |
| 连不上 `localhost:7687` | 服务没起来 / 进程被终端关闭杀死 | 确认日志有 `Started.`；用独立窗口跑 |
| `CredentialsExpired` | 首次登录必须改密码 | 已改为 `neo4j123456` |
| 浏览器打不开 7474 | 服务没启动或端口被占 | `Test-NetConnection 127.0.0.1 -Port 7474` |
| 查询结果为空 | 图谱没写入 | 跑 `neo4j_store --load` 写入 |
| 图谱节点乱码 | 编码问题 | 启动时已带 `-Dfile.encoding=UTF-8` |
