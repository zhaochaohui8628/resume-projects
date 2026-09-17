# agent —— 编排 + UI + 报告（产品主入口）

把 rag2 / ner2 / rules 组装成可用产品：**意图路由调度中心 + 方案审查 subagent（唯一，内部三路）+ LLM 汇总**（v7.2）。

## 调度架构（v7.2）

```
用户输入 / 上传方案
   │
   ▼
调度中心 orchestrator/dispatcher.py
   ├─ intent_route()：意图路由（问什么查什么；intent / agents / checks / scope_terms）
   │     hazard_level  是否危大·超规模·需论证        → review（内部只走判档路）
   │     technical     施工工艺/工序/做法/参数是否合规 → review（内部只走技术核对路）
   │     basis         编制依据/废止引用            → review（内部只走依据路 C1）
   │     elements      缺项/九章要素                → review（内部只走要素路 C4+C2）
   │     qa            无方案 / 一般条文咨询         → qa
   │     review        全面审查                    → review（内部五路全开）
   ├─ execute() / execute_async()：并行执行（AsyncPipeline + 令牌桶 + 共享上下文）
   │     review ── ★ 方案审查**唯一** subagent，内部三路，输出一份统一风险清单
   │               · 判档路    ：阈值表查表（不检索）
   │               · 技术核对路：多形态短查询并集 → comparator 比限值 → LLM 兜底（做法类）
   │               · 依据/要素路：规则库 C1 废止引用 / C4 九章 / C2 必备内容
   │     qa     ── RAG 条文检索（直接用 query，不做方案上下文截断）
   ├─ summarize()：LLM 汇总 → 最终答复
   └─ trace：全链路步骤（input → dispatch → execute[review 内部各步] → aggregate → done）
```

> **v7.2 结构变更（2026-09-13，用户定稿）**：
> ① **方案审查收敛为一个 subagent（`review`）**——判档 / 技术核对 / 依据与要素三路都在它**内部**完成，
>  产出**一份**统一风险清单；原先拆出的 `hazard` subagent 与独立"规则工具节点"已**删除**
>  （规则检查不再是一个单独的执行单元）。
> ② **删除"必须走的全量规则审查"**：v6.1 每次输入都先跑一遍 C1/C2/C4 再按策略注入，用户问
>  "施工工艺有什么不符合规范"时跑的是依据/缺项检查，答非所问。v7 起改为 **用户输入 → 调度中心**，
>  由意图路由决定 review 内部走哪几路（`ctx.checks`）。
> ③ **严格模式（默认 `GlobalOpts.allow_degrade=False`）**：NER 模型或检索不可用时**直接报错**，
>  不静默降级成"纯规则层 + 空检索"——那会把漏报包装成"无风险"。判档路不依赖模型与检索，永远可用。
> ④ **删除 `rules_checker.SCALE_PATTERNS`**（基坑5m/模板8m/脚手架50m 硬编码 + 全文取首个匹配），
>  危大/超规模判档统一由 `tools/hazard_level.py` 阈值表完成（9 类 / 22 组维度，输入为工况三元组）。
> ⑤ **`rag_client` 只对接 rag2**（择塔优先级 `dual_mix` → `dual_gold` → `tower_base`，RRF 融合）——
>  此前 agent 一直指向已作废的 `data/vector_db`（旧 rag 索引，2026-09-13 已从磁盘清除），
>  导致技术核对路**永远检索不到条文**；RagClient 构造绝不抛异常，失败只记 `errors`。
> **v7（2026-09-11）**：NER 全面切换到 **ner2** 项目（级联管道：规则层 + 微调模型层），
> **完全舍弃三层漏斗**（旧 plan_funnel 已删除）；review/qa 均不对方案做收敛/截断——
> review 对完整方案全文做 **ner2 全量分块识别**（10 万字级 ~5s）。

## v6 新特性（并发控制 / 防御性控制流 / 评测 Benchmark）

### 1. 并发控制：异步事件驱动 + 令牌桶 + 共享内存上下文 + 动态依赖裁剪

不再「盲目多线程并行发送请求」（会带来限流/超时雪崩/状态竞争）。新增
`agent/src/concurrency/`（纯标准库 asyncio，零新增依赖）：

- **TokenBucket**：令牌桶限流大模型推理并发。`llm_rate>0` 时路由/汇总/ReAct
  等所有 LLM 调用统一经 `RateLimitedLLM`（同步）或 `acomplete()`（异步）限流；
  调用失败自动回滚令牌。
- **SharedContext**：共享内存上下文。`publish()/subscribe()` 实现多 Agent 间
  **非阻塞状态同步**（如 compliance 检出的危大类型 → 注入 hazard，避免重复规则自检）；
  `wait()/await_()` 支持依赖等待；`has()/get(default)` 提供**动态依赖裁剪**判定依据。
- **AsyncPipeline**：异步事件驱动流水线。节点按依赖 DAG 执行；**硬依赖**任一失败
  → 下游自动裁剪（skipped）；**软依赖**等待上游完成（成败都继续）配合共享状态做
  非阻塞同步；asyncio.Semaphore 控制最大并发、单节点支持 timeout/retries。
- **run_async()**：全异步编排入口（`asyncio.run(run_async(...))`），返回结构
  与 `run()` 完全一致；同步 `run()` 保持兼容。

### 2. 防御性控制流：图状态机 + 节点转换校验器 + 状态向量余弦拦截（v6.1 无硬编码截断）

**已删除硬编码 5 步工具调用截断（max_tool_calls）**——它治标不治本（第 3 步陷入死胡同
会被强行截断 → 输出残缺/逻辑崩塌）。防死循环完全由图状态机承担：

- 把 ReAct 循环建模为**图状态机**（THINK→TOOL→OBSERVE→ANSWER/GUARD/DEGRADE）；
- **TransitionValidator** 挂在 THINK→TOOL 转换边：重复调用同一工具 ≥2 次、或
  2-3 轮中间结论的状态向量（特征哈希词袋）余弦 ≥ 阈值（默认 0.92）→ 拦截；
- 拦截后**主动引导**：注入「换一个技能 / 换检索角度 / 直接降级 answer」的纠偏指令
  （max_recover 次）；纠偏耗尽仍不收敛 → **降级输出**（返回已收集证据 + 明确说明未收敛）；
- `max_iterations` 仅作整体推理轮数的循环上界（防御性保险，非工具调用截断）；
- trace 记录 `guard`（拦截）步骤与 `guard_reasons`；返回 `degraded` 标志。

### 3. 多 Agent 级联容错评测 Benchmark（agent/bench/）

评估不止于「合成 12 项零漏报 / 真实方案 24 项」。新增 `agent/bench/`：

- `cases.py`：12 类**对抗性测试用例**（空输入 / 二进制乱码 / 90K 超长 / 重复文本 /
  提示注入 / 语义混淆 / 文本截断 / 自相矛盾 / 领域外 / 简繁全角编码混杂 /
  模板字符嵌套 / 多危大工程），带 golden 标注（风险片段 / 实体 / 条文源）；
- `cases_real.py`：**真实施工方案用例集**（11 本真实方案 → **89 用例**，交叉验证标注）
  —— 数据源 `data/raw/plans_internal`（6 本）+ `data/raw/plans_xproj`（5 本）：
  33 条「概况+编制依据」精标 + 56 条主题切片（基坑/吊装/高支模/脚手架/拆除/临电/有限空间/塔吊/应急），
  合计 **101 条（含合成 12 条）**。golden 经「规则引擎机械标注 ↔ 模型人工标注」交叉验证，
  `data/REVIEW_GUIDE.md` 列出规则缺陷/存疑版本/口径议题供领域专家复核；
- `metrics.py`：全链路指标拆解——**审查漏报率 FNR / 误报率 FPR / NER 实体 F1 /
  RAG recall@k / 耗时三级拆解**：
  ① `total_ms` 端到端；② `subagent_ms` 各 subagent **单跑基准**（非并行分段，与 total 不可相加）；
  ③ `phase_ms` **阶段拆解**（从 orchestrator trace 的 `ms` 聚合）——`route_ms` 意图路由 /
  `execute_ms` subagent 执行（NER 分块 + 检索精排 + 阈值与限值比对）/ `summary_ms` LLM 汇总 /
  `other_ms` 输入准备。⚠️ **未带 `--llm` 时路由走关键词规则（≈0 ms）、汇总直接跳过（0 ms）**，
  所以 895 ms 这类端到端数字**不含 LLM 往返**；带 `--llm deepseek` 才有真实模型耗时；
- `runner.py` + `report.py`：端到端跑 orchestrator，产出 Markdown 报告（含阶段拆解行）；
- CLI：`python agent/bench/run_benchmark.py --quick [--out] [--json]`（零依赖可跑），
  `--real` 追加真实施工方案用例集，`--llm deepseek` 覆盖真实路由/汇总链路。

> 耗时埋点在 `orchestrator/dispatcher.py`：`emit()` 给每条 trace 步自动写 `ms`
> （= 自上一步以来的耗时），`subagent_step` 的批次回放步标 `ms=None` 不计入。
> 端到端耗时 = 各步 `ms` 累加，无需额外埋点即可拆解路由 / 执行 / 汇总。

## 两个 subagent（review / qa）

| subagent | 职责 | 只管 | 明确不管 |
|---|---|---|---|
| `review` | **方案审查（唯一审查出口）**，内部三路 | 判档（查阈值表）+ 工艺/参数/做法相符性（短查询 + comparator + LLM 兜底）+ 依据/要素（C1/C4/C2） | 把方案拆成多份分别审 / 多实体拼一条 query |
| `qa` | 规范问答 | RAG 检索规范库条文 | 方案内容审查 |

> 路由示例（v7.2 问什么查什么，**都进同一个 review**）：
> `"审查方案编制依据"` → checks=[basis]（只走依据路 C1）；
> `"帮我核查危大方案施工工艺内容"` → checks=[technical]（只走技术核对路）；
> `"这个基坑开挖深度算超规模吗"` → checks=[hazard_level]（只走判档路，**不检索**）；
> `"全面审查方案"` → checks=[五路全开]。

## v5 新特性

1. **subagent 完整描述注册**：每个 subagent 在 `dispatcher.AGENT_DOC_FULL` 里写全输入/输出/适用场景/边界，路由 prompt 一次性交给 LLM，语义精确路由。
2. **全局运行时开关**（`GlobalOpts`）：主界面 RAG 精排开关、NER CRF 开关，透传到所有相关 subagent 统一生效。
3. **RAG 溯源下沉**：检索层（rag 项目）统一补 `rank / score / source / clause_no / source_path`，subagent 直接透传，界面 Dataframe + JSON 展示。
4. **记忆 flush**：`MemoryManager.flush()` 主动强制落盘（fsync）+ 返回摘要，UI 提供按钮。
5. **全链路 trace 可视化**：`dispatcher.run()` 返回 `trace`（输入 → 调度 → subagent 内部流程 → LLM 汇总），UI 用 Mermaid + 折叠明细渲染。

## 目录

```
src/orchestrator/   dispatcher（意图路由 + global_opts + trace + 异步编排 v7.2）
                    · intent（问句意图 → intent/agents/checks/scope_terms）
                    · aggregator（LLM 汇总，接收 review 的统一风险清单）
src/subagents/      base（SubAgentResult + trace/rag_hits）· review（唯一审查 subagent，内部三路）· qa
src/concurrency/    ★ v6 并发控制：token_bucket（令牌桶限流）· shared_context（非阻塞状态同步）
                    · pipeline（异步事件驱动 DAG + 动态依赖裁剪）
src/tools/          rules_checker(C1/C4/C2，规则库查表 · 被 review 内部调用) · hazard_level(危大阈值判档)
                    · param_extractor(裸量名+句内数值 → 工况三元组) · comparator(条文限值比对 + LLM 兜底)
                    · rag_client(只对接 rag2 + 溯源补齐) · ner_client(ner2 全量级联)
                    （范围收窄由 review 内部按问句章节完成，无独立工具节点）
src/harness/        skills（渐进披露）· memory（四层 + flush）· react_agent（防御性控制流 v6.1，无硬编码截断）
                    · state_machine（图状态机 + 转换校验器 + 余弦拦截）
src/llm/            base / DeepSeek（api_key 可注入）/ Mock
src/report/         Markdown / Excel 导出
src/fastapi_app.py  ★ Web UI 服务（FastAPI + uvicorn：任务队列 + SSE 流式；替代旧 http.server web_app）
src/ui/index.html   ★ 前端页面（纯 HTML/CSS/JS：设计系统 + 卡片布局 + 交互反馈 + 自绘链路）
src/agent/          compliance_pipeline（确定性主链路封装，CLI 与 UI 共用）
bench/              ★ v6 多 Agent 级联容错评测 Benchmark（cases 对抗用例 / metrics / runner / report / CLI）
scripts/run_check.py  CLI 合规自查
data/rules/         ★ 3 份规则库 JSON（C1/C2/C4 判定依据）
data/memory/        harness 记忆落盘（episodic.jsonl / semantic.json / procedural.json）
```

## 运行

> 以下命令在 **Windows PowerShell** 中、项目根目录执行。

```powershell
# 本机唯一推荐解释器（带 CUDA）；项目根目录也有 .\start.ps1 可一步到位
$PY = "C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe"

# 【一键】环境自检 + 启动 UI（等价于下面手动两条）
.\start.ps1

# 【UI 唯一入口】FastAPI 部署（任务队列 + SSE 流式；旧 http.server web_app 已回收）
& $PY agent/src/fastapi_app.py            # -> http://127.0.0.1:7860/（自动打开浏览器，Ctrl+C 停止）
#   换端口： $env:AGENT_UI_PORT = "8000"; & $PY agent/src/fastapi_app.py
#   或：     & $PY -m uvicorn agent.src.fastapi_app:app --host 127.0.0.1 --port 7860
#   接口：/api/health · /api/upload · /api/stream(SSE) · /api/tasks(任务队列) · /api/flush · /docs

# CLI 合规自查（确定性主链路，真实 RAG 依据；NER 走 ner2 级联全量识别）
& $PY agent/scripts/run_check.py agent/tests/fixtures/sample_plan.txt
#   -> agent/outputs/report.md（+ report.xlsx；高危项带真实规范条文依据）
#   换成真实方案就是同一个命令： & $PY agent/scripts/run_check.py .\方案.txt（10 万字级 ~5s）
#   （想走真实 DeepSeek：末尾加 --deepseek）

# 设置 DeepSeek Key（不设则 LLM 路由/汇总/ReAct 走 mock；检索/规则/判档/NER 不受影响）
$env:DEEPSEEK_API_KEY = "sk-..."
```

### 部署技术选型（v7.3）

**FastAPI + uvicorn**（替代旧 `http.server` web_app，旧部署层已回收）：

- **任务队列**：`POST /api/tasks` 创建后台编排任务（asyncio.Task），
  `GET /api/tasks/{id}` 查状态/结果，`POST /api/tasks/{id}/cancel` 取消；
- **异步并发 IO**：`dispatcher.run_async` 全 asyncio（AsyncPipeline 事件驱动 + 令牌桶限流），
  非阻塞 SSE 推送；
- **图状态机余弦拦截**：harness 内（react_agent + state_machine）承担，trace 原样推送；
- 前端仍为纯 HTML/CSS/JS（`agent/src/ui/index.html`），FastAPI 仅作后端部署层。

接口一览：

| 接口 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 前端页面 |
| `/api/health` | GET | 健康检查（含已载入方案字数） |
| `/api/upload` | POST | 上传方案（JSON：`{filename, b64}`） |
| `/api/stream` | GET | SSE 流式编排（事件：`ping` / `step` / `token` / `done` / `error`；`token` = 终答逐块增量，review 与 qa 两路都有） |
| `/api/tasks` | POST | 创建后台编排任务（返回 task_id） |
| `/api/tasks/{id}` | GET | 任务状态与结果 |
| `/api/tasks/{id}/cancel` | POST | 取消任务 |
| `/api/flush` | POST | 记忆 flush（返回落盘摘要） |
| `/docs` | GET | OpenAPI 自动文档 |

## 测试

`agent/tests/`：
- `test_orchestrator_v5.py` —— v7 调度：意图路由表 / **按需**规则工具（不再是固定前置）/ 技术类问句
  不触发规则工具 / 范围收窄 / RagClient 构造不抛异常 / trace / 溯源 / 记忆 flush
- `test_triage.py` —— P0 分诊链路：三元组绑定（语序·距离·量纲）/ 阈值查表（含脚手架子类型消歧、
  单位换算、标高负值纠偏）/ 条文限值比对（含 OCR "7. Om" 清洗）/ LLM 兜底与安全降级
  （HIGH 强制·用户提及·低风险抛弃）/ GlobalOpts 透传 / trace（含 rules 步骤）/ RAG 溯源 /
  hazard 不再自检 / 记忆 flush（16 项）
- `test_concurrency.py` —— v6 并发控制：令牌桶 / RateLimitedLLM 回滚 / SharedContext 非阻塞同步
  / AsyncPipeline 硬依赖裁剪 + 软依赖共享 / dispatcher.run_async 端到端（12 项）
- `test_harness.py` —— 渐进披露 / 防御性控制流（重复工具拦截·相似结论拦截·纠偏后收敛·降级输出，
  无硬编码工具上限）/ 四层记忆（10 项）
- `test_bench_smoke.py` —— 评测框架：漏报/误报 / NER F1 / RAG recall@k / 耗时拆解 + **阶段拆解** / 汇总（8 项）
- `test_agent_mock`（确定性主链路 12 风险 + ReAct）、`test_rules`

```powershell
& $PY -m pytest agent/tests
```

## 评测 Benchmark（v6）

```powershell
# 快速全量（12 对抗性用例，零依赖确定性路径）
& $PY agent/bench/run_benchmark.py --quick --out --json
# 追加真实施工方案用例集（11 本真实方案 → 33 用例）
& $PY agent/bench/run_benchmark.py --quick --real --out --json
# 指定用例
& $PY agent/bench/run_benchmark.py --cases empty_input,real_04_tadiao_a
# 带真实 LLM 链路（路由/汇总/ReAct，需 DEEPSEEK_API_KEY）
& $PY agent/bench/run_benchmark.py --llm deepseek --real --out
# 报告输出：agent/bench/outputs/bench_report_<时间戳>.md
```
