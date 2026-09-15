# 施工方案合规审查（rag2 + ner2 + agent + grpo）

> 📊 **简历口径数据（唯一事实源）→ [`RESUME_NUMBERS.md`](RESUME_NUMBERS.md)**：本文与各子 README 只引用该文件，不另写口径数字。
> 📁 **文档索引 → [`docs/README.md`](docs/README.md)**（历史文档在 `docs/archive/`）
> 📦 **仓库版说明 → [`REPO_NOTES.md`](REPO_NOTES.md)**：本仓库为公开发布版，**不含**模型权重 / 规范 PDF / 检索索引 / 真实项目方案数据——差异清单与恢复方式见该文件。

面向建筑工程领域的**施工方案（专项方案）合规预审**：自动识别四类问题——C1 引用已废止规范 / C2 危大工程缺项·超规模未论证 / C3 强条疑似违反 / C4 编制要素缺失。交付 = 自查报告（Markdown/Excel）+ 可交互 Web UI。

## 一、30 秒跑起来

> 所有命令都在 **Windows PowerShell** 中、**项目根目录**执行（即同时含 `rag2/`、`agent/`、`data/` 的那一层）。

```powershell
cd <REPO_ROOT>

# 一键：环境自检 -> 启动 Web UI（自动打开浏览器）
.\start.ps1
```

就这一条。脚本会自动挑 Python 解释器、检查索引与模型是否在位，然后起服务（`http://127.0.0.1:7860/`）。

另有一个**独立 demo**（不属于主链路）：GraphRAG 规范知识图谱，一键启动：

```powershell
.\graphrag\start_demo.ps1        # 自动起 Neo4j + 建图 + 前端（http://127.0.0.1:7870/）
```

如果 PowerShell 提示"无法加载文件…因为在此系统上禁止运行脚本"，先放行一次：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 二、启动脚本 `start.ps1`

| 命令 | 作用 |
|---|---|
| `.\start.ps1` | 环境自检 + 启动 Web UI（默认动作） |
| `.\start.ps1 check` | 只做体检：依赖 / 数据指纹 / 模型 / 冒烟检索（不启动服务） |
| `.\start.ps1 ui` | 只启动 Web UI |
| `.\start.ps1 test` | 跑全量回归测试（agent + rag2 + ner2 + grpo） |
| `.\start.ps1 review .\方案.txt` | CLI 合规自查一份方案 → `agent/outputs/report.md` |
| `.\start.ps1 demo` | 用内置样例方案演示 CLI 自查（零配置、无 Key 可跑） |
| `.\start.ps1 help` | 打印帮助 |

常用可选参数：

```powershell
.\start.ps1 ui -Port 8000            # 换端口（默认 7860）
.\start.ps1 -Py "C:\path\to\python.exe"   # 手动指定解释器
.\start.ps1 check -Install           # 体检发现缺依赖时自动 pip install
.\start.ps1 review .\方案.docx -Llm   # 走真实 DeepSeek（需先设 Key，见下）
```

> ⚠️ 脚本会设 `OPENBLAS_NUM_THREADS=1 / OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1`。
> **原因**：本机 16 核 + 16 G 内存，OpenBLAS 默认按核数分配线程缓冲，
> 在「加载 torch + sentence-transformers + FAISS 索引」时会报
> `OpenBLAS error: Memory allocation still failed after 10 retries` 或 faiss `MemoryError: std::bad_alloc`。
> 限制为单线程后内存占用线性下降，检索 / 自检 / NER 全部正常。
> 手动跑 `python rag2/scripts/eval_retrieval.py` 或 `env/bootstrap.py check` 时，记得先设这三个变量。

## 三、环境

**本机唯一推荐解释器（带 CUDA，RTX 4060 Ti）**：

```powershell
$PY = "C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe"
```

- `start.ps1` 会按 `torch_gpu → 托管 venv → PATH 上的 python` 顺序自动探测，一般不用手填。
- ⚠️ **别用** `C:\Users\<用户名>\anaconda3\python.exe`（base）——那是 CPU 版 torch，检索/精排/NER 会慢很多且可能加载失败。
  若某台机器上**没有** `envs\torch_gpu`，脚本会一路退到 PATH 上的 `python`（常是 base），此时自检会提示「建议改用 torch_gpu 环境」——
  用 `-Py` 显式指定那台机器的正确解释器即可，脚本不会替你改环境。
- 依赖清单见 [`requirements.txt`](requirements.txt)；装依赖：

```powershell
& $PY -m pip install -r requirements.txt
```

**DeepSeek Key（可选）**：不设 Key 时 UI 与 CLI 走确定性 mock 路径，检索、规则、判档、NER 全部照常工作；只有"LLM 路由/汇总/ReAct 自然语言描述"会退化。

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
```

## 四、四个模块

| 目录 | 定位 | 状态 | 入口 |
|---|---|---|---|
| [`rag2/`](rag2/README.md) | 规范检索：条款级父子块 + BM25 ∥ 双塔向量 + RRF 融合 + CrossEncoder 精排 | ✅ 可用 | `rag2/README.md` |
| [`ner2/`](ner2/README.md) | 实体抽取：规则/词典层 + 微调 CRF 模型层两层级联，6 类实体 | ✅ 可用 | `ner2/README.md` |
| [`agent/`](agent/README.md) | 编排 + UI + 报告：意图路由 → `review`（唯一审查 subagent，内部三路）/ `qa` → LLM 汇总 | ✅ 端到端 | `agent/src/fastapi_app.py` |
| [`grpo/`](grpo/README.md) | Qwen2.5-3B 的 GRPO→SFT→GRPO 两轮迭代强化学习（四维连续奖励） | ✅ 训练完成 | `grpo/README.md` |

数据流：

```
82 本规范 PDF ──条款级三级切分──> 13661 父块 / 14916 检索单元（81 本入库）
                                   ├── BM25  ∥  双塔 bge 向量(FAISS)  ──> RRF 融合 ──> CrossEncoder 精排 ──> top-k
agent/data/rules/*.json（3 份规则库）──> C1 废止引用 / C2 危大阈值判档 / C4 九章要素（确定性查表，不走检索）
                                   │
        rag2（检索）      ner2（实体）      agent（编排 + UI + 报告）
                                   │
                    用户输入 → 意图路由 → review（判档 / 技术核对 / 依据要素，三路内部完成）/ qa → LLM 汇总
```

## 五、手动命令（不想用脚本时）

```powershell
$PY = "C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe"

# Web UI（自动打开 http://127.0.0.1:7860/）
& $PY agent/src/fastapi_app.py

# CLI 合规自查 -> agent/outputs/report.md（+ report.xlsx）
& $PY agent/scripts/run_check.py agent/tests/fixtures/sample_plan.txt

# 全量回归测试（分模块跑；四个 tests/ 目录同名，一次全给 pytest 会互相覆盖收集）
.\start.ps1 test
#   等价于逐条：
foreach ($t in 'agent/tests','rag2/tests','ner2/tests','grpo/tests') { & $PY -m pytest $t -q }

# 环境/数据自检（换机器后第一件事；注意内存紧张时先设 BLAS 单线程，见 §七）
& $PY env/bootstrap.py check

# 检索质量评估（黄金集）
& $PY rag2/scripts/eval_retrieval.py --tag gold --mode convex --index dual_gold.faiss
```

## 六、目录约定

| 路径 | 内容 |
|---|---|
| `agent/data/rules/*.json` | C1/C2/C4 规则库（**事实真相源**，grpo 训练数据也由它派生） |
| `agent/src/fastapi_app.py` · `agent/src/ui/index.html` | Web UI 后端 / 前端 |
| `agent/outputs/` | CLI 报告输出（`report.md` / `report.xlsx`） |
| `rag2/data/corpus/` · `rag2/data/index/` | 条款语料 / BM25 + FAISS 索引 |
| `rag2/data/models/` | 双塔（`dual_mix` 现行 / `dual_gold` 备选）+ CrossEncoder（`cross_v2_ep4`） |
| `ner2/models/` | 实体模型（`s2_crf_param_v3` 现行） |
| `data/raw/` | 规范 PDF（`standards/` `guidelines/` `plans_internal/` `plans_xproj/`） |
| `data/models/` | 基座与训练产物（`Qwen2.5-3B-Instruct`、`qwen-grpo/r2_final`、`bge-small-zh-v1.5`、`bert-base-chinese`） |
| `config/config.yaml` | 检索/精排运行配置 |
| `env/` | 环境快照与自举（`bootstrap.py`） |
| `docs/` | 文档索引；`docs/archive/` 为已归档历史文档 |
| `graphrag/` | **独立 demo（非落地链路）**：GraphRAG 规范知识图谱 —— Neo4j 后端 + 单路/双路前端 + 一键启动 `start_demo.ps1` |
| `neo4j-community-4.4.8/` | 仅 GraphRAG demo 使用（需 JDK 11）；主链路运行时不依赖 |

## 七、常见问题

| 现象 | 处理 |
|---|---|
| 自检报「缺依赖 sentence_transformers」或「不可用（1 个致命问题）」 | 该机器上没有 `envs\torch_gpu`，脚本退到了 PATH 上的 python（多为 base，CPU 版 torch 且缺 sentence-transformers）。用 `-Py` 显式指定：<br>`.\start.ps1 check -Py "<该机器 torch_gpu 的 python.exe>"` |
| 脚本输出中文变成「鐩綍濂戠害」这类乱码 | Python 侧写 UTF-8、PowerShell 5.1 按系统代码页(GBK)解码不一致所致。脚本已统一设 `PYTHONIOENCODING=utf-8` + `[Console]::OutputEncoding=UTF8`；手动跑 python 时先 `$env:PYTHONIOENCODING = "utf-8"` |
| UI/trace 里显示 `CE精排=关`，但开关是打开的 | **精排被降级**了（CrossEncoder 加载失败会静默降级，只记 `rag_client.errors["rerank"]`）。常见原因是本机**提交内存不足**：`OSError: 页面文件太小，无法完成操作。(os error 1455)`。处理：调大 Windows 虚拟内存（页面文件 16–32 GB）或关掉占内存程序；`OPENBLAS_NUM_THREADS=1` 已由脚本设好 |
| `OpenBLAS error: Memory allocation still failed after 10 retries` 或 faiss `MemoryError: std::bad_alloc` | BLAS 按 16 核分配缓冲、内存不够。设 `$env:OPENBLAS_NUM_THREADS="1"`（`start.ps1` 已自动设）。**索引文件本身没问题**——已用文件头校验：`IxFI` + d=512 + ntotal=14916，长度精确等于 45+14916×512×4 |
| `faiss` 报 `DLL load failed` | 已知问题：faiss 依赖 `libiomp5md.dll`（在 torch/lib）。**不要裸 `import faiss`**，统一走 `rag2/src/index/dense.py::import_faiss()` |
| `import torch/transformers` 报 `No module named 'xxx'` | torch_gpu 环境缺依赖（本机曾缺 `huggingface_hub`、`pydantic-core`）。补装：<br>`& $PY -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple huggingface_hub accelerate "pydantic>=2.13.5" iniconfig` |
| 报 `SystemError: installed pydantic-core version ... incompatible` | pydantic 与 pydantic-core 版本错配，用上面同一条命令升级即可 |
| 从工具宿主启动的终端里报 `No module named 'typing_extensions'`（但包里明明有） | 宿主注入的 `PYTHONPATH` 里带 `sitecustomize.py` 在拦截导入，属假故障。`start.ps1` 已自动净化；手动跑时先 `$env:PYTHONPATH = ""` |
| 检索命中为空 / `rag2` 后端不可用 | 跑 `.\start.ps1 check` 看索引与模型是否在位；`dual_mix.faiss` 缺失时 rag_client 会退到 `dual_gold` → `tower_base` |
| UI 中文乱码 / 端口占用 | 换端口：`.\start.ps1 ui -Port 8000` |
| 报告说"无风险"但你觉得漏了 | 检查是否 `DEEPSEEK_API_KEY` 未设导致 LLM 环节退化；严格模式（默认）下 NER/检索不可用会**直接报错**而不是静默降级 |
| 改了语料/索引后自检报"低于下界阈值" | 那是真信号——按 `env/README.md` 重跑 `snapshot` 刷新基线并复核下界 |

## 八、更多文档

| 文档 | 内容 |
|---|---|
| [`RESUME_NUMBERS.md`](RESUME_NUMBERS.md) | 简历口径数字 + 边界声明（**唯一事实源**） |
| [`docs/README.md`](docs/README.md) | 文档索引 |
| [`agent/README.md`](agent/README.md) | 调度架构（v7.2）、接口表、测试与 Benchmark |
| [`rag2/README.md`](rag2/README.md) | 检索栈、索引构建、评估命令 |
| [`ner2/README.md`](ner2/README.md) | 级联抽取架构、7 阶段路线、训练与评估 |
| [`grpo/README.md`](grpo/README.md) | 强化学习训练链、8G 显存工程、评测矩阵 |
| [`env/README.md`](env/README.md) | 换机器 / 环境快照 / 自检原理 |
| [`graphrag/README.md`](graphrag/README.md) | **独立 demo**：GraphRAG 规范知识图谱（Neo4j 后端 · 单路/双路开关 · 一键启动 · 面试演示脚本） |
| [`graphrag/docs/NEO4J_GUIDE.md`](graphrag/docs/NEO4J_GUIDE.md) | Neo4j 部署、Cypher 查询与排障手册（demo 专用） |
| [`02_规范清单_对账.md`](02_规范清单_对账.md) | 规范库对账（82 本 → 81 本入库） |
| [`docs/archive/`](docs/archive/README.md) | 已归档历史文档 + 2026-09-13 清理记录 |

> **GraphRAG 的位置**：它是**独立演示模块**，不属于 rag2 / agent 的落地链路——
> rag2（检索）与 agent（编排）中不含任何图谱/Neo4j 代码或依赖，agent 的结构化 QA 走纯 RAG。
> 演示链路强制走 Neo4j（不回退 JSON），一键启动：`.\graphrag\start_demo.ps1`（前端 7870 / Neo4j Browser 7474）。
