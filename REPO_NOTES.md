# 仓库版说明（REPO_NOTES.md）

> 本仓库是 **施工方案合规审查** 项目的**公开发布版**。
> 内容 = 全部源代码 + 文档 + 公开规范语料（条款级 JSONL）+ 合成 / 脱敏后的评测数据。
> **不含** 模型权重、规范 PDF 原文、检索索引、真实项目方案数据 —— 原因与恢复方式见下。

---

## 一、相对本地开发版的差异

| 未入仓内容 | 原体积 | 原因 | 本地恢复方式 |
|---|---|---|---|
| `data/models/` | ~13.8 GB | 基座与训练权重（`Qwen2.5-3B-Instruct`、`qwen-grpo/*`、`bge-small-zh-v1.5`、`bert-base-chinese`） | 从 HuggingFace（国内可用 `hf-mirror.com`）下载同名模型放入该目录 |
| `data/raw/` | ~525 MB | 82 本规范 PDF，含版权内容 | `python scripts/scg_fetch.py`，清单见 `scripts/fetch_list.txt`（来源：上海建工标准库） |
| `rag2/data/models/` | ~1.4 GB | 双塔（`dual_gold` / `dual_mix`）与 CrossEncoder（`cross_v2_ep4`）训练产物 | 用 `rag2/src/train/dual_tower`、`rag2/src/train/cross_encoder` 重训，或 `rag2/scripts/train_cross_encoder.py` |
| `rag2/data/index/` | ~154 MB | BM25 + FAISS 索引，可由语料确定性重建 | `python rag2/scripts/build_index.py`（双塔索引：`build_dual_index.py`） |
| `rag2/data/corpus/_baseline_clauses.jsonl` | ~11 MB | baseline 对照副本（与 `clauses.jsonl` 重复） | 由 `clauses.jsonl` 复制 |
| `rag2/data/eval/*.jsonl` | ~1 MB | 评测中间产物 | 重跑 `rag2/scripts/eval_*.py` 生成 |
| `ner2/models/` | ~776 MB | NER 检查点（`s2_crf_param_v3` 等） | `python ner2/scripts/train_ner.py` |
| `neo4j-community-4.4.8/` | ~139 MB | 第三方二进制，仅重建 GraphRAG 图谱时需要 | 自行下载 Neo4j 4.4.x 社区版 |
| `agent/bench/data/real_fragments*` | ~1.4 MB | **真实项目方案原文切片**（含具体项目名称） | 不提供 |
| `agent/bench/data/real_cases*.jsonl`、`rule_engine_mechanical*.json` | ~200 KB | 同上，标注含真实项目标识 | 不提供 |
| `agent/data/memory/` | ~20 KB | 运行时记忆，存有真实方案正文 | 首次运行自动生成 |
| `简历模板/` | ~0.4 MB | 个人简历，与代码无关 | 不提供 |

> `agent/bench/` 保留了框架代码（`runner.py` / `metrics.py` / `report.py` / `cases.py`），
> 其中 `cases.py` 是**合成用例**（已写作"某医院"），可直接跑通 benchmark 流程。

## 二、最小可运行路径

只跑「代码 + 公开语料」这条最小链路时：

```powershell
# 1) 解释器（推荐带 CUDA 的环境；纯 CPU 也能跑，只是慢）
$PY = "C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe"

# 2) 依赖
& $PY -m pip install -r requirements.txt

# 3) 条款语料已随仓库提供（rag2/data/corpus/clauses.jsonl，12720 条款），
#    只需重建检索索引即可跑通检索：
& $PY rag2/scripts/build_index.py            # 建 BM25 + FAISS 索引

# 4) 不依赖检索的部分（规则判档 / 提示词构造 / 报告渲染）装完依赖即可跑
```

## 三、数据脱敏说明

仓库中的评测数据来自真实工程方案，已做**项目标识脱敏**：

- 医院 / 项目名称 → 通用占位（`某医院`、`某医疗中心`、`某医学中心`）
- 施工单位名称 → `某施工单位`
- 数据源标识统一为工序名（如 `04_高支模`），不再附带项目后缀
- 共处理 **3741 处**替换，同步重算了 NER 标注的字符偏移量
  （校验：80162 条记录 / 36240 个实体，JSON 零错误、偏移零越界）

未脱敏的是**公开规范条文**（`rag2/data/corpus/clauses.jsonl`）以及**规范编制单位名单**——
这些属于标准文本的公开信息。

## 四、边界声明

- 本仓库是**研究与工程演示**用途，不构成任何合规审查的最终结论。
- 文档中的绝对路径已统一替换为占位符（`<REPO_ROOT>`、`C:\Users\<用户名>`），
  首次运行时按本机实际情况调整。
- `env/bootstrap.py check` 校验的是**完整开发机**的数据指纹；
  本仓库裁剪了模型/PDF/索引，该命令不会通过，属预期行为。
