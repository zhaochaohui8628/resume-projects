# archive —— 归档文档（不再维护）

> 这里的东西**只作历史证据/追溯用途**。内容写作时是准确的，但项目已演进，
> 文中出现的路径、目录名、命令、指标口径**可能已不存在或已失效**。
> 需要权威口径时请看 [`../../RESUME_NUMBERS.md`](../../RESUME_NUMBERS.md) 与各模块 README。

| 文件 | 原位置 | 归档原因 |
|---|---|---|
| `01_规范知识库清单.md` | 项目根 | v1 采购/选库清单（2026-08-31）。使命已完成——82 本 PDF 已下载并入库（81 本进语料），对账结论沉淀在 `02_规范清单_对账.md`，已无维护价值 |
| `rag2_MIGRATION.md` | `rag2/MIGRATION.md` | rag2 换机器迁移指南。迁移已完成；文中引用的 `dual_p1` 旧塔、`data/models/dual_p1` 目录已按 2026-09-13 清理删除。迁移/自检的现行入口改为 `env/README.md` + `env/bootstrap.py` |
| `rag2_NEO4J_GUIDE.md` | `rag2/docs/NEO4J_GUIDE.md` | GraphRAG 建图用的 Neo4j 部署说明。运行时不依赖 Neo4j 服务（agent 的 qa 路直接读离线图谱文件 `rag2/data/graphrag/demo_graph_v2.json`），仅重建图谱时才需要 |
| `cleanup_20260913.py` | `scripts/_cleanup_20260913.py` | 2026-09-13 项目清理脚本（一次性）。保留作清理审计证据；下方记录本次实际删除内容 |

## 2026-09-13 清理记录

**执行方式**：`cleanup_20260913.py --apply`（先 `--dry-run` 出清单、用户确认后执行）

**删除 85 项 / 32.13 GB**：

| 类别 | 内容 | 体积 |
|---|---|---|
| 缓存 | `__pycache__/`（agent/src 下 7 处）、`.pytest_cache/`（根 + rag2） | <1 MB |
| 日志 | `grpo/scripts/*.log`（17 个训练/调试日志）、`agent/_ui*.log`、`neo4j*/**/*.log` | ~1 MB |
| 临时脚本 | `scripts/_*.py`（30 个一次性环境诊断脚本）、空目录 `_env_fix_trash` / `_stdlib_backup` | <1 MB |
| 备份文件 | `data/train/*.bak_*`、`data/vector_db/bm25.json.bak_preparent` | ~1 MB |
| 旧检索路线 | `data/vector_db/`（旧 rag 的 BM25+FAISS 索引，v7.3.1 已移除代码引用）、`scripts/eval_real.py` | 96.8 MB |
| 旧塔与旧索引 | `rag2/data/models/dual_p1/`、`rag2/data/index/` 下 9 个废弃 faiss（`dual_p1*` / `dual_distill*` / `*_bak_cls`）及孤立 meta | ~445 MB |
| GRPO 中间权重 | `data/models/qwen-grpo/` 的 `s1_grpo` `s1_sft` `final` `r2_grpo` `r2_sft` 五档全量权重 | 31.7 GB |

**保留**：`data/models/qwen-grpo/r2_final`（唯一交付/评测权重）+ 6 个 LoRA adapter（训练链证据）；
`neo4j-community-4.4.8/`（用户要求保留）；`rag2/data/index/{dual_mix,dual_gold,tower_base}.faiss`（当前后端 + 兜底）。

**结果**：项目从约 49 GB 降到 **16.66 GB**。

**同步改动的引用**（删完不留死链）：
- `env/bootstrap.py`：自检项 `dual_p1` → `dual_mix`/`dual_gold`；`bge-reranker-base` → `rag2/data/models/cross_v2_ep4`
- `config/config.yaml`：移除旧 `retrieval.persist_dir`、`domain`、`training`、`paths` 段，新增 `retrieval.rag2` 段
- `agent/README.md`、根 `README.md`：目录表与"历史对照"链接改指 `docs/archive/`

## 顺手修复的环境问题（2026-09-13）

清理后跑 `env/bootstrap.py check` 暴露 **`torch_gpu` 环境多处依赖缺失 / 包体文件被剥离**。
经逐项核对，这些与本次清理**无关**（删除项全部在项目目录内），是环境自身的既有损坏：

| 症状 | 缺失 / 损坏 | 修复 |
|---|---|---|
| `import transformers` → `No module named 'huggingface_hub'` | huggingface_hub、accelerate 缺失 | `pip install huggingface_hub accelerate` |
| `import fastapi` → `SystemError: pydantic-core version ... incompatible` | pydantic 2.13.4 配 pydantic-core 2.46.4 | `pip install --upgrade pydantic pydantic-core` → 2.13.5 / 2.46.5 |
| `import sklearn` → `No module named 'threadpoolctl'` | threadpoolctl 缺失 | `pip install threadpoolctl` |
| `import sentence_transformers` → `cannot import name 'PeftMixedModel' from 'peft'` | peft 包体不完整 | `pip install --force-reinstall --no-deps peft` → 0.20.0 |
| `import PIL` → `cannot import name '_version'`（文件被剥离） | pillow 包体不完整 | `pip install --force-reinstall --no-deps pillow` → 12.3.0 |
| `import fitz` 失败 / UI 无法解析 PDF | pymupdf 包体不完整 | `pip install --force-reinstall --no-deps pymupdf` → 1.28.2 |
| `import pytest` 异常 | pytest 包体不完整 + 缺 iniconfig | `pip install --force-reinstall --no-deps pytest iniconfig` |
| `import jieba` 失败（ner2 规则层可选加速） | jieba 缺失 | `pip install jieba` |
| rich / sqlalchemy 缺传递依赖 | markdown-it-py、greenlet | `pip install markdown-it-py greenlet` |

> `ragas` 的一堆缺失依赖（langchain / datasets / instructor 等）**故意不装**——
> ragas 是可选评估通道（`rag2/scripts/eval_ragas.py --ragas`），不影响主链路。

安装源统一用清华镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`。

## 清理后的验证结论（2026-09-13）

**环境体检** `.\start.ps1 check`（= `env/bootstrap.py check`）

| 项 | 结果 |
|---|---|
| 1–6 项 | ✅ 全绿：目录契约 / 依赖（torch 2.7.1+cu118、transformers 5.17.0、sentence-transformers 6.0.1、faiss 1.15.0）/ GPU（RTX 4060 Ti，CUDA 可用）/ 模型（dual_mix、dual_gold、cross_v2_ep4 权重齐）/ 数据指纹 / 语料 **13661 条款 · 81 规范 · 14916 单元**（与快照一致） |
| 7 项 BM25 冒烟 | ✅ hit@5 **0.860** / MRR **0.806**（下界 0.800/0.750） |
| 7 项 双塔冒烟（dual_mix + RRF，线上配置） | 内存充裕时实测 hit@5 **0.930** / MRR **0.814**；内存紧张时会因提交内存不足偶发失败（见上一节） |

**回归测试**（⚠️ 四个模块的 `tests/` 目录同名，**必须分模块逐个跑**，一次全给 pytest 会互相覆盖收集）

| 模块 | 结果 |
|---|---|
| `agent/tests` | ✅ **83 passed** |
| `rag2/tests` | ✅ **27 passed** |
| `ner2/tests` | ✅ **39 passed** |
| `grpo/tests` | ✅ **12 passed** |

**清理未造成数据损坏**：三个 faiss 索引文件头校验为 `IxFI` + `d=512` + `ntotal=14916`，
文件长度 30,548,013 B **精确等于** 45 + 14916×512×4（已用独立读取路径反序列化成功并跑出检索结果）。

**本次一并修的真实缺陷**：`rag2/scripts/eval_retrieval.py` 的防呆只补 `--query-tower`、不补 `--doc-tower`
→ `dual_*.faiss` 漏给 doc 塔时会退回基座 doc 塔（与微调索引不同向量空间）→ **静默偏低**
（实测 hit@5 0.86 → 0.79）。已改为按 index 名成对补齐两塔，bootstrap 冒烟也显式给两塔。

