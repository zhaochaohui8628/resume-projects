# docs —— 项目文档索引

> 本项目所有"给人看"的文档入口。各子模块 README 在各自目录下。

## 读什么

| 你的目的 | 看哪里 |
|---|---|
| **跑起来** | 根目录 [`../README.md`](../README.md) §快速开始，或直接 `.\start.ps1`（见 §启动脚本） |
| 简历数字口径（唯一事实源） | [`../RESUME_NUMBERS.md`](../RESUME_NUMBERS.md) |
| 检索模块（rag2） | [`../rag2/README.md`](../rag2/README.md) |
| 实体抽取（ner2） | [`../ner2/README.md`](../ner2/README.md) |
| 编排 + UI + 报告（agent） | [`../agent/README.md`](../agent/README.md) |
| GRPO 强化学习训练 | [`../grpo/README.md`](../grpo/README.md) |
| GraphRAG 知识图谱演示（**独立 demo，非落地链路**） | [`../graphrag/README.md`](../graphrag/README.md) · Neo4j 手册 [`../graphrag/docs/NEO4J_GUIDE.md`](../graphrag/docs/NEO4J_GUIDE.md) |
| 换机器 / 环境自检 | [`../env/README.md`](../env/README.md) |
| 规范库对账（82 本 → 81 本入库） | [`../02_规范清单_对账.md`](../02_规范清单_对账.md) |
| rag2 开发进度与交接 | [`../rag2/docs/PROGRESS.md`](../rag2/docs/PROGRESS.md) |
| ner2 开发进度与参数口径 | `../ner2/docs/PROGRESS.md`、`../ner2/docs/PARAM_SPEC.md` |

## 归档

已失效但仍留作证据的历史文档在 [`archive/`](archive/README.md)。归档文档**不再维护**，
其中的文件路径、目录名、命令可能指向已删除的旧模块（如旧检索索引 `data/vector_db/`、旧迁移指南），
读的时候以当前代码为准（现行模块 = `rag2` / `ner2` / `agent` / `grpo`）。

> 2026-09-15：原归档文档 `rag2_NEO4J_GUIDE.md` 随 GraphRAG 抽离，已迁至独立 demo
> [`../graphrag/docs/NEO4J_GUIDE.md`](../graphrag/docs/NEO4J_GUIDE.md)。GraphRAG **不属于落地链路**，
> 相关演示步骤、单路/双路说明、Neo4j 排障都在 `graphrag/` 内。
