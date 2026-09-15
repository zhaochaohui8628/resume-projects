# rag2 —— 规范检索（条款级父子块 + BM25 ∥ 双塔 + RRF + CrossEncoder 精排）

> 替代已作废的旧 `rag/`（2026-09-13 起旧索引 `data/vector_db/` 已从磁盘清理）。
> 本模块是**唯一检索后端**，agent 通过 `agent/src/tools/rag_client.py` 调用。
> 简历口径数字见 [`../RESUME_NUMBERS.md`](../RESUME_NUMBERS.md) §1；开发进度与交接见 [`docs/PROGRESS.md`](docs/PROGRESS.md)。

## 检索栈（当前线上配置）

```
81 本规范 PDF
  └─ build_corpus.py   条款级三级切分（章 → 条 → 长条滑窗）
       └─ 13661 父块（条款级）
            └─ build_index.py   检索单元化：≤1200 字整条直用，>1200 切 700/100 滑窗
                 └─ 14916 检索单元 + bm25.json + tower_base.faiss

查询 ──┬─ BM25（惰性倒排，词面）
       └─ 双塔 bge 向量（FAISS IndexFlatIP，doc/query 均 L2 归一化，语义）
                 │
                 ├─ 融合：RRF（默认，用户裁定）/ 凸组合（温莎截断 min-max 全局归一，α=0.3）
                 │        ↑ 两路分数尺度差异大，必须先归一；BM25 边界由真实 query 命中分分布标定
                 └─ CrossEncoder 精排（cross_v2_ep4）：召回 top_k×oversample → 打分 → 截断 top_k
                         │
                         └─ 统一补齐溯源字段（rank / score / source / clause_no / source_path）
```

**双塔择塔优先级**（`rag_client._build_rag2`）：`dual_mix` → `dual_gold` → `tower_base`;对应索引同名 `.faiss`。

| 塔 | 含义 | 状态 |
|---|---|---|
| `dual_mix` | 黄金集 1000 + 混合 1422 条重训（v2） | ★ 现行（用户裁定） |
| `dual_gold` | 纯黄金集 1000 条重训（v1），RRF 表现更好 | 备选 |
| `tower_base` | 未微调的 bge-small-zh-v1.5 | 兜底 |

> `dual_p1` = **P1 冷启动塔的产物名**，不是"已上线产物"：它作为重训起点被消费完后已于 2026-09-13 清理，
> 需要时用 `src/train/dual_tower/train.py` 重训（约 47 分钟 CPU / 数分钟 GPU）。**各脚本的读路径默认值已统一改为 `dual_mix`。**

## 目录

```
rag2/
├── src/
│   ├── common/        paths（全部产物路径集中定义）· limits · normalize
│   ├── corpus/        pdf_reader · recursive_split（递归切分）· clause_split（条款级）
│   ├── index/         dense（FAISS + import_faiss 兜底）· bm25 · tokenize
│   ├── retrieval/     hybrid（Corpus/EncoderPair/HybridRetriever/load_retriever）· rerank · llm_summarize
│   ├── eval/          metrics（hit@k / recall@k / MRR，零依赖）
│   ├── train/         dual_tower（InfoNCE）· cross_encoder（BCE）· distill（软 margin 蒸馏，备选）
│   └── serve/         llm_client
├── scripts/           见下表
├── data/
│   ├── corpus/        clauses.jsonl（条款级父块）
│   ├── index/         bm25.json · units.jsonl · {dual_mix,dual_gold,tower_base}.faiss(+dim/meta)
│   ├── models/        dual_mix/ · dual_gold/ · cross_v2_ep4/（各含 doc_encoder、query_encoder）
│   ├── phase1/        冷启动训练/验证集（人工撰写 query）
│   ├── phase2/        P2 难负例候选与清洗判定
│   ├── phase7/        黄金集（gold_train / gold_eval / gold_eval_clean）与融合对照结果
│   └── authoring/     问句撰写工作区
├── docs/PROGRESS.md   进度 / 数字 / 交接
└── tests/             切分器 / 指标 / 汇总 单测
```

### 脚本速查

| 阶段 | 脚本 |
|---|---|
| 语料 | `build_corpus.py`（PDF→条款父块）、`sample_clauses.py`、`build_phase1_data.py` |
| 索引 | `build_index.py`（BM25+基座 FAISS+单元表）、`build_dual_index.py`（换塔重建）、`calibrate_bounds.py`（BM25 归一边界标定） |
| P2 难负例 | `mine_negatives.py`（多路召回并集）→ `filter_candidates.py`（收敛为可判定清单） |
| P3 精排 | `build_ce_data.py` / `build_ce_data_full.py` → `train_cross_encoder.py` → `eval_ce_versions.py`、`eval_rerank.py` |
| P4 蒸馏（备选） | `gen_distill_data.py` → `src/train/distill/train.py` → `eval_teacher_ab.py` |
| P6 融合 | `eval_fusion_gold.py`（RRF vs 凸组合 + α 网格）、`eval_pipeline_rerank.py`（端到端召回+精排） |
| P7 黄金集 | `sample_gold_sources.py` → 人工撰写 → `build_gold_train_data.py`、`gold_fp_screen.py` |
| 评估 | `eval_retrieval.py`（多路对照）、`eval_ragas.py`（需 Key）、`ab_chunk_size.py` / `ab_pooling.py`（消融） |

## 运行（Windows PowerShell，项目根目录）

```powershell
# 带 CUDA 的解释器：检索/训练都建议用它
$PY = "C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe"

# —— 评估 ——
# 单路/融合对照（--tag 必填）
& $PY rag2/scripts/eval_retrieval.py --tag gold --mode convex --index dual_mix.faiss `
    --query-tower data/models/dual_mix/query_encoder --top-k 5
# 融合方式对照（RRF vs 凸组合，黄金集）
& $PY rag2/scripts/eval_fusion_gold.py --index dual_mix.faiss
# 端到端（召回 top20 → CE 精排）
& $PY rag2/scripts/eval_pipeline_rerank.py

# —— 重建产物（改了语料或换了塔才需要）——
& $PY rag2/scripts/build_corpus.py
& $PY rag2/scripts/build_index.py
& $PY rag2/scripts/build_dual_index.py --doc-tower data/models/dual_mix/doc_encoder --out dual_mix.faiss
& $PY rag2/scripts/calibrate_bounds.py            # 换语料后必须重标 BM25 归一边界

# —— 训练（GPU）——
& $PY rag2/src/train/dual_tower/train.py --out data/models/dual_mix
& $PY rag2/scripts/build_gold_train_data.py --index dual_mix.faiss    # 挖 hard negatives
& $PY rag2/scripts/train_cross_encoder.py --out data/models/cross_v2_ep4

# —— 测试 ——
& $PY -m pytest rag2/tests
```

## 硬约定（改代码前先读）

1. **`MAX_PARENT_LEN = 1200` 同时是切分阈值**，回归测试锁死——不要退回 700。
2. **融合的归一化边界必须用"真实 query 命中分"标定**（`calibrate_bounds.py`，BM25 lo=0 / hi=p99）；
   `self-score` 分布量级远高于真实命中，用它标定会得到错误边界。
3. **黄金集不可在检索结果里人工挑 gold**（等于用被测系统定义正确答案）；
   正确做法是回溯问句的**生成来源条款**。gold 口径 = `source::clause_no`。
4. **融合结论随塔强度变化**：重训双塔后必须重扫 α / 重跑 RRF 对照，不能沿用旧结论。
5. ⚠️ **不要裸 `import faiss`**——Windows 上会 `DLL load failed`（缺 `libiomp5md.dll`）。
   统一走 `src/index/dense.py::import_faiss()`；且 faiss 无法写非 ASCII 路径，
   落盘用 `serialize_index` + Python 文件 IO。

## 相关

- agent 侧接入（溯源字段、精排开关、择塔）→ [`../agent/README.md`](../agent/README.md)
- 环境 / 数据完整性自检 → [`../env/README.md`](../env/README.md)
- 旧迁移指南（含已删 `dual_p1` 路径，勿照抄）→ [`../docs/archive/rag2_MIGRATION.md`](../docs/archive/rag2_MIGRATION.md)
