# rag2 迁移与续跑指南

> 目标：把本项目从当前机器（Windows / 12 核 CPU / 无 GPU）整体搬到另一台机器继续开发与训练。
>
> **最省事的路径**：拷整个项目文件夹 → 在新机器根目录执行
> `python env/bootstrap.py check --install` → 全绿即可开工。
> 新机器的**第一入口是项目根的 [`../START_HERE.md`](../START_HERE.md)**（阅读顺序、目录地图、
> 硬约定、报错处置都在那里）；本文件补充迁移细节。

---

## 0. 一句话结论

**必须拷的 5 类**：代码（`src/ scripts/ tests/ docs/`）、我人工撰写的问句（`data/authoring/`）、
训练/评估集（`data/phase1/`）、**已训好的双塔（`data/models/dual_p1/`，184 M）**、
基座模型（`data/models/bge-small-zh-v1.5`）。

另外**必须随包带走**（都在项目文件夹内，直接整包拷就自动包含）：
`START_HERE.md`、`env/`（自举脚本 + 环境快照 + 冻结依赖）、
`.workbuddy/memory/`（项目记忆）、`.workbuddy/identity_snapshot/`（机器级人格/画像/技能快照）。

其余（条款库、BM25/FAISS 索引、向量缓存、难负例候选）都能用命令重建——但重建条款库需要 PDF。

最小传输量 **约 300 MB**（不含 PDF、不含精排基座 1.1 G）。

---

## 1. 目录结构与体积（实测）

```
施工方案合规审查/                    3.5 G（整盘）
├── START_HERE.md                        ★ 新机器第一入口（阅读顺序/目录地图/硬约定）
├── env/                              1.4 M  ★ 自举与快照
│   ├── bootstrap.py                         snapshot / check / install 三个子命令
│   ├── environment.json                     环境快照：依赖版本 + 产物 sha256 + 语料统计
│   └── requirements-frozen.txt              旧机全部 401 个包的精确版本
├── .workbuddy/
│   ├── memory/                       250 K  ★ 项目级记忆（MEMORY.md + 9 份日志）
│   └── identity_snapshot/            1.4 M  ★ 机器级快照（SOUL/IDENTITY/USER/MEMORY + skills）
├── data/raw/                         526 M  ○ 语料 PDF（74 本规范 + 7 本沪地标）
│   ├── standards/                           ○ 主语料
│   └── guidelines/                          ○ 上海地标 7 本
├── data/models/                      2.3 G
│   ├── bge-small-zh-v1.5/             92 M  ★ 必须拷（双塔基座，可离线跑）
│   ├── bge-reranker-base/            1.1 G  △ 精排基座；也可在目标机用镜像拉取
│   ├── bert-base-chinese/            393 M  ✗ 与 rag2 无关（旧 NER 模块用）
│   ├── bert-ner/                     777 M  ✗ 与 rag2 无关
│   └── domain/ domain_natural/       7.4 M  ✗ 旧项目的微调产物，已被 rag2 取代
└── rag2/                             396 M  ← 本次新建的重塑版项目
    ├── src/                           77 K  ★ 必须拷
    ├── scripts/                       58 K  ★ 必须拷（10 个 CLI）
    ├── tests/                          5 K  ★ 必须拷（11 条零依赖单测）
    ├── docs/PROGRESS.md                    ★ 必须拷（进度与交接）
    ├── MIGRATION.md                        ★ 本文件
    ├── requirements.txt / .gitignore       ★ 必须拷
    ├── data/authoring/               652 K  ★ 必须拷（人工撰写的 465 条 query，**不可复现**）
    ├── data/phase1/                  1.2 M  ★ 必须拷（训练/验证集）
    ├── data/models/dual_p1/          184 M  ★ 必须拷（已训双塔；CPU 重训要 47 分钟）
    ├── data/phase2/                   52 M  ○ 建议拷（P2 候选 12660 条；重建依赖 dual_p1.faiss）
    ├── data/corpus/                   12 M  ○ 可重建（需 PDF，约 3~6 分钟）
    ├── data/index/                   129 M  ○ 可重建（BM25 27M + base.faiss 32M + dual.faiss 32M
    │                                          + npy 缓存 32M + units 10M，约 6 分钟）
    └── data/eval/                    180 K  ○ 评估结果 JSON，可重建
```

★ = 必须拷　○ = 建议拷或可重建　△ = 按需　✗ = 不必拷

> 整包拷贝时 ★ 与 ○ 会自动带上，无需挑选；只需注意**别把 `data/models/bert-*` 那 1.17 G
> 与 rag2 无关的旧产物误当成必需**（拷了也无害，只是白占体积）。

---

## 2. 拷贝清单

### 方案 A：整包拷（**你选的就是这个**，3.5 G）
直接复制整个 `施工方案合规审查` 目录即可，**★ 与 ○ 会自动带上**：

```bash
# 拷完后在新机器上验证（一条命令）
cd <新位置的>/施工方案合规审查
python env/bootstrap.py check --install
```

唯一的契约：**`rag2/` 与 `data/` 保持同级兄弟目录**（见第 3 节）。

### 方案 B：最小集（只想要核心，约 300 MB + PDF）

```bash
# 旧机打包
tar -czf core.tgz START_HERE.md env .workbuddy rag2/src rag2/scripts rag2/tests rag2/docs \
    rag2/MIGRATION.md rag2/requirements.txt rag2/.gitignore \
    rag2/data/authoring rag2/data/phase1 rag2/data/phase2 \
    rag2/data/corpus rag2/data/index rag2/data/models
tar -czf models.tgz data/models/bge-small-zh-v1.5 data/models/bge-reranker-base
tar -czf raw.tgz data/raw/standards data/raw/guidelines   # 可选，约 250 M
```

### 方案 C：git（适合只搬代码与人工数据）
`rag2/.gitignore` 已把可重建的大产物排除；`data/authoring`、`data/phase1` 会入库。
模型与大索引仍走外部拷贝。**注意 `.workbuddy/identity_snapshot` 含个人偏好，不要推到公开仓库。**

---

## 3. 路径约定（迁移后唯一要检查的地方）

`rag2/src/common/paths.py` 用**相对位置**推导根目录，不写死绝对路径：

```python
RAG2_ROOT    = <rag2 目录>            # 由本文件位置推出
PROJECT_ROOT = RAG2_ROOT.parent       # 预期这里还有 data/models、data/raw
BASE_TOWER_MODEL = PROJECT_ROOT/data/models/bge-small-zh-v1.5
BASE_CROSS_MODEL = PROJECT_ROOT/data/models/bge-reranker-base
```

因此：

1. **保持 `rag2/` 与 `data/` 是同级兄弟目录**。若目标机想把 PDF 放别处，设环境变量覆盖：

   ```bash
   export RAG2_RAW_DIR=/path/to/standards
   ```
   模型路径无环境变量，需改请直接编辑 `paths.py` 的 `BASE_TOWER_MODEL` / `BASE_CROSS_MODEL`。

2. **faiss + 非 ASCII 路径的坑已处理**：`faiss.write_index` 走 C++ `fopen`，Windows 下无法处理
   含中文的路径（本项目路径即含中文）。`FaissStore.save/load` 已改为
   `faiss.serialize_index` + Python 文件 IO 绕开（`rag2/src/index/dense.py`）。Linux 无此问题。

3. **离线策略**：所有脚本都用本地模型目录，不联网。若没拷 `bge-reranker-base`：

   ```bash
   export HF_ENDPOINT=https://hf-mirror.com     # huggingface.co 在本网络常 502
   huggingface-cli download BAAI/bge-reranker-base --local-dir data/models/bge-reranker-base
   ```

---

## 4. 目标机环境准备

### 4.1 Python

要求 3.10+（本机验证于 3.13）。**用独立 venv**：

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### 4.2 依赖

```bash
# 1) GPU 版 torch（按目标机 CUDA 版本选 cu121 / cu124 / cu126）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2) 其余
pip install -r rag2/requirements.txt
```

核心：`pymupdf`（解析 PDF）、`numpy`、`transformers`、`sentence-transformers`、`faiss-cpu`
（建库不是 GPU 瓶颈，CPU 版即可）。

### 4.3 冒烟验证（迁移后第一件事）

**首选：一条命令做完全部体检**（在项目根目录，即含 `rag2/` 与 `data/` 的那一层）：

```bash
python env/bootstrap.py check --install
# 期望末行：结论：✓ 全部通过，环境与快照一致，可直接开工
# （CPU 机器上会有「无 CUDA」告警，属预期）
```

它会校验目录契约 / 依赖 / GPU / 4 个模型目录 / **关键数据文件 sha256** / 语料统计 /
跑两路冒烟检索。退出码 `0` 通过、`1` 致命、`2` 有告警。

**手工逐项（bootstrap 不可用时的退路）**：

```bash
cd rag2
python -m pytest tests -q                       # 期望 11 passed
python scripts/build_phase1_data.py             # 期望「有效 query 465；无效 0」
python scripts/eval_retrieval.py --tag smoke --mode bm25
# 期望 ≈ hit@5 0.884 / mrr 0.807（43 条 val）
python scripts/eval_retrieval.py --tag smoke_dual --mode convex --alpha 0.4 \
  --index dual_p1.faiss --query-tower data/models/dual_p1/query_encoder
# 期望 ≈ hit@5 1.000 / mrr 0.895 —— 这条最能证明「模型与环境都没搬坏」
```

> 注意：以上命令都用**当前解释器**，脚本内部按相对位置解析路径，**没有写死任何机器路径**。
> 唯一要保证的是 `rag2/` 与 `data/` 同级（见第 3 节）。

---

## 5. 从零重建索引（仅当没拷 `data/index/`）

```bash
cd rag2
python scripts/build_corpus.py       # 82 本 PDF -> data/corpus/clauses.jsonl（3~6 分钟）
python scripts/build_index.py        # BM25 + bge 向量 + units（CPU 约 6 分钟）
python scripts/calibrate_bounds.py   # 标定 BM25 温莎边界（**必须**，否则融合失效）
# 若也搬了双塔，重建微调索引：
python scripts/build_dual_index.py \
  --doc-tower data/models/dual_p1/doc_encoder --out dual_p1.faiss
```

> `build_index.py` 会把向量缓存到 `data/index/tower_base.npy`；换基座或改单元切分时加 `--force`。

---

## 6. 在新机器上继续（GPU 命令）

```bash
cd rag2

# ── P2 遗留：清洗完候选后组装精排数据（脚本待写 scripts/build_ce_data.py）
#    candidates.jsonl 已生成（12660 条）；清洗判定落盘 judgments.jsonl

# ── P3：CrossEncoder 精排（**GPU 必须**，CPU 约 1 样本/秒）
python scripts/train_cross_encoder.py \
  --data data/phase2/ce_train.jsonl --out data/models/cross_p1 \
  --device cuda --epochs 3 --batch-size 32 --lr 1e-5 --max-len 256
python scripts/eval_rerank.py --data data/phase2/ce_dev.jsonl --tag base
python scripts/eval_rerank.py --data data/phase2/ce_dev.jsonl --tag cross_p1 \
  --model data/models/cross_p1

# ── 重训双塔（若需要；GPU 上约 1~2 分钟）
python -m src.train.dual_tower.train --data data/phase1/train.jsonl \
  --out data/models/dual_p1 --epochs 12 --batch-size 32 --lr 3e-5 --max-len 224
```

显存参考：`bge-reranker-base`(278M) + max_len 256 + batch 32 约需 6~8 G；
不足就 `--batch-size 8 --grad-accum 4`（等效 batch 32）。

---

## 7. 如何继续这次对话

新机器上开新会话时，把下面这段话原样发给助手（自包含）：

> 我在做「施工方案合规审查」项目的 RAG 重塑（7 阶段：冷启动双塔 → 难负例挖掘清洗 →
> CrossEncoder 教师 → pairwise 软 margin 蒸馏 → 迭代蒸馏 → RRF/凸组合融合寻优 → 黄金集评估）。
> 项目在 `<项目绝对路径>`。请先读 `START_HERE.md`，然后执行 `python env/bootstrap.py check`，
> 再读 `rag2/docs/PROGRESS.md` 第 5 节，从「★ 下一会话第一步」接着做，不要重复已完成的部分。

助手会自动读到：`START_HERE.md`（入口）→ `.workbuddy/memory/MEMORY.md`（项目长期约定）
→ `.workbuddy/memory/2026-09-11.md`（最近日志）→ `rag2/docs/PROGRESS.md`（进度与下一步）。
机器级的人格与用户画像在 `.workbuddy/identity_snapshot/`，按其中的 README 放回用户主目录后重启会话即可生效。

---

## 8. 迁移检查清单

**自动项**（跑一条命令就全查到，不用手工对）

```bash
cd <项目根> && python env/bootstrap.py check --install
```

它会逐项核对下面这些，并给出 `✓ / ! / ✗` 与统一结论：

- [ ] 目录契约：`rag2/` 与 `data/` 同级，10 个关键文件存在
- [ ] 依赖：numpy / torch / transformers / sentence-transformers / faiss / pymupdf 版本与快照一致
- [ ] GPU：`torch.cuda.is_available()` 是否为 True（CPU 机器会给告警，属预期）
- [ ] 模型：4 个目录关键权重文件存在（base_tower / base_cross / dual_p1 两塔）
- [ ] 数据指纹：5 个关键文件 sha256 与快照一致（防拷贝截断）＋ 8 个大产物字节数
- [ ] 语料统计：条款 12720 / 规范 77 / 索引单元 15742 与快照一致
- [ ] 冒烟检索：BM25 `hit@5 0.884 / MRR 0.807`；双塔凸组合 α=0.4 `hit@5 1.000 / MRR 0.895`

**手工项**（脚本查不到，只有你能确认）

- [ ] venv 已建并用它跑 `bootstrap.py`（别用系统 Python 直接装包）
- [ ] 若要 GPU 训练：torch 装的是 CUDA 版（`bootstrap check` 第 3 节会显示 `cuda_available`）
- [ ] 机器级配置已恢复：把 `.workbuddy/identity_snapshot/machine_config/` 与 `skills/`
      放回用户主目录 `.workbuddy/`（方法见该目录的 `README.md`），然后**重启会话**
- [ ] 新会话已按 `START_HERE.md` 第 6 节的话术发起，并确认它读到 `PROGRESS.md` 第 5 节
