# env —— 项目自举与环境快照

> 目的：让**整个项目文件夹拷到新机器后能一键确认"环境齐不齐、数据坏没坏"**，
> 不依赖写死的解释器路径，也不需要人工逐条对照文档。

## 三个命令（Windows PowerShell，项目根目录执行）

```powershell
# 推荐解释器（带 CUDA）：torch_gpu
$PY = "C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe"

& $PY env/bootstrap.py snapshot          # ① 旧机器 / 基线变更后：冻结环境与产物指纹
& $PY env/bootstrap.py check             # ② 新机器：全套体检（缺依赖时加 --install）
& $PY env/bootstrap.py check --install   # ③ 体检 + 缺依赖时自动 pip install
& $PY env/bootstrap.py install           #    只装依赖
```

也可以直接用启动脚本，不用记这些：

```powershell
.\start.ps1 check              # 等价于 bootstrap.py check
.\start.ps1 check -Install     # 等价于 check --install
```

退出码：`0` 全通过 / `1` 有致命问题 / `2` 可用但有告警。

## 自检项

| # | 项目 | 说明 |
|---|---|---|
| 1 | 目录契约 | `rag2/` 与 `data/` 同级；`REQUIRED` 列表里的关键文件在位 |
| 2 | Python 与依赖 | 版本对比快照；核心包 `numpy/torch/transformers/sentence_transformers/faiss`，可选包 `fitz`（仅重建语料用） |
| 3 | GPU | `torch.cuda.is_available()`；CPU 机器给告警而非报错（判档/规则/NER 规则层不依赖 GPU） |
| 4 | 模型目录 | `base_tower`(bge-small-zh-v1.5) / `dual_mix` 双塔（现行）/ `dual_gold` 双塔（备选）/ `cross_v2_ep4`（精排），检查是否含权重文件 |
| 5 | 数据指纹 | 5 个小文件 **sha256**（防拷贝截断/改坏）+ 10 个大产物字节数 |
| 6 | 语料统计 | 条款数 / 规范本数 / 检索单元数与快照比对 |
| 7 | 冒烟检索 | BM25 与**线上配置双塔（dual_mix + RRF）**各跑一遍，**低于下界阈值**才告警 —— 说明"模型或环境搬坏了" |

> 第 7 节的阈值语义是**下界**（≥ 即通过），不是精度回归断言。语料/塔版本微调会让精确值小幅漂移，
> 用下界可避免天天误报；"搬坏了"这类大幅下降依然会被抓住。改语料或索引后请重跑 `snapshot` 并复核该下界。
>
> 实测基线（2026-09-13，`data/phase1/val.jsonl` n=43，单线程 BLAS）：
> BM25 **0.860/0.806**、dual_mix+RRF **0.930/0.814**、dual_mix+convex 0.884/0.816、dual_gold+convex 0.791/0.724。
> ⚠ 该集是 P1 冷启动分布，**不是简历口径**（真口径 = 黄金集，见 `RESUME_NUMBERS.md` §1）；
> `dual_gold` 在此集天然偏低（按黄金集口径重训），低不代表塔坏了。
>
> ⚠️ 若第 7 节报 `OpenBLAS error: Memory allocation still failed after 10 retries` 或
> faiss `MemoryError: std::bad_alloc`，**不是索引坏了**，是 BLAS 按 16 核分配缓冲超出可用内存。
> 先设环境变量再跑：
> ```powershell
> $env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
> & $PY env/bootstrap.py check
> ```
> `start.ps1` 会自动设好这三个变量。

## 文件

| 文件 | 作用 |
|---|---|
| `bootstrap.py` | 自举脚本本体（纯标准库，**任何 Python 3.10+ 都能跑**；重活才懒加载第三方库） |
| `environment.json` | `snapshot` 产出：解释器、依赖版本、GPU、模型目录、产物 sha256/字节数、语料统计 |
| `requirements-frozen.txt` | 旧机器 `pip freeze` 全量包，用于"精确复现"场景 |

## 为什么 sha256 很有用

整包拷贝最常见的两种事故是**文件被截断**（大文件拷一半）和**拷到了旧版本**，
这两者都不会报错，只会在训练时给出诡异结果。`check` 用 sha256 直接把它们挡在前面。

`environment.json` 里记录的 `artifact_bytes` 覆盖那些太大而不便算哈希的文件
（如 1.1 G 的 `cross_v2_ep4/model.safetensors`），只比字节数，够用。

## 何时该重跑 snapshot

以下情况过后，用 `snapshot` 刷新基线，否则 `check` 会一直报"不一致"：

- 重建了语料 / 索引（`rag2/scripts/build_corpus.py` / `build_index.py` / `build_dual_index.py`）
- 重新训练了双塔（`dual_mix` / `dual_gold` 变了）
- 重新训练了 CrossEncoder（`cross_v2_ep4` 变了）
- 扩充了人工问句或训练集（`rag2/data/authoring` / `rag2/data/phase1` 变了）
- 升级了依赖

## 相关

- 换机器时的完整清单与顺序 → [`../README.md`](../README.md) §环境
- rag2 进度与数字 → [`../rag2/docs/PROGRESS.md`](../rag2/docs/PROGRESS.md)
- 历史迁移指南（已归档，含 `dual_p1` 旧塔路径，勿照抄）→ [`../docs/archive/rag2_MIGRATION.md`](../docs/archive/rag2_MIGRATION.md)
