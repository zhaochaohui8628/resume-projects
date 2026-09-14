---
name: ner2
description: 施工方案实体抽取（NER）—— 从方案文本抽取 工程类型/工序/设备/参数/规范编号/危大类别 6 类实体。采用「规则/词典层 + 微调 BERT+CRF 模型层」级联管道；完整方案（含 10 万字级 PDF）全量分块识别，不收敛、不截断、无三层漏斗。当需要抽取方案中的实体、危大类别识别、参数指标提取，或 agent 的 hazard subagent 需要实体供给时使用。
---

# ner2 —— 施工方案实体抽取（NER）

> 取代旧 `ner/`（旧 RuleNER/PromptNER/BertNER 三后端已废弃）。核心：**级联混合管道** +
> **全量分块识别**（不做任何收敛/筛选/截断，与旧 agent 的三层漏斗彻底切割）。

## 1. 能力与产物

| 实体类型 | 例 |
|---|---|
| 工程类型 | 基坑、地下连续墙、钻孔灌注桩、塔吊基础 |
| 工序 | 开挖、吊装、浇筑、拆除 |
| 设备 | 塔吊、履带吊、混凝土输送泵、钢丝绳 |
| 参数 | 施工荷载、混凝土强度、桩长、跨度、垂直度、坍落度、∅800@600、C35 |
| 规范编号 | GB50010-2010、JGJ80-2016、建质〔2009〕87号 |
| 危大类别 | 基坑工程、模板支撑、起重吊装、脚手架 |

**微调模型产物**（`ner2/models/`，基座 bert-base-chinese，BIOE+X 20 标签）：
- `s1_softmax_stage1/stage2` —— 方案1：BERT+Softmax+focal CE 两轮
- `s2_crf_stage1/stage2` —— 方案2：BERT+CRF（pytorch-crf）两轮
- **`s2_crf_param_v3`（默认生产模型）** —— CRF + 参数类专项微调（低 lr 5e-6），
  gold_test 543 严格 F1 **0.9055**（参数类 0.632→0.714）
- 层1 规则/词典基线：gold_test 0.7607（零依赖、可解释，级联中优先占用字符网格）

## 2. 调用方式（全量分块识别）

**CLI**（GPU 环境 `C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe`）：
```
python ner2/scripts/extract_full.py 方案.txt [--out 实体.jsonl] [--rule-only]
python ner2/scripts/extract_full.py 方案.pdf [--out 实体.jsonl]
python ner2/scripts/extract_full.py 方案.txt --positions --out 全量.jsonl
                          # 默认输出「去重实体清单」；--positions 输出带句位置的完整实体
```

**去重收尾（默认）**：10 万字方案同一实体常跨句重复出现（实测一本方案重复率 ~60%）。
`extract_full.py` 默认按 (type, text) 聚合输出唯一实体清单（含 count / first_sent_idx /
前 3 句佐证）；需要定位原文时用 `--positions` 拿完整带位置列表。
Python 侧等价函数：`ner2.src.pipeline.full_text.dedup_entities(ents)`。

**Python API**（agent 等外部调用方）：
```python
from ner2.src.pipeline.full_text import FullTextExtractor
fx = FullTextExtractor()                          # 默认 s2_crf_param_v3 + 规则层级联
ents = fx.extract_text(plan_text)                 # 全量：分块 → 逐句推理 → 级联合并
# ents: [{"type","text","start","end","layer","conf","sent_idx","sent_text"}]
```

**关键语义**：
- **全量**：完整文本按行清洗切句（与训练语料同口径），每一句都过模型，不筛选/不收敛/不截断；
- **级联**：层1 规则/词典（规范编号正则 + 危大关键词 + 高置信专有名词最长匹配）优先占用字符网格，
  层2 微调模型只补空缺（开放词表/边界歧义/未登录词）；
- **性能**：10 万字符 ≈ 2439 句，RTX 4060 Ti batch=64 实测 **~4.7s**（520 句/s）；
- **PDF**：PyMuPDF 抽文字层（图片/扫描忽略），需 `pip install pymupdf`；
- `--rule-only` / `model_path=None`：纯规则模式，零依赖、无 GPU 可跑（精度低于模型级联）。

## 3. agent hazard subagent 对接契约

- hazard subagent **必须**调用本项目的 `FullTextExtractor`（经 `agent/src/tools/ner_client.py` 适配，
  默认 `s2_crf_param_v3`），对**完整方案全文**做全量识别；
- **禁止**使用任何三层漏斗/收敛逻辑（旧 `agent/src/tools/plan_funnel.py` 已删除）；
- 实体输出直接作为 hazard 的实体供给（危大类别驱动 RAG 条文检索、参数/工序/设备驱动关键参数核对）。

## 4. 数据与训练流程（可审计）

- 数据链：11 份方案语料 → 11876 清洗句 → 远程监督弱标 3974 句/6000 实体 → LLM 二次校验 1400 条
  → 银标 silver_train 1077 / val 167（与金标逐句 hash 零泄漏）→ 黄金集 1043 句共识
  （三专家交叉校验+投票）→ 500 并入训练 / **gold_test 543** 独立测试；
- 参数专项：口径 `docs/PARAM_SPEC.md`（定稿）—— 只标裸指标量名 + ∅/C35/P8 封闭规格值，
  泛化量名不裸标、量名与值分离；词表 `data/phase6/param_lexicon.json`（102 个逐条核验量名）；
- 复跑入口：`docs/PROGRESS.md`（全流程命令 + 实测数字）；单测 `ner2/tests`（34 条，torch_gpu 跑）。
