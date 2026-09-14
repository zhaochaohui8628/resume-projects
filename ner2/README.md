# ner2 —— 实体抽取（远程监督 + LLM 清洗重建版）

> 替代旧 `ner/`（2026-09-11 起）。核心差异：**数据清洗由模型逐条人工执行**（扮演大模型
> 角色对规则弱标二次校验与纠错），而非代码调 LLM API 批量生成。代码与手工产物格式同构、可互换。

从施工方案文本抽取 6 类实体：**工程类型 / 工序 / 设备 / 参数 / 规范编号 / 危大类别**。

## 抽取架构（2026-09-11 定稿：级联混合管道）

**砍掉 PromptNER（纯 few-shot）**——维护三套底层逻辑成本高，且施工领域未做指令调优 → 效果差、延迟高。
改为**两层级联**：

| 层 | 机制 | 定位 |
|---|---|---|
| 层1 确定性规则/词典 | 规范编号正则 + 危大类别关键词 + 高置信封闭专有名词（设备/工程类型/工序/参数）最长匹配 | **高精度拦截**：零延迟、可解释、可复跑（`src/pipeline/rule_layer.py`） |
| 层2 微调轻量实体模型 | ner2 银标微调的 BertNER | **兜底复杂上下文**：开放词表/边界歧义/类型歧义/未登录词（`src/pipeline/model_layer.py`） |

合并规则：**层1 优先占用网格，层2 只在未占用位置补充**（杜绝两套逻辑打架）。
入口：`src/pipeline/cascade.py::CascadeExtractor`（`model_path=None` 即纯规则模式）。

## 7 阶段路线（详见 `docs/PROGRESS.md`）

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 语料 + 词典底座 | ✅ |
| P1 | 远程监督弱标注（词典+正则，纯规则） | ✅ 3974 弱标句 / 6000 实体 |
| P2 | LLM 二次校验与纠错（模型手工逐条判定） | ✅ 1400 条：keep 1036 / fix 364 |
| P3 | 冲突过滤 + 银标构建（零泄漏） | ✅ train 1077 / val 167 / conflicts 156 |
| P4 | **两套全量微调方案 × 各两轮**（softmax+focal CE / CRF） | ✅ 见下方结果 |
| P5 | 黄金测试集（1000+ 多源异构，交叉校验+多专家投票） | ✅ 共识 1043 → 训练并入 500 + 测试 543 |
| P5 | 黄金测试集评估（严格/宽容 F1 + 分类型 + 95%CI） | ✅ 最佳 0.8849 |
| P6 | 迭代（参数类短板 / 软标降噪 / 词典补词） | ⏸ |
| P7 | 简历报告 | ✅ 口径已定稿（`RESUME_NUMBERS.md` §2） |

### 训练与评估结果（gold_test **543 句 / 841 实体**，零泄漏）

| 方案 | 第一轮（冻结主权重） | 第二轮（解冻全量微调） | val 严格 F1 |
|---|---|---|---|
| 方案1 **BERT+Softmax+focal CE** | 仅训分类头（**15,380** 参数） | 1100 句（第一轮高置信 495 + 伪标 105 + 专家 500） | 0.6710 → **0.7214** |
| 方案2 **BERT+CRF**（pytorch-crf） | 仅训 **CRF 转移矩阵（440 参数）** | 1046 句（504 + 42 + 500） | 0.7509 → **0.8175** |

| 模型 | 严格 F1 | 95% CI | 宽容 F1 |
|---|---|---|---|
| 层1 规则/词典（零模型基线） | 0.7607 | [0.7333, 0.7851] | 0.7607 |
| s1_softmax_stage1 | 0.7179 | [0.6894, 0.7438] | 0.7688 |
| s1_softmax_stage2 | 0.8145 | [0.7885, 0.8377] | 0.8667 |
| s2_crf_stage1 | 0.8353 | [0.8150, 0.8555] | 0.8573 |
| **s2_crf_stage2** | **0.8849** | [0.8629, 0.9054] | **0.9047** |

分类型（严格 F1，s2_crf_stage2）：工程类型 0.955 / 危大类别 0.954 / 设备 0.916 / 工序 0.877 /
规范编号 0.800 / **参数 0.632**（短板）。

## 目录

```
ner2/
├── docs/PROGRESS.md      # 进度 + 实测数字（换机器后第一入口）
├── src/
│   ├── common/           # paths / types（6 类白名单、优先级）
│   ├── weak/             # lexicon（词典）+ remote_label（远程监督标注）
│   ├── review/           # prompts（LLM 校验提示词）+ parser（输出兜底）+ conflict（冲突判定）
│   ├── pipeline/         # ★级联混合管道：rule_layer + model_layer + cascade
│   ├── bert/             # ★P4 模型栈：data_utils(BIOE+X 20标签) / losses(focal CE)
│   │                     #   / crf(pytorch-crf 适配器) / model(双头+冻结) / decode / engine
│   └── eval/             # metrics（严格/宽容 F1、标签集 Jaccard）
├── scripts/
│   ├── build_lexicon.py         # P0 词典底座
│   ├── weak_label.py            # P1 远程监督弱标（可复跑）
│   ├── sample_for_review.py     # P2 分层抽样待复核池
│   ├── llm_review.py            # P2 LLM 校验（写但不运行；或 future 有 key 后跑）
│   ├── apply_manual_judgments.py# P2 模型逐条判定的落盘（片段->偏移机械化）
│   ├── filter_conflicts.py      # P3 冲突过滤 + 银标 + 泄漏剔除
│   ├── eval_silver.py           # P3 清洗评估报告
│   ├── prepare_review_v2.py     # P2 扩量：按 text 复用旧判定 + 列未判定清单
│   ├── apply_manual_judgments_v2.py # P2 扩量判定落盘（1400 条）
│   ├── build_gold_candidates.py # P5 黄金集候选（多源异构抽样）
│   ├── gen_auto_experts.py      # P5 自动专家 A/B（规则层 / 弱标层）
│   ├── make_model_expert.py     # P5 专家 C 模型精标
│   ├── multi_expert_vote.py     # P5 交叉校验 + 多专家投票 + 权威仲裁
│   ├── build_gold_split.py      # P5 划分（500 并入训练 / 其余测试）+ 零重叠校验
│   ├── train_ner.py             # ★P4 训练入口（--head softmax|crf × --stage 1|2）
│   ├── mine_pseudo.py           # ★P4 高置信度伪标挖掘（margin / CRF 解码得分 + 去泄漏）
│   ├── apply_pseudo_review.py   # ★P4 LLM 交叉验证清洗（策略可复跑）
│   ├── filter_round1.py         # ★P4 第一轮数据高置信度子集清洗
│   ├── eval_gold.py             # ★P5 黄金集评估（严格/宽容 F1 + 分类型 + bootstrap CI）
│   └── run_two_schemes.py       # ★P4 全流程编排（7 步）
├── data/
│   ├── phase1/           # weak.jsonl + unlabeled_sentences.jsonl
│   ├── phase2/           # review_pool / judgments(jsonl/_v2) / silver_* / conflicts
│   ├── phase4/           # 伪标(pseudo_hc_*/pseudo_clean_*/hc_round1_*) + 报告 + 指标
│   └── phase5/           # gold_candidates / expert_* / gold_consensus / gold_test / gold_train_portion
├── models/               # 训练产物（磁盘现存：s2_crf_param_v3 现行 / s1_softmax_stage2 对照）
└── tests/                # 34 条单测（含 pytorch-crf 与暴力枚举一致性）
```

## 运行（Windows PowerShell，项目根目录）

```powershell
# 训练/推理建议用带 CUDA 的解释器
$PY = "C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe"

# 训练（方案2 CRF 两轮；第一轮冻结主干只训 CRF 转移矩阵，第二轮解冻全量微调）
& $PY ner2/scripts/train_ner.py --head crf --stage 1
& $PY ner2/scripts/train_ner.py --head crf --stage 2
#   方案1 softmax 同理： --head softmax --stage 1|2
#   小显存机可加 --accum <n>（梯度累积）、--max-len 96（默认，p99=53 够用）

# 黄金集评估（严格/宽容 F1 + 分类型 + 句级 bootstrap 95%CI + 规则层基线）
& $PY ner2/scripts/eval_gold.py --with-rule-baseline `
    --model s2_crf_param_v3=ner2/models/s2_crf_param_v3/model.pt

# 单份方案全量抽取（现行级联；加 --rule-only 可零依赖跑纯规则层）
& $PY ner2/scripts/extract_full.py .\方案.txt --out ner2/data/entities.jsonl

# 单测（含 pytorch-crf 与暴力枚举一致性）
& $PY -m pytest ner2/tests
```

> agent / CLI 里不需要手动调本模块：`agent/src/tools/ner_client.py` 默认走 ner2 级联
> （规则层 + `s2_crf_param_v3`，全量分块识别，10 万字级 ~5s）。
> 严格模式（默认）下模型缺失会**直接报错**而不是静默降级——静默降级会把漏报包装成"无风险"。

### ⚠️ 模型检查点现状（2026-09-13）

磁盘现存两个：`models/s2_crf_param_v3/`（**现行**，参数专项微调后 gold_test 严格 F1 最高）、
`models/s1_softmax_stage2/`（对照）。

脚本里出现的 `s1_softmax_stage1` / `s2_crf_stage1` / `s2_crf_stage2` 是**历史实验检查点**
（用于复现下方"两套方案 × 各两轮"的逐轮数字），当前不在磁盘上，需要时用 `train_ner.py` 重训生成。

## 硬约束

1. 数据清洗/校验由**模型逐条撰写执行**（扮演 LLM），不得用代码调 LLM API 批量生成；
   `llm_review.py` 保留完整实现但不运行，手工产物 `judgments.jsonl` 与之同构。
2. 远程监督走词典+正则（纯规则，可复跑，无需 API）。
3. silver 训练集与金标 `gold_eval_v2` **零泄漏**（逐句 hash 校验强制）。
4. 黄金测试集与训练集**零重叠**（逐句 hash 校验强制）。
5. **CRF 用安装库 pytorch-crf，不自实现**（`pip install pytorch-crf`）；
   ⚠️ 该库 `forward` 返回的是**对数似然**（gold − logZ），取负才是损失，`crf.py` 已封装并单测锁死。
6. **两套方案均需执行、各两轮**：第一轮冻结主权重（softmax 只训头 / CRF 只训转移矩阵），
   第二轮解冻全量微调；CRF 第一轮必须从方案1第一轮的检查点续训（否则发射层随机）。
