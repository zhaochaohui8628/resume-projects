# ner2 进度与交接（PROGRESS）

> 面向"换机器后继续做"的场景：这份文档说清 **已经做到哪、数字是多少、下一步敲哪条命令**。

最后更新：2026-09-11（P4 两套全量微调方案 × 各两轮跑通 + gold_test 543 评估出简历口径）

> **本机解释器（项目统一）**：P0~P3/P5 纯标准库脚本用
> `C:\Users\<用户名>\.workbuddy\binaries\python\versions\3.13.12\python.exe`；
> **P4 BERT 微调统一用** `C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe`（torch 2.7.1+cu118 / CUDA 可用 /
> 已装 transformers 5.10.1 + **pytorch-crf 0.7.2**）；单测也走 torch_gpu 环境（`-m pytest ner2/tests`）。

---

## 1. 目标（7 阶段路线）

| 阶段 | 内容 | 目标 |
|---|---|---|
| P0 | 语料 + 词典底座（沿用 6 份方案语料 + 旧 lexicon.json 种子） | 可复跑的标注输入 |
| P1 | **远程监督标注**：词典/正则对语料打弱标签 | `weak.jsonl`（规则标签，冲突网格去重叠） |
| P2 | **LLM 二次校验与纠错**：大模型对规则弱标逐条判定 keep/fix/drop + 修正边界/类型 | `judgments.jsonl`（判定+修正+理由） |
| P3 | **冲突过滤 + 银标构建**：过滤冲突样本 → silver_train/val + 清洗对照 | 干净训练集 + 清洗收益数字 |
| P4 | BERT 微调（本机 GPU；**微调流程待用户给定**） | 领域模型 |
| P5 | **黄金测试集**：1000+ 多源异构 + 大模型交叉校验 + 多专家投票（500 并入训练 / 500+ 测试） | 简历口径测试集 |
| P6 | 迭代：坏例回注 → 换句重标 → 词典补词 → 再校验再训 | 指标爬升 |
| P7 | 简历报告（宽容/严格 F1，分类型，95%CI） | 简历口径定稿 |

**抽取架构（2026-09-11 定稿，取代旧 ner 的三后端并存）**：
**砍掉 PromptNER（纯 few-shot）**——维护三套底层逻辑成本高，且施工领域未做指令调优 → 效果差、延迟高。
改为**两层级联混合管道**（`src/pipeline/`）：层1 确定性规则/词典（规范编号正则 + 危大关键词 +
高置信封闭专有名词最长匹配，高精度低延迟）→ 层2 微调轻量实体模型（兜底开放词表/边界歧义/未登录词）。
合并规则：**层1 优先占用网格，层2 只补空缺**。

**硬约束**（用户明确要求）：
1. **数据清洗/校验由模型逐条撰写执行（扮演大模型角色），不得用代码调 LLM API 批量生成**；
   `scripts/llm_review.py` 代码写好但**实际不运行**，手工判定产物格式与它完全一致（可替换）。
2. 远程监督仍走词典+正则（纯规则，可复跑，无需 API）。
3. 实体类型沿用 6 类：**工程类型 / 工序 / 设备 / 参数 / 规范编号 / 危大类别**。
4. 训练数据格式与旧 `ner/src/bert/data_utils.py` 兼容：`{"text","entities":[{type,start,end}]}`。
5. **黄金测试集与训练集零重叠**（逐句 hash 强制）。

---

## 2. 状态总表

| 阶段 | 状态 | 产物 / 备注 |
|---|---|---|
| P0 语料与词典 | ✅ 完成 | 11 份方案 txt（内部 6 + 外部 5）+ 旧 lexicon 种子 + 编号正则 |
| P1 远程监督弱标 | ✅ 完成 | `data/phase1/weak.jsonl`（**3974 弱标句 / 6000 实体**），见 3.1 |
| P2 LLM 二次校验 | ✅ 完成（模型手工执行） | `data/phase2/judgments_v2.jsonl`（**1400 条**：keep 1036 / fix 364），见 3.2 |
| P3 冲突过滤 + 银标 | ✅ 完成 | `silver_train.jsonl`（**1077**）/ `silver_val.jsonl`（**167**）/ conflicts 156（零泄漏），见 3.3 |
| P4 BERT 微调 | ✅ 完成 | **两方案 × 各两轮**全跑通（softmax+focal CE / CRF）；产物 `models/{s1_softmax_stage1,s1_softmax_stage2,s2_crf_stage1,s2_crf_stage2}/`，见 3.6 |
| P5 黄金测试集 | ✅ 完成 | 共识 **1043 句** → 测试集 **543** + 并入训练 **500**（训练集达 **1577**），见 3.5 |
| P5 评估 | ✅ 完成 | gold_test 543 严格/宽容 F1 + 分类型 + 95%CI，见 3.7 |
| P6 参数专项 | ✅ 完成 | `s2_crf_param_v3`：参数类 0.632→**0.714**、整体 0.8849→**0.9055**（全类型最优），见 3.8 |
| P7 简历报告 | ✅ 口径已定稿 | `RESUME_NUMBERS.md` §2 已替换为 ner2 数字 |

---

## 3. 已完成部分的实测数字

### 3.1 P1 远程监督弱标注（`scripts/weak_label.py`）

- 语料（**2026-09-11 扩源**）：`data/raw/plans_internal/*.txt` 6 份 + `data/raw/plans_xproj/*.txt` 5 份 = **11 份方案**
  清洗句子 **11876**（原 6 份时 6745）；规则同旧 `_dirty`（去页眉伪影/超长/非中文行）
- 词典：旧 `data/ner/lexicon.json`（工程类型 84 / 工序 127 / 设备 84 / 参数 163 / 危大类别 13 / 规范编号 4）
- 规则：词典最长匹配 + 编号正则先行 + 类型优先级网格（设备 > 工程类型 > 危大类别 > 参数 > 工序）去重叠
- 输出：`weak.jsonl`（**3974 含实体句**）+ `unlabeled_sentences.jsonl`（11876 清洗句）
- 实体分布（全量 **6000**）：工序 2181 / 设备 1381 / 工程类型 941 / 参数 874 / 危大类别 603 / 规范编号 20
  （扩源后危大类别 170→603 显著提升，规范编号 5→20）

### 3.2 P2 LLM 二次校验（`data/phase2/judgments_v2.jsonl`，模型手工逐条判定）

判定 schema：`{"qid","text","verdict":"keep|fix|drop","entities":[修正后],"issues":[],"reason"}`

抽样池 `sample_for_review.py`（每类保底 + 可疑模式优先：附图标题/短实体/多义/多实体/稀有）→ **1400 条**
（含复用首轮 400 条判定；新增 928 条由模型逐条补判）

| 项 | 值 |
|---|---|
| 判定条数 | **1400**（覆盖 6 类实体） |
| keep（规则标签正确） | **1036（74.0%）** |
| fix（需修正：边界/类型/漏标） | **364（26.0%）** |
| 空修正（fix 后无实体，视同剔除） | 139 |
| 金标泄漏剔除 | 17（与 gold_eval_v2 重叠，逐句 hash 校验） |

**清洗主要收益**（1400 句口径，规则 vs 银标，实体级）：规则独有 43 / 银标新增 47 / 共有 449 / **Jaccard 0.833**
修正热点：①管理/泛化动词被误标为工序（检查/布置/布设/调整）；②桩型与墙体对象由工序改工程类型
（钻孔灌注桩/三轴搅拌桩/地下连续墙）；③"连通道/地下连通道"边界补全；④参数带值补全；
⑤语境误标剔除（级配/功率/电压/通道）。

### 3.3 P3 冲突过滤与银标（`scripts/filter_conflicts.py` + `scripts/eval_silver.py`）

- `silver_train.jsonl`：**1077 句**（keep 原样 + fix 用修正标签，漏标已补）
- `silver_val.jsonl`：**167 句**（15%，按句子 hash 稳定划分）
- `conflicts.jsonl`：156（139 empty-fix + 17 gold-leak）
- **泄漏检查：silver_train vs gold_eval_v2 = 0 泄漏**（程序化逐句 hash 校验，硬约束达成）

- 边界修正幅度：fix 样本平均规则独有 1.09 / 银标新增 0.51 / 标签 Jaccard 0.268（说明 fix 多为"剔除错误实体"为主、补漏为辅）

### 3.4 运行环境与测试

- 纯标准库即可跑 P0~P3、P5；**单测与 P4 训练统一用 torch_gpu 环境**
  （`C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe -m pytest ner2/tests -q`）
- `ner2/tests`：**34 条单测全绿**——远程标注最长匹配/去重叠/编号正则、LLM 解析容错、
  sanitize 白名单+越界 clamp、冲突判定三态、严格/宽容 F1、标签集 Jaccard、级联管道（规则层/合并优先级/模型层接口）、
  **BIOE+X 数据对齐与字符级实体解码**、**pytorch-crf 与暴力枚举一致性**（logZ/NLL/Viterbi/mask/归一化得分）

### 3.5 P5 黄金测试集（1000+ 多源异构 + 大模型交叉校验 + 多专家投票）

**流程**（`build_gold_candidates.py` → `gen_auto_experts.py` → `make_model_expert.py` →
`multi_expert_vote.py` → `build_gold_split.py`）：

1. **候选抽取**：11 份方案（多源异构）× 排除 silver/gold_eval_v2/已判定池 → **1200 句候选**
   （含 1905 规则实体；覆盖 11 个来源；可疑模式优先，来源均衡补齐）
2. **三位专家独立标注**：
   - 专家A `expert_rule`（确定性规则/词典层，1880 实体）
   - 专家B `expert_weak`（远程监督弱标层，1905 实体）
   - 专家C `expert_model`（**模型逐条精标**，1637 实体；相对规则层 244 句修正）
3. **交叉校验（两两 Jaccard）**：C|B = **0.835**、C|A = 0.565、A|B = 0.685
   （A/B 同源词典故相关度高；C 的差异即清洗收益）
4. **多专家投票 + 权威仲裁**（`--authoritative expert_model`）：精标为资深标注，其判定为最终黄金集；
   规则/弱标一致的实体被精标否决者 → `conflicts.jsonl`（**633 条词典假阳性**，交叉校验发现）
5. **产物**：`gold_consensus.jsonl` **1043 句 / 1637 实体** → 去重 + 互斥过滤后
   `gold_train_portion.jsonl` **500 句**（并入训练集）+ `gold_test.jsonl` **543 句**（测试集）

| 集合 | 句子数 | 说明 |
|---|---|---|
| gold_consensus | 1043 | 投票共识（含 n_votes 票数） |
| **gold_train_portion** | **500** | 并入训练集（`src="gold_portion"`） |
| **gold_test** | **543** | 黄金测试集（≥500） |
| conflicts | 633 | 被精标否决的词典假阳性 |

**零重叠硬校验（全部通过）**：`train ∩ val`、`train ∩ gold_test`、`val ∩ gold_test`、
`gold_test ∩ gold_eval_v2` 均为 **0**；543 句测试集 **零非法 span / 零重叠**。
**训练集终量**：`silver_train` 1077 + 并入 500 = **1577 句**（备份 `silver_train_pre_gold.jsonl`）。

黄金测试集实体分布：设备 278 / 工序 217 / 危大类别 161 / 参数 92 / 工程类型 87 / 规范编号 6。

---

### 3.6 P4 两套全量微调方案（`src/bert/` + `scripts/train_ner.py`，GPU 实测）

**模型栈**：`data_utils`（BIOE + 子词 X，**20 标签**）/`losses`（focal CE，O 降权 0.25、γ=2）/
`crf.py`（**安装库 pytorch-crf 0.7.2** 的适配器，**不自实现 CRF**）/`model.py`（softmax|crf 双头 + 三档冻结）/
`decode.py`/`engine.py`（训练 + 预测 + 实体级评估 + 高置信度挖掘）。基座 `data/models/bert-base-chinese`。

| 方案 | 第一轮（冻结主权重） | 第二轮（解冻全量微调） |
|---|---|---|
| **方案1** BERT+Softmax+**focal CE** | 冻结主干、仅训分类头（可训 **15,380** / 总 101,692,436 = 0.02%）；数据 silver_train 1577；val 严格 F1 **0.6710** | 全量微调；数据 495+105+500 = **1100**；val 严格 F1 **0.7214** |
| **方案2** BERT+**CRF**（pytorch-crf） | 冻结主干+发射层、仅训 **CRF 转移矩阵（440 参数）**；数据同左；val 严格 F1 **0.7509** | 全量微调；数据 504+42+500 = **1046**；val 严格 F1 **0.8175** |

> ⚠️ **CRF 第一轮必须 `--init` 方案1 第一轮的检查点**：若发射层是随机初始化，"只训转移矩阵"毫无意义；
> 先用同一批 1000+ 数据把发射层训好（方案1 第一轮），再冻结它只学标签转移，才是干净的 CRF 增益隔离。

**第二轮数据来源**（第一步见下表，第二步见 `hc_round1_{head}`）：

- `hc_round1_{head}`：第一轮训练数据中「模型预测与标注严格一致 + span 内逐 token margin 达标」子集
  （softmax 495 句 / CRF 504 句）
- `pseudo_clean_{head}`：无标注池 6498 句 → 高置信挖掘 → **LLM 交叉验证清洗**后
- `gold_train_portion` 500（LLM 交叉校验 + 多领域专家）

**高置信度挖掘与 LLM 交叉验证实测**（`scripts/mine_pseudo.py` → `scripts/apply_pseudo_review.py`）

| 方案 | 挖掘判据 | 挖掘产出 | LLM 交叉验证后 | 剔除构成 |
|---|---|---|---|---|
| softmax | 逐 token 相对置信度 **margin = p_top1 − p_top2**：实体 span 内全 token ≥0.25、O 侧中位 ≥0.15 | 364 句 | **105 句**（保留率 28.9%） | 单字碎片 154 / 图表标题 16 / 通用动词 38 / 无值参数 22 / 无词典·规则共识 81 / 清空 243 |
| CRF | **归一化解码对数似然** `crf_score_per_token` ≥ p60 分位 + Viterbi 路径 == 发射 argmax | 72 句 | **42 句**（保留率 58.3%） | 通用动词 16 / 无共识 13 / 图表标题 3 / 清空 27 |

**关键结论**：第一轮（冻结主干、只训头）的 margin 只代表"很自信"，不代表"正确"——挖掘软标保留率仅 29%，
噪声集中在**单字碎片**（"电""基""监"）、**图表标题**（"附图001：…"）与**通用动词**（"布置/检查"）；
CRF 的归一化解码得分筛出的软标明显更干净（58%）。
→ **高置信度筛选必须叠加 LLM 交叉验证 + 词典/规则共识**，不能只靠置信度阈值。

**踩坑（已修，勿回退）**：①`max_len` 曾默认 256，而本语料句长 p99≈53、max=66 → 白烧约 4 倍显存；
改为 96 后全量微调稳定。②每轮把整份 state_dict 深拷到 CPU 作 `best_state`，在
Windows WDDM（显存溢出到系统内存）+ 主机内存紧张时会**硬崩溃** → 改为**改进即落盘**
（`torch.save` 逐张量转 CPU，峰值仅一个张量）。③全量微调用 `--batch 8/16 --accum 2~4` 梯度累积，
显存峰值 ~2.0 G。

### 3.7 P5 黄金测试集评估（`scripts/eval_gold.py`，gold_test **543 句 / 841 实体**）

口径：**严格** = 类型 + 字符 span 完全一致；**宽容** = 类型一致 + span 重叠；95% CI 为**句级 bootstrap 500 次**。

| 模型 | 严格 P | 严格 R | 严格 F1 | 95% CI | 宽容 F1 |
|---|---|---|---|---|---|
| L1 规则/词典（零模型基线） | 0.7435 | 0.7788 | 0.7607 | [0.7333, 0.7851] | 0.7607 |
| s1_softmax_stage1（仅训头） | 0.6478 | 0.8050 | 0.7179 | [0.6894, 0.7438] | 0.7688 |
| s1_softmax_stage2（全量微调） | 0.7787 | 0.8537 | 0.8145 | [0.7885, 0.8377] | 0.8667 |
| s2_crf_stage1（仅训 CRF） | 0.8154 | 0.8561 | 0.8353 | [0.8150, 0.8555] | 0.8573 |
| **s2_crf_stage2（全量微调）** | 0.8658 | 0.9049 | **0.8849** | [0.8629, 0.9054] | **0.9047** |

分类型（严格 F1）：工程类型 0.955(n=87) / 危大类别 0.954(n=161) / 设备 0.916(n=278) /
工序 0.877(n=217) / 规范编号 0.800(n=6) / **参数 0.632(n=92)**。

**三条结论**：① 第二轮全量微调相对第一轮 **+5.0（CRF）~ +9.7（softmax）** 点；
② **CRF 两轮均优于 softmax**——第一轮仅训 440 个转移参数即 0.8353，已超过 softmax 两轮的 0.8145；
③ **层1 规则基线 0.7607 高于方案1 第一轮（0.7179）**，印证级联架构「规则层打底 + 模型补空缺」的必要性，
模型的主要增益在开放词表召回（工序/参数/工程类型）。
④ 短板：**参数类 0.632**（数值 + 单位 + 括号口径多变），是 P6 迭代的首要目标。

### 3.8 P6 参数类专项微调（`s2_crf_param_v3`，gold_test 543 全类型最优）

**流程**：反推参数口径 → 错误分析 → 挖量名候选逐条核验 → 机械生成专项数据（词表+规则）→
非参数标签高置信补全 → 组装「专项 + 通用回放」训练集 → **低学习率 5e-6**（为第二轮 2e-5 的 1/4）全量微调 8 轮。

**口径**（`docs/PARAM_SPEC.md` 定稿）：只标**裸指标量名**（A）+ ∅/C35/P8 规格（SPEC）；
泛化量名（荷载/厚度/高度/深度/速度/间距/直径/系数/压力/温度/沉降/轴力/力矩/标高/埋深/强度等级…）
移入 reject_words 不裸标；量名与值**分离**标注（`安全间距≥500mm` → 两个实体；`桩长60m` 只标 `桩长`）。
关键教训：**专项数据必须对齐权威测试集口径**，自造 dev 指标（与训练同源）不可作判断依据。

**最终结果（gold_test 543）**：

| 模型 | 严格 F1 | 参数类 | 工序 | 设备 | 危大 | 工程类型 |
|---|---|---|---|---|---|---|
| s2_crf_stage2（未专项） | 0.8849 | 0.632 | 0.877 | 0.916 | 0.954 | 0.955 |
| **s2_crf_param_v3（专项）** | **0.9055** | **0.714** | 0.893 | 0.927 | 0.978 | 0.954 |

- 参数类 **+8.2pt**（0.632→0.714），整体 **+2.1pt**（0.8849→0.9055），宽容 F1 0.9234；CI 不相交（[0.8882,0.9228] vs [0.8629,0.9054]）。
- 数据：`data/phase6/param_train.jsonl` 1240 句（专项 640 + 回放 600）/ 词表 `param_lexicon.json`（定稿，102 个已核验量名）/ 口径 `docs/PARAM_SPEC.md`。
- 全流程脚本：`param_spec_analysis.py` → `param_error_analysis.py` → `param_lexicon_candidates.py` →
  `build_param_specialist.py` → `complete_param_labels.py` → `build_param_train_set.py` → `train_ner.py --lr 5e-6`。
- **踩坑**：① 训练中段 Windows 页面文件（虚拟内存）不足导致**非确定性段错误**（commit 只剩 ~5G，
  崩溃点在进程加载期/`opt.step`，非代码 bug）——解法：`--batch 8 --accum 4` + **改进即原子落盘**
  （`.tmp`+`os.replace`）+ 分块 2 轮续训，崩溃最多丢 2 轮；② 低学习率 5e-6 8 轮未学崩，
  训练 loss 平滑下降（4.17→0.41），val F1 单调爬升。

### 3.9 全量分块识别（`src/pipeline/full_text.py`，舍弃三层漏斗）

- **背景**（2026-09-11 定稿）：真实施工方案动辄 10 万字；**不再使用三层漏斗**（旧 agent 的
  PlanFunnel 已完全舍弃），完整方案文本**全量分块**送入 ner2 级联管道抽取全部实体。
- **FullTextExtractor**（`ner2/src/pipeline/full_text.py`）：
  - 分块 = 行级清洗切句 + 超长行按句号/分号拆分（与训练语料同口径）；
  - 全量 = 每一句都过模型（batch 推理），不筛选、不收敛、不截断；
  - 级联 = 规则层优先占用字符网格，模型层（默认 `s2_crf_param_v3`）只补空缺。
- **实测性能**（RTX 4060 Ti，batch=64/max_len=96）：10 万字符 → 2439 句 → **~4.7s**（520 句/s ≈ 2.1 万字符/s）；
  模型加载一次性 ~3-5s（常驻后摊销）。CLI：`ner2/scripts/extract_full.py`（支持 txt/pdf）。
- **去重收尾**（2026-09-12）：全量按句抽取后同一实体跨句大量重复（实测一本方案
  122 全量 → 47 唯一，**重复率 61.5%**）。`extract_full.py` 默认按 (type,text) 聚合输出
  唯一实体清单（count/first_sent_idx/前 3 句佐证），`--positions` 取完整带位置列表；
  等价函数 `ner2.src.pipeline.full_text.dedup_entities`。

---

## 4. 命令速查（Windows PowerShell，项目根 `<REPO_ROOT>`）

```powershell
$PY = "C:\Users\<用户名>\.workbuddy\binaries\python\versions\3.13.12\python.exe"

# P1 远程监督弱标（纯规则，可复跑；默认 internal + xproj 两目录）
& $PY ner2/scripts/weak_label.py --lexicon data/ner/lexicon.json

# P2 LLM 二次校验 —— 正式版调 API（本机默认不运行，见硬约束 1）
& $PY ner2/scripts/llm_review.py --weak ner2/data/phase1/weak.jsonl --out ner2/data/phase2/judgments.jsonl
#    手工判定产物与上同构，直接喂 P3 即可

# P3 冲突过滤 + 银标（扩量判定走 judgments_v2.jsonl）
& $PY ner2/scripts/filter_conflicts.py --judgments ner2/data/phase2/judgments_v2.jsonl
& $PY ner2/scripts/eval_silver.py --judgments ner2/data/phase2/judgments_v2.jsonl

# P5 黄金集：候选 -> 自动专家 -> 模型精标 -> 多专家投票 -> 划分
& $PY ner2/scripts/build_gold_candidates.py
& $PY ner2/scripts/gen_auto_experts.py
& $PY ner2/scripts/make_model_expert.py
& $PY ner2/scripts/multi_expert_vote.py --experts expert_rule.jsonl expert_weak.jsonl expert_model.jsonl
& $PY ner2/scripts/build_gold_split.py

# P4 训练（GPU 环境；两方案 × 两轮。$PY2 = anaconda3\envs\torch_gpu\python.exe）
# 方案1：BERT + Softmax + focal CE
& $PY2 ner2/scripts/train_ner.py --head softmax --stage 1          # 一轮：冻结主干，仅训分类头
& $PY2 ner2/scripts/mine_pseudo.py --ckpt ner2/models/s1_softmax_stage1/model.pt --head softmax `
        --out ner2/data/phase4/pseudo_hc_softmax.jsonl
& $PY  ner2/scripts/apply_pseudo_review.py --in ner2/data/phase4/pseudo_hc_softmax.jsonl --head softmax
& $PY2 ner2/scripts/filter_round1.py --ckpt ner2/models/s1_softmax_stage1/model.pt --head softmax `
        --out ner2/data/phase4/hc_round1_softmax.jsonl
& $PY2 ner2/scripts/train_ner.py --head softmax --stage 2 --batch 8 --accum 4    # 二轮：全量微调

# 方案2：BERT + CRF（pytorch-crf）
& $PY2 ner2/scripts/train_ner.py --head crf --stage 1              # 一轮：冻结主干+发射层，仅训 CRF 转移矩阵
& $PY2 ner2/scripts/mine_pseudo.py --ckpt ner2/models/s2_crf_stage1/model.pt --head crf `
        --out ner2/data/phase4/pseudo_hc_crf.jsonl
& $PY  ner2/scripts/apply_pseudo_review.py --in ner2/data/phase4/pseudo_hc_crf.jsonl --head crf
& $PY2 ner2/scripts/filter_round1.py --ckpt ner2/models/s2_crf_stage1/model.pt --head crf `
        --out ner2/data/phase4/hc_round1_crf.jsonl
& $PY2 ner2/scripts/train_ner.py --head crf --stage 2 --batch 16 --accum 2        # 二轮：全量微调

# 全流程编排（等价于上面 1~7 步；伪标清洗缺产物时会中止并提示）
& $PY2 ner2/scripts/run_two_schemes.py

# P5 评估（gold_test 543；严格/宽容 F1 + 分类型 + 句级 bootstrap 95%CI + 规则基线）
& $PY2 ner2/scripts/eval_gold.py --with-rule-baseline `
        --model s1_softmax_stage1=ner2/models/s1_softmax_stage1/model.pt `
        --model s1_softmax_stage2=ner2/models/s1_softmax_stage2/model.pt `
        --model s2_crf_stage1=ner2/models/s2_crf_stage1/model.pt `
        --model s2_crf_stage2=ner2/models/s2_crf_stage2/model.pt `
        --model s2_crf_param_v3=ner2/models/s2_crf_param_v3/model.pt

# P6 参数专项（低 lr 5e-6 全量微调 8 轮；数据/词表/口径见 data/phase6 + docs/PARAM_SPEC.md）
& $PY2 ner2/scripts/train_ner.py --head crf --stage 2 `
        --init ner2/models/s2_crf_stage2/model.pt `
        --train ner2/data/phase6/param_train.jsonl --val ner2/data/phase6/param_val.jsonl `
        --lr 5e-6 --epochs 8 --batch 8 --accum 4 --out-dir ner2/models/s2_crf_param_v3

# ★ 全量分块识别（对外服务入口；舍弃三层漏斗，完整方案全量送入，10 万字级 ~5s）
& $PY2 ner2/scripts/extract_full.py 方案.txt [--out 实体.jsonl] [--rule-only]
& $PY2 ner2/scripts/extract_full.py 方案.pdf [--out 实体.jsonl]   # PyMuPDF 抽文字层
```

---

## 5. 设计与口径（同旧项目约定，避免踩坑）

- **标注单位**：字符偏移，`start`/`end` 半开区间，`text[start:end]` 即实体原文；`document_span` 只要字符，不做 token 级（训练对齐在 P4 做）。
- **类型多义消解**：同一词串命中多个词典类型时，按优先级 `设备 > 工程类型 > 危大类别 > 参数 > 工序` 取其一；规范编号正则先行占用网格。
- **弱标签噪声来源**（清洗重点）：①词典词串在句中不是实体语境（如"拆除"出现在禁止性表述）；②边界偏短（词典只含词干，金标风格是完整短语）；③类型歧义（"基坑降水"既是工程类型又是工序）；④数值单位残缺（参数类）。
- **宽容/严格口径**（P5 沿用）：严格=类型+span 完全一致；宽容=类型一致+span 重叠。
- **LLM 校验提示词红线**（`prompts.py` 内固化）：只输出 JSON；type 限 6 类白名单；实体 span 必须与原文逐字一致；explain 一句话写清"为什么不是/是实体"。