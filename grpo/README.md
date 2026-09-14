# grpo —— Qwen GRPO 强化学习领域微调（LoRA 合并链）

目标从"Qwen LoRA 领域微调"升级为 **GRPO 强化学习**：让 Qwen 学会施工合规审查的固定答案题
（规范废止 / 危大分级 / 阈值数值 / 方案缺章），**答案可代码判分**，训练全程无需人工标注。

> **2026-09-11 重构**：稀疏 4 档离散奖励 → **四维连续奖励**（格式/cot/basis/answer 加权求和，
> 平滑奖励曲线抑制梯度震荡）；数据 → **真实 agent 场景模拟**（方案片段+模拟 RAG 证据+CoT 金链+
> 漏报/误报标签）；评测 → **业务对齐矩阵**（漏报率/误报率/溯源准确率/CoT 可解释性）。

## 两阶段流程（用户定稿）

```
阶段一  ① GRPO-LoRA（基座 Qwen-Instruct）──采样(8条,T=1.0)──四维连续判分──z-score 优势──更新
          结束 → merge → s1_grpo
        ② 轨迹回收：GRPO 过程采样 (query,回答,reward+四维分解) 落盘 →
          每组按 reward 降序取 **top2** 高分（reward≥0.95 原样学 / 0.5~0.95 金据重建合规 JSON / <0.5 丢弃）
          混入 ~10% 预训练条款续写样本（减少对齐税）
        ③ SFT-LoRA（基座 = s1_grpo）克隆高奖励轨迹，只对 assistant 段（cot+证据+结论）算 CE
          → merge → s1_sft
阶段二  ④ GRPO-LoRA（基座 = s1_sft，ref = s1_sft 冻结，β 加大）→ merge → final
```

**LoRA 合并链**：每次微调独立 adapter、结束 merge 进主权重；下一阶段基座 = 上次合并产物
（`grpo/config.py: WEIGHT_CHAIN = s1_grpo → s1_sft → final`）。ref = 阶段开始权重，用 peft
`disable_adapter` 实现（LoRA 冻结基座 → 单份显存）。

> ⚠️ **权重盘点（2026-09-13 清理后）**：`data/models/qwen-grpo/` 现存 **`r2_final`**（两轮迭代终点，
> 唯一交付/评测权重）+ 6 个 LoRA adapter（`s1_adapter` / `s1_sft_adapter` / `s2_adapter` /
> `r2_grpo_adapter` / `r2_sft_adapter` / `r2_final_adapter`）。中间 merge 档
> `s1_grpo` / `s1_sft` / `final` / `r2_grpo` / `r2_sft` 共 5 档（约 31.7 GB）已删除——
> 需要时用下方训练命令 + `data/grpo/train.jsonl` 重跑即可复现（GPU 数小时）。
> 简历表里的 R1/R2 跨阶段对照是**评测结果**，不依赖中间权重文件。

## 5 要素定稿

1. **数据**：`scripts/gen_rl_data.py` 生成**真实 agent 场景数据集**——每条样本 = 待审方案片段
   (`plan_excerpt`) + 用户提问 + 模拟 RAG 检索证据 + 金标准 agent 输出（含 `cot_steps` 推理链）；
   **证据来源（2026-09-12 定稿）**：C1（废除/现行）取 `rag2/data/corpus/clauses.jsonl` 真实条款、
   C4（编制指南）取 31/48 号文原文，C2/C3（危大分级/阈值）因 37 号令附件1/2 不在语料内、
   一律**规则合成附件条款**（来源与 golden_basis 一致、语义精准，真实条款检索必命中无关条款）；
   4 类固定答案题
   （C1 废止 / C2 危大分级 / C3 阈值 / C4 缺章）+ 负样本（现行规范/低于阈值/章节齐全），
   judge_meta 携带 `golden_basis`/`golden_cot`/`violation_expected` → 判分器自检满分率 100%。
   **规模（默认 `--scale 5`）：共 1578 条，train 1341 / val 237**（零重复 query，val 按来源隔离 15%）。
   扩充按「参数维度 × 自动采样值 × 多问法模板 × 多工程场景」组合，而非重复模板：
   C2 覆盖住建部令第37号 **9 类危大 × 12 参数维度**（含模板支撑跨度/荷载、悬挑与附着脚手架等），
   每维自动采样数值点（非危大/危大/超规模三档，与阈值留 ≥15% 裕度，危大档在 [h,s] 内按
   15%/85% 插值采样，哨兵维度以 VAL_FLOOR 抬到工程合理下限）；
   C4 含缺 1/2/3 章组合 + 建办质〔2021〕48号九类细化要素；
   C1 条文废止放开至 173 条全量 + 语料 81 本现行规范负样本。
2. **system prompt**：`prompts.py` 任务定制 + 强制纯 JSON（cot_steps+conclusion+basis+explanation）
   + conclusion 白名单（与判分器同源）+ 检索证据注入。
3. **采样**：每组 G=8、temperature=1.0（阶段二 0.9）。
4. **判分**：`reward.py` **四维连续奖励** `Σ w_i·dim`（format 0.15/cot 0.20/basis 0.25/answer 0.40）：
   格式合规分（JSON 语法+字段+白名单分级）、CoT 推理连贯分（步数/衔接词/证据引用，金链全命中满分）、
   依据溯源精准分（命中金据−捏造惩罚）、核心判分结论分（set_match/numeric 连续化）；V=组均值，
   优势 A=(Q−V)/std 组内 z-score。
5. **损失**：`loss.py` 论文原版——`E[min(ρA, clip(ρ)A)] − β·k3KL`，token 级、每组内更新 3 次。

## 运行（GPU）

> 以下命令在 **Windows PowerShell** 中、项目根目录执行。

```powershell
# 本机可跑（CPU 即可）：数据生成 + 判分器自检 + 评测矩阵自检 + 全模块单测
$PY = "C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe"
& $PY grpo/scripts/gen_rl_data.py --scale 5          # -> train 1341 / val 237
& $PY grpo/scripts/evaluate.py --checker-only        # 判分器/业务矩阵自检
& $PY -m pytest grpo/tests

# GPU（权重链，Qwen2.5-3B + LoRA，8G 显存适配）：
#   3B bf16 权重 ~6.4G 全上 GPU（8G 卡峰值 ~7.5G，安全）；7B 需 offload 慢 3-8 倍，不推荐。
#   8G 显存下自动启用逐条采样 + logp 切块（见 train_grpo.py 环境变量，默认开启）。
& $PY grpo/scripts/verify_qwen3b.py                                  # 校验 3B 完整性 + 8G 显存加载
& $PY grpo/scripts/train_grpo.py --base-model data/models/Qwen2.5-3B-Instruct --stage s1   # → merge s1_grpo
& $PY grpo/scripts/train_sft.py  --base-model data/models/qwen-grpo/s1_grpo                 # → merge s1_sft
& $PY grpo/scripts/train_grpo.py --base-model data/models/qwen-grpo/s1_sft --stage s2       # → merge final
& $PY grpo/scripts/run_r2_v2.py                                                     # 第二轮整链：final → r2_final
& $PY grpo/scripts/evaluate.py --model data/models/qwen-grpo/r2_final                # 四维 + 业务矩阵
& $PY grpo/scripts/evaluate.py --model data/models/qwen-grpo/r2_final --judge-llm     # CoT 用 DeepSeek 裁判
```

## 关键超参（`config.py`）

GRPO：G=8 · T=1.0 · clip ε=0.2 · KL β=0.04（阶段二 0.06~0.1）· lr 7e-6 · inner 3 ·
LoRA r32/α64/all-linear · SFT lr 1.5e-5 epochs 5 ·
奖励权重 `REWARD_WEIGHTS`（format .15 / cot .20 / basis .25 / answer .40）·
SFT 轨迹回收 `SFT_KEEP_MIN=0.95 / SFT_REBUILD_MIN=0.5`（**每组 top-k=2 高分**，`--top-k`）·
SFT 混入预训练 `--pretrain-ratio 0.1`（条款续写，减少对齐税）·
评测阈值 `EVAL_ANSWER_OK_MIN=0.9 / EVAL_FMT_OK_MIN=0.8`。详见 `config.py` 与 `prompts.py` 内注释。

## 依赖与边界

- 训练需 GPU + Qwen 权重（`--base-model` 或 `GRPO_BASE_MODEL`；下载建议 ModelScope/hf-mirror 镜像）
- **8G 显存适配**（train_grpo.py 已内置，无需改代码）：
  - `GRPO_SAFE_SAMPLE=1` 逐条采样（默认一次 G=8 生成，8G 下可设此环境变量防 OOM）
  - `GRPO_LOGP_CHUNK=<n>` logp 前向切块（默认 2，8G 下 3B 峰值 ~7.5G 安全；若 OOM 调小）
  - 3B bf16 权重 ~6.4G 全上 GPU；7B 需 offload + 32G 内存且慢 3-8 倍，不推荐
- 本机（Windows/CPU）止于：数据生成、判分/评测自检、loss/merge 单测、train 脚本 `--dry-run`
- 真实条款证据依赖 `rag2/data/corpus/clauses.jsonl`（缺失时自动退化为规则合成证据，不影响跑通）
- 修复记录：merge 产物必须 `.eval()`（否则 dropout 致重载不一致）；`resp_mask` 需含首响应 token 与 eos（off-by-one）

## 简历口径

- **数字与边界声明** → [`../RESUME_NUMBERS.md`](../RESUME_NUMBERS.md) §4（R1/R2 跨阶段对照、漏报率/误报率/F1/溯源准确率）
- **讲述材料**（实现细节、设计决策、踩坑与对策、30 秒口头版与高频追问）→
  [`../简历模板/GRPO项目四_重构描述.md`](../简历模板/GRPO项目四_重构描述.md)、
  [`../简历模板/GRPO项目四_简历版_纯文本.md`](../简历模板/GRPO项目四_简历版_纯文本.md)
- 诚实边界：真实 RL 训练需 GPU 执行（本机 RTX 4060 Ti 已完成两轮 GRPO→SFT→GRPO 迭代）。
