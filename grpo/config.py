"""GRPO 领域微调（Qwen，LoRA 合并链）——全局配置。

权重链约定（用户指定）：
    每次 LoRA 微调结束即 merge 进主权重 -> 下一阶段以「上次合并后的模型」为新基座。
    ref 策略 = 每个 GRPO 阶段开始时的权重（合并结果）冻结副本。
评分档位（用户指定 4 档）：
    全对 1.0 / 格式对答案错 0.3 / 格式错答案对 0.7 / 全错 0.0
"""
from __future__ import annotations

import os

# ---------------- 路径 ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根
GRPO_DIR = os.path.join(ROOT, "grpo")
DATA_DIR = os.path.join(ROOT, "data", "grpo")                        # 生成数据/轨迹
MODELS_ROOT = os.path.join(ROOT, "data", "models")

# 真实条款语料（rag2 分块产物，81 本规范 13661 条款）：数据集生成器用其构造
# 「模拟 RAG 检索证据」，让样本贴近真实 agent 场景；缺失时退化为规则合成的证据。
CORPUS_CLAUSES = os.path.join(ROOT, "rag2", "data", "corpus", "clauses.jsonl")

# LoRA 合并链产物（按序 merge 落盘，每阶段基座 = 上一阶段产物）
WEIGHT_CHAIN = {
    "s1_grpo": os.path.join(MODELS_ROOT, "qwen-grpo", "s1_grpo"),    # 阶段一 GRPO 后 merge
    "s1_sft": os.path.join(MODELS_ROOT, "qwen-grpo", "s1_sft"),      # SFT 后 merge（阶段二基座+ref）
    "final": os.path.join(MODELS_ROOT, "qwen-grpo", "final"),        # 阶段二 GRPO 后 merge
}

# 训练基座：显式传入或环境变量（GPU 机需自行下载放置）
#   例: Qwen2.5-7B-Instruct / Qwen2.5-3B-Instruct
DEFAULT_BASE = os.environ.get(
    "GRPO_BASE_MODEL", os.path.join(MODELS_ROOT, "Qwen2.5-3B-Instruct"))
# 本机 8G 显存实测：3B bf16 权重 5.75G + LoRA/优化器 ~0.7G + 激活，余量约 1.5G
HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

# ---------------- 四维连续奖励（reward.py 使用，勿改值）----------------
# 旧 4 档离散奖励 {1.0,0.7,0.3,0.0} 已废弃（稀疏、跨度大、梯度震荡）。
# 新奖励 = 四维连续评分加权求和（每维 ∈ [0,1]），形成平滑奖励曲线：
#   格式合规分 format  —— JSON 语法 + 必需字段 + conclusion 白名单（分级 1.0/0.8/0.6/0.4/0.2/0）
#   CoT 推理连贯分 cot —— 思维链结构/衔接词/数值与依据引用（确定性代理，可选 LLM 裁判）
#   依据溯源精准分 basis —— 预测依据与 golden_basis 的命中率 − 捏造惩罚
#   核心判分结论分 answer —— 与旧 answer_ok 同源，但 set_match/numeric 改为连续打分
REWARD_WEIGHTS = {"format": 0.15, "cot": 0.20, "basis": 0.25, "answer": 0.40}  # 求和=1
# 兼容旧口径的常量（供报告/测试对照，训练不再使用）
Q_FULL = 1.0
Q_FMT_OK_ANS_WRONG = 0.3
Q_FMT_WRONG_ANS_OK = 0.7
Q_ZERO = 0.0

# ---------------- 多维评测阈值（evaluate.py 使用）----------------
EVAL_ANSWER_OK_MIN = 0.9   # answer 维 ≥ 此值视为「答对」（正确率/漏报/误报口径）
EVAL_FMT_OK_MIN = 0.8      # format 维 ≥ 此值视为「格式合规」
SFT_REWARD_MIN = 0.85      # SFT 轨迹回收下限（旧 0.7 档重建逻辑升级为连续阈值）
SFT_KEEP_MIN = 0.95        # 轨迹 reward ≥ 此值：直接原样学（近满四维）
SFT_REBUILD_MIN = 0.5      # 轨迹 reward ∈ [0.5, 0.95)：按 judge_meta 金据重建合规 JSON 学（教规范格式）
                           # reward < 0.5：丢弃（学它有害）

# ---------------- 采样（要素③：T=1.0, 每组 G=8）----------------
G = 8
TEMPERATURE = 1.0        # 阶段二建议 0.9（--temperature 覆盖）
TOP_P = 0.95
MAX_NEW_TOKENS = 512
MIN_RESPONSE_TOKENS = 8   # 过短视为无效(记格式分并过滤轨迹)

# ---------------- GRPO loss（要素⑤）----------------
CLIP_EPS = 0.2           # clip 裁剪 ε
KL_BETA = 0.04           # 阶段二建议 0.05~0.1（--beta 覆盖）
GRPO_INNER_EPOCHS = 3    # 每组数据原地更新次数（用户定稿：本项目用 k3 KL 无偏估计，
                         # inner 可保持 3；前提是**小学习率** GRPO_LR=7e-6 控制单批次内策略漂移）
GRPO_LR = 7e-6
MAX_SEQ_LEN = 1024

# ---------------- LoRA（每次微调新建 adapter，结束后 merge）----------------
# 8G 显存实测：r=32 时固定开销 = 权重 5.75G + LoRA fp32 参数 0.24G + 梯度 0.24G
# + AdamW 状态 0.48G = 6.71G，仅剩 1.2G 给激活/图 → s2 第 9 组 backward OOM。
# r=16（alpha 同步减半，保持 alpha/r=2）把这三项各减半，省 0.36G。
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0    # 必须为 0：训练走 model.train()（启用 gradient checkpointing
                      # 的必要条件），dropout 会破坏 importance ratio 的精确性
LORA_TARGETS = "all-linear"   # peft: q,k,v,o,gate,up,down 全线性层

# ---------------- SFT（阶段一数据回收后）----------------
SFT_LR = 1.5e-5
SFT_EPOCHS = 5
SFT_BATCH = 8

# ---------------- 训练步数（预算受限可减）----------------
GRPO_STEPS = 300          # 阶段二建议 150（--steps 覆盖）
QUERY_BATCH = 8           # 每 step 的 query 数（有效样本 = ×G）

# ---------------- 数据生成 ----------------
GEN_SCALE = 1             # 扰动网格放大倍数（越大样本越多）
VAL_RATIO = 0.15          # 验证集切分（按来源隔离）
GEN_SEED = 42
