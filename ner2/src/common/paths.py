"""路径常量：ner2 全部脚本按相对位置解析，不依赖 cwd。"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))          # ner2/src/common
SRC = os.path.dirname(HERE)                                 # ner2/src
NER2 = os.path.dirname(SRC)                                 # ner2/
ROOT = os.path.dirname(NER2)                                # 项目根

DATA = os.path.join(NER2, "data")
PHASE1 = os.path.join(DATA, "phase1")
PHASE2 = os.path.join(DATA, "phase2")
PHASE4 = os.path.join(DATA, "phase4")          # P4 训练：伪标、检查点元信息、报告
PHASE5 = os.path.join(DATA, "phase5")          # P5 黄金集
PHASE6 = os.path.join(DATA, "phase6")          # P6 迭代
LEXICON_DIR = os.path.join(DATA, "lexicon")
DOCS = os.path.join(NER2, "docs")
SCRIPTS = os.path.join(NER2, "scripts")
MODELS = os.path.join(NER2, "models")          # 训练产物（softmax/crf × 两轮）

# 基座（本机已有，无需联网）
BASE_BERT = os.path.join(ROOT, "data", "models", "bert-base-chinese")
# 旧 ner 微调产物（仅作对照参考，不参与训练）
LEGACY_BERT_NER = os.path.join(ROOT, "data", "models", "bert-ner", "m1_r11")

# 训练/测试数据
GOLD_TEST = os.path.join(PHASE5, "gold_test.jsonl")
GOLD_TRAIN_PORTION = os.path.join(PHASE5, "gold_train_portion.jsonl")
GOLD_CONSENSUS = os.path.join(PHASE5, "gold_consensus.jsonl")

# 复用旧 ner 的独立资产（词典种子、金标、历史弱标参考）
LEGACY_LEXICON = os.path.join(ROOT, "data", "ner", "lexicon.json")
LEGACY_GOLD = os.path.join(ROOT, "data", "ner", "gold_eval_v2.jsonl")
LEGACY_CLEAN = os.path.join(ROOT, "data", "ner", "clean_r11.jsonl")

# 语料（6 份方案 txt + 外部项目方案 txt，2026-09-11 起扩源）
PLANS_DIR = os.path.join(ROOT, "data", "raw", "plans_internal")
PLANS_XPROJ_DIR = os.path.join(ROOT, "data", "raw", "plans_xproj")

# 产物
WEAK = os.path.join(PHASE1, "weak.jsonl")
UNLABELED = os.path.join(PHASE1, "unlabeled_sentences.jsonl")
JUDGMENTS = os.path.join(PHASE2, "judgments.jsonl")
SILVER_TRAIN = os.path.join(PHASE2, "silver_train.jsonl")
SILVER_VAL = os.path.join(PHASE2, "silver_val.jsonl")
CONFLICTS = os.path.join(PHASE2, "conflicts.jsonl")
