"""ner2 BERT-NER 数据层：标注加载 + BIOE(+X) 标签映射 + 字符偏移→token 对齐。

标签方案沿用旧 `ner/src/bert/data_utils.py`（BIOE + 子词 X），保证与历史指标可比：

    ["O", "X"] + {B,I,E}-{工程类型/工序/设备/参数/规范编号/危大类别}   # 2 + 3×6 = 20 类

约定：
- **子词（##x）标 X**：CE 场景置 -100（不进 loss）；CRF 场景由模型侧映射为 O
  （线性链 CRF 无法 ignore 中间 token，中文按字切分基本不产 X，偶发英文子词无损）。
- **词首 token** 继承它覆盖首字符的字级 BIOE 标签（offset_mapping 精确对齐）。
- 特殊符 [CLS]/[SEP] 一律 -100 / 名为 O。
"""
from __future__ import annotations

import json
import os
import random

ENTITY_TYPES = ("工程类型", "工序", "设备", "参数", "规范编号", "危大类别")


def _build_tags() -> list[str]:
    tags = ["O", "X"]
    for et in ENTITY_TYPES:
        for pre in ("B", "I", "E"):
            tags.append(f"{pre}-{et}")
    return tags


TAGS: list[str] = _build_tags()
TAG2ID: dict[str, int] = {t: i for i, t in enumerate(TAGS)}
ID2TAG: dict[int, str] = {i: t for t, i in TAG2ID.items()}
NUM_TAGS: int = len(TAGS)
X_ID: int = TAG2ID["X"]
O_ID: int = TAG2ID["O"]
IGNORE: int = -100


# --------------------------------------------------------------------------- IO
def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, rows: list[dict]) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------- 字级 BIOE
def char_bioe(text: str, entities: list[dict]) -> list[str]:
    """字符偏移实体 → 逐字 BIOE 标签数组（len == len(text)）。"""
    n = len(text)
    labels = ["O"] * n
    # 长度升序写入，长实体最后落笔（重叠时以长者为尊；本语料已验证无重叠）
    for ent in sorted(entities, key=lambda e: int(e["end"]) - int(e["start"])):
        et = ent.get("type")
        if et not in ENTITY_TYPES:
            continue
        s = max(0, min(int(ent["start"]), n))
        e = max(s, min(int(ent["end"]), n))
        if e - s <= 0:
            continue
        labels[s] = f"B-{et}"
        if e - s == 1:
            continue
        for i in range(s + 1, e - 1):
            labels[i] = f"I-{et}"
        labels[e - 1] = f"E-{et}"
    return labels


# ------------------------------------------------------------------ 样本编码
def encode(text: str, entities: list[dict], tokenizer, max_len: int = 256) -> dict:
    """单样本编码为 token 级张量素材。

    返回 {"input_ids", "attention_mask", "labels", "gold_names", "offsets", "text"}
      - labels：CE 用；子词/特殊符/pad 一律 IGNORE(-100)
      - gold_names：token 级标签名（子词为 "X"），供 CRF 映射与实体评估
      - offsets：每个 token 在原文的 [start, end) 字符区间
    """
    enc = tokenizer(text, truncation=True, max_length=max_len,
                    return_offsets_mapping=True, add_special_tokens=True)
    ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    char_labels = char_bioe(text, entities)
    labels = [IGNORE] * len(ids)
    gold_names = ["O"] * len(ids)
    for i, (tok_str, (s, e)) in enumerate(zip(tokens, offsets)):
        if (s, e) == (0, 0):                      # [CLS]/[SEP]/[PAD]
            continue
        if tok_str.startswith("##"):              # 非词首子词
            gold_names[i] = "X"
            labels[i] = IGNORE
            continue
        name = char_labels[s] if 0 <= s < len(char_labels) else "O"
        gold_names[i] = name
        labels[i] = TAG2ID.get(name, O_ID)
    return {"input_ids": ids, "attention_mask": [int(x) for x in enc["attention_mask"]],
            "labels": labels, "gold_names": gold_names, "offsets": offsets, "text": text}


def char_entities_to_token_names(text: str, entities: list[dict], tokenizer,
                                 max_len: int = 256) -> list[str]:
    """便捷函数：字符实体 → token 级标签名序列（评估时复算 gold 用）。"""
    return encode(text, entities, tokenizer, max_len)["gold_names"]


def split_train_eval(samples: list[dict], eval_ratio: float = 0.15, seed: int = 42):
    """按比例切分（可复现）。"""
    arr = list(samples)
    random.Random(seed).shuffle(arr)
    n_eval = max(1, int(len(arr) * eval_ratio)) if len(arr) > 1 else 0
    return arr[n_eval:], arr[:n_eval]
