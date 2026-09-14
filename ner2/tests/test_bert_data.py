"""P4 数据层单测：字符 BIOE、token 对齐、字符级实体解码、高置信度过滤判据。

用字符级 stub tokenizer（每字一个 token，特殊符 offset=(0,0)），无需下载模型。
"""
from __future__ import annotations

from ner2.src.bert.data_utils import (
    IGNORE, O_ID, TAGS, TAG2ID, char_bioe, encode,
)
from ner2.src.bert.decode import token_tags_to_char_entities
from ner2.src.bert.engine import collect_high_confidence


class CharTokenizer:
    """最小可行 tokenizer：逐字切分，首尾加特殊符（offset=(0,0)）。"""

    pad_token_id = 0

    def __call__(self, text, truncation=True, max_length=256, return_offsets_mapping=True,
                 add_special_tokens=True):
        chars = list(text)
        if truncation and len(chars) > max_length - 2:
            chars = chars[: max_length - 2]
        ids = [101] + [1000 + i for i in range(len(chars))] + [102]
        offsets = [(0, 0)] + [(i, i + 1) for i in range(len(chars))] + [(0, 0)]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "offset_mapping": offsets}

    def convert_ids_to_tokens(self, ids):
        out = []
        for i in ids:
            if i == 101:
                out.append("[CLS]")
            elif i == 102:
                out.append("[SEP]")
            else:
                out.append(f"t{i}")
        return out


def test_tag_table_shape():
    assert len(TAGS) == 2 + 3 * 6
    assert TAGS[0] == "O" and TAGS[1] == "X"
    assert "B-工序" in TAG2ID and "E-危大类别" in TAG2ID


def test_char_bioe_single_and_multi():
    text = "采用塔吊施工"
    labels = char_bioe(text, [{"type": "设备", "start": 2, "end": 4}])
    assert labels == ["O", "O", "B-设备", "E-设备", "O", "O"]
    # 单字实体只有 B
    labels2 = char_bioe("基坑", [{"type": "工程类型", "start": 0, "end": 1}])
    assert labels2 == ["B-工程类型", "O"]
    # 三字实体 B/I/E
    labels3 = char_bioe("abcde", [{"type": "工序", "start": 1, "end": 4}])
    assert labels3 == ["O", "B-工序", "I-工序", "E-工序", "O"]


def test_encode_alignment_and_ignore():
    tok = CharTokenizer()
    text = "采用塔吊施工"
    e = encode(text, [{"type": "设备", "start": 2, "end": 4}], tok, max_len=64)
    assert e["labels"][0] == IGNORE and e["labels"][-1] == IGNORE      # CLS/SEP
    assert e["gold_names"][3] == "B-设备" and e["labels"][3] == TAG2ID["B-设备"]
    assert e["gold_names"][4] == "E-设备"
    assert e["labels"][1] == O_ID
    assert len(e["offsets"]) == len(e["labels"]) == len(e["input_ids"])


def test_token_tags_to_char_entities_roundtrip():
    tok = CharTokenizer()
    text = "采用塔吊和基坑工程施工"
    ents = [{"type": "设备", "start": 2, "end": 4},
            {"type": "工程类型", "start": 5, "end": 9}]
    e = encode(text, ents, tok, max_len=64)
    got = token_tags_to_char_entities(e["offsets"], e["gold_names"])
    assert [(x["type"], x["start"], x["end"]) for x in got] == \
           [("设备", 2, 4), ("工程类型", 5, 9)]


def test_decode_handles_malformed_and_specials():
    # 无 E 收尾（模型漏 E）→ 容错到类型变化处；特殊符 offset (0,0) 跳过
    offsets = [(0, 0), (0, 1), (1, 2), (2, 3), (0, 0)]
    tags = ["O", "B-工序", "I-工序", "O", "O"]
    out = token_tags_to_char_entities(offsets, tags)
    assert out == [{"type": "工序", "start": 0, "end": 2}]


def test_collect_high_confidence_margin_gate():
    rows = [
        # 高置信实体句：保留
        {"text": "塔吊", "offsets": [(0, 0), (0, 1), (1, 2), (0, 0)],
         "tags": ["O", "B-设备", "E-设备", "O"], "margins": [0.9, 0.8, 0.7, 0.9]},
        # span 内 margin 不足 → 剔除
        {"text": "塔吊", "offsets": [(0, 0), (0, 1), (1, 2), (0, 0)],
         "tags": ["O", "B-设备", "E-设备", "O"], "margins": [0.9, 0.05, 0.7, 0.9]},
        # O 侧中位 margin 不足 → 剔除
        {"text": "塔吊", "offsets": [(0, 0), (0, 1), (1, 2), (0, 0)],
         "tags": ["O", "B-设备", "E-设备", "O"], "margins": [0.05, 0.9, 0.9, 0.02]},
        # CRF 解码不稳定 → 剔除
        {"text": "塔吊", "offsets": [(0, 0), (0, 1), (1, 2), (0, 0)],
         "tags": ["O", "B-设备", "E-设备", "O"], "margins": [0.9, 0.9, 0.9, 0.9],
         "crf_consistent": False},
    ]
    kept, stat = collect_high_confidence(rows, sp_margin=0.25, o_margin=0.15,
                                        require_crf_consistent=True, keep_negatives=0.0)
    assert len(kept) == 1
    assert kept[0]["text"] == "塔吊"
    assert stat["drop_span_margin"] == 1 and stat["drop_o_margin"] == 1
    assert stat["drop_inconsistent"] == 1
