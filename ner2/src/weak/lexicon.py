"""词典加载与规范化。

词典 = 远程监督的种子知识源（封闭类目）。
来源优先级：用户增补 > 规则生成 > 旧 lexicon 种子。规范编号不依赖词典（走正则）。
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

from ner2.src.common.paths import LEXICON_DIR, LEGACY_LEXICON
from ner2.src.common.types import VALID_TYPES

# 规范编号正则（远程监督先行）：GB50010-2010 / JGJ80-2016 / DG/TJ08-...
STD_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z]{1,6}/?[A-Z]{0,4}\s*)?\d{2,4}[-—]\d{4}")
# 公文号式（沪地标常见）：DG/TJ 08-2023-XX 之类由 STD_RE 覆盖；此处兜底"〔年〕号"
DOC_RE = re.compile(r"[^\s，。；、]{1,8}〔\d]{4}〕\d{1,3}\s*号")

_DEFAULT = {t: [] for t in VALID_TYPES}


def _norm_term(w: str) -> str:
    return w.replace(" ", "").replace("\u3000", "").strip()


def load_legacy_lexicon() -> dict:
    """读旧 ner 的词典种子（data/ner/lexicon.json）。"""
    if not os.path.exists(LEGACY_LEXICON):
        return dict(_DEFAULT)
    data = json.load(open(LEGACY_LEXICON, encoding="utf-8"))
    out = dict(_DEFAULT)
    for t, ws in data.items():
        if t in out:
            out[t] = [w for w in (_norm_term(x) for x in ws) if w]
    return out


def load_user_lexicon() -> dict:
    """读用户/人工增补词表（data/lexicon/user.json），可选。"""
    p = os.path.join(LEXICON_DIR, "user.json")
    if not os.path.exists(p):
        return dict(_DEFAULT)
    data = json.load(open(p, encoding="utf-8"))
    out = dict(_DEFAULT)
    for t, ws in data.items():
        if t in out:
            out[t] = [w for w in (_norm_term(x) for x in ws) if w]
    return out


def merge_lexicons(*parts: dict) -> dict:
    """合并多来源词典（后者优先级高，按类型合并去重保持顺序）。"""
    out = {t: [] for t in VALID_TYPES}
    for part in parts:
        for t in VALID_TYPES:
            for w in part.get(t, []):
                if w not in out[t]:
                    out[t].append(w)
    return out


def build_lexicon(include_user: bool = True) -> dict:
    """默认组装：旧种子 + 用户增补。规则生成词表由 build_lexicon.py 另行并入。"""
    lex = load_legacy_lexicon()
    if include_user:
        lex = merge_lexicons(lex, load_user_lexicon())
    return lex


def summary(lex: dict) -> dict:
    return {t: len(ws) for t, ws in lex.items() if t != "规范编号"}
