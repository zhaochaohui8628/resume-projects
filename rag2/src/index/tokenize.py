"""中文检索分词（零依赖）：CJK 单字 + 二元组，混合字母数字与条款号。

不做词性/词典切分是刻意的：施工规范里大量术语（"扣件式钢管脚手架""碗扣节点"）在通用
词典里不存在，二元组能稳定覆盖；同时保留 `gb55037` / `3.0.1` / `c30` / `10mpa` 这类
字母数字 token（条款号与规范号检索强依赖它们）。
"""
from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ALNUM_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

# 极高频虚词，作为单个字出现时无检索价值（二元组仍保留）
_STOP_UNIGRAM = set("的了和是在与及或对其为以等中上这有")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    t = text.lower()
    toks: list[str] = []
    for m in _CJK_RE.finditer(t):
        run = m.group(0)
        if len(run) == 1:
            if run not in _STOP_UNIGRAM:
                toks.append(run)
            continue
        toks.extend(ch for ch in run if ch not in _STOP_UNIGRAM)
        toks.extend(run[i:i + 2] for i in range(len(run) - 1))
    for m in _ALNUM_RE.finditer(t):
        s = m.group(0)
        if s and not s.isspace():
            toks.append(s)
            # 规范号归一：gb55037 / gb 55037 / gbt50010 -> 都补一个去连字符形态
            bare = s.replace("-", "").replace(".", "")
            if bare != s:
                toks.append(bare)
    return toks
