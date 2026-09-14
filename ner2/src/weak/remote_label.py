"""远程监督标注核心：词典最长匹配 + 正则 + 类型优先级网格去重叠。

设计（与旧 ner 一致并增强）：
- 规范编号：正则先行，占用网格；
- 词典词：按 (长度降序, 类型优先级升序) 全词扫描，最长匹配优先；
- 占用网格：字符级 used[]，重叠 span 丢弃（先到先得）；
- 每个实体附带 weak 元信息 {source: regex|dict, term: 命中的词典词}，
  供 P2 LLM 二次校验追溯"边界偏短/类型歧义"的根因；正式训练格式兼容
  {"type","start","end"}（data_utils 只读这三个字段）。

句子清洗规则同旧 weak_label._dirty：剔除页眉伪影/超长/非中文开头。
"""
from __future__ import annotations

import re
from collections import Counter

from ner2.src.common.types import TYPE_PRIORITY, TYPE_SET
from ner2.src.weak.lexicon import DOC_RE, STD_RE

_CN = re.compile(r"^[\u4e00-\u9fff]")


def is_dirty(s: str) -> bool:
    """判为噪声行：伪影字符 / 过长 / 过短 / 非中文开头。"""
    if re.search(r"[\.·•]{3,}|…{3,}|[-—]{6,}|┬|┌|└|│|┃|□|◆|〔[^\]]*〕\s*\d+\s*号\s*$", s):
        return True
    if len(s) > 240 or len(s) < 6:
        return True
    if not _CN.match(s):
        return True
    return False


def split_long(s: str):
    """超长行按句末标点切分；子句仍做 is_dirty 过滤。"""
    for sub in re.split(r"[。；!?！？]+", s):
        sub = sub.strip()
        if not sub or is_dirty(sub):
            continue
        yield sub


def normalize_text(text: str) -> str:
    """去空白，得到"紧凑文本"用于词典匹配。"""
    return text.replace(" ", "").replace("\u3000", "")


def char_map(text: str) -> list[int]:
    """紧凑文本字符 -> 原文本字符索引。"""
    return [i for i, ch in enumerate(text) if ch not in (" ", "\u3000")]


class RemoteLabeler:
    """词典+正则远程监督标注器。"""

    def __init__(self, lexicon: dict):
        # 扁平化词典词表：(term, type)，按长度降序 + 类型优先级升序
        self.terms = []
        for t, ws in lexicon.items():
            if t == "规范编号" or t not in TYPE_SET:
                continue
            for w in ws:
                w = w.replace(" ", "").replace("\u3000", "")
                if w:
                    self.terms.append((w, t))
        self.terms.sort(key=lambda x: (-len(x[0]), TYPE_PRIORITY.get(x[1], 9)))

    def tag(self, text: str) -> list[dict]:
        """对单句打弱标签，返回实体列表（含 weak 元信息）。"""
        norm = normalize_text(text)
        mp = char_map(text)
        n = len(norm)
        if n == 0:
            return []
        used = [False] * n
        ents = []

        # 1) 规范编号正则（先行）
        for m in STD_RE.finditer(norm):
            s, e = m.start(), m.end()
            if not any(used[s:e]):
                for k in range(s, e):
                    used[k] = True
                ents.append({
                    "type": "规范编号",
                    "start": mp[s], "end": mp[e - 1] + 1,
                    "source": "regex", "term": norm[s:e],
                })

        # 2) 词典最长匹配
        for w, t in self.terms:
            i = 0
            while True:
                j = norm.find(w, i)
                if j < 0:
                    break
                s, e = j, j + len(w)
                if not any(used[s:e]):
                    for k in range(s, e):
                        used[k] = True
                    ents.append({
                        "type": t,
                        "start": mp[s], "end": mp[e - 1] + 1,
                        "source": "dict", "term": w,
                    })
                i = j + 1

        # 3) 排序 + 网格冲突丢弃（先到先得已保证，这里仅稳定输出顺序）
        ents.sort(key=lambda x: (x["start"], x["end"] - x["start"]))
        return ents

    def stats(self, weak: list[dict]) -> dict:
        c = Counter(e["type"] for w in weak for e in w["entities"])
        by_src = Counter(e["source"] for w in weak for e in w["entities"])
        return {"sent": len(weak), "ents": sum(c.values()),
                "by_type": dict(c), "by_source": dict(by_src)}
