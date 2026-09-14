"""级联层 1：确定性规则/词典层（高精度拦截）。

职责（只做"确定性可判"的部分，宁缺勿滥）：
  1) 规范编号：标准号 / 公文号 正则（封闭形态，正则 100% 可靠）；
  2) 危大类别：关键词表（封闭类目，与 agent 侧 rules_checker 单一事实源）；
  3) 高置信词典词：设备型号、工程类型专有名词等"封闭专有名词"最长匹配。

本层不处理开放词表 / 复杂上下文（那交给层 2 微调模型兜底）。
输出带 `layer="rule"` 与 `conf`（置信度，规则命中恒为 1.0）。
"""
from __future__ import annotations

import re

from ner2.src.common.types import TYPE_PRIORITY, TYPE_SET
from ner2.src.weak.lexicon import DOC_RE, STD_RE

_CN = re.compile(r"^[\u4e00-\u9fff]")

# 危大类别关键词（封闭类目；与 agent/src/tools/rules_checker.py 的 CATEGORY_KEYWORDS 语义一致）
CATA_KEYWORDS = {
    "基坑工程": ["基坑工程", "基坑", "开挖", "围护", "支护", "降水", "地下连续墙", "灌注桩", "SMW"],
    "模板支撑": ["模板支撑", "支撑体系", "满堂支架", "高支模"],
    "起重吊装": ["起重吊装", "吊装", "起重机", "塔吊", "履带吊", "汽车吊", "起重"],
    "脚手架": ["脚手架", "扣件式", "盘扣", "悬挑架"],
    "拆除": ["拆除", "爆破"],
    "暗挖": ["暗挖", "盾构", "顶管", "矿山法"],
    "幕墙安装": ["幕墙"],
    "人工挖孔桩": ["人工挖孔", "挖孔桩"],
    "钢结构安装": ["钢结构", "钢构", "网架"],
}

# 高置信"封闭专有名词"类型白名单：这类词做词典最长匹配最可靠
_CLOSED_TYPES = ("设备", "工程类型", "工序", "参数")


class RuleLayer:
    """确定性规则/词典层。"""

    def __init__(self, lexicon: dict, use_lexicon: bool = True):
        self.use_lexicon = use_lexicon
        self.terms = []
        for t, ws in lexicon.items():
            if t == "规范编号" or t not in TYPE_SET or t not in _CLOSED_TYPES:
                continue
            for w in ws:
                w = w.replace(" ", "").replace("\u3000", "")
                if w:
                    self.terms.append((w, t))
        # 长词优先；同长按类型优先级
        self.terms.sort(key=lambda x: (-len(x[0]), TYPE_PRIORITY.get(x[1], 9)))

    def extract(self, text: str) -> list[dict]:
        """返回实体列表（字符偏移，不重叠；layer=rule, conf=1.0）。"""
        norm = text.replace(" ", "").replace("\u3000", "")
        mp = [i for i, ch in enumerate(text) if ch not in (" ", "\u3000")]
        n = len(norm)
        if n == 0:
            return []
        used = [False] * n
        ents = []

        def _add(typ, s, e, term):
            for k in range(s, e):
                used[k] = True
            ents.append({"type": typ, "start": mp[s], "end": mp[e - 1] + 1,
                         "layer": "rule", "conf": 1.0, "term": term})

        # 1) 规范编号正则（先行）
        for m in STD_RE.finditer(norm):
            s, e = m.start(), m.end()
            if not any(used[s:e]):
                _add("规范编号", s, e, norm[s:e])
        for m in DOC_RE.finditer(text):
            s, e = m.start(), m.end()
            ns = len([1 for ch in text[:s] if ch not in (" ", "\u3000")])
            ne = ns + (e - s)
            if ne <= n and not any(used[ns:ne]):
                _add("规范编号", ns, ne, text[s:e])

        # 2) 危大类别关键词（封闭类目）
        for cat, kws in CATA_KEYWORDS.items():
            best = None
            for kw in kws:
                j = norm.find(kw)
                if j >= 0 and not any(used[j:j + len(kw)]):
                    if best is None or len(kw) > len(best[0]):
                        best = (kw, j)
            if best:
                kw, j = best
                _add("危大类别", j, j + len(kw), kw)

        # 3) 高置信词典专有名词（最长匹配）
        if self.use_lexicon:
            for w, t in self.terms:
                i = 0
                while True:
                    j = norm.find(w, i)
                    if j < 0:
                        break
                    if not any(used[j:j + len(w)]):
                        _add(t, j, j + len(w), w)
                    i = j + 1

        ents.sort(key=lambda x: (x["start"], x["end"] - x["start"]))
        return ents
