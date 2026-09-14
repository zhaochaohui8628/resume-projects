"""施工方案合规自查 - 规则引擎（C1 / C2 / C4）

纯标准库实现，零依赖，可在无 GPU / 无 API Key 环境下直接运行与测试。
依赖 agent/data/rules/ 下的三份 JSON 规则库（均来自住建部官方公告，已核实）。

检查能力：
  C1 过期/废止规范引用：正则 + 查表，准确率 100%，不走 RAG
  C2 危大工程缺项：基于危大类别关键词 + 阈值 + 必备要素
  C4 方案编制要素缺失：建办质〔2018〕31号 九章 + 建办质〔2021〕48号 逐类细化

本模块作为 Agent 项目的一个工具（tool）被 ReAct 链路调用。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Optional

# agent/src/tools/rules_checker.py -> workspace/agent/data/rules
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
DEFAULT_RULES_DIR = os.path.join(ROOT, "agent", "data", "rules")


def _norm(s: str) -> str:
    s = s.strip().upper()
    s = re.sub(r"\s+", "", s)
    return s


# ---------------------------------------------------------------- 规则加载
class RuleStore:
    def __init__(self, rules_dir: str):
        self.rules_dir = rules_dir
        self.abolished = self._load("abolished_clauses.json")
        self.hazardous = self._load("hazardous_work_types.json")
        self.sections = self._load("required_sections.json")
        self._build_abolish_index()

    def _load(self, name):
        with open(os.path.join(self.rules_dir, name), "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_abolish_index(self):
        self.full_abolished = {}  # norm_std_no -> item
        self.clause_abolished = {}  # norm_std_no -> set(clauses)
        for it in self.abolished.get("full_doc_abolished", []):
            self.full_abolished[_norm(it["std_no"])] = it
        for grp in self.abolished.get("clause_abolished", []):
            for item in grp["items"]:
                n = _norm(item["std_no"])
                self.clause_abolished.setdefault(n, set()).update(item["clauses"])


# ---------------------------------------------------------------- 风险项
@dataclass
class Risk:
    check: str  # C1 / C2 / C4
    severity: str  # HIGH / MEDIUM / LOW
    title: str
    detail: str
    evidence: str = ""
    suggestion: str = ""
    source: str = "rules"  # 标记由规则引擎产出

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- 抽取
STD_CAND = re.compile(r"[A-Za-z]+(?:/[A-Za-z]+)?\s*\d[\d\-\s]{2,}\d{3,4}")
DOC_NO = re.compile(
    r"(建质|建办质|住建部令|沪建规?范?|沪建质安|沪建管|沪建交联|沪水务)"
    r"\s*[〔\[]?\s*(\d{4})\s*[〕\]]?\s*(\d+)\s*号"
)
CLAUSE_RE = re.compile(r"第\s*(\d+(?:\.\d+)*)\s*(?:\((\d+)\))?\s*条")


def _extract_standard_refs(text: str):
    refs, seen = [], set()
    for m in STD_CAND.finditer(text):
        raw = m.group(0)
        n = _norm(raw)
        if n not in seen:
            seen.add(n)
            refs.append((raw, n))
    return refs


# ---------------------------------------------------------------- C1
def check_c1(text: str, store: RuleStore):
    risks = []
    for raw, n in _extract_standard_refs(text):
        if n in store.full_abolished:
            it = store.full_abolished[n]
            risks.append(
                Risk(
                    check="C1",
                    severity="HIGH",
                    title=f"引用已废止/失效标准：{raw}",
                    detail=f'{it["std_name"]} 已于 {it["effective_date"]} 废止，被 {it["replaced_by"]} 取代。',
                    evidence=raw,
                    suggestion=f'将编制依据中的 {raw} 替换为现行版本 {it["replaced_by"]}。',
                )
            )
        elif n in store.clause_abolished:
            abolished_clauses = store.clause_abolished[n]
            idx = text.upper().find(raw.upper())
            window = text[max(0, idx - 40) : idx + len(raw) + 60]
            cited = CLAUSE_RE.findall(window)
            hit = []
            for a, b in cited:
                cstr = a + (f"({b})" if b else "")
                if cstr in abolished_clauses:
                    hit.append(cstr)
            if hit:
                risks.append(
                    Risk(
                        check="C1",
                        severity="HIGH",
                        title=f"引用已废止的具体条文：{raw} 第 {', '.join(hit)} 条",
                        detail="该标准整体未废止，但所列条文已被全文强制规范废止，不得作为依据。",
                        evidence=f"{raw} 第 {', '.join(hit)} 条",
                        suggestion="删除/替换这些废止条文，改引全文强制规范的对应要求。",
                    )
                )
            else:
                risks.append(
                    Risk(
                        check="C1",
                        severity="LOW",
                        title=f"引用了含废止条文的标准：{raw}",
                        detail=f"该标准未被整本废止，但 GB 550xx 全文强制规范已废止其部分条文（共 {len(abolished_clauses)} 条）。若方案引用了具体条文号需核对。",
                        evidence=raw,
                        suggestion="核对编制依据中引用的具体条文是否落在废止清单内。",
                    )
                )
    for m in DOC_NO.finditer(text):
        prefix, year, num = m.group(1), m.group(2), m.group(3)
        if prefix == "建质" and year == "2009" and num == "87":
            risks.append(
                Risk(
                    check="C1",
                    severity="HIGH",
                    title="引用已废止文件：建质〔2009〕87号",
                    detail="《危险性较大的分部分项工程安全管理办法》（建质〔2009〕87号）已于 2018-06-01 废止，被住建部令第37号 + 建办质〔2018〕31号 取代。",
                    evidence=m.group(0),
                    suggestion="将文件引用更新为住建部令第37号及建办质〔2018〕31号。",
                )
            )
    return risks


# ---------------------------------------------------------------- C2
CATEGORY_KEYWORDS = {
    "基坑工程": ["基坑", "开挖", "围护", "支护", "降水", "地下连续墙", "灌注桩", "SMW"],
    "模板支撑": ["模板", "支撑体系", "满堂支架", "高支模", "排架"],
    "起重吊装": ["吊装", "起重机", "塔吊", "履带吊", "汽车吊", "起重"],
    "脚手架": ["脚手架", "扣件式", "盘扣", "悬挑架"],
    "拆除": ["拆除", "爆破"],
    "暗挖": ["暗挖", "盾构", "顶管", "矿山法"],
    "幕墙安装": ["幕墙"],
    "人工挖孔桩": ["人工挖孔", "挖孔桩"],
    "钢结构安装": ["钢结构", "钢构", "网架"],
}
# ⚠️ 2026-09-13：原 SCALE_PATTERNS（基坑5m/模板8m/脚手架50m + re.search 全文取首个匹配）
#    已**删除**。它只有 3 类、取全文第一个数字、与 NER/RAG 零交互，是漏检来源。
#    危大/超规模判档统一由 `tools/hazard_level.py` 的阈值表完成（9 类 / 22 组维度，
#    以工况三元组为输入）。C2 在此只保留「必备内容缺失」检查。


def _any_in(text, kws):
    return any(k in text for k in kws)


def check_c2(text: str, store: RuleStore):
    """C2：危大工程**必备内容缺失**检查（阈值判档不在此处，见 tools/hazard_level.py）。"""
    risks = []
    cat_by_key = {c["key"]: c for c in store.hazardous["categories"]}
    detected = [k for k, kws in CATEGORY_KEYWORDS.items() if _any_in(text, kws)]
    for key in detected:
        c = cat_by_key.get(key)
        if not c:
            continue
        for must in c.get("must_have", []):
            if not _any_in(text, [must]):
                risks.append(
                    Risk(
                        check="C2",
                        severity="MEDIUM",
                        title=f'【{c["name"]}】疑似缺少必备内容：{must}',
                        detail=f'方案被识别为危大工程「{c["name"]}」，按 37 号令/31 号文应至少包含「{must}」。',
                        evidence="",
                        suggestion=f'补充「{must}」章节或内容。',
                    )
                )
    return risks


# ---------------------------------------------------------------- C4
def check_c4(text: str, store: RuleStore, detected_types=None):
    risks = []
    sec = store.sections
    for ch in sec["required_chapters"]:
        if not _any_in(text, ch["keywords"]):
            risks.append(
                Risk(
                    check="C4",
                    severity="MEDIUM",
                    title=f'缺少编制要素（建办质〔2018〕31号九章）：{ch["name"]}',
                    detail="专项施工方案应包含九章内容，未检测到本章对应关键词。",
                    evidence="",
                    suggestion=f'补充「{ch["name"]}」章节。',
                )
            )
    per = sec.get("per_type_sections", {})
    if detected_types:
        for key in detected_types:
            for sub in per.get(key, []):
                if not _any_in(text, sub["keywords"]):
                    risks.append(
                        Risk(
                            check="C4",
                            severity="LOW",
                            title=f"缺少细化要素（建办质〔2021〕48号）：{key} - {sub['name']}",
                            detail="按编制指南，该类型危大工程方案应包含此细化要素。",
                            evidence="",
                            suggestion=f'补充「{sub["name"]}」内容。',
                        )
                    )
    return risks


# ---------------------------------------------------------------- 入口
def run_checks(text: str, rules_dir: Optional[str] = None):
    rules_dir = rules_dir or DEFAULT_RULES_DIR
    store = RuleStore(rules_dir)
    detected = [k for k, kws in CATEGORY_KEYWORDS.items() if _any_in(text, kws)]
    risks = []
    risks += check_c1(text, store)
    risks += check_c2(text, store)
    risks += check_c4(text, store, detected)
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    risks.sort(key=lambda r: order.get(r.severity, 3))
    return {
        "detected_types": detected,
        "risk_count": len(risks),
        "risks": [r.to_dict() for r in risks],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    print(json.dumps(run_checks(text), ensure_ascii=False, indent=2))
