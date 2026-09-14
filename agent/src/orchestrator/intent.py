"""问句意图路由（v7）—— 用户输入直接进调度中心，**不再有强制前置的全量规则审查**。

## 为什么改
旧链路（v6.1）无论用户问什么，都先把整篇方案过一遍规则引擎（C1/C2/C4），
再把结果"按 HIGH 强制 / 用户提及"决定要不要注入汇总。问题：
- 用户问"施工工艺有什么不符合规范"时，跑的是依据/缺项检查，答非所问；
- 全量规则审查是**无差别**的，而 NER 本身已经是全量识别（逐句不收敛），
  真正需要"按问题决定查什么"的是**路由**，不是固定的前置。

## 五类意图（**只有两个 subagent：review（方案审查）/ qa**）
| intent | 触发 | agents | checks（review 内部三路） |
|---|---|---|---|
| `hazard_level` | 是否危大/超规模/需专家论证 | review | hazard_level |
| `technical` | 施工工艺/工序/做法/参数/材料 是否符合规范（含开放问句） | review | technical |
| `basis` | 引用规范是否废止/失效、编制依据 | review | basis(C1) |
| `elements` | 缺项/九章/编制要素是否齐全 | review | elements(C4)、content(C2缺项) |
| `qa` | 无方案，或一般性条文咨询 | qa | — |
| `review` | 全面审查/整体合规自查 | review | hazard_level+technical+basis+elements+content |

⚠️ 方案审查**只有一个 subagent**：三路是 review 内部的分支，不拆成多个 subagent。

`scope_terms`：从问句里抽出**章节关键词 / 危大类别词**，供 review 把范围收窄到对应章节。

降级：无 LLM 时走关键词表；有 LLM 时优先 LLM，失败回落关键词。
"""
from __future__ import annotations

import json
import os
import re

INTENT_HAZARD_LEVEL = "hazard_level"
INTENT_TECHNICAL = "technical"
INTENT_BASIS = "basis"
INTENT_ELEMENTS = "elements"
INTENT_QA = "qa"
INTENT_REVIEW = "review"
INTENT_STRUCTURED = "structured"

INTENTS = (INTENT_HAZARD_LEVEL, INTENT_TECHNICAL, INTENT_BASIS,
           INTENT_ELEMENTS, INTENT_QA, INTENT_REVIEW, INTENT_STRUCTURED)

# 各意图 → (subagents, review 内部三路开关)
_DISPATCH = {
    INTENT_HAZARD_LEVEL: (["review"], ["hazard_level"]),
    INTENT_TECHNICAL: (["review"], ["technical"]),
    INTENT_BASIS: (["review"], ["basis"]),
    INTENT_ELEMENTS: (["review"], ["elements", "content"]),
    INTENT_QA: (["qa"], []),
    INTENT_REVIEW: (["review"], ["hazard_level", "technical", "basis", "elements", "content"]),
    INTENT_STRUCTURED: (["qa"], []),   # 复合结构化问句 → qa 内部走拆解+组装
}

_KW = {
    # 注意：不含裸词"危大"——"危大工程"是名词（可触发 structured/review），
    # "算不算危大/是否危大"才是判定类意图。裸词会误抢。
    INTENT_HAZARD_LEVEL: ("超规模", "超过一定规模", "专家论证", "论证",
                          "危大级别", "是否属于危大", "算危大", "算不算危大",
                          "属于危大吗", "危大吗", "分档", "需不需要论证"),
    INTENT_BASIS: ("废止", "失效", "编制依据", "引用", "标准号", "规范编号", "版本"),
    INTENT_ELEMENTS: ("缺项", "缺少", "缺失", "九章", "编制要素", "要素", "齐全", "完整"),
    INTENT_TECHNICAL: ("施工工艺", "工艺", "工序", "做法", "参数", "材料", "措施",
                       "符合", "不符合", "强条", "合规", "安全技术", "构造"),
    INTENT_REVIEW: ("全面", "整体", "全部", "所有", "全量", "审查方案", "帮我审查",
                    "合规自查", "检查方案", "综合"),
}

_ROUTER_SYSTEM = """你是"施工方案合规审查"调度中心的意图识别器。把用户输入归到唯一意图：

- hazard_level：判断危大级别／是否超过一定规模／是否需要专家论证（查阈值表）
- technical：施工工艺、工序、做法、参数、材料、措施**是否与规范相符**（开放问句也算）
- basis：编制依据、引用规范是否废止/失效
- elements：方案缺项、九章要素是否齐全
- qa：无方案上下文的一般性条文咨询
- structured：复合结构化问句——一句话里问多个维度（风险点/方案编制内容/管控清单/验收节点等），
  且含危大类别（如"130t 汽车吊钢栈桥吊装危大工程有哪些风险点、方案编制内容、管控清单、验收节点"）
- review：要求对方案做全面/整体合规审查

另外从用户输入里抽出 scope_terms（用于把审查范围收窄到方案对应章节）：
只抽**方案章节名/工法类别**（如"施工工艺技术""模板支撑""基坑""塔吊"），抽不到就给空数组。

只输出 JSON：{"intent": "...", "scope_terms": ["..."]}，不要解释。"""


def _chapters() -> list[dict]:
    """九章清单（含关键词），用于 scope_terms 与范围定位。"""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))          # agent/src/orchestrator -> workspace
    path = os.path.join(root, "agent", "data", "rules", "required_sections.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("required_chapters", [])
    except Exception:
        return []


def _keyword_intent(q: str) -> str:
    if not q:
        return INTENT_REVIEW                              # 传了方案没给指令 → 全面审查
    # 危大判定类优先（"算不算危大/超规模/要不要论证" 是单点判定，不是要四维框架）
    if any(k in q for k in _KW[INTENT_HAZARD_LEVEL]):
        return INTENT_HAZARD_LEVEL
    # 复合结构化问句（风险点/管控/验收等多维度 + 危大类别 → 要框架清单）
    try:
        from qa.decompose import is_structured_query
        if is_structured_query(q):
            return INTENT_STRUCTURED
    except Exception:
        pass
    for name in (INTENT_REVIEW, INTENT_BASIS,
                 INTENT_ELEMENTS, INTENT_TECHNICAL):
        if any(k in q for k in _KW[name]):
            return name
    return INTENT_TECHNICAL                               # 有方案、问句不明确 → 按技术核对


def scope_terms(q: str) -> list[str]:
    """抽章节关键词/危大类别词，作为技术路的范围收窄依据。"""
    from tools.hazard_level import CATEGORY_KEYWORDS
    out: list[str] = []
    for ch in _chapters():
        for kw in ch.get("keywords", []) or []:
            if kw and kw in q and kw not in out:
                out.append(kw)
    for cat, kws in CATEGORY_KEYWORDS.items():
        if cat in q and cat not in out:
            out.append(cat)
        for k in kws:
            if len(k) >= 2 and k in q and k not in out:
                out.append(k)
    return out


def _extract_json(resp: str) -> dict:
    m = re.search(r"\{.*\}", resp or "", re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def intent_route(query: str, has_plan: bool, llm=None) -> dict:
    """返回 {"intent","agents","checks","scope_terms","source"}。

    `checks` = review subagent 内部要走的三路（hazard_level / technical / basis / elements / content）。
    """
    q = (query or "").strip()
    if not has_plan:
        # 无方案：复合结构化问句仍走结构化链路（不依赖方案）；
        # 危大判定类无方案无法查表 → 落到 qa（走 RAG/结构化兜底）。
        intent = _keyword_intent(q)
        if intent == INTENT_STRUCTURED:
            return _pack(INTENT_STRUCTURED, scope_terms(q), "no_plan_structured")
        return _pack(INTENT_QA, scope_terms(q), "no_plan")

    source = "keyword"
    intent = None
    terms: list[str] = []
    if llm is not None:
        try:
            hint = f"是否已上传方案：是\n用户输入：{q or '（未给指令）'}"
            resp = llm.complete([{"role": "system", "content": _ROUTER_SYSTEM},
                                 {"role": "user", "content": hint}])
            d = _extract_json(resp)
            if d.get("intent") in INTENTS:
                intent = d["intent"]
                source = "llm"
                terms = [t for t in (d.get("scope_terms") or []) if isinstance(t, str)]
        except Exception:
            intent = None                                     # 回落关键词

    if intent is None:
        intent = _keyword_intent(q)
    if not terms:
        terms = scope_terms(q)
    return _pack(intent, terms, source)


def _pack(intent: str, terms: list[str], source: str) -> dict:
    agents, checks = _DISPATCH[intent]
    return {"intent": intent, "agents": list(agents), "checks": list(checks),
            "scope_terms": list(terms), "source": source}


# ---------------- 兼容旧接口 ----------------
def route(query: str, has_plan: bool, llm=None) -> list:
    """旧接口：只返回 subagent 列表。"""
    return intent_route(query, has_plan, llm)["agents"]


def _fallback(query: str, has_plan: bool) -> list:
    """旧接口：关键词降级路由。"""
    return intent_route(query, has_plan, llm=None)["agents"]
