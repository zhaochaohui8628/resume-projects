"""复合结构化 QA —— 拆解器（维度识别 + 危大类别识别）。

把用户复合问句（如"130t 汽车吊钢栈桥吊装危大工程，有哪些风险点？对应的
方案编制内容、风险管控清单、验收节点分别是什么？"）拆成结构化槽位：

    {
      "category": "起重吊装",          # 命中的危大类别（hazardous_work_types.json key）
      "subject":   "钢栈桥",           # 吊装对象
      "equipment": "汽车吊",           # 设备
      "capacity":  "130",              # 吨位数值（仅提取）
      "unit":      "t",
      "dimensions": [                  # 命中的维度槽位
        {"key": "risks",       "label": "风险点",       "enabled": True},
        {"key": "plan",        "label": "方案编制内容",  "enabled": True},
        {"key": "controls",    "label": "风险管控清单",  "enabled": True},
        {"key": "acceptance",  "label": "验收节点",      "enabled": True},
      ],
      "is_super_scale": True           # 起重吊装 130t 是否超规模（>100kN 起重量）
    }

设计要点：
- 纯规则 + 关键词（零依赖，可测），LLM 版留作后续可选增强。
- 维度识别：匹配问句里的维度关键词（风险/编制/管控/验收）。
- 危大类别识别：优先取"起重吊装/塔吊/汽车吊/履带吊/吊装"类关键词，
  再回落到 hazardous_work_types.json 的类别名。
- 超规模判定：仅对起重吊装做吨位→kN 折算（1t≈9.8kN，超规模线 100kN）；
  其余类别超规模判定留给 hazard_level（不在此处重复）。
"""
from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "rules")  # agent/src/qa -> agent/data/rules

# 维度槽位（顺序即输出顺序）
DIMENSIONS = [
    {"key": "risks",      "label": "风险点",       "kw": ("风险", "危险源", "隐患")},
    {"key": "plan",       "label": "方案编制内容",  "kw": ("方案编制", "编制内容", "编制要求", "编制")},
    {"key": "controls",   "label": "风险管控清单",  "kw": ("管控", "控制措施", "预防措施", "清单")},
    {"key": "acceptance", "label": "验收节点",      "kw": ("验收", "检查节点", "验评")},
]

# 危大类别关键词 → hazardous_work_types.json 的 key
CATEGORY_KW = {
    "基坑工程":   ("基坑", "开挖", "支护", "围护"),
    "模板支撑":   ("模板", "支撑体系", "高支模", "满堂架"),
    "起重吊装":   ("吊装", "起重", "塔吊", "汽车吊", "履带吊", "龙门吊", "起重机"),
    "脚手架":     ("脚手架", "悬挑架", "爬架", "附着式升降"),
    "拆除":       ("拆除", "爆破"),
    "暗挖":       ("暗挖", "盾构", "顶管", "隧道"),
    "幕墙安装":   ("幕墙",),
    "人工挖孔桩": ("人工挖孔", "挖孔桩"),
    "钢结构安装": ("钢结构安装", "钢构安装", "网架"),
}

# 设备/对象抽取
_EQUIP_KW = ("汽车吊", "履带吊", "塔吊", "塔式起重机", "龙门吊", "轮胎吊", "起重机")
_OBJECT_KW = ("钢栈桥", "栈桥", "钢梁", "钢构件", "预制梁", "预制构件", "网架", "设备")

_CAP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(t|吨|kN|千牛)", re.I)


def _load_categories() -> dict:
    """读 hazardous_work_types.json 的 categories key 列表。"""
    p = os.path.join(DATA, "hazardous_work_types.json")
    try:
        with open(p, encoding="utf-8") as f:
            return {c["key"]: c for c in json.load(f).get("categories", [])}
    except Exception:
        return {}


def _detect_category(q: str) -> str | None:
    """关键词识别危大类别，返回 key 或 None。"""
    best, best_len = None, -1
    for cat, kws in CATEGORY_KW.items():
        for kw in kws:
            if kw in q and len(kw) > best_len:
                best, best_len = cat, len(kw)
    return best


def _detect_dimensions(q: str) -> list[dict]:
    out = []
    for d in DIMENSIONS:
        hit = any(k in q for k in d["kw"])
        out.append({"key": d["key"], "label": d["label"], "enabled": hit})
    return out


def _detect_equipment(q: str) -> str | None:
    for k in _EQUIP_KW:
        if k in q:
            return k
    return None


def _detect_object(q: str) -> str | None:
    for k in _OBJECT_KW:
        if k in q:
            return k
    return None


def _detect_capacity(q: str) -> tuple[float | None, str | None]:
    m = _CAP_RE.search(q)
    if not m:
        return None, None
    return float(m.group(1)), m.group(2)


def _is_super_scale(category: str, capacity: float | None, unit: str | None) -> bool:
    """起重吊装超规模判定：起重量 ≥ 300kN 或单件起吊 ≥ 100kN。
    1t ≈ 9.8kN。仅起重吊装做吨位折算，其余类别 False（交给 hazard_level）。"""
    if category != "起重吊装" or capacity is None:
        return False
    kn = capacity * 9.8 if unit in ("t", "吨") else capacity
    return kn >= 100.0


def decompose(query: str, use_llm: bool = True) -> dict:
    """主入口：把复合问句拆成结构化槽位。

    use_llm=True 时优先本地 Qwen 拆解（显存不足/失败自动回落纯规则），
    返回槽位带 `llm` 标记（True=LLM 拆解，False/缺失=规则拆解）。
    """
    if use_llm:
        try:
            from .qwen_decomposer import llm_decompose
            llm_slot = llm_decompose(query)
            if llm_slot is not None:
                return llm_slot
        except Exception:
            pass
    q = (query or "").strip()
    category = _detect_category(q)
    dimensions = _detect_dimensions(q)
    cap, unit = _detect_capacity(q)
    # 兜底：识别到危大类别但问句没写清维度 → 默认全开（问句可能只写"危大工程有哪些内容"）。
    # 未识别类别且无维度命中 → 维持全关（如"JGJ130 6.2.4 说什么"是单条文查询，不是复合问句）。
    if category is not None and not any(d["enabled"] for d in dimensions):
        for d in dimensions:
            d["enabled"] = True
    return {
        "category": category,
        "category_name": (_load_categories().get(category) or {}).get("name", "") if category else "",
        "subject": _detect_object(q),
        "equipment": _detect_equipment(q),
        "capacity": cap,
        "unit": unit,
        "dimensions": dimensions,
        "is_super_scale": _is_super_scale(category, cap, unit),
        "query": q,
        "llm": False,
    }


# ---------- 判断是否是复合结构化问句（供 intent 路由） ----------
def is_structured_query(query: str) -> bool:
    """复合问句 = 命中 ≥2 个维度 或 (命中危大类别 + ≥1 个维度)。

    强制规则版（use_llm=False）：路由判据要快且确定，不触发模型加载/显存判断。
    """
    d = decompose(query, use_llm=False)
    n = sum(1 for x in d["dimensions"] if x["enabled"])
    return (n >= 2) or (d["category"] is not None and n >= 1)
