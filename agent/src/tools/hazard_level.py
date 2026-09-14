"""危大分档判定器 —— 判定类问题**走查表，不走 RAG**。

## 为什么必须有这个文件
37 号令附件1/附件2 的阈值条款**不在 rag2 语料里**（13661 条款 / 81 规范，`grpo/scripts/gen_rl_data.py:256`
已注明）。实测：即使把判定意图拼进 query（`基坑开挖深度8m 是否属于超过一定规模的危大工程`），
检索 top3 也召不回任何阈值条款，只会召回"深基坑定义""3m 勘察要求"等无关条文。
→ **判定类问题的正确答案来自结构化阈值表，不是语义检索。查表零幻觉、可复现、可溯源。**

## 阈值口径来源
`agent/data/rules/hazardous_work_types.json`（住建部令第37号附件1/附件2 + 建办质〔2018〕31号），
并与 `grpo/scripts/gen_rl_data.py:198-250` 的 DANGER_DIMS 同口径（含 h=0.1 哨兵语义）。
**改动本表时必须同步这两处。**

## 哨兵约定（与 grpo 一致）
- `h = 0.1`：该类别**本身即属危大工程**，无数值危大线（附件1 定性条目，如幕墙安装、人工挖孔桩）。
- `s = None`：无统一超规模线（如拆除、暗挖）。

## 单位
统一折算到基准单位比较：长度→m，力→kN（t×10，kg×0.01，N/1000），
面荷载→kPa（kN/m²=1，MPa×1000），线荷载→kN/m，比例→%，时间→d。
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- 单位归一
_TO_M = {"m": 1.0, "mm": 0.001, "cm": 0.01, "km": 1000.0}
_TO_KN = {"kN": 1.0, "N": 0.001, "t": 10.0, "kg": 0.01}
_TO_KPA = {"kN/m2": 1.0, "kN/m²": 1.0, "kPa": 1.0, "MPa": 1000.0, "Pa": 0.001}
_TO_KNM = {"kN/m": 1.0}
_TO_PCT = {"%": 1.0}
_TO_DAY = {"d": 1.0, "h": 1 / 24, "min": 1 / 1440, "s": 1 / 86400}

BASE_UNIT = {**{u: ("m", v) for u, v in _TO_M.items()},
             **{u: ("kN", v) for u, v in _TO_KN.items()},
             **{u: ("kPa", v) for u, v in _TO_KPA.items()},
             **{u: ("kN/m", v) for u, v in _TO_KNM.items()},
             **{u: ("%", v) for u, v in _TO_PCT.items()},
             **{u: ("d", v) for u, v in _TO_DAY.items()}}


def canon_unit(u: str) -> str:
    """单位字面归一：大小写与全角面积符号（方案常写 26T、15KN/㎡）。"""
    return (u or "").lower().replace("㎡", "m2").replace("㎥", "m3").replace("％", "%")


def normalize(value: float, unit: str) -> tuple[float, str] | None:
    """折算到基准单位。未知单位返回 None（由调用方判为「无法判定」）。"""
    got = BASE_UNIT.get(canon_unit(unit))
    if not got:
        return None
    base, factor = got
    return value * factor, base


# ---------------------------------------------------------------- 类别关键词
# 与 rules_checker.CATEGORY_KEYWORDS 同义，但按判定需要排序/补充：
# 「排架」归模板支撑（排架=模板支撑架/满堂排架，非脚手架），这是方案用词与规范用词的差异点。
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "模板支撑": ["模板", "支撑体系", "满堂支架", "满堂支撑", "高支模", "排架", "支模"],
    "脚手架": ["脚手架", "扣件式", "盘扣", "碗扣", "悬挑架", "落地架", "外架"],
    "基坑工程": ["基坑", "开挖", "围护", "支护", "降水", "地下连续墙", "地墙", "灌注桩", "SMW"],
    "起重吊装": ["吊装", "起重机", "塔吊", "履带吊", "汽车吊", "起重", "起吊"],
    "幕墙安装": ["幕墙"],
    "人工挖孔桩": ["人工挖孔", "挖孔桩"],
    "钢结构安装": ["钢结构", "钢构", "网架"],
    "拆除": ["拆除", "爆破"],
    "暗挖": ["暗挖", "盾构", "顶管", "矿山法"],
}

# 脚手架细分（同一量名「搭设高度」在三类脚手架上阈值不同，必须消歧）
SCAFFOLD_SUBTYPE = {
    "附着式": ["附着", "升降", "爬架"],
    "悬挑式": ["悬挑"],
    "落地式": ["落地", "扣件式", "盘扣", "碗扣"],
}


def detect_category(text: str) -> str | None:
    """在整句（或上下文窗口）里判定危大类别。命中多个时取关键词最长者（更具体）。"""
    best, best_len = None, 0
    for cat, kws in CATEGORY_KEYWORDS.items():
        for k in kws:
            if k in text and len(k) > best_len:
                best, best_len = cat, len(k)
    return best


def detect_scaffold_subtype(text: str) -> str:
    for sub, kws in SCAFFOLD_SUBTYPE.items():
        if any(k in text for k in kws):
            return sub
    return "落地式"          # 无修饰词时按落地式，并在 note 里标注假设


# ---------------------------------------------------------------- 阈值表
# (类别, 子类型, 量名正则, 基准单位, 危大线 h, 超规模线 s, 前置条件关键词, 依据)
THRESHOLDS: list[tuple] = [
    ("基坑工程", None, r"开挖深度|基坑深度|挖深|落深", "m", 3.0, 5.0, None, "附件1三(一)/附件2三(一)"),
    ("模板支撑", None, r"搭设高度|支模高度|排架高度", "m", 5.0, 8.0, None, "附件1二(二)/附件2二(二)"),
    ("模板支撑", None, r"跨度", "m", 10.0, 18.0, None, "附件1二(二)/附件2二(二)"),
    ("模板支撑", None, r"施工总荷载|总荷载|面荷载", "kPa", 10.0, 15.0, None, "附件1二(二)/附件2二(二)"),
    ("模板支撑", None, r"集中线荷载|线荷载", "kN/m", 15.0, 20.0, None, "附件1二(二)/附件2二(二)"),
    ("脚手架", "落地式", r"搭设高度|架体高度", "m", 24.0, 50.0, None, "附件1四(一)/附件2四(一)"),
    ("脚手架", "悬挑式", r"搭设高度|分段架体搭设高度", "m", 0.1, 20.0, None, "附件1四(二)/附件2四(二)"),
    ("脚手架", "附着式", r"提升高度|搭设高度", "m", 0.1, 150.0, None, "附件1四(三)/附件2四(三)"),
    ("起重吊装", None, r"单件起吊重量|起重量|吊重", "kN", 10.0, 100.0, "非常规起重设备|非常规", "附件1六(一)/附件2六(一)"),
    ("起重吊装", None, r"起重量|额定起重量", "kN", None, 300.0, None, "附件2六(一)"),
    ("幕墙安装", None, r"施工高度|安装高度", "m", 0.1, 50.0, None, "附件1七(一)/附件2七(一)"),
    ("人工挖孔桩", None, r"开挖深度|孔深|桩长", "m", 0.1, 16.0, None, "附件1五/附件2五"),
    ("钢结构安装", None, r"跨度", "m", 0.1, 36.0, None, "附件1七(二)/附件2七(二)"),
]

LEVEL_NONE = "非危大"
LEVEL_HAZ = "危大工程"
LEVEL_SUPER = "超规模危大工程"
LEVEL_UNKNOWN = "无法判定"


def _match_rule(category: str, subtype: str | None, metric: str, unit_base: str):
    """挑命中的阈值规则：类别 → 子类型 → 量名 → 单位一致。"""
    cands = []
    for cat, sub, pat, u, h, s, pre, basis in THRESHOLDS:
        if cat != category or not re.search(pat, metric):
            continue
        if sub and subtype and sub != subtype:
            continue
        if u != unit_base:
            continue
        cands.append((cat, sub, pat, u, h, s, pre, basis))
    if not cands:
        return None
    # 有前置条件的规则要求句内出现该条件，否则退选无条件规则
    return cands[0]


def judge(category: str | None, metric: str, value: float, unit: str,
          context: str = "") -> dict:
    """判定单个工况参数的分档。

    context：所在句原文（用于脚手架子类型消歧 + 前置条件判定）。
    返回 dict：level / category / threshold / basis / need_review / note
    """
    out = {"level": LEVEL_UNKNOWN, "category": category, "metric": metric,
           "value": value, "unit": unit, "threshold": None,
           "basis": "", "need_review": False, "note": ""}
    if not category:
        out["note"] = "未识别出危大类别，无法判定"
        return out

    norm = normalize(value, unit)
    if norm is None:
        out["note"] = f"单位 {unit} 无法折算到基准单位"
        return out
    v, base = norm

    subtype = detect_scaffold_subtype(context) if category == "脚手架" else None
    rule = _match_rule(category, subtype, metric, base)
    if rule is None:
        out["note"] = f"「{category}」无「{metric}({base})」对应的阈值条目"
        return out
    _cat, sub, _pat, _u, h, s, pre, basis = rule

    # 前置条件（如"非常规起重设备、方法"）
    if pre and context and not re.search(pre, context):
        no_pre = [x for x in THRESHOLDS
                  if x[0] == category and re.search(x[2], metric) and x[6] is None and x[3] == base]
        if no_pre:
            rule = no_pre[0]
            _cat, sub, _pat, _u, h, s, pre, basis = rule
            out["note"] = "未满足前置条件（非常规起重设备/方法），按无条件条目判定"
        else:
            out["note"] = f"未满足前置条件（{pre}），且无备选条目"
            return out
    if sub:
        out["note"] = (out["note"] + "；" if out["note"] else "") + f"脚手架子类型={sub}"

    has_h = h is not None and h >= 1
    if s is not None and v >= s:
        out["level"] = LEVEL_SUPER
        out["threshold"] = f"{metric} ≥ {s}{base}"
        out["need_review"] = True
    elif has_h and v >= h:
        out["level"] = LEVEL_HAZ
        out["threshold"] = f"{metric} ≥ {h}{base}"
    elif not has_h and h is not None and h < 1:
        out["level"] = LEVEL_HAZ          # 哨兵：本身即危大
        out["threshold"] = "列入附件1范围（无数值判定线）"
    elif h is None and s is not None:
        out["level"] = LEVEL_NONE
        out["threshold"] = f"{metric} < {s}{base}"
    else:
        out["level"] = LEVEL_NONE
        out["threshold"] = f"{metric} < {h}{base}" if h else None

    out["basis"] = f"《危险性较大的分部分项工程安全管理规定》（住建部令第37号）{basis}"
    return out
