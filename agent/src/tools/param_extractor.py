"""工况三元组抽取 —— 把 ner2 的**裸量名**实体与句内数值绑定成结构化三元组。

## 为什么需要它
`ner2/docs/PARAM_SPEC.md` 明文规定「量名与紧邻值不合并」（`桩长60m` 只标 `桩长`），
所以 NER 出口**天然没有值**，下游拿不到"10m"，无法做任何数值判定。
本模块不改 NER 口径（标注与已训模型都不动），只在 NER 之上派生：

    {category, subject, metric, value, unit, sent_idx, sent_text}

## 绑定的三条约束（2026-09-12 实测标定，11 份方案 2514 句）
1. **语序**：值默认在量名之后（中文占绝大多数，如"开挖深度 8m"）；
   值在量名之前时，要求紧贴连接词（为/不小于/不大于/≥/≤/= /：）。
2. **距离**：≤10 字。放宽到 15 字会引入跨短语误绑。
3. **量纲相容**（最有效）：量名关键词 → 允许单位集合。"起重量"只能配 kN/N/t/kg，
   "深度/高度"只能配长度单位。实测**量纲过滤拦下的 10 条全部是真误绑**
   （起重量=51m、混凝土强度=30cm、水泥掺量=1.0MPa、搭接长度=20% …）。

## 两条审查路（`triage()` 分派，**不是二选一**）
本模块输出的是**全量**工况三元组；过滤只发生在其中的一路：

| 路 | 内容 | 判定依据 | 规模（实测/篇） |
|---|---|---|---|
| `hazard_level` | 量名命中阈值表（开挖深度/搭设高度/跨度/总荷载/线荷载/起重量…） | **查表**（37号令附件1/2） | 十余条 |
| `technical` | 其余全部（允许偏差/垂直度/桩长/桩径/壁厚/水泥掺量/坍落度/强度…） | **检索条文 + comparator/LLM 比对** | 几十~百余条 |

→ 所以「本方案施工工艺有什么不符合规范要求」这类开放问句走 `technical`，
不存在"只能审阈值表内容"的问题；`judgeable()` 只服务于自动判档那一路。

## 已知边界：无量纲规格值不产生三元组
`C35` / `P8` / `∅800@600` 按 PARAM_SPEC 是合法的参数实体，但**没有可比较的量纲**，
绑不出 `{value, unit}`，因此进不了 comparator。这类只能走**实体级检索**
（拿实体本身当查询词，把条文捞回来看要求），不要强行给它们编造单位。

## 规模实测
参数实体 628 → 绑定 121 → 去重 92 → 其中阈值表维度 11 / 技术核对类 81。
"""
from __future__ import annotations

import re
from collections import defaultdict

from .hazard_level import CATEGORY_KEYWORDS, detect_category

# ---------------------------------------------------------------- 数值与单位
# 大小写不敏感（方案常写 26T / 15KN/㎡）；㎡ 归到 m2
UNIT = (r"(?:kN/m2|kN/m²|kN/㎡|kN/m|kN|MPa|kPa|Pa|m2|m²|㎡|m3|m³|mm|cm|km|m|mm2|N|t|kg|%|°|d|h|min|s)")
_NUM = r"(-?\d+(?:\.\d+)?)"
# 负向预查只挡拉丁字母（防止 "30min" 里的 m 被当成米），**不能挡中文**
# ——施工方案里单位后面几乎总是接中文（"10m时""8m后"），挡中文会把绑定率砍到 1/3。
NUM_RE = re.compile(_NUM + r"\s*(" + UNIT + r")(?![a-zA-Z])", re.IGNORECASE)

_LEN = {"m", "mm", "cm", "km"}
_FORCE = {"kN", "N", "t", "kg"}
_PRESS = {"kN/m2", "kN/m²", "kPa", "MPa", "Pa"}
_LINE = {"kN/m"}
_RATIO = {"%"}
_ANGLE = {"°"}
_TIME = {"d", "h", "min", "s"}
_AREA = {"m2", "m²", "mm2"}
_VOL = {"m3", "m³"}

# (量名关键词, 允许单位集合)。**顺序敏感**：先窄后宽（线荷载 先于 荷载）。
DIM_RULES: list[tuple[str, set]] = [
    (r"线荷载", _LINE),
    (r"总荷载|面荷载|压力|强度", _PRESS),
    (r"重量|吊重|起重量|轴力|承载力|吨位|力", _FORCE | _LINE | _PRESS),
    (r"深度|高度|长度|厚度|间距|距离|跨度|桩长|半径|直径|宽度|埋深|孔深|水位|超灌|标高|落差|落深|步距|纵距|横距", _LEN),
    (r"偏差|位移|沉降|垂直度|平整度|水平度|误差|挠度|变形", _LEN | _RATIO),
    (r"率|掺量|孔隙率|配筋率|坡度|含水量|坍落度|系数", _RATIO | _ANGLE),
    (r"角度|倾角", _ANGLE | _RATIO),
    (r"时间|天数|龄期|周期", _TIME),
    (r"面积", _AREA),
    (r"方量|体积|用量", _VOL),
]

# 值在量名之前时要求紧贴的连接词
LINKER = re.compile(r"(?:为|不小于|不大于|不应超过|不得超过|不宜超过|大于等于|小于等于|≥|≤|=|：|:)\s*$")

MAX_DIST = 10


def dim_ok(metric: str, unit: str) -> bool:
    """量纲相容性检查。无可依据规则时放行（宁可多绑，不误杀）。"""
    from .hazard_level import canon_unit
    u = canon_unit(unit)
    for kw, allowed in DIM_RULES:
        if re.search(kw, metric):
            return unit in allowed
    return True


def _pick_value(sent: str, lo: int, hi: int, max_dist: int = MAX_DIST,
                require_after: bool = True):
    """就近取值：优先量名之后；值在量名之前时要求紧跟连接词。"""
    best = None
    for m in NUM_RE.finditer(sent):
        vs, ve = m.span()
        if vs >= hi:
            dist = vs - hi
        elif require_after:
            continue
        else:
            if not LINKER.search(sent[max(0, lo - 2):lo] or ""):
                continue
            dist = lo - ve
        if dist > max_dist:
            continue
        if best is None or dist < best[0]:
            best = (dist, m.group(1), m.group(2), vs)
    return best


def _subject_of(ents_in_sent, lo: int) -> str:
    """主体 = 量名之前最近的 工程类型/设备/工序（按此优先级）。

    注意：实测主体绑定质量一般（常取到动词误标的"工序"），因此主体仅作展示/佐证，
    **判定一律以 category（整句关键词匹配，更鲁棒）为准**。
    """
    for t in ("工程类型", "设备", "工序"):
        cand = [e for e in ents_in_sent if e.get("type") == t and e["end"] <= lo]
        if cand:
            return max(cand, key=lambda e: e["end"])["text"]
    return ""


def extract_triples(ents: list[dict], window: int = 1) -> list[dict]:
    """输入 ner2 `FullTextExtractor.extract_text` 的实体列表，输出三元组。

    ents 需含 {type, text, start, end, sent_idx, sent_text}。
    category 在本句内判定；本句无类别关键词时向前后各看 window 句。
    """
    by_sent: dict[int, list[dict]] = defaultdict(list)
    sent_text: dict[int, str] = {}
    for e in ents:
        sid = e.get("sent_idx")
        if sid is None:
            continue
        by_sent[sid].append(e)
        if sid not in sent_text:
            sent_text[sid] = e.get("sent_text", "")
    if not sent_text:
        return []

    max_sid = max(sent_text)
    out = []
    for sid, es in sorted(by_sent.items()):
        es = sorted(es, key=lambda x: x["start"])
        # 类别上下文：本句 → 前后 window 句
        ctx = sent_text.get(sid, "")
        if not detect_category(ctx):
            for d in range(1, window + 1):
                for nb in (sid - d, sid + d):
                    if 0 <= nb <= max_sid and detect_category(sent_text.get(nb, "")):
                        ctx = ctx + " " + sent_text[nb]
                        break
        category = detect_category(ctx)
        for p in [e for e in es if e.get("type") == "参数"]:
            lo, hi = p["start"], p["end"]
            got = _pick_value(sent_text.get(sid, ""), lo, hi)
            if got is None:
                continue
            _d, val, unit, _vs = got
            if not dim_ok(p["text"], unit):
                continue
            try:
                value = float(val)
            except ValueError:
                continue
            # 负值纠偏：深度/高度类不会出现负值，"开挖深度-22.25m" 的负号来自**标高口径**
            # （实测把 -22.25m 判成"非危大"，是硬伤）；"标高"本身允许为负，不处理。
            if value < 0 and re.search(r"深度|高度|长度|厚度|间距|距离|跨度|桩长|埋深|孔深", p["text"]):
                value = abs(value)
            out.append({"category": category, "subject": _subject_of(es, lo),
                        "metric": p["text"], "value": value, "unit": unit,
                        "sent_idx": sid, "sent_text": sent_text.get(sid, ""),
                        "context": ctx})
    return out


def dedup(triples: list[dict]) -> list[dict]:
    """按 (category, metric, value, unit) 去重。"""
    seen, out = set(), []
    for t in triples:
        k = (t["category"], t["metric"], t["value"], t["unit"])
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def keep_worst(triples: list[dict]) -> list[dict]:
    """同一 (category, metric) 只保留**最不利**值（绝对值最大）。

    方案里同一量名常出现多处（多个基坑/多段架体），判定时应取最不利工况，
    而不是每个值都报一条风险。
    """
    best: dict[tuple, dict] = {}
    for t in triples:
        k = (t["category"], t["metric"])
        cur = best.get(k)
        if cur is None or abs(t["value"]) > abs(cur["value"]):
            best[k] = t
    return list(best.values())


def judgeable(triples: list[dict]) -> list[dict]:
    """**只服务于「自动判档」那一路**：保留量名命中阈值表维度的三元组。

    ⚠️ 这不是全系统的过滤器。`extract_triples` 的输出是**全量**工况
    （实测含 允许偏差/垂直度/桩长/桩径/壁厚/水泥掺量/电压… 等），
    坍落度、混凝土强度、垂直度这类"技术核对类"参数**照样要审**，
    只是走 `<-- technical -->` 那条路（检索 + comparator/LLM 比对），
    判定依据来自条文而不是阈值表。用 `triage()` 一次拿到两条路的分派结果。
    """
    from .hazard_level import THRESHOLDS
    keep = []
    for t in triples:
        if not t["category"]:
            continue
        for cat, _sub, pat, _u, _h, _s, _pre, _b in THRESHOLDS:
            if cat == t["category"] and re.search(pat, t["metric"]):
                keep.append(t)
                break
    return keep


def _key(t: dict) -> tuple:
    return (t["category"], t["metric"], t["value"], t["unit"])


def triage(triples: list[dict]) -> dict[str, list[dict]]:
    """**分诊（不是筛选）**：把全量三元组分派到两条审查路。

    - `"hazard_level"`：量名命中阈值表 → 走**查表判档**（是否危大/超规模/需论证），
      不检索。覆盖阈值表 22 组维度，规模十余条/篇。
    - `"technical"`：其余全部 → 走**工况检索 + comparator/LLM 比对**（工艺、做法、
      允许偏差、材料指标…），判定依据来自检索到的条文。规模几十~百余条/篇。

    用户问「本方案施工工艺有什么不符合规范要求」时用的是 `"technical"`（可再按
    方案章节/工序二次收窄检索范围）；自动巡检则两条都跑，`"hazard_level"` 出档位结论、
    `"technical"` 出条文比对结论。
    """
    jd = judgeable(triples)
    jd_keys = {_key(t) for t in jd}
    return {"hazard_level": jd,
            "technical": [t for t in triples if _key(t) not in jd_keys]}
