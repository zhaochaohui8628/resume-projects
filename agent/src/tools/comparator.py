"""条文比对器（comparator）—— 把「检索到的条文」和「方案里的数值」真正比一次。

## 为什么需要它
现状链路把 RAG 命中结果写进 `r["basis"]` **仅作展示**，代码里没有任何比较动作
（`agent/src/agent/compliance_pipeline.py:41-43`）。实测证明检索侧其实够用：
查询「开挖深度 允许值 应符合」的 top1 是 DG-TJ08-61 16.2.1
「放坡开挖的基坑开挖深度不宜超过 7.0m」——方案写的是 8m，
**这条本来足以直接判"疑似违反"，只是没人去比**。

## 设计：规则优先，LLM 兜底（兜底不在本文件）
数值型限值用正则抽取 + 单位归一后直接比较，零幻觉、可复现；
只有工艺/做法类（无数值）才交给 LLM，且输入严格限定为「三元组 + 该条原文」。

## 语料 OCR 噪声
实测 DG-TJ08-61 16.2.1 的限值被 OCR 成「7. Om」（0 被识别成字母 O），
直接抽取会漏掉这条最有用的条款 → 抽取前先做定向清洗 `clean_ocr_number`。
"""
from __future__ import annotations

import json
import re

UNIT = (r"(?:kN/m2|kN/m²|kN/㎡|kN/m|kN|MPa|kPa|Pa|m2|m²|㎡|m3|m³|mm|cm|km|m|N|t|kg|%|°|d|h|min|s)")

# 上限类：方案值 > 限值 → 违反
UPPER_OPS = ["不宜超过", "不应超过", "不得超过", "不得大于", "不应大于",
             "不宜大于", "不应高于", "不得超过", "不超过", "不大于", "小于等于"]
# 下限类：方案值 < 限值 → 违反
LOWER_OPS = ["不宜小于", "不应小于", "不得小于", "不应低于", "不小于", "大于等于"]

_OPS = sorted(set(UPPER_OPS + LOWER_OPS), key=len, reverse=True)
LIMIT_RE = re.compile(
    r"(" + "|".join(re.escape(o) for o in _OPS) + r")"
    r"\s*(\d+(?:\s*[.．]\s*\d+)?(?:\s*[Oo]\d)?)\s*(" + UNIT + r")"
)

VERDICT_OK = "符合"
VERDICT_BAD = "疑似违反"
VERDICT_NA = "无法判定（条文无可用限值）"


def clean_ocr_number(text: str) -> str:
    """定向修 OCR 噪声：「7. Om」→「7.0m」（语料中 0 被识别为字母 O）。"""
    text = re.sub(r"(\d)\s*[.．]\s*[Oo](?=\s*[a-zA-Z\u4e00-\u9fa5])", r"\1.0", text)
    text = re.sub(r"(\d)\s*[.．]\s+(?=[a-zA-Z\u4e00-\u9fa5])", r"\1.0 ", text)
    return text


def _to_float(s: str) -> float:
    s = s.replace(" ", "").replace("．", ".").replace("O", "0").replace("o", "0")
    return float(s)


def extract_limits(clause_text: str) -> list[dict]:
    """从条文正文抽数值限值。返回 [{op, value, unit, direction, snippet}]。"""
    txt = clean_ocr_number(clause_text or "")
    out = []
    for m in LIMIT_RE.finditer(txt):
        op = m.group(1)
        direction = "upper" if op in UPPER_OPS else ("lower" if op in LOWER_OPS else "")
        if not direction:
            continue
        try:
            val = _to_float(m.group(2))
        except ValueError:
            continue
        out.append({"op": op, "value": val, "unit": m.group(3),
                    "direction": direction,
                    "snippet": txt[max(0, m.start() - 12):m.end() + 12]})
    return out


def compare(triple: dict, clause: dict | str) -> dict:
    """三元组 vs 单条条文 → 判定。

    triple: {metric, value, unit, ...}   clause: 检索结果（含 text/metadata）或纯文本
    返回 {verdict, limit, direction, evidence, source, clause_no}
    """
    text = clause.get("text", "") if isinstance(clause, dict) else str(clause)
    meta = (clause.get("metadata", {}) if isinstance(clause, dict) else {}) or {}
    res = {"verdict": VERDICT_NA, "limit": None, "direction": None,
           "evidence": "", "source": meta.get("source", ""),
           "clause_no": meta.get("clause_no", "")}

    from .hazard_level import normalize
    tv = normalize(triple.get("value", 0.0), triple.get("unit", ""))
    if tv is None:
        return res
    v, v_base = tv

    for lim in extract_limits(text):
        lv = normalize(lim["value"], lim["unit"])
        if lv is None:
            continue
        l, l_base = lv
        if l_base != v_base:
            continue                       # 量纲不同，不可比
        bad = (v > l) if lim["direction"] == "upper" else (v < l)
        res.update({"verdict": VERDICT_BAD if bad else VERDICT_OK,
                    "limit": f"{lim['op']} {lim['value']}{lim['unit']}",
                    "direction": lim["direction"],
                    "evidence": lim["snippet"]})
        if bad:
            break                          # 命中一条违反即可，不必再看
    return res


def compare_all(triple: dict, clauses: list) -> dict:
    """对 top-k 条文逐条比较：出现任一「疑似违反」即报违反；全符合则报符合。"""
    results = [compare(triple, c) for c in clauses]
    bad = [r for r in results if r["verdict"] == VERDICT_BAD]
    ok = [r for r in results if r["verdict"] == VERDICT_OK]
    if bad:
        return {"verdict": VERDICT_BAD, "picked": bad[0], "n_checked": len(results)}
    if ok:
        return {"verdict": VERDICT_OK, "picked": ok[0], "n_checked": len(results)}
    return {"verdict": VERDICT_NA, "picked": None, "n_checked": len(results)}


# ---------------------------------------------------------------- LLM 兜底
# 规则 comparator 只能处理**数值型限值**（"不宜超过7.0m" vs 8m）。
# 工艺/做法类要求（"支撑架应每4~6步距拉结"、"拆除应由上而下"）没有可比数值，
# 正则抽不出限值 → 这一类的兜底交 LLM。**输入严格限定为本工况句 + 条文原文**，
# 不允许模型引入条文之外的要求（防幻觉），并要求给引文片段。
_LLM_SYSTEM = """你是施工安全合规审查员。判断「方案中的做法/参数」是否与「给定的规范条文」相符。

判定规则：
1) **只依据给出的条文判断**，不得引入条文之外的要求或你自己的经验；
2) 条文未涉及该事项 → 判 "无法判定"；不得因为"没写"就判不符合；
3) evidence 必须是条文里的**原句片段**（照抄，不得改写、不得拼接无关内容）；
4) 只输出 JSON，不要解释：
{"verdict": "符合|不符合|无法判定", "reason": "一句话理由", "evidence": "条文原句片段"}"""


def llm_compare(triple: dict, clauses: list, llm, max_clauses: int = 3) -> dict:
    """规则 comparator 判不了时的 **LLM 兜底比对**（做法/工艺类）。

    输入被严格限定：工况句 + 量名/值/单位 + top-k 条文原文。
    返回 {"verdict","reason","evidence","source","clause_no"}；
    llm 为空或调用失败 → verdict=无法判定（绝不臆造结论）。
    """
    res = {"verdict": VERDICT_NA, "reason": "", "evidence": "",
           "source": "", "clause_no": ""}
    if llm is None or not clauses:
        return res
    picks = [c for c in clauses[:max_clauses] if isinstance(c, dict)]
    if not picks:
        return res
    docs = []
    for i, c in enumerate(picks, 1):
        md = c.get("metadata", {}) or {}
        head = f"[{i}] {md.get('source', '?')} {md.get('clause_no', '')}".strip()
        docs.append(f"{head}\n{(c.get('text', '') or '').strip()[:600]}")
    user = (f"方案中的工况：\n{(triple.get('sent_text') or '').strip()[:400]}\n\n"
            f"抽取到的指标：量名={triple.get('metric')} 值={triple.get('value')}"
            f"{triple.get('unit', '')}\n\n"
            f"相关规范条文（共 {len(picks)} 条）：\n" + "\n\n".join(docs))
    try:
        resp = llm.complete([{"role": "system", "content": _LLM_SYSTEM},
                             {"role": "user", "content": user}])
    except Exception:
        return res
    m = re.search(r"\{.*\}", resp or "", re.DOTALL)
    if not m:
        return res
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return res
    verdict = str(d.get("verdict", "")).strip()
    res["verdict"] = VERDICT_BAD if verdict in ("不符合", "疑似违反") else (
        VERDICT_OK if verdict == "符合" else VERDICT_NA)
    res["reason"] = str(d.get("reason", ""))
    res["evidence"] = str(d.get("evidence", ""))
    res["source"] = (picks[0].get("metadata", {}) or {}).get("source", "")
    res["clause_no"] = (picks[0].get("metadata", {}) or {}).get("clause_no", "")
    return res
