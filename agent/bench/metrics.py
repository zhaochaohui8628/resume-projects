"""评测指标（零依赖）：漏报率 / 误报率 / NER F1 / RAG recall@k / 耗时拆解。

口径说明：
- 审查（risks）：golden 用「标题关键词片段」子串匹配（宽松、可复现）；
  漏报率 FNR = 未命中的 golden / golden 总数；
  误报率 FPR = 预测中不在 golden 的条数 / 预测总数。
- NER（entities）：golden 用实体关键词文本匹配；
  Precision/Recall/F1 按 golden↔预测双向包含判定。
- RAG：golden_rag {query, sources}，hit = golden source 前缀出现在 top-k 命中 source 中；
  recall@k = 命中 golden 数 / golden sources 数（k 取实际命中数）。
- 耗时：总耗时 + 各 subagent 单独耗时（runner 计时）。
"""
from __future__ import annotations


# ---------------- 通用 ----------------
def _contains(hay: str, needle: str) -> bool:
    """宽松子串匹配（忽略大小写与空白）。"""
    if not needle:
        return True
    h = "".join((hay or "").upper().split())
    n = "".join(needle.upper().split())
    return n in h


def _f1(p: int, r: int, total_golden: int, total_pred: int) -> dict:
    """由命中数推 precision/recall/F1。"""
    precision = p / total_pred if total_pred else 0.0
    recall = r / total_golden if total_golden else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "hit": p, "golden": total_golden, "pred": total_pred}


# ---------------- 审查指标（漏报/误报） ----------------
def risk_metrics(golden_risks: list[dict], predicted_risks: list[dict]) -> dict:
    """golden: [{check?, severity?, frag}]；predicted: rules 输出的 risk dict。"""
    if not golden_risks:
        return {"fnr": None, "fpr": None, "hit": 0, "golden": 0, "pred": len(predicted_risks)}
    hit = 0
    for g in golden_risks:
        frag = g.get("frag", "")
        for p in predicted_risks:
            hay = " ".join([str(p.get("check", "")), str(p.get("severity", "")),
                            str(p.get("title", "")), str(p.get("detail", ""))])
            if _contains(hay, frag):
                hit += 1
                break
    fn = len(golden_risks) - hit
    fp = max(0, len(predicted_risks) - hit)   # 预测条数 - 命中 golden 数（宽松估计）
    fnr = fn / len(golden_risks) if golden_risks else 0.0
    fpr = fp / len(predicted_risks) if predicted_risks else 0.0
    return {"fnr": round(fnr, 4), "fpr": round(fpr, 4),
            "missed": fn, "false_positive": fp,
            "hit": hit, "golden": len(golden_risks), "pred": len(predicted_risks)}


# ---------------- 多维拆解（v2） ----------------
CHECKS = ("C1", "C2", "C3", "C4")
CHECK_NAME = {"C1": "废止规范引用", "C2": "危大工程缺项",
              "C3": "强条疑似违反", "C4": "编制要素缺失"}
SEVERITIES = ("HIGH", "MEDIUM", "LOW")


def _group_metrics(golden: list[dict], pred: list[dict], key: str) -> dict:
    """按 key（check / severity）分组算 FNR，并给出该组的 golden/pred 计数。"""
    out = {"fnr": None, "hit": 0, "golden": len(golden), "pred": len(pred)}
    if not golden:
        return out
    hit = 0
    for g in golden:
        frag = g.get("frag", "")
        for p in pred:
            hay = " ".join([str(p.get("check", "")), str(p.get("severity", "")),
                            str(p.get("title", "")), str(p.get("detail", ""))])
            if _contains(hay, frag):
                hit += 1
                break
    out["hit"] = hit
    out["fnr"] = round((len(golden) - hit) / len(golden), 4)
    return out


def risk_metrics_by_check(golden_risks: list[dict], predicted_risks: list[dict]) -> dict:
    """按四类合规检查（C1 废止规范 / C2 危大缺项 / C3 强条违反 / C4 编制要素）拆解漏报率。

    诊断价值：总体 FNR 高时，可定位是哪一类检查失效（如 C2 判档无输入 → 三元组为空）。
    """
    res = {}
    for c in CHECKS:
        g = [x for x in golden_risks if x.get("check") == c]
        p = [x for x in predicted_risks if x.get("check") == c]
        res[c] = {"name": CHECK_NAME[c], **_group_metrics(g, p, "check")}
    return res


def risk_metrics_by_severity(golden_risks: list[dict], predicted_risks: list[dict]) -> dict:
    """按严重度（HIGH/MEDIUM/LOW）拆解漏报率——高危漏报比低危漏报代价更高。"""
    res = {}
    for s in SEVERITIES:
        g = [x for x in golden_risks if (x.get("severity") or "").upper() == s]
        p = [x for x in predicted_risks if (x.get("severity") or "").upper() == s]
        res[s] = _group_metrics(g, p, "severity")
    return res


def traceability_metrics(predicted_risks: list[dict]) -> dict:
    """依据可溯源率：产出的风险项中，带规范依据（basis/source/clause_no）的比例。

    审查结论必须"可溯源到条文"，否则无法人工复核——这是合规系统的硬要求。
    """
    if not predicted_risks:
        return {"rate": None, "with_basis": 0, "total": 0}
    with_b = 0
    for p in predicted_risks:
        basis = p.get("basis") or p.get("source") or p.get("clause_no") or ""
        if isinstance(basis, (list, tuple)):
            basis = " ".join(str(x) for x in basis)
        if str(basis).strip():
            with_b += 1
    return {"rate": round(with_b / len(predicted_risks), 4),
            "with_basis": with_b, "total": len(predicted_risks)}


def pipeline_metrics(n_entities: int, n_params: int, n_triples: int,
                     n_risks: int) -> dict:
    """链路贯通标记（无需 golden）：本用例是否走通 NER → 参数 → 三元组 → 风险。"""
    return {"has_entity": n_entities > 0, "has_param": n_params > 0,
            "has_triple": n_triples > 0, "has_risk": n_risks > 0,
            "n_entities": n_entities, "n_params": n_params,
            "n_triples": n_triples, "n_risks": n_risks}


# ---------------- NER F1 ----------------
def ner_metrics(golden_entities: list[dict], predicted_entities: list[dict]) -> dict:
    """golden: [{type?, text}]；predicted: NER 输出 [{type, text, ...}]。"""
    if not golden_entities:
        return {"precision": None, "recall": None, "f1": None,
                "golden": 0, "pred": len(predicted_entities)}
    hit = 0
    for g in golden_entities:
        gtext = g.get("text", "")
        for p in predicted_entities:
            if _contains(str(p.get("text", "")), gtext) or _contains(gtext, str(p.get("text", ""))):
                hit += 1
                break
    return _f1(hit, hit, len(golden_entities), len(predicted_entities))


# ---------------- RAG recall@k ----------------
def rag_recall(golden_rag: dict | None, rag_hits: list[dict]) -> dict:
    """golden_rag: {query, sources:[前缀]}；rag_hits: qa 检索 top-k（[source,...]）。

    k = len(rag_hits)（由调用方传 top-5 之类固定窗口）；golden sources 命中即算 hit。
    """
    if not golden_rag or not golden_rag.get("sources"):
        return {"recall@k": None, "hit": 0, "golden": 0, "k": len(rag_hits)}
    gold = golden_rag["sources"]
    hit = 0
    for g in gold:
        for h in rag_hits:
            src = str(h.get("source", ""))
            if _contains(src, g) or _contains(g, src):
                hit += 1
                break
    k = len(rag_hits)
    return {"recall@k": round(hit / len(gold), 4) if gold else None,
            "hit": hit, "golden": len(gold), "k": k,
            "top_sources": [h.get("source", "?") for h in rag_hits[:3]]}


# ---------------- 耗时拆解 ----------------
def time_metrics(total_ms: float, subagent_ms: dict[str, float]) -> dict:
    return {"total_ms": round(total_ms, 1),
            "subagent_ms": {k: round(v, 1) for k, v in subagent_ms.items()},
            "slowest": (max(subagent_ms, key=subagent_ms.get) if subagent_ms else None)}
