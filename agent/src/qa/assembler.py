"""复合结构化 QA —— 四维取数 + 模板组装。

输入：decompose() 的槽位 dict
输出：四段式结构化答案（markdown），每段带来源标注。

四维取数：
  risks      风险点          ← risk_sources 知识库（工程管理知识）
  plan       方案编制内容     ← 编制要素表（建办质〔2018〕31号 九章 + 48号 细化要素）
  controls   风险管控清单     ← 风险点→管控要点（知识库）+ RAG 条文
  acceptance 验收节点        ← 规则模板（分阶段验收框架）+ RAG 条文

设计原则：
- 不依赖 RAG 也能出第一版（规则 + 知识库）。
- 每段标注来源（知识库 / 规范依据）。
- 输出为 markdown，可直接喂给 LLM 汇总或直接展示。
"""
from __future__ import annotations

import json
import os

from .decompose import decompose            # noqa: E402
from .risk_sources import risk_table        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "rules")  # agent/src/qa -> agent/data/rules


def _load_required_sections() -> dict:
    p = os.path.join(DATA, "required_sections.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _chapter_names() -> list[str]:
    return [c["name"] for c in _load_required_sections().get("required_chapters", [])]


def _per_type_sections(category: str | None) -> list[str]:
    if not category:
        return []
    pts = _load_required_sections().get("per_type_sections", {})
    # 兼容 key 与中文名
    for k, v in pts.items():
        if k == category or (category and k in category):
            return [s["name"] for s in v]
    return []


def _plan_content(slot: dict) -> list[str]:
    """方案编制内容：九章 + 该类别细化要素。"""
    ch = _chapter_names()
    extra = _per_type_sections(slot.get("category"))
    if slot.get("is_super_scale") and "专家论证" not in ch:
        ch = ch + ["（超规模）专家论证"]
    return ch + extra


def _controls(slot: dict, ctx: dict | None = None) -> tuple[list[str], str]:
    """风险管控清单：知识库管控要点 + RAG 条文（RAG 可用时）。"""
    rows = risk_table(slot.get("category"))
    items = [f"{r['source']} → {r['control']}" for r in rows]
    source = "风险源知识库（管控要点）"
    if not slot.get("no_retrieve"):
        try:
            from .retrieve import retrieve_clauses
            dr = retrieve_clauses(slot.get("category"),
                                  f"{slot.get('category') or ''} 吊装安全技术措施 钢丝绳 指挥 大风",
                                  top_k=3)
            if dr["items"]:
                items = items + dr["items"]
                source += " + RAG 条文"
        except Exception:
            pass
    return items, source


def _acceptance(slot: dict, ctx: dict | None = None) -> tuple[list[str], str]:
    """验收节点：规则模板 + RAG 条文（RAG 可用时）。"""
    cat = slot.get("category")
    base = [
        "施工准备阶段验收：方案交底、人员持证、设备/吊具检查",
        "过程验收：关键工序完成后验收（如吊装前试吊、支撑体系搭设后检查）",
        "隐蔽/分项验收：按检验批验收（吊点加固、地基处理、连接节点）",
        "整体验收：全部完成后，组织监理/业主验收并留存记录",
    ]
    if cat == "起重吊装":
        items = ["吊装前检查验收：地基处理、吊车支腿、吊具钢丝绳、试吊",
                 "构件安装过程验收：吊点、就位、临时固定、校正",
                 "整体验收：钢栈桥/构件验收，焊缝/连接节点检查，荷载试验（如需）"] + base[1:]
    elif cat == "基坑工程":
        items = ["支护结构分阶段验收：围护、支撑、降排水（随开挖分层验收）",
                 "监测点布设验收：监测点完好、初始值采集",
                 "开挖完成验收：坑底标高、支护完整性"] + base[1:]
    else:
        items = list(base)
    source = "分阶段验收规则模板"
    if not slot.get("no_retrieve"):
        try:
            from .retrieve import retrieve_clauses
            dr = retrieve_clauses(cat, f"{cat or ''} 验收 检查 检验批 记录", top_k=3)
            if dr["items"]:
                items = items + dr["items"]
                source += " + RAG 条文"
        except Exception:
            pass
    return items, source


def assemble(slot: dict, ctx: dict | None = None) -> dict:
    """四维组装 → {title, sections: [{key,label,items,source}], markdown}。

    ctx: 可选 {rag: RagClient | None}，控制 controls/acceptance 是否接 RAG。
         slot['no_retrieve'] = True 时跳过 RAG（纯规则，测试/离线用）。
    """
    sections = []
    for d in slot["dimensions"]:
        if not d["enabled"]:
            continue
        key = d["key"]
        if key == "risks":
            items = [f"{r['source']}：{r['hazard']}（管控：{r['control']}）"
                     for r in risk_table(slot.get("category"))]
            source = "风险源知识库（工程管理框架）"
        elif key == "plan":
            items = _plan_content(slot)
            source = "建办质〔2018〕31号 九章 + 建办质〔2021〕48号 细化要素"
        elif key == "controls":
            items, source = _controls(slot, ctx)
        elif key == "acceptance":
            items, source = _acceptance(slot, ctx)
        else:
            items, source = [], ""
        sections.append({"key": key, "label": d["label"], "items": items, "source": source})

    # 标题
    subject = slot.get("subject") or "该工程"
    eq = slot.get("equipment") or ""
    cap = f"{slot.get('capacity')}{slot.get('unit')}" if slot.get("capacity") else ""
    head = f"{cap} {eq} {subject} 危大工程" if (eq or cap) else f"{subject} 危大工程"
    title = f"{head} — 管理框架（结构化问答）"

    # markdown
    md = [f"## {title}", ""]
    md.append(f"- 危大类别：{slot.get('category_name') or slot.get('category') or '未识别'}"
              + ("（**超规模，需专家论证**）" if slot.get("is_super_scale") else "（危大级别）"))
    md.append("")
    for s in sections:
        md.append(f"### {s['label']}（{len(s['items'])} 项）")
        md.append(f"> 来源：{s['source']}")
        for it in s["items"]:
            md.append(f"- {it}")
        md.append("")
    return {"title": title, "sections": sections, "markdown": "\n".join(md)}


def run(query: str, ctx: dict | None = None) -> dict:
    """复合结构化 QA 入口：拆解 → 组装。"""
    slot = decompose(query)
    return assemble(slot, ctx)
