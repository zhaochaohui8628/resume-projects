"""报告导出：Markdown（必选）+ Excel（守卫式，需 openpyxl）。"""
from __future__ import annotations


def to_markdown(report: dict) -> str:
    lines = ["# 施工方案合规自查报告", "", f"> {report.get('summary', '')}", ""]
    lines.append("## 风险清单")
    lines.append("| 级别 | 检查项 | 风险 | 依据/建议 |")
    lines.append("| --- | --- | --- | --- |")
    for r in report.get("risks", []):
        basis = "；".join(r.get("basis", [])[:1]) if r.get("basis") else ""
        sug = r.get("suggestion", "") or (r.get("description_nl", "")[:60] if r.get("description_nl") else "")
        lines.append(f"| {r['severity']} | {r['check']} | {r['title']} | {sug} {basis} |")
    return "\n".join(lines)


def to_excel(report: dict, path: str):
    try:
        from openpyxl import Workbook
    except ImportError:
        raise RuntimeError("导出 Excel 需要 openpyxl：pip install openpyxl")
    wb = Workbook()
    ws = wb.active
    ws.title = "风险清单"
    ws.append(["级别", "检查项", "风险", "细节", "依据/建议"])
    for r in report.get("risks", []):
        ws.append([r["severity"], r["check"], r["title"], r.get("detail", ""), r.get("suggestion", "")])
    wb.save(path)


def export(report: dict, out_dir: str, name: str = "report"):
    import os

    os.makedirs(out_dir, exist_ok=True)
    md = os.path.join(out_dir, f"{name}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(to_markdown(report))
    try:
        xlsx = os.path.join(out_dir, f"{name}.xlsx")
        to_excel(report, xlsx)
        return [md, xlsx]
    except RuntimeError as e:
        return [md, str(e)]
