"""文本清洗：PDF 抽取文本的噪声治理（页眉页脚 / 水印 / 目录 / 拍板粘连 / OCR 花字）。

设计原则：只做可解释、可回归的确定性清洗，不做"猜测式改写"。
清洗强度分级：
  - clean_text()     句子级轻清洗（用于已确定是正文的片段）
  - strip_boilerplate() 页级清洗（去水印、页码、重复页眉页脚）
  - garbage_ratio()   乱码占比（用于过滤扫描件伪影块）
"""
from __future__ import annotations

import re
from collections import Counter

# 常见水印 / 站点标记
_WATERMARK_PATTERNS = [
    r"www\.[a-z0-9\-]+\.(?:com|cn|net)",
    r"免费下载|标准下载|建工之家|工标网|道客巴巴|豆丁网",
    r"文档来源为.*?下载",
]
_WATERMARK_RE = re.compile("|".join(_WATERMARK_PATTERNS), re.I)

# 页码行：纯数字 / 罗马数字 / 带短横线的页码
_PAGE_NUM_RE = re.compile(r"^\s*[-—–·•]?\s*(?:\d{1,4}|[ivxlcIVXLC]{1,5})\s*[-—–·•]?\s*$")

# 点线目录行：'范围…..' / '一般要求·' 等
_TOC_DOT_RE = re.compile(r"[…·.]{3,}")

# PDF 抽取常见粘连（全角/半角混用产生的伪字符）
_CHAR_FIX = {
    "\u00a0": " ", "\u3000": " ",
    "（": "(", "）": ")", "，": "，", "；": "；",
    "â": "", "\x00": "", "\ufffd": "",
}

# 全角数字 -> 半角
_FULLWIDTH_DIGITS = {ord("０") + i: ord("0") + i for i in range(10)}

# 允许的字符集合：CJK、常见标点、字母数字、工程符号
_ALLOWED_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf"          # CJK
    r"0-9A-Za-z"
    r"\s"
    r"()\[\]{}<>《》〈〉（）【】"
    r"，。、；：？！“”‘’\"'·—–\-_/\\|+=*%‰℃°±×÷≤≥≠≈~^&@#$:;,.?!"
    r"φΦΔδαβγθπΩΩμ"
    r"]"
)

# 高置信乱码片段：连续出现的非字典拉丁 + 非 ASCII 混排
_GARBLE_RE = re.compile(r"[^\u4e00-\u9fff\w\s]{2,}")


def strip_boilerplate(text: str, *, min_repeat: int = 4) -> str:
    """去页眉/页脚/水印：按行频次统计，重复 >= min_repeat 且长度 < 40 的短行视为样板行。"""
    lines = [ln.rstrip() for ln in text.split("\n")]
    cnt = Counter(ln.strip() for ln in lines if ln.strip() and len(ln.strip()) < 40)
    boiler = {ln for ln, c in cnt.items() if c >= min_repeat}
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append("")
            continue
        if s in boiler:
            continue
        if _PAGE_NUM_RE.match(s):
            continue
        if _WATERMARK_RE.search(s) and len(s) < 60:
            continue
        out.append(ln)
    return "\n".join(out)


def clean_text(text: str) -> str:
    """句子级清洗：去水印、修粘字、折行归一。"""
    if not text:
        return ""
    t = text
    for k, v in _CHAR_FIX.items():
        t = t.replace(k, v)
    t = _WATERMARK_RE.sub("", t)
    # 行内多余空白
    t = re.sub(r"[ \t]+", " ", t)
    # 中文字符之间不应有空格（PDF 抽取常见断裂）
    t = re.sub(r"(?<=[\u4e00-\u9fff]) +(?=[\u4e00-\u9fff])", "", t)
    # 章节标题粘连：'5. 1 . 1' -> '5.1.1'
    t = re.sub(r"(\d)\s*\.\s*(?=\d)", r"\1.", t)
    # 重排间距 PDF：'J G J 2 1 5' -> 'JGJ215'、'2 0 1 0' -> '2010'
    t = re.sub(r"(?<![.\d])((?:[A-Za-z0-9]\s){2,}[A-Za-z0-9])",
               lambda m: m.group(1).replace(" ", ""), t)
    # 去掉行首行尾的装饰符号
    t = re.sub(r"^[\s\-–—·•*#]+", "", t)
    t = re.sub(r"[\s\-–—·•*#]+$", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def garbage_ratio(text: str) -> float:
    """乱码占比：非允许字符占比。>0.12 基本可判为扫描件伪影。"""
    if not text:
        return 1.0
    bad = sum(1 for ch in text if not _ALLOWED_RE.match(ch))
    return bad / max(1, len(text))


def is_toc_like(text: str) -> bool:
    """目录/索引页特征：点线 + 行数多 + 平均行长短。"""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return True
    dotted = sum(1 for ln in lines if _TOC_DOT_RE.search(ln))
    if dotted / len(lines) > 0.3:
        return True
    avg = sum(len(ln) for ln in lines) / len(lines)
    return avg < 12 and len(lines) > 15


def collapse(text: str) -> str:
    """把折行压平成单段（条款正文用）。"""
    t = re.sub(r"\n+", "\n", text)
    parts = [p.strip() for p in t.split("\n") if p.strip()]
    return "".join(parts) if parts and all(len(p) < 40 for p in parts) else "\n".join(parts)
