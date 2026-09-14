"""条款级切分：把规范正文切成「章 - 条」结构。

## 旧实现（`rag/src/ingestion/chunker.py`）的问题

旧版用 `re.finditer(r"第\\s*([\\d.]+)\\s*条")` 取相邻命中做条款边界，三类系统性错位：
  a) 现行 GB/JGJ 规范用数字条款号（`5.2.3`），压根没有"第X条"→ 整篇退化为巨大片段
     （旧产物里 `6.2.3` 的正文实际是 `7.2.6` 的内容）；
  b) 正文交叉引用（"应符合第9.2.1 条的规定"）被当成新条款起点；
  c) 目次/前言/公告里的条款号清单（"3.2.3 、3.2.4 、…"）被当成条款。

## 本实现要处理的三种真实排版

1. **号文同行**：`4.1.4燃油或燃气锅炉…`（GB550xx 全文强制规范，号与正文紧贴）
2. **号独占一行**：`1.0.1` / 换行 / `为预防建筑火灾…`（建工社 PDF 常见，行被排版拆开）
3. **号在句末**：`…观测记录。6.2.3成孔的控制深度应符合…`（一条里塞两条，需行内切分）

## 定位方式

**正文 = 文档中最长的一段"条款号严格递增链"**（>= 20 条）。条款号清单（号后接"、"、"条"）
不参与成链；条文说明的条款号从 1.0.1 重启、与正文最大号构成回退，天然断开。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..common.normalize import clean_text, collapse, garbage_ratio, is_toc_like

_CJK = r"\u4e00-\u9fff"

# 号与正文同行（号后紧跟中文）
NUM_CLAUSE_RE = re.compile(rf"^(\d{{1,2}}(?:\.\d{{1,2}}){{1,3}})\s*(?=[{_CJK}])")
APP_CLAUSE_RE = re.compile(rf"^([A-Z]\.\d{{1,2}}(?:\.\d{{1,2}}){{0,2}})\s*(?=[{_CJK}])")
CN_CLAUSE_RE = re.compile(
    rf"^第\s*(\d+(?:\.\d+)*|[一二三四五六七八九十百]+)\s*条\s*(?=[{_CJK}])")
# 号独占一行（需下一行以中文开头，排除前言的条款号清单）
EXACT_CLAUSE_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,3})$")
EXACT_APP_RE = re.compile(r"^([A-Z]\.\d{1,2}(?:\.\d{1,2}){0,2})$")
# 行内条款号：前接句读，后接中文（排除 "3.0m"、"表5.5.4"、"第9.2.1条"）
INLINE_CLAUSE_RE = re.compile(rf"(?<=[。；;：:！？!?])(\d{{1,2}}(?:\.\d{{1,2}}){{1,3}})\s*(?=[{_CJK}])")
INLINE_APP_RE = re.compile(rf"(?<=[。；;：:！？!?])([A-Z]\.\d{{1,2}}(?:\.\d{{1,2}}){{0,2}})\s*(?=[{_CJK}])")

CHAPTER_RE = re.compile(r"^(\d{1,2})\s+([^\d\s].{0,40})$")
CHAPTER_NUM_RE = re.compile(r"^(\d{1,2})$")
APPENDIX_RE = re.compile(r"^附录\s*([A-Z])\s*(.{0,30})$")

_REF_PREFIX = ("条", "款", "项", "的", "规定", "要求", "执行", "进行", "所列", "等", "及",
               "和", "或", "与", "、", "，", ",", "～", "~", "）", ")", "中", "内", "所")
_BAD_TITLE_CHARS = set("。，；：、？！()（）[－—]")

# 目次里的点线（'……120' / '.....'）：用来把目次中的附录清单与正文中的真附录区分开
_DOTS_RE = re.compile(r"[…·.]{3,}")

_MIN_CHAIN = 20      # 正文成链所需最少条款数
# 「号独占一行」时，下一行至少要有这么长才算条款正文。
# 取值要兼顾两头：
#   太大 -> 把"术语条"整段误杀——术语条排版是 `2.1.1` / `桩基` / `pile foundation`，
#           正文首行常常只有两个字（实测丢 200+ 条）；
#   太小 -> 表格单元格（`7.5` 后面跟 `关10`）被当条款。
# 所以主判据是"数字占比"（`关10` 占比 0.67 直接出局），长度只做最低下限。
_MIN_BODY_LEN = 2


@dataclass
class Clause:
    source: str
    clause_no: str
    text: str
    chapter: str = ""
    part: str = "正文"          # 正文 | 附录 | 条文说明
    appendix: str = ""
    children: list[str] = field(default_factory=list)

    @property
    def clause_id(self) -> str:
        return f"{self.source}::{self.clause_no}"


_CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
              "八": 8, "九": 9, "十": 10, "百": 100}


def _cn_to_int(s: str) -> int | None:
    """中文数字 -> 整数（支持 一~九十九 与 一百 这类常见条款号写法）。"""
    if not s or any(c not in _CN_DIGITS for c in s):
        return None
    if s == "十":
        return 10
    total, unit = 0, 1
    for c in reversed(s):
        v = _CN_DIGITS[c]
        if v >= 10:
            unit = v
        else:
            total += v * unit
    return total or None


def _num_key(no: str) -> tuple[int, ...]:
    """条款号 -> 可比较的键。

    带字母的附录号（`A.0.1`）必须把**字母本身**编进键：旧实现剥掉字母只看数字，
    于是 `B.0.1` 的键 (0,1) 小于 `A.0.9` 的 (0,9)，被当成"号回退"而拒绝开新条，
    整个附录 B 的内容就挂到附录 A 最后一条下面。
    """
    if not no:
        return (0,)
    head = no[0]
    # ASCII 字母前缀（附录 A.0.1 / 表 B.1）：字母编进键
    if head.isascii() and head.isalpha():
        nums = tuple(int(x) for x in no.split(".")[1:] if x.isdigit())
        return (ord(head.upper()),) + nums
    seg = no.split(".")[0]
    if seg.isdigit():
        nums = tuple(int(x) for x in no.split(".") if x.isdigit())
        return nums or (0,)
    # 中文数字条款号（第X条）：转成整数；转不了就退化为 (0,)，与旧行为一致
    v = _cn_to_int(seg)
    return (v,) if v is not None else (0,)


def _gt(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) > b + (0,) * (n - len(b))


def _ge(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return a == b or _gt(a, b)


def _is_ref(rest: str) -> bool:
    return any(rest.startswith(p) for p in _REF_PREFIX)


def _cjk_head(s: str) -> bool:
    return bool(s) and bool(re.match(rf"^[{_CJK}]", s))


def _looks_like_body(ln: str) -> bool:
    """「号独占一行」时用来判断下一行是不是真的条款正文。

    旧实现只要求"下一行以中文开头"，这个判据太弱：表格里一个单元格
    `7.5`、下一行是 `关10`（以中文开头、仅 3 字）就会被误认成条款 7.5，
    随后续到下一个真条款之前的全部表格内容都挂到它名下（实测制造出
    122193 字符的巨块）。真条款正文是一句完整的话或一个术语名，因此要求：
    以中文起头 + 数字占比不过半（挡数值/单位行如 `关10`、`O.50`）。
    """
    s = ln.strip()
    if len(s) < _MIN_BODY_LEN or not _cjk_head(s):
        return False
    return sum(1 for c in s if c.isdigit()) / len(s) <= 0.5


# 附录条款号的 OCR 混淆形态：`A.O.l` / `D. 0.1` / `c. 0.1`（数字 0/1 被抽成字母 O/l/I）。
_APPX_NUM_RE = re.compile(r"^[A-Za-z]\s*\.\s*[0-9OoIl]")
_OCR_DIGIT = {"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8", "Z": "2"}
# 行首的附录号（含 OCR 混淆字符），第 1 组字母 / 2、3 组各级数字 / 4 组号后正文
_APPX_HEAD_RE = re.compile(
    r"^([A-Za-z])\s*\.\s*([0-9OoIlSsBZ]{1,2})(?:\s*\.\s*([0-9OoIlSsBZ]{1,2}))?\s*(.*)$")


def _appendix_head(ln: str, nxt: str | None, in_appendix: bool
                   ) -> tuple[int, str, str] | None:
    """识别附录条款起点，容忍 OCR 把 0/1 抽成 O/l。

    JGJ94 附录 A 的条款号在 PDF 里是 `A.O.l`（字母 O、小写 l），标准正则全部失效
    → 附录里再没有条款起点 → 第一条把整本附录吞成 119004 字符的巨块。
    这里把混淆字符还原成数字再匹配；`in_appendix` 之外的正文不走这条路径，
    避免把正文里的英文缩写误当条款号。
    """
    if not in_appendix:
        return None
    m = _APPX_HEAD_RE.match(ln)
    if not m:
        return None
    num = m.group(1).upper() + "." + _OCR_DIGIT.get(m.group(2), m.group(2))
    if m.group(3):
        num += "." + _OCR_DIGIT.get(m.group(3), m.group(3))
    rest = m.group(4).strip()
    # 号后必须接中文正文（或号独占一行、下一行像正文）。
    # 不这么卡的话，表格里的十进制数会被误判成附录条号：
    #   `o. 99966 | 0.00640 …` -> 伪条款 `O.99`；`O. 92 X 1.4uμ 8 h2` -> 伪条款 `O.92`。
    # 伪条号一出现就会重置条款号链，把整块附录吞进一个巨块（实测最大 140824 字符）。
    if rest:
        if not _looks_like_body(rest):
            return None
    elif not _looks_like_body(nxt or ""):
        return None
    return (0, num, rest)


def _appendix_header(lines: list[str], i: int, body_count: int = 10 ** 9,
                     min_body: int = _MIN_CHAIN) -> re.Match | None:
    """判断 lines[i] 是不是**正文里**的附录标题（而不是目次里的附录清单）。

    只认标题字符串是不够的——目次同样有 `附录A` / `附录B` 这样的行。旧实现因此
    在 JGJ94 上完全没有识别出附录（part=附录 全语料仅 125 条），正文最后一条把
    附录表格与条文说明一路吞进自己（最长 122193 字符）；反过来判据放松又会把
    目次当真附录（DGJ08-11 目次里的 `附录R` 紧跟着"本标准用词说明"），一旦误
    触发就整段进入附录模式，正文条款号再也不开新条，**实测丢 467 条正文**。

    所以五重排除，任一不满足即判定为"不是真附录"：
      a) 标题本身不能带交叉引用（`附录B 的规定。` 是正文里的引用，不是标题）；
      b) 正文尚未成链（条款数 < min_body）——真附录都在正文之后；
      c) 标题后 8 行内出现点线（`……120`）；
      d) 标题后 8 行内又出现另一个附录标题（目次里附录是扎堆列出的）；
      e) 标题后 8 行内找不到真附录的内容特征（`表A…` 或 `A.0.1` 形式的附录条款号）。
    """
    ma = APPENDIX_RE.match(lines[i])
    if not ma or len(lines[i]) >= 40 or _DOTS_RE.search(lines[i]):
        return None
    if _is_ref(ma.group(2).strip()):
        return None
    if body_count < min_body:
        return None
    window = [x.strip() for x in lines[i + 1:i + 9]]
    if any(_DOTS_RE.search(x) for x in window):
        return None
    if any(APPENDIX_RE.match(x) and len(x) < 40 for x in window):
        return None
    has_table = any(re.search(r"表\s*[A-Z]", x) for x in window)
    has_num = any(_APPX_NUM_RE.match(x) for x in window)
    if not (has_table or has_num):
        return None
    return ma


def find_heads(ln: str, nxt: str | None = None,
               in_appendix: bool = False) -> list[tuple[int, str, str]]:
    """该行的条款起点：[(字符位置, 条款号, 号后正文), ...]。号独占一行时正文为空串。"""
    heads: list[tuple[int, str, str]] = []
    m = NUM_CLAUSE_RE.match(ln)
    if not m and in_appendix:
        m = APP_CLAUSE_RE.match(ln)
    if m:
        heads.append((0, m.group(1), ln[m.end():]))
    else:
        me = EXACT_CLAUSE_RE.match(ln)
        if not me and in_appendix:
            me = EXACT_APP_RE.match(ln)
        if me is not None and _looks_like_body(nxt or ""):
            heads.append((0, me.group(1), ""))
        else:
            m2 = CN_CLAUSE_RE.match(ln)
            if m2:
                heads.append((0, m2.group(1), ln[m2.end():]))
            else:
                # 最后的兜底：附录号可能被 OCR 打坏（`A.O.l`），标准正则认不出来
                ah = _appendix_head(ln, nxt, in_appendix)
                if ah:
                    heads.append(ah)
    for im in INLINE_CLAUSE_RE.finditer(ln):
        heads.append((im.start(), im.group(1), ln[im.end():]))
    if in_appendix:
        for im in INLINE_APP_RE.finditer(ln):
            heads.append((im.start(), im.group(1), ln[im.end():]))
    heads.sort(key=lambda x: x[0])
    return heads


def _iter_lines(pages: list[str]) -> list[str]:
    out: list[str] = []
    for p in pages:
        for ln in p.split("\n"):
            s = clean_text(ln)
            if s:
                out.append(s)
    return out


def find_body_start(lines: list[str], *, scan_limit: int = 8000) -> int:
    """正文档位 = **第一段**长度 >= _MIN_CHAIN 的条款号递增链的起点行。

    取"第一段"而非"最长段"：条文说明的条款号从 1.0.1 重启，其递增链同样可以很长，
    但它排在正文之后；正文永远是文档中第一段足够长的递增链。
    """
    run: list[int] = []
    prev_key: tuple[int, ...] | None = None
    for i, ln in enumerate(lines[:scan_limit]):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        for _pos, no, rest in find_heads(ln, nxt):
            if _is_ref(rest) or (rest and len(rest) < 4):
                continue
            key = _num_key(no)
            if prev_key is not None and _gt(key, prev_key):
                run.append(i)
            else:
                if len(run) >= _MIN_CHAIN:
                    return max(0, run[0] - 4)
                run = [i]
            prev_key = key
    if len(run) >= _MIN_CHAIN:
        return max(0, run[0] - 4)
    return 0


def _chapter_at(lines: list[str], i: int, expect: int) -> tuple[int, int] | None:
    """章标题：(序号, 消耗的行数)。支持 '1 总则' 单行与 '1' / '总则' 两行。"""
    ln = lines[i]
    m = CHAPTER_RE.match(ln)
    if m and not find_heads(ln):
        n, title = int(m.group(1)), m.group(2).strip()
        if len(title) <= 16 and not any(c in _BAD_TITLE_CHARS for c in title):
            if not expect or n == expect:
                return n, 1
    m2 = CHAPTER_NUM_RE.match(ln)
    if m2 and i + 1 < len(lines):
        title = lines[i + 1]
        n = int(m2.group(1))
        if (len(title) <= 16 and not find_heads(title, lines[i + 2] if i + 2 < len(lines) else "")
                and not any(c in _BAD_TITLE_CHARS for c in title)
                and not any(ch.isdigit() for ch in title)):
            if not expect or n == expect:
                return n, 2
    return None


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff") / len(text)


def split_clauses(pages: list[str], source: str, *, min_len: int = 24,
                  max_garbage: float = 0.10, min_cjk_ratio: float = 0.30,
                  strict_body: bool = True,
                  lines: list[str] | None = None) -> list[Clause]:
    """切条款。`lines` 可注入预处理过的行（见 reassemble_number_lines 的第二遍解析）。"""
    if lines is None:
        lines = _iter_lines(pages)
    lines = [ln for ln in lines if not is_toc_like(ln)]
    if strict_body:
        lines = lines[find_body_start(lines):]

    clauses: list[Clause] = []
    chapter, chapter_num = "", 0
    part, appendix, in_appendix = "正文", "", False
    cur: Clause | None = None
    prev_key: tuple[int, ...] | None = None
    skip_until = 0

    def flush():
        nonlocal cur
        if cur is not None:
            body = collapse(cur.text)
            if len(body) >= min_len and garbage_ratio(body) <= max_garbage:
                cur.text = body
                clauses.append(cur)
        cur = None

    def open_clause(no: str):
        nonlocal cur, prev_key
        flush()
        cur = Clause(source=source, clause_no=no, text="",
                     chapter=chapter, part=part, appendix=appendix)
        prev_key = _num_key(no)

    for idx, ln in enumerate(lines):
        if idx < skip_until:
            if cur is not None:
                cur.text += "\n" + ln
            continue

        if part != "条文说明" and "条文说明" in ln and len(ln) <= 30:
            flush()
            part, in_appendix = "条文说明", False
            continue

        ma = _appendix_header(lines, idx, body_count=len(clauses))
        if ma and part == "正文":
            flush()
            appendix = ma.group(1)
            title = ma.group(2).strip()
            nxt_ln = lines[idx + 1] if idx + 1 < len(lines) else ""
            if not title and 0 < len(nxt_ln) <= 30 and not find_heads(nxt_ln):
                title = nxt_ln
            part, in_appendix = "附录", True
            chapter = f"附录{appendix} {title}".strip()
            prev_key = None
            continue

        if not in_appendix:
            ch = _chapter_at(lines, idx, chapter_num + 1 if chapter_num else 1)
            if ch:
                n, used = ch
                flush()
                chapter_num = n
                chapter = ln if used == 1 else f"{ln} {lines[idx + 1]}"
                part, appendix, prev_key = "正文", "", None
                skip_until = idx + used
                continue

        nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
        cursor = 0
        for pos, no, rest in find_heads(ln, nxt, in_appendix):
            key = _num_key(no)
            ok = prev_key is None or (_gt(key, prev_key) if not in_appendix
                                      else _ge(key, prev_key))
            if _is_ref(rest) or (rest and len(rest) < 3) or not ok:
                continue
            if cur is not None:
                cur.text += "\n" + ln[cursor:pos]
            open_clause(no)
            cursor = pos
        if cur is not None:
            cur.text += "\n" + ln[cursor:]
    flush()
    return clauses


def make_children(text: str, size: int = 700, overlap: int = 100,
                  threshold: int | None = None) -> list[str]:
    """超长条款切子块（父块始终保留全文，供回扩）。

    ⚠️ **切分触发阈值 ≠ 子块尺寸**：只有 `len(text) > threshold` 的"过长条款"才切分，
    threshold 默认取 `MAX_PARENT_LEN`（由长度分布标定 ≈p99，代表"整条送 LLM 的合理上限"）。
    长度在合理范围内的条款**整条直用，不做滑窗、不做父子块**——这是第四轮重分块的核心口径。

    用递归字符切分而不是定长滑窗：定长会把句子拦腰截断，子块大量以半个句子开头，
    检索命中后拼给 LLM 的上下文边界是碎的。
    """
    from .recursive_split import recursive_split

    if threshold is None:
        from ..common.limits import MAX_PARENT_LEN
        threshold = MAX_PARENT_LEN
    if len(text) <= threshold:
        return []
    return recursive_split(text, size=size, overlap=overlap)


def make_children_spans(text: str, size: int = 700, overlap: int = 100,
                        threshold: int | None = None
                        ) -> list[tuple[int, int]]:
    """子块的字符偏移 [(start, end), ...]。

    索引层需要偏移而不是字符串：超长父块回扩兜底时要按**命中子块在父块里的位置**
    取周边窗口，没有偏移就只能从头截断。

    ⚠️ 与 `make_children` 同一口径：`len(text) <= threshold`（默认 `MAX_PARENT_LEN`）
    时返回整条 `[(0, len(text))]`——**不切分**；只有超过阈值才按 size/overlap 递归切分。
    """
    from .recursive_split import recursive_spans

    if threshold is None:
        from ..common.limits import MAX_PARENT_LEN
        threshold = MAX_PARENT_LEN
    if len(text) <= threshold:
        return [(0, len(text))] if text else []
    return recursive_spans(text, size=size, overlap=overlap)


# ---------------------------------------------------------------- 第二遍解析

_NUM_FRAG_RE = re.compile(r"^\d{1,2}$")
_SUB_FRAG_RE = re.compile(r"^[.．·]\s*\d{1,2}$")
_ALPHA_FRAG_RE = re.compile(r"^[A-Z]$")


def reassemble_number_lines(lines: list[str]) -> list[str]:
    """把被排版拆散的条款号拼回一行。

    有些 PDF 的条款号是逐字符成行的（`GB50550-2010`）：

        7 / ．2 / ．2 / 原构件表面凿毛后，应按设计的规定涂刷结构界面胶(剂)。

    `clean_text` 的行内合并规则处理不了跨行拆散，于是 `NUM_CLAUSE_RE` 全线失效、
    正文递增链一个都成不了链 → `find_body_start` 返回 0 → **整本 17.1 万字的规范
    被静默丢弃**。这里把「裸数字/单字母 + 若干 `.NN` 片段 + 紧随的中文正文」拼成
    `7.2.2原构件表面凿毛后…`，让第一遍的解析器能复用。

    仅作为第一遍失败后的**第二遍**使用（见 build_corpus），避免影响已能正确切分的规范。
    """
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i].strip()
        if _NUM_FRAG_RE.match(ln) or _ALPHA_FRAG_RE.match(ln):
            j, subs = i + 1, []
            while j < n and len(subs) < 3 and _SUB_FRAG_RE.match(lines[j].strip()):
                subs.append(re.sub(r"^[.．·]\s*", ".", lines[j].strip()))
                j += 1
            if subs:
                body = ""
                if j < n and _cjk_head(lines[j]):
                    body, j = lines[j], j + 1
                out.append(ln + "".join(subs) + body)
                i = j
                continue
        out.append(lines[i])
        i += 1
    return out


def split_fixed(pages: list[str], source: str, *, size: int = 700, overlap: int = 100,
                part: str = "无条款") -> list[Clause]:
    """无条款结构的规范：整本按固定大小（700/100）递归切分。

    切不出条款号的规范（文本层重排伪影、或条款号完全无法定位）不能再走"整本当一条"，
    否则语料里会出现 10 万字符级的伪条款。这里退化为定长块，`clause_no` 用
    `C0001` 形式的序号——它只是**父块标识**，不冒充真实条款号。
    """
    from .recursive_split import recursive_split

    lines = _iter_lines(pages)
    # 目次/索引行与点线行不进正文
    lines = [ln for ln in lines if not is_toc_like(ln) and not _DOTS_RE.search(ln)]
    text = collapse("\n".join(lines))
    chunks = recursive_split(text, size=size, overlap=overlap)
    return [
        Clause(source=source, clause_no=f"C{k:04d}", text=c, chapter="", part=part)
        for k, c in enumerate(chunks, 1) if c.strip()
    ]
