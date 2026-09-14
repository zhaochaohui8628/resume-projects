"""递归字符切分（RecursiveCharacterTextSplitter 思路的零依赖实现）。

## 为什么要「递归」

固定滑窗（每 700 字切一刀）会把句子、条款、表格拦腰截断，切出来的块大量以半个句子开头。
递归切分的做法是**按分隔符优先级逐层退让**：

    段落空行 `\\n\\n`  →  换行 `\\n`  →  句末标点 `。；！？`  →  分句标点 `，、`  →  空格  →  字符

先在最高优先级（语义边界最强）的分隔符上切；切出来的片段若仍然超长，才降一级再切。
这样只要文本里存在句末标点，块就几乎总能在句子边界处收口。

## 为什么保留分隔符

切分后把分隔符**贴回前一片段**（`。` 归上一句）。否则"…应大于 5m"和"下一句…"会被切成
两段各自缺标点，在检索里表现为句末无标点、语义边界消失。

## 为什么返回 span 而不是字符串

结果要同时供三处使用：索引单元文本、父子块的**偏移量**（超长父块兜底时按命中的子块
算周边窗口，必须知道它在父块里的位置）、以及调试期回溯。所以内核产出
`(start, end)` 区间，字符串版本只是它的一层薄封装。
"""
from __future__ import annotations

# 分隔符优先级：越靠前语义边界越强。末尾的 "" 是兜底字符级切分。
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n", "\n", "。", "；", "！", "？", "；", "\n", "，", "、", " ", "",
)

SIZE = 700          # 子块目标长度（字符）
OVERLAP = 100       # 相邻子块重叠（字符）


def _split_keep(seg: str, sep: str) -> list[str]:
    """按 sep 切分并把 sep 贴回前一片段（保证句末标点不丢）。"""
    if sep == "":
        return list(seg)
    parts = seg.split(sep)
    out: list[str] = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            out.append(p + sep)
        elif p:
            out.append(p)
    return [p for p in out if p]


def _recursive(seg: str, seps: tuple[str, ...], size: int,
               base: int, out: list[tuple[int, int]]) -> None:
    """把 seg 递归切到 <= size，结果以 (start, end) 形式追加到 out。

    `base` 是 seg 在原文中的起始偏移——递归会逐层缩小子串，必须一路带着偏移，
    否则最后拿不到相对于**父块**的绝对位置。
    """
    if not seg:
        return
    if len(seg) <= size:
        out.append((base, base + len(seg)))
        return
    if not seps:
        # 分隔符耗尽（理论上被 "" 兜住）：硬按 size 切，保证不再超限
        for i in range(0, len(seg), size):
            out.append((base + i, base + min(len(seg), i + size)))
        return
    sep, rest = seps[0], seps[1:]
    if sep not in seg:
        _recursive(seg, rest, size, base, out)
        return
    # 逐片段递归；偏移用累加推进（片段长度 + 分隔符长度）
    cur = base
    for piece in _split_keep(seg, sep):
        _recursive(piece, rest, size, cur, out)
        cur += len(piece)


def recursive_spans(text: str, size: int = SIZE, overlap: int = OVERLAP,
                    separators: tuple[str, ...] | None = None) -> list[tuple[int, int]]:
    """递归切分，返回 [(start, end), ...]（相对 text 的字符偏移，左闭右开）。

    契约：
      - 每个 span 覆盖非空文本，且 `end - start <= size`；
      - 相邻 span 若来自同一次合并，重叠区长度 <= overlap；
      - 所有 span 的并集覆盖 text 的全部非空白字符（不丢内容）。
    """
    if not text or not text.strip():
        return []
    seps = tuple(separators) if separators is not None else DEFAULT_SEPARATORS
    raw: list[tuple[int, int]] = []
    _recursive(text, seps, size, 0, raw)
    return _merge(raw, text, size, overlap)


def _merge(spans: list[tuple[int, int]], text: str, size: int, overlap: int
           ) -> list[tuple[int, int]]:
    """自左向右贪心合并，块内总长不超过 size；换块时从上一块尾部回退 overlap。

    以 `text[e0-overlap:e]` 为新块起点，这样重叠区是**原文里真实存在的一段**，
    不会出现"凭空复制字符"的伪重叠（后者会让向量索引里出现不存在的文本）。
    """
    if not spans:
        return []
    out: list[tuple[int, int]] = []
    s0, e0 = spans[0]
    for s, e in spans[1:]:
        if e - s0 <= size:
            e0 = e
            continue
        out.append((s0, e0))
        s0 = e0 - overlap if e0 - overlap > s0 else e0
        e0 = e
    out.append((s0, e0))
    return out


def recursive_split(text: str, size: int = SIZE, overlap: int = OVERLAP,
                    separators: tuple[str, ...] | None = None) -> list[str]:
    """递归切分的字符串版本（严格等于 `[text[s:e] for s, e in recursive_spans(...)]`）。"""
    return [text[s:e] for s, e in recursive_spans(text, size, overlap, separators)]


def window_around(span: tuple[int, int], total: int, limit: int) -> tuple[int, int]:
    """在长度 `total` 的父块里，取「恰好 limit 长、且尽量以 span 为中心」的窗口。

    用于超长父块的兜底：不整篇回扩，只给命中子块周边一段，总量与父块最长限制一致。
    两侧余量分配规则：先按命中位置居中，再贴边收拢（保证窗口一定落在 [0, total] 内）。
    """
    s, e = span
    limit = min(limit, total)
    if e - s >= limit:               # 命中子块本身就超限（不该发生，防御性处理）
        return s, s + limit
    left = (limit - (e - s)) // 2
    ws = s - left
    we = ws + limit
    if ws < 0:
        we, ws = we - ws, 0
    if we > total:
        ws, we = max(0, ws - (we - total)), total
    return ws, we
