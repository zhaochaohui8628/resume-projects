"""token 级标签序列 → 字符级实体解码（BIOE 状态机）。

与 `data_utils.encode` 的 offsets 对齐：每个 token 贡献自己的 [start, end)，
实体区间取起点 token 的 start 与末点 token 的 end，因此天然精确到字符。
"""
from __future__ import annotations


def token_tags_to_char_entities(offsets: list[tuple[int, int]], tags: list[str]) -> list[dict]:
    """BIOE 标签名 + token 字符偏移 → [{"type","start","end"}]。

    - `B-` 开新实体，`I-` 续接，`E-` 收尾（缺 E 容错收到类型变化处）；
    - 特殊符 / 空区间（offset == (0,0)）自动跳过。
    """
    ents: list[dict] = []
    i, L = 0, len(tags)
    while i < L:
        tg = tags[i]
        if not tg.startswith("B-"):
            i += 1
            continue
        t = tg[2:]
        s = int(offsets[i][0])
        e = int(offsets[i][1])
        j = i + 1
        while j < L:
            tj = tags[j]
            if tj == f"E-{t}":
                e = max(e, int(offsets[j][1]))
                j += 1
                break
            if tj == f"I-{t}":
                e = max(e, int(offsets[j][1]))
                j += 1
                continue
            break
        if e > s:
            ents.append({"type": t, "start": s, "end": e})
        i = j
    return ents


def tags_to_entities(tag_names: list[str]):
    """标签名序列 → [(type, i, j)]（token 下标区间，供 token 级 F1 用）。"""
    ents = []
    i, n = 0, len(tag_names)
    while i < n:
        tg = tag_names[i]
        if tg.startswith("B-"):
            t = tg[2:]
            j = i + 1
            while j < n:
                tj = tag_names[j]
                if tj == f"I-{t}":
                    j += 1
                elif tj == f"E-{t}":
                    j += 1
                    break
                else:
                    break
            ents.append((t, i, j))
            i = j
        else:
            i += 1
    return ents
