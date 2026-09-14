"""建索引：clauses.jsonl -> BM25 索引 + FAISS 向量索引 + 索引单元表。

## 父子块模型

检索的最小单位是**子块（unit）**，不是条款：

  - 父块（clause）长度 <= MAX_PARENT_LEN（≈p99，由长度分布标定）时，**整条即唯一子块**
    （`cid#0`）——合理范围内的条款**不做滑窗、不做父子块**，直接整条进索引；
  - 只有**过长条款**（> MAX_PARENT_LEN）才按 `CHUNK_SIZE/CHUNK_OVERLAP`（700/100）
    用**递归字符切分**切成多个子块，全部共享同一个 `clause_id`。

⚠️ **切分触发阈值 ≠ 子块尺寸**：`CHUNK_SIZE=700` 只是子块的目标长度，**不是**"是否切分"的
判据；判据是 `MAX_PARENT_LEN`（见 `src/common/limits.py` 与 `make_children_spans` 的
`threshold` 参数）。旧实现把 700 同时当阈值用，导致 700~1200 字的条款被无谓切成多块。

检索侧按 `clause_id` 聚合：同一父块的多个子块只保留最高分的一条（去重），
再按 `MAX_PARENT_LEN` 决定是整条回扩还是只给命中子块周边窗口。

## 为什么子块要带字符偏移

`start`/`end` 是子块在父块内的位置。超长父块兜底回扩时必须知道"命中的是父块的第几段"
才能取周边窗口；没有偏移就只能从父块开头截，等于永远只回扩开头那一截。
即使当前父块没超限，也一并记下（`start=0, end=len`），让检索侧不需要分支判断。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.limits import CHUNK_OVERLAP, CHUNK_SIZE  # noqa: E402
from src.common.paths import BASE_TOWER_MODEL, data_dir  # noqa: E402
from src.corpus.clause_split import make_children_spans  # noqa: E402
from src.index.bm25 import BM25Index  # noqa: E402
from src.index.dense import DenseEncoder, FaissStore, save_units  # noqa: E402

FORCE = "--force" in sys.argv


def make_units(clauses: list[dict]) -> list[dict]:
    units: list[dict] = []
    for c in clauses:
        text, cid = c["text"], c["id"]
        meta = c["metadata"]
        spans = make_children_spans(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for k, (s, e) in enumerate(spans):
            seg = text[s:e]
            if not seg.strip():
                continue
            units.append({
                "uid": f"{cid}#{k}",
                "clause_id": cid,
                "text": seg,
                "start": s,
                "end": e,
                "parent_len": len(text),
                "source": meta["source"],
                "clause_no": meta["clause_no"],
                "part": meta.get("part", "正文"),
                "chunk_mode": meta.get("chunk_mode", "clause"),
            })
    return units


def main() -> None:
    src = data_dir("corpus", "clauses.jsonl")
    clauses = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    print(f"父块 {len(clauses)}")
    units = make_units(clauses)
    n_multi = sum(1 for c in clauses
                  if len(make_children_spans(c["text"], CHUNK_SIZE, CHUNK_OVERLAP)) > 1)
    print(f"索引单元 {len(units)}（{n_multi} 个超长父块被切成多子块，"
          f"窗口 {CHUNK_SIZE}/{CHUNK_OVERLAP}）")
    save_units(data_dir("index", "units.jsonl"), units)

    print("构建 BM25 ...")
    bm = BM25Index().build([u["uid"] for u in units], [u["text"] for u in units])
    bm.save(data_dir("index", "bm25.json"))
    print(f"  BM25 词表 {len(bm.postings)}，avgdl {bm.avgdl:.1f}")

    print("编码向量（bge-small-zh-v1.5）...")
    vec_path = data_dir("index", "tower_base.npy")
    if Path(vec_path).exists() and not FORCE:
        import numpy as np
        vecs = np.load(vec_path)
        print(f"  复用缓存 {vecs.shape}")
    else:
        import numpy as np
        enc = DenseEncoder(BASE_TOWER_MODEL)
        vecs = enc.encode([u["text"] for u in units], show_progress=True)
        np.save(vec_path, vecs)
    print(f"  向量 {vecs.shape}")
    store = FaissStore().build(vecs)
    store.save(data_dir("index", "tower_base.faiss"))
    with open(data_dir("index", "index_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "clauses": len(clauses), "units": len(units),
            "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
            "multi_child_parents": n_multi,
            "model": "bge-small-zh-v1.5", "dim": int(vecs.shape[1]),
        }, f, ensure_ascii=False, indent=2)
    print("完成 -> rag2/data/index/")


if __name__ == "__main__":
    main()
