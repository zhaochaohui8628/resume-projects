"""用指定 doc 塔重新编码索引单元并落 FAISS（微调后重建索引用）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import BASE_TOWER_MODEL, data_dir  # noqa: E402
from src.index.dense import DenseEncoder, FaissStore, load_units  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-tower", default=BASE_TOWER_MODEL)
    ap.add_argument("--out", required=True, help="输出 faiss 文件名（data/index/ 下）")
    ap.add_argument("--max-len", type=int, default=512)
    a = ap.parse_args()

    units = load_units(data_dir("index", "units.jsonl"))
    enc = DenseEncoder(a.doc_tower, max_seq_length=a.max_len)
    print(f"编码 {len(units)} 个单元 -> {a.out}")
    vecs = enc.encode([u["text"] for u in units], show_progress=True)
    FaissStore().build(vecs).save(data_dir("index", a.out))
    with open(data_dir("index", a.out + ".meta.json"), "w", encoding="utf-8") as f:
        json.dump({"doc_tower": a.doc_tower, "units": len(units),
                   "dim": int(vecs.shape[1]), "max_len": a.max_len},
                  f, ensure_ascii=False, indent=2)
    print("ok", np.asarray(vecs).shape)


if __name__ == "__main__":
    main()
