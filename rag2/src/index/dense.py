"""稠密向量索引：sentence-transformers 编码 + FAISS IndexFlatIP（L2 归一化后即余弦）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_st(model_path: str, max_seq_length: int = 512, device: str | None = None):
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    return SentenceTransformer(model_path, device=device), max_seq_length


def import_faiss():
    """导入 faiss，并绕开 Windows 上的一个环境坑。

    坑：`faiss/_swigfaiss.pyd` 隐式依赖 Intel OpenMP（`libiomp5md.dll`）。某些 conda 环境
    （如只装了 GPU 版 torch 的 env）把这个 DLL 只放在 `torch/lib/` 下，而该目录仅在
    `import torch` 之后才进入 DLL 搜索路径 —— 于是「先导 faiss 再导 torch」会直接
    `ImportError: DLL load failed while importing _swigfaiss`，「先导 torch」却正常。
    这种「能不能 import 取决于导入顺序」的 bug 极难排查，所以这里不赌顺序：
    失败时主动把 torch 的 lib 目录挂上 DLL 搜索路径再重试一次。
    """
    try:
        import faiss
        return faiss
    except ImportError:
        pass
    try:
        import os
        import torch
        lib = Path(torch.__file__).resolve().parent / "lib"
        if lib.is_dir() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(lib))
    except Exception:  # noqa: BLE001  环境修复尽力而为，失败则原样抛出下面的错误
        pass
    import faiss
    return faiss


class DenseEncoder:
    """双塔编码器：tower='doc' 用于建库，tower='query' 用于检索（无微调时同一模型）。"""

    def __init__(self, model_path: str, max_seq_length: int = 512, batch_size: int = 64,
                 device: str | None = None):
        self.model, self.max_seq_length = _load_st(model_path, max_seq_length, device)
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def encode(self, texts: list[str], *, batch_size: int | None = None,
               show_progress: bool = False) -> np.ndarray:
        v = self.model.encode(
            texts, batch_size=batch_size or self.batch_size,
            normalize_embeddings=True, show_progress_bar=show_progress,
            convert_to_numpy=True)
        return np.asarray(v, dtype="float32")


class FaissStore:
    def __init__(self, dim: int = 0):
        self.dim = dim
        self.index = None

    def build(self, vectors: np.ndarray) -> "FaissStore":
        faiss = import_faiss()

        vecs = np.ascontiguousarray(vectors.astype("float32"))
        # 再归一化一次，保证内积 == 余弦
        norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        vecs = vecs / norm
        self.dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)
        return self

    def search(self, query_vecs: np.ndarray, top_k: int = 10):
        q = np.ascontiguousarray(query_vecs.astype("float32"))
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        q = q / norm
        scores, idx = self.index.search(q, top_k)
        return idx, scores

    def save(self, path: str | Path) -> None:
        faiss = import_faiss()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 注意：faiss.write_index 走 C++ fopen，Windows 下无法处理非 ASCII 路径
        # （本项目路径含中文），故序列化成字节流后由 Python 写盘。
        buf = faiss.serialize_index(self.index)
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        with open(str(path) + ".dim", "w", encoding="utf-8") as f:
            f.write(str(self.dim))

    @classmethod
    def load(cls, path: str | Path) -> "FaissStore":
        faiss = import_faiss()

        path = Path(path)
        with open(path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype="uint8")
        st = cls()
        st.index = faiss.deserialize_index(buf)
        st.dim = st.index.d
        return st


def save_units(path: str | Path, units: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")


def load_units(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
