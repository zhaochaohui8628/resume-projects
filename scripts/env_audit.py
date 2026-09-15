"""torch_gpu 环境全盘体检（只读，不改任何东西）。

输出三类问题：
  A. 半安装包：有 *.dist-info 但代码文件没落地（沙箱拦截 pip 写入的症状）
  B. 版本漂移：与已验证基线不符的关键包
  C. import 可用性：关键模块能否真正导入

用法：python scripts/env_audit.py
"""
from __future__ import annotations

import os
import subprocess
import sys

TG = r"C:\Users\<用户名>\anaconda3\envs\torch_gpu"
SP = os.path.join(TG, "Lib", "site-packages")
PY = os.path.join(TG, "python.exe")

# dist-info 名（规范化）→ import 模块名；不在表中则按 -/_ 规则推导
NAME_MAP = {
    "python-dateutil": "dateutil",
    "sentence-transformers": "sentence_transformers",
    "scikit-learn": "sklearn",
    "pytorch-crf": "torchcrf",
    "python-docx": "docx",
    "pyyaml": "yaml",
    "rank-bm25": "rank_bm25",
    "pymupdf": "fitz",
    "faiss-cpu": "faiss",
    "annotated-doc": "annotated_doc",
    "python-dotenv": "dotenv",
    "pydantic-core": "pydantic_core",
    "hf-xet": "hf_xet",
    "markupsafe": "markupsafe",
    "pygments": "pygments",
    "typing-extensions": "typing_extensions",
    "charset-normalizer": "charset_normalizer",
    "markdown-it-py": "markdown_it",
    "scikit-network": "sknetwork",
    "beautifulsoup4": "bs4",
    "pillow": "PIL",
    "opencv-python": "cv2",
}
# 纯共享库/元数据，无 Python 模块，不算半安装
NO_MODULE = {"mkl", "tbb", "tcmlib", "umf", "onemkl-license",
             "intel-openmp", "intel-cmplr-lib-ur", "intel-cmplr-lic-rt"}

# 已验证基线（昨天 torch_gpu 修好时的版本）
BASELINE = {
    "torch": "2.7.1+cu118",
    "numpy": "2.5.3",
    "sentence_transformers": "6.0.1",
    "transformers": "5.17.0",
    "fastapi": "0.141",
    "pandas": "2.x",          # 昨天 2.x，今天被升到 3.0.5
    "openai": "?",            # 未 pin，今天 3.13.0
}


def norm(name: str) -> str:
    return name.lower().replace("_", "-")


def mod_name(dist: str) -> str | None:
    d = norm(dist)
    if d in NO_MODULE:
        return None
    if d in NAME_MAP:
        return NAME_MAP[d]
    return d.replace("-", "_")


def installed_version(dist_dir: str) -> str:
    """从 dist-info 目录名解析版本。"""
    base = os.path.basename(dist_dir)[: -len(".dist-info")]
    parts = base.split("-")
    return parts[-1] if len(parts) > 1 else "?"


def main() -> int:
    print("=" * 70)
    print("torch_gpu 环境全盘体检（只读）")
    print("=" * 70)

    dists = [d for d in os.listdir(SP) if d.endswith(".dist-info")]
    half: list[tuple[str, str, str]] = []      # (dist, module, version)
    for d in sorted(dists):
        dist = d[: -len(".dist-info")]
        # 去掉版本尾巴得到包名
        parts = dist.split("-")
        pkg = "-".join(parts[:-1]) if len(parts) > 1 else dist
        m = mod_name(pkg)
        if m is None:
            continue
        has_dir = os.path.isdir(os.path.join(SP, m))
        has_py = os.path.exists(os.path.join(SP, m + ".py"))
        has_so = any(os.path.exists(os.path.join(SP, m + ext))
                     for ext in (".pyd", ".so"))
        if not (has_dir or has_py or has_so):
            half.append((dist, m, installed_version(d)))

    print(f"\n【A】半安装包（有 dist-info 无代码）：{len(half)} 个")
    for dist, m, v in half:
        print(f"   - {dist:<38s} 缺模块 {m:<24s} v{v}")

    # 【B】版本核对
    print("\n【B】关键包实际版本（对照基线）")
    checks = ["torch", "numpy", "sentence_transformers", "transformers",
              "fastapi", "pandas", "openai", "faiss", "pydantic", "scipy"]
    code = ("import importlib,sys\n"
            "for m in %r:\n"
            "    try:\n"
            "        mod=importlib.import_module(m)\n"
            "        print('   ', m, getattr(mod,'__version__','?'))\n"
            "    except Exception as e:\n"
            "        print('   ', m, 'FAIL', type(e).__name__, str(e)[:60])\n" % (checks,))
    r = subprocess.run([PY, "-E", "-c", code], capture_output=True, text=True, timeout=300)
    out = (r.stdout or "").strip()
    print(out if out else (r.stderr or "")[:400])

    print("\n【C】基线对照（昨天验证通过的版本）")
    for k, v in BASELINE.items():
        print(f"   {k:<22s} 基线 {v}")

    print("\n" + "=" * 70)
    print(f"半安装包总数：{len(half)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
