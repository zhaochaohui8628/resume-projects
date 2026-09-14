"""P4 全流程编排：两方案 × 两轮 + 伪标挖掘 + 黄金集评估。

步骤
----
1. softmax stage1（冻结主干，仅训 softmax 头；数据 = silver_train）
2. mine_pseudo(softmax)  → `data/phase4/pseudo_hc_softmax.jsonl`
   ⇢ 人工/LLM 交叉验证清洗 → `data/phase4/pseudo_clean_softmax.jsonl`（本项目由模型逐条复核落盘）
3. softmax stage2（解冻全量微调；数据 = 伪标 + gold_train_portion 500）
4. crf stage1（冻结主干 + 发射层，仅训 CRF 转移矩阵；发射层来自步骤 1）
5. mine_pseudo(crf)      → `data/phase4/pseudo_hc_crf.jsonl`
   ⇢ 同上清洗 → `data/phase4/pseudo_clean_crf.jsonl`
6. crf stage2（解冻全量微调；数据同步骤 3 口径）
7. eval_gold（四模型 + 规则基线）

用法（GPU 环境）
    $PY ner2/scripts/run_two_schemes.py                # 全流程
    $PY ner2/scripts/run_two_schemes.py --from 3 --to 3  # 只跑步骤 3
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import MODELS, PHASE4  # noqa: E402

SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)))
S1A = os.path.join(MODELS, "s1_softmax_stage1", "model.pt")
S1B = os.path.join(MODELS, "s1_softmax_stage2", "model.pt")
S2A = os.path.join(MODELS, "s2_crf_stage1", "model.pt")
S2B = os.path.join(MODELS, "s2_crf_stage2", "model.pt")
PSEUDO_HC = {h: os.path.join(PHASE4, f"pseudo_hc_{h}.jsonl") for h in ("softmax", "crf")}
PSEUDO_CLEAN = {h: os.path.join(PHASE4, f"pseudo_clean_{h}.jsonl") for h in ("softmax", "crf")}


def _run(py: str, *args):
    cmd = [py] + list(args)
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"[error] 步骤失败（exit {r.returncode}）：{' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--from", dest="frm", type=int, default=1)
    ap.add_argument("--to", dest="to", type=int, default=7)
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args()
    py = args.python

    def want(step: int) -> bool:
        return args.frm <= step <= args.to

    if want(1):
        _run(py, f"{SCRIPTS}/train_ner.py", "--head", "softmax", "--stage", "1")
    if want(2):
        _run(py, f"{SCRIPTS}/mine_pseudo.py", "--ckpt", S1A, "--head", "softmax",
             "--out", PSEUDO_HC["softmax"])
        if not os.path.exists(PSEUDO_CLEAN["softmax"]):
            raise SystemExit(
                f"[stop] 缺 {PSEUDO_CLEAN['softmax']}：高置信度伪标需经 LLM 交叉验证清洗"
                "（本项目由模型逐条复核后落盘）后方可进入第二轮训练。")
    if want(3):
        _run(py, f"{SCRIPTS}/train_ner.py", "--head", "softmax", "--stage", "2")
    if want(4):
        _run(py, f"{SCRIPTS}/train_ner.py", "--head", "crf", "--stage", "1")
    if want(5):
        _run(py, f"{SCRIPTS}/mine_pseudo.py", "--ckpt", S2A, "--head", "crf",
             "--out", PSEUDO_HC["crf"])
        if not os.path.exists(PSEUDO_CLEAN["crf"]):
            raise SystemExit(f"[stop] 缺 {PSEUDO_CLEAN['crf']}：同上，需先清洗。")
    if want(6):
        _run(py, f"{SCRIPTS}/train_ner.py", "--head", "crf", "--stage", "2")
    if want(7):
        specs = []
        for name, path in (("s1_softmax_stage1", S1A), ("s1_softmax_stage2", S1B),
                           ("s2_crf_stage1", S2A), ("s2_crf_stage2", S2B)):
            if os.path.exists(path):
                specs += ["--model", f"{name}={path}"]
        _run(py, f"{SCRIPTS}/eval_gold.py", *specs, "--with-rule-baseline",
             "--bootstrap", str(args.bootstrap))
    print("\n[done] 全流程结束", flush=True)


if __name__ == "__main__":
    main()
