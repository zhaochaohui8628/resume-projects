"""P4 训练入口：BERT + softmax(focal CE) / BERT + CRF（pytorch-crf），各两轮。

方案与轮次（用户 2026-09-11 定稿）
--------------------------------
方案1 `--head softmax`（损失 = focal loss CE，O 降权 0.25 + γ=2）
  stage1  冻结主权重（BERT 编码器），**仅训 softmax 分类头**    数据 = LLM 清洗出的 1000+（silver_train）
  stage2  解冻主权重，**全量微调**                              数据 = 第一轮高置信度伪标 + gold_train_portion 500
方案2 `--head crf`（损失 = CRF NLL）
  stage1  冻结主权重 + 发射层，**仅训 CRF 转移矩阵**            数据 = 同上（发射层由方案1 stage1 提供）
  stage2  解冻主权重，**全量微调**                              数据 = 第一轮高得分解码伪标 + gold_train_portion 500

为什么 CRF stage1 要 `--init` 方案1 stage1：若发射层是随机初始化，只训转移矩阵毫无意义；
先用同一批 1000+ 数据把发射层训好（方案1 stage1），再冻结它只学标签转移，才是干净的 CRF 增益隔离。

用法（GPU 环境）
    $PY ner2/scripts/train_ner.py --head softmax --stage 1
    $PY ner2/scripts/train_ner.py --head softmax --stage 2
    $PY ner2/scripts/train_ner.py --head crf     --stage 1
    $PY ner2/scripts/train_ner.py --head crf     --stage 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.bert.data_utils import load_jsonl  # noqa: E402
from ner2.src.bert.engine import TrainCfg, train  # noqa: E402
from ner2.src.common.paths import (  # noqa: E402
    BASE_BERT, GOLD_TRAIN_PORTION, MODELS, PHASE4, SILVER_TRAIN, SILVER_VAL,
)

# (head, stage) -> 默认计划；out 为 ner2/models/<out>/
PLAN = {
    ("softmax", 1): dict(freeze="encoder", lr=1e-3, epochs=20, patience=5,
                         out="s1_softmax_stage1", init=""),
    ("softmax", 2): dict(freeze="none", lr=2e-5, epochs=12, patience=3,
                         out="s1_softmax_stage2", init="s1_softmax_stage1"),
    ("crf", 1): dict(freeze="encoder_and_head", lr=5e-2, epochs=20, patience=5,
                     out="s2_crf_stage1", init="s1_softmax_stage1"),
    ("crf", 2): dict(freeze="none", lr=2e-5, epochs=12, patience=3,
                     out="s2_crf_stage2", init="s2_crf_stage1"),
}


def _default_train(head: str, stage: int) -> list[str]:
    """第二轮数据 = 第一轮高置信度数据（自清洗）+ 无标注池高置信度伪标（LLM 交叉验证）
                   + gold_train_portion 500（LLM 交叉校验 + 多领域专家）。"""
    if stage == 1:
        return [SILVER_TRAIN]
    return [os.path.join(PHASE4, f"hc_round1_{head}.jsonl"),
            os.path.join(PHASE4, f"pseudo_clean_{head}.jsonl"),
            GOLD_TRAIN_PORTION]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, choices=["softmax", "crf"])
    ap.add_argument("--stage", required=True, type=int, choices=[1, 2])
    ap.add_argument("--train", nargs="+", default=None, help="训练 jsonl（默认见 PLAN）")
    ap.add_argument("--val", default=SILVER_VAL)
    ap.add_argument("--init", default=None, help="续训 ckpt（默认：stage2 自动接上一轮）")
    ap.add_argument("--freeze", default=None, choices=["none", "encoder", "encoder_and_head"])
    ap.add_argument("--base-model", default=BASE_BERT)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--head-lr-mult", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="梯度累积步数（小显存机用）")
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--o-weight", type=float, default=0.25)
    ap.add_argument("--crf-aux-focal", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    plan = PLAN[(args.head, args.stage)]
    freeze = args.freeze or plan["freeze"]
    lr = args.lr if args.lr is not None else plan["lr"]
    epochs = args.epochs if args.epochs is not None else plan["epochs"]
    patience = args.patience if args.patience is not None else plan["patience"]
    out_dir = args.out_dir or os.path.join(MODELS, plan["out"])
    train_files = args.train or _default_train(args.head, args.stage)

    for p in train_files:
        if not os.path.exists(p):
            raise SystemExit(f"[error] 训练数据缺失：{p}")
    if not os.path.exists(args.val):
        fallback = SILVER_VAL if os.path.exists(SILVER_VAL) else None
        if not fallback:
            raise SystemExit(f"[error] 验证集缺失：{args.val}")
        args.val = fallback

    init_from = args.init
    if init_from is None and plan["init"]:
        cand = os.path.join(MODELS, plan["init"], "model.pt")
        init_from = cand if os.path.exists(cand) else None
    if args.head == "crf" and args.stage == 1 and not init_from and freeze == "encoder_and_head":
        print("[warn] CRF stage1 未找到方案1 stage1 发射层检查点 → 退化为 freeze=encoder"
              "（发射层 + CRF 联合训练）", flush=True)
        freeze = "encoder"

    rows = []
    for p in train_files:
        r = load_jsonl(p)
        print(f"[data] {os.path.basename(p)}: {len(r)} 句", flush=True)
        rows.extend(r)
    val_rows = load_jsonl(args.val)
    print(f"[data] 合计 train={len(rows)} val={len(val_rows)}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    cfg = TrainCfg(head=args.head, freeze=freeze, base_model=args.base_model,
                   epochs=epochs, lr=lr, head_lr_mult=args.head_lr_mult, batch=args.batch,
                   accum=args.accum, max_len=args.max_len, gamma=args.gamma,
                   o_weight=args.o_weight, crf_aux_focal=args.crf_aux_focal,
                   patience=patience, warmup_ratio=args.warmup_ratio, seed=args.seed,
                   device=args.device)

    # 小显存机的兜底：CUDA OOM 时自动折半 micro-batch 重试（最多 3 次）
    import torch
    res = None
    for attempt in range(4):
        try:
            res = train(cfg, tokenizer, rows, val_rows, out_dir, init_from=init_from)
            break
        except torch.cuda.OutOfMemoryError as e:
            if cfg.batch <= 4 or attempt == 3:
                raise
            print(f"[warn] CUDA OOM（batch={cfg.batch}）：{str(e).splitlines()[0]}", flush=True)
            cfg.batch = max(4, cfg.batch // 2)
            cfg.accum = min(8, args.accum * 2)
            torch.cuda.empty_cache()
            print(f"[warn] 折半重试 micro_batch={cfg.batch} accum={cfg.accum}", flush=True)
    summary = {"head": args.head, "stage": args.stage, "freeze": freeze, "lr": lr,
               "epochs": epochs, "batch": cfg.batch, "accum": cfg.accum,
               "train_files": train_files, "n_train": len(rows),
               "n_val": len(val_rows), "init_from": init_from,
               "ckpt": res["ckpt"], "best_val_f1": res["best_val_f1"],
               "best_epoch": res["best_epoch"],
               "trainable_params": res["trainable_params"],
               "total_params": res["total_params"]}
    with open(os.path.join(out_dir, "train_result.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("[result] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
