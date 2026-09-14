"""蒸馏：用教师(CrossEncoder)分数做监督，margin ranking loss 训练双塔。

## 关键设计（用户明确要求）

1. **margin 不设固定值**，取教师模型对 (query, pos) 与 (query, neg) 打分的差：
   `margin_i = σ(s_pos) − σ(s_neg)`（σ=sigmoid，把 CE logit 压到 [0,1]，与学生余弦差同量级）。
2. 学生 = 双塔（query/doc），从 dual_mix 继续训练（**必须以检查点目录为 base 建塔**，
   不能"先建基座塔再 load_state_dict"，Windows 上会静默 segfault）。
3. 损失（margin ranking loss）：
   `loss = max(0, margin_i − (sim(q,pos) − sim(q,neg)))`
   即学生要把正负例余弦差拉到至少等于教师的 margin。教师越确信（margin 大），学生要拉开越多。
4. 内积即余弦（CLS + L2 归一），temperature 概念融入 margin 中，不需额外缩放。

## 数据

`data/phase7/distill_data.jsonl`：{qid, query, pos_text, neg_text, s_pos, s_neg, margin}
（由 `scripts/gen_distill_data.py` 用已训 CE 打分生成）。

用法：
  python -m src.train.distill.train \
    --data data/phase7/distill_data.jsonl \
    --out data/models/dual_distill \
    --init-doc data/models/dual_mix/doc_encoder \
    --init-query data/models/dual_mix/query_encoder \
    --epochs 4 --lr 1e-5 --batch-size 4 --max-len 160 --device cuda
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.common.paths import BASE_TOWER_MODEL, data_dir  # noqa: E402
from src.train.dual_tower.model import build_dual_tower  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def margin_rank_loss(q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor,
                     margins: torch.Tensor) -> torch.Tensor:
    """margin_i = σ(s_pos) − σ(s_neg)；要求学生余弦差 ≥ margin。"""
    sim_pos = (q * pos).sum(dim=-1)      # 已 L2 归一，内积即余弦
    sim_neg = (q * neg).sum(dim=-1)
    gap = sim_pos - sim_neg
    loss = torch.clamp(margins - gap, min=0.0).mean()
    return loss, gap


def train(*, data: str, out_dir: str, epochs: int = 4, batch_size: int = 4,
          lr: float = 1e-5, max_len: int = 160, warmup_ratio: float = 0.1,
          seed: int = 42, device: str | None = None,
          init_doc: str | None = None, init_query: str | None = None) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_jsonl(Path(data))
    # 去掉教师认为"负例胜出"的极端样本（margin < -0.05，可能是清洗漏网的假阴性）
    rows = [r for r in rows if r["margin"] > -0.05]
    print(f"蒸馏样本 {len(rows)}（过滤 margin<=-0.05 后）")
    model = build_dual_tower(BASE_TOWER_MODEL, max_len=max_len,
                             doc_base=init_doc, query_base=init_query)
    model.to(device)
    tok = model.tokenizer
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, math.ceil(len(rows) / batch_size))
    total = steps_per_epoch * epochs
    warmup = max(1, int(total * warmup_ratio))

    def lr_at(s: int) -> float:
        if s < warmup:
            return lr * (s + 1) / warmup
        return lr * max(0.02, 1 - (s - warmup) / max(1, total - warmup))

    hist, step, t0 = [], 0, time.time()
    for ep in range(epochs):
        model.train()
        idx = list(range(len(rows)))
        random.shuffle(idx)
        tot, gap_sum, nb = 0.0, 0.0, 0
        for i in range(0, len(idx), batch_size):
            chunk = [rows[j] for j in idx[i:i + batch_size]]
            enc_q = tok([r["query"] for r in chunk], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
            enc_p = tok([r["pos_text"] for r in chunk], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
            enc_n = tok([r["neg_text"] for r in chunk], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
            margins = torch.tensor([r["margin"] for r in chunk], device=device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            qe = model.encode_query(**enc_q)
            pe = model.encode_doc(**enc_p)
            ne = model.encode_doc(**enc_n)
            loss, gap = margin_rank_loss(qe, pe, ne, margins)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            tot += float(loss)
            gap_sum += float(gap.mean())
            nb += 1
            step += 1
            if step % 20 == 0:
                print(f"  step {step}/{total} loss {tot/nb:.4f} gap {gap_sum/nb:.4f} "
                      f"lr {lr_at(step):.2e} ({time.time()-t0:.0f}s)")
        hist.append({"epoch": ep + 1, "loss": tot / max(1, nb),
                     "mean_gap": gap_sum / max(1, nb)})
        print(f"epoch {ep+1}: loss {hist[-1]['loss']:.4f} gap {hist[-1]['mean_gap']:.4f}")

    # 导出（与双塔 train.py 的 export 相同逻辑）
    from sentence_transformers import SentenceTransformer, models
    for name, tower in (("doc_encoder", model.doc_tower), ("query_encoder", model.query_tower)):
        word = models.Transformer(BASE_TOWER_MODEL)
        word.max_seq_length = tower.max_len
        pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="cls")
        st = SentenceTransformer(modules=[word, pool])
        st[0].auto_model.load_state_dict(tower.transformer.state_dict())
        p = Path(out_dir) / name
        p.mkdir(parents=True, exist_ok=True)
        st.save(str(p))
        print(f"导出 {p}")

    hist_path = Path(out_dir) / "train_history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({"history": hist, "rows": len(rows), "epochs": epochs,
                   "batch_size": batch_size, "lr": lr, "max_len": max_len,
                   "data": data, "init_doc": init_doc, "init_query": init_query},
                  f, ensure_ascii=False, indent=2)
    return {"history": hist, "out_dir": out_dir}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(data_dir("phase7", "distill_data.jsonl")))
    ap.add_argument("--out", default=str(data_dir("models", "dual_distill")))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--device", default=None)
    ap.add_argument("--init-doc", default=None)
    ap.add_argument("--init-query", default=None)
    a = ap.parse_args()
    train(data=a.data, out_dir=a.out, epochs=a.epochs, batch_size=a.batch_size,
          lr=a.lr, max_len=a.max_len, device=a.device,
          init_doc=a.init_doc, init_query=a.init_query)
