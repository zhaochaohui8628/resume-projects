"""双塔训练：InfoNCE 微调 → 导出 doc/query 两塔为 SentenceTransformer 目录。

导出后 `src/retrieval/hybrid.EncoderPair(doc, query)` 可直接加载；doc 塔用于重建索引，
query 塔用于在线检索。
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
from src.train.dual_tower.loss import info_nce  # noqa: E402
from src.train.dual_tower.model import build_dual_tower  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def make_batches(rows: list[dict], tok, batch_size: int, max_len: int):
    idx = list(range(len(rows)))
    random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [rows[j] for j in idx[i:i + batch_size]]
        q = tok([r["query"] for r in chunk], padding=True, truncation=True,
                max_length=max_len, return_tensors="pt")
        pos = tok([r["pos_text"] for r in chunk], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        negs: list[str] = []
        for r in chunk:
            negs.extend(r["neg_texts"])
        k = len(chunk[0]["neg_texts"]) if chunk else 0
        neg = (tok(negs, padding=True, truncation=True, max_length=max_len,
                   return_tensors="pt") if negs else None)
        yield q, pos, neg, k, len(chunk)


def train(*, data: str, out_dir: str, epochs: int = 10, batch_size: int = 16,
          lr: float = 2e-5, temperature: float = 0.05, max_len: int = 224,
          warmup_ratio: float = 0.1, seed: int = 42, tie_weights: bool = False,
          log_every: int = 5, device: str | None = None,
          init_doc: str | None = None, init_query: str | None = None) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_jsonl(Path(data))
    # 继续训练：直接以检查点目录为 base 建塔（见 model.DualTower 的注释——
    # 不要改成"建基座塔再 load_state_dict 覆盖"，Windows 上会静默 segfault）
    model = build_dual_tower(BASE_TOWER_MODEL, max_len=max_len, tie_weights=tie_weights,
                             doc_base=init_doc, query_base=init_query)
    if init_doc or init_query:
        print(f"继续训练模式：doc ← {init_doc or BASE_TOWER_MODEL}"
              f" | query ← {init_query or BASE_TOWER_MODEL}")
    model.to(device)
    tok = model.tokenizer
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, math.ceil(len(rows) / batch_size))
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return lr * max(0.02, 1 - prog)

    print(f"训练样本 {len(rows)}，batch {batch_size}，epochs {epochs}，"
          f"steps {total_steps}，可训参数 {n_train/1e6:.2f}M/{n_all/1e6:.2f}M")
    history = []
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot, acc_sum, nb = 0.0, 0.0, 0
        for q, pos, neg, k, b in make_batches(rows, tok, batch_size, max_len):
            q = {kk: vv.to(device) for kk, vv in q.items()}
            pos = {kk: vv.to(device) for kk, vv in pos.items()}
            if neg is not None:
                neg = {kk: vv.to(device) for kk, vv in neg.items()}
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            qe = model.encode_query(**q)
            de = model.encode_doc(**pos)
            labels = torch.arange(b, device=device)
            docs = de
            if neg is not None:
                ne = model.encode_doc(**neg)
                docs = torch.cat([de, ne], dim=0)
            loss = info_nce(qe, docs, labels, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            with torch.no_grad():
                pred = (qe @ docs.t()).argmax(dim=-1)
                acc_sum += float((pred == labels).float().mean())
            tot += float(loss)
            nb += 1
            step += 1
            if step % log_every == 0:
                print(f"  step {step}/{total_steps} loss {tot/nb:.4f} "
                      f"batch_acc {acc_sum/nb:.3f} lr {lr_at(step):.2e} "
                      f"({time.time()-t0:.0f}s)")
        rec = {"epoch": ep + 1, "loss": tot / max(1, nb), "batch_acc": acc_sum / max(1, nb)}
        history.append(rec)
        print(f"epoch {ep+1}: loss {rec['loss']:.4f} batch_acc {rec['batch_acc']:.3f}")

    export(model, out_dir)
    hist_path = Path(out_dir) / "train_history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({"history": history, "rows": len(rows), "epochs": epochs,
                   "batch_size": batch_size, "lr": lr, "temperature": temperature,
                   "max_len": max_len, "tie_weights": tie_weights,
                   "data": data, "init_doc": init_doc, "init_query": init_query},
                  f, ensure_ascii=False, indent=2)
    return {"history": history, "out_dir": out_dir}


def export(model, out_dir: str) -> None:
    """把两塔导出成可由 SentenceTransformer 直接加载的目录。"""
    from sentence_transformers import SentenceTransformer, models

    base = BASE_TOWER_MODEL
    for name, tower in (("doc_encoder", model.doc_tower), ("query_encoder", model.query_tower)):
        word = models.Transformer(base)
        word.max_seq_length = model.doc_tower.max_len
        pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="cls")
        st = SentenceTransformer(modules=[word, pool])
        st[0].auto_model.load_state_dict(tower.transformer.state_dict())
        p = Path(out_dir) / name
        p.mkdir(parents=True, exist_ok=True)
        st.save(str(p))
        print(f"导出 {p}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(data_dir("phase1", "train.jsonl")))
    ap.add_argument("--out", default=str(data_dir("models", "dual_p1")),
                    help="输出目录。dual_p1 = P1 冷启动塔（默认产物名，需训练生成）；"
                         "现行线上塔为 dual_mix / dual_gold，勿直接覆盖")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=224)
    ap.add_argument("--tie-weights", action="store_true")
    ap.add_argument("--init-doc", default=None,
                    help="doc 塔初始权重目录（继续训练；不给则从基座 bge 从头训）")
    ap.add_argument("--init-query", default=None, help="query 塔初始权重目录")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    train(data=a.data, out_dir=a.out, epochs=a.epochs, batch_size=a.batch_size,
          lr=a.lr, temperature=a.temperature, max_len=a.max_len,
          tie_weights=a.tie_weights, device=a.device,
          init_doc=a.init_doc, init_query=a.init_query)
