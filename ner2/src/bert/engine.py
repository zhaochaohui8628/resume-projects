"""ner2 BERT 训练/推理引擎。

一条主链路：编码 → 训练（冻结策略 + 实体级 F1 早停）→ 预测（含逐 token 相对置信度
与 CRF 归一化解码得分）→ 字符级实体评估（严格 / 宽容 / 分类型）。
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ner2.src.bert.data_utils import (
    ID2TAG, IGNORE, NUM_TAGS, O_ID, TAGS, TAG2ID, encode,
)
from ner2.src.bert.decode import token_tags_to_char_entities
from ner2.src.bert.model import HEAD_CRF, HEAD_SOFTMAX, BertTokenModel
from ner2.src.eval.metrics import evaluate as eval_entities


# --------------------------------------------------------------------- 配置
@dataclass
class TrainCfg:
    head: str = HEAD_SOFTMAX
    freeze: str = "none"
    base_model: str = ""
    epochs: int = 15
    lr: float = 1e-3
    head_lr_mult: float = 1.0
    batch: int = 32
    # 本语料句长 p99≈53、max=66（实测），96 足以完全覆盖；旧默认 256 白烧 ~4 倍显存
    # （全量微调 101.7M 参数在 8 G 卡上会 OOM 崩溃）。
    max_len: int = 96
    accum: int = 1          # 梯度累积步数（小显存机上用 micro-batch × accum 顶等效 batch）
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 4
    seed: int = 42
    device: str = ""
    gamma: float = 2.0
    o_weight: float = 0.25
    crf_aux_focal: float = 0.0
    grad_clip: float = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------- 数据装载
class EncodedDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.items = [encode(r["text"], r.get("entities", []), tokenizer, max_len)
                      for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id: int):
    def _collate(batch):
        T = max(len(b["input_ids"]) for b in batch)
        ids, mask, lab = [], [], []
        for b in batch:
            n = T - len(b["input_ids"])
            ids.append(torch.tensor(b["input_ids"] + [pad_id] * n, dtype=torch.long))
            mask.append(torch.tensor(b["attention_mask"] + [0] * n, dtype=torch.long))
            lab.append(torch.tensor(b["labels"] + [IGNORE] * n, dtype=torch.long))
        return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(mask),
                "labels": torch.stack(lab)}
    return _collate


# ----------------------------------------------------------------- 模型装载
def build_model(base_dir: str, head: str, gamma: float = 2.0, o_weight: float = 0.25,
                crf_aux_focal: float = 0.0) -> BertTokenModel:
    from transformers import AutoModelForTokenClassification
    base = AutoModelForTokenClassification.from_pretrained(
        base_dir, num_labels=NUM_TAGS,
        id2label=dict(enumerate(TAGS)), label2id=dict(TAG2ID))
    return BertTokenModel(base, NUM_TAGS, head=head, gamma=gamma,
                          o_weight=o_weight, crf_aux_focal=crf_aux_focal)


def load_ckpt(ckpt_path: str, device: str = "cpu", override_head: str | None = None,
              fallback_base: str | None = None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = override_head or ck.get("head", HEAD_SOFTMAX)
    base_dir = ck.get("base_model") or fallback_base
    if not base_dir or not os.path.exists(base_dir):
        if not fallback_base:
            raise FileNotFoundError(f"ckpt 内 base_model 不可用：{base_dir}")
        base_dir = fallback_base
    model = build_model(base_dir, head, ck.get("gamma", 2.0), ck.get("o_weight", 0.25),
                        ck.get("crf_aux_focal", 0.0))
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    model.to(device)
    return model, ck, {"missing": sorted(missing), "unexpected": sorted(unexpected)}


# --------------------------------------------------------------------- 训练
def train(cfg: TrainCfg, tokenizer, train_rows, val_rows, save_dir: str,
          init_from: str | None = None) -> dict:
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.seed)
    model = build_model(cfg.base_model, cfg.head, cfg.gamma, cfg.o_weight, cfg.crf_aux_focal)
    load_info = None
    if init_from:
        ck = torch.load(init_from, map_location="cpu", weights_only=False)
        li = model.load_state_dict(ck["state"], strict=False)
        load_info = {"missing": sorted(li.missing_keys), "unexpected": sorted(li.unexpected_keys)}
        print(f"[train] 续训自 {init_from}（载入 {len(ck['state'])} 张量；"
              f"缺 {len(li.missing_keys)} / 多 {len(li.unexpected_keys)}）", flush=True)
    n_train = model.apply_freeze(cfg.freeze)
    model.to(device)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[train] head={cfg.head} freeze={cfg.freeze} device={device} "
          f"可训参数 {n_train:,} / 总 {n_all:,}（{n_train / n_all:.2%}）", flush=True)

    groups = model.param_groups()
    head_lr = cfg.lr * cfg.head_lr_mult
    pgs = []
    if groups["encoder"]:
        pgs.append({"params": groups["encoder"], "lr": cfg.lr})
    if groups["head"]:
        pgs.append({"params": groups["head"], "lr": head_lr})
    if groups["crf"]:
        pgs.append({"params": groups["crf"], "lr": head_lr})
    if not pgs:
        raise RuntimeError("没有可训练参数，检查 --freeze 与 --head 组合")
    opt = torch.optim.AdamW(pgs, lr=cfg.lr, weight_decay=cfg.weight_decay)

    pad_id = tokenizer.pad_token_id or 0
    ds_tr = EncodedDataset(train_rows, tokenizer, cfg.max_len)
    loader = DataLoader(ds_tr, batch_size=cfg.batch, shuffle=True,
                        collate_fn=make_collate(pad_id))
    accum = max(1, int(cfg.accum))
    steps_per_epoch = max(1, math.ceil(len(loader) / accum))
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup = int(total_steps * cfg.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warmup) if s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    print(f"[train] micro_batch={cfg.batch} accum={accum} "
          f"等效 batch={cfg.batch * accum} | steps/epoch={steps_per_epoch}", flush=True)

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "model.pt")

    def _dump(extra: dict) -> None:
        """改进即落盘（**原子写**）。**不留 CPU 副本**：主机内存紧张 + Windows WDDM 下
        `{k: v.cpu() for ...}` 这类整份深拷会触发硬崩溃；`torch.save` 逐张量转 CPU，
        峰值仅一个张量。先写 `.tmp` 再 `os.replace`——进程若在写入中被杀，也不会毁掉
        上一份完好的最佳权重（曾因中途崩溃留下 0 字节 model.pt）。"""
        tmp = ckpt_path + ".tmp"
        torch.save({"state": model.state_dict(), "tags": TAGS, "head": cfg.head,
                    "base_model": cfg.base_model, "max_len": cfg.max_len,
                    "gamma": cfg.gamma, "o_weight": cfg.o_weight,
                    "crf_aux_focal": cfg.crf_aux_focal, "freeze": cfg.freeze,
                    "trainable_params": n_train, "total_params": n_all,
                    "init_from": init_from, "load_info": load_info,
                    "cfg": asdict(cfg), **extra}, tmp)
        os.replace(tmp, ckpt_path)

    def _dump_history() -> None:
        with open(os.path.join(save_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump({"cfg": asdict(cfg), "history": hist,
                       "best_val_f1": best_f1, "best_epoch": best_ep,
                       "trainable_params": n_train, "total_params": n_all,
                       "init_from": init_from}, f, ensure_ascii=False, indent=1)

    hist, best_f1, best_ep, bad = [], -1.0, 0, 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lab = batch["labels"].to(device)
            out = model(ids, mask, lab)
            (out["loss"] / accum).backward()
            tot += float(out["loss"].item())
            nb += 1
            if (bi + 1) % accum == 0 or (bi + 1) == len(loader):
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.trainable_params(),
                                                   cfg.grad_clip, foreach=False)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        val = corpus_eval(model, tokenizer, val_rows, device=device, max_len=cfg.max_len,
                          batch=min(32, cfg.batch * 2))
        f1 = val["strict"]["f1"]
        hist.append({"epoch": ep, "loss": round(tot / max(1, nb), 4), "val": val})
        info = ""
        if device.type == "cuda":
            free_b, total_b = torch.cuda.mem_get_info()
            info = (f" | peakGPU={torch.cuda.max_memory_allocated() / 2 ** 30:.2f}G "
                    f"free={free_b / 2 ** 30:.2f}G/{total_b / 2 ** 30:.2f}G")
        print(f"[train] ep{ep}/{cfg.epochs} loss={tot / max(1, nb):.4f} "
              f"| val strict P={val['strict']['precision']:.4f} "
              f"R={val['strict']['recall']:.4f} F1={f1:.4f}{info}", flush=True)
        if f1 > best_f1 + 1e-6:
            best_f1, best_ep, bad = f1, ep, 0
            _dump({"best_val_f1": f1, "best_epoch": ep})       # 原子落盘，不留主机副本
        else:
            bad += 1
        _dump_history()                                        # 每轮可续看，崩溃不丢进度
        gc.collect()                                           # CRF 的 Python 循环产生大量小对象
        if bad >= cfg.patience:
            print(f"[train] 早停：连续 {cfg.patience} 轮验证 F1 未提升", flush=True)
            break

    if not os.path.exists(ckpt_path):                          # 理论上不会发生
        _dump({"best_val_f1": best_f1, "best_epoch": best_ep})
    _dump_history()
    print(f"[train] 保存 best(F1={best_f1:.4f} @ep{best_ep}) → {ckpt_path}", flush=True)
    return {"ckpt": ckpt_path, "best_val_f1": best_f1, "best_epoch": best_ep,
            "trainable_params": n_train, "total_params": n_all, "history": hist}


# --------------------------------------------------------------------- 推理
@torch.no_grad()
def predict_batch(model: BertTokenModel, tokenizer, texts: list[str], max_len: int = 256,
                  batch: int = 64, device="cpu") -> list[dict]:
    """批量预测。返回每句 {text, offsets, tags, confs, margins[, crf_*]}。

    逐 token 相对置信度 `margins = p_top1 − p_top2`（发射 softmax，两套方案通用）；
    CRF 场景额外给 `crf_score_per_token`（归一化解码对数似然）。
    """
    model.eval()
    pad_id = tokenizer.pad_token_id or 0
    out_rows: list[dict] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        encs = [encode(t, [], tokenizer, max_len) for t in chunk]
        coll = make_collate(pad_id)(encs)
        ids = coll["input_ids"].to(device)
        mask = coll["attention_mask"].to(device)
        logits = model.emissions(ids, mask)
        probs = F.softmax(logits.float(), dim=-1)
        top2 = probs.topk(2, dim=-1).values
        margins = (top2[..., 0] - top2[..., 1]).clamp(min=0.0)
        confs = top2[..., 0]
        crf_scores = crf_ok = None
        am = logits.argmax(-1)
        if model.head == HEAD_CRF:
            paths = model.crf.decode(logits, mask)
            lab = torch.full_like(ids, O_ID)
            for b, p in enumerate(paths):
                if p:
                    lab[b, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
            crf_scores = model.crf.path_logprob_per_token(logits, lab, mask)
            crf_ok = [list(p) == am[b, :len(p)].tolist() for b, p in enumerate(paths)]
        for b, e in enumerate(encs):
            L = len(e["input_ids"])
            if model.head == HEAD_CRF:
                names = [ID2TAG[int(x)] for x in paths[b]]
            else:
                names = [ID2TAG[int(x)] for x in am[b, :L].tolist()]
            row = {"text": chunk[b], "offsets": e["offsets"], "tags": names, "n_tokens": L,
                   "confs": [round(float(x), 6) for x in confs[b, :L].cpu().tolist()],
                   "margins": [round(float(x), 6) for x in margins[b, :L].cpu().tolist()]}
            if model.head == HEAD_CRF:
                s = float(crf_scores[b].item())
                row["crf_score_per_token"] = round(s, 6)
                row["crf_logprob"] = round(s * max(1, L), 6)
                row["crf_consistent"] = bool(crf_ok[b])
            out_rows.append(row)
    return out_rows


def predict_entities(model, tokenizer, rows: list[dict], max_len: int = 256,
                     batch: int = 64, device="cpu"):
    """返回 (字符级实体预测, 原始预测行) 二元组。"""
    pr = predict_batch(model, tokenizer, [r["text"] for r in rows], max_len, batch, device)
    ents = [token_tags_to_char_entities(r["offsets"], r["tags"]) for r in pr]
    return ents, pr


# --------------------------------------------------------------------- 评估
def _prf(c: Counter) -> dict:
    p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def corpus_eval(model, tokenizer, rows: list[dict], max_len: int = 256,
                batch: int = 64, device="cpu", return_preds: bool = False):
    """语料级实体 P/R/F1（严格 = 类型+字符 span 完全一致；宽容 = 类型一致 + span 重叠）。"""
    preds, _ = predict_entities(model, tokenizer, rows, max_len, batch, device)
    golds = [r.get("entities", []) for r in rows]
    agg = {"strict": Counter(), "lenient": Counter()}
    per_type = {k: defaultdict(Counter) for k in ("strict", "lenient")}
    for pe, g in zip(preds, golds):
        for key, relaxed in (("strict", False), ("lenient", True)):
            m = eval_entities(pe, g, relaxed=relaxed)
            for k in ("tp", "fp", "fn"):
                agg[key][k] += m[k]
            # 分类型：以 gold 类型为准累计 tp/fn；fp 归预测类型
            gt = Counter()
            for ge in g:
                gt[ge["type"]] += 1
            pt = Counter()
            for pe_ in pe:
                pt[pe_["type"]] += 1
            for t in set(gt) | set(pt):
                sub_p = [x for x in pe if x["type"] == t]
                sub_g = [x for x in g if x["type"] == t]
                mm = eval_entities(sub_p, sub_g, relaxed=relaxed)
                for k in ("tp", "fp", "fn"):
                    per_type[key][t][k] += mm[k]
    out = {k: _prf(v) for k, v in agg.items()}
    out["n_sent"] = len(rows)
    out["per_type"] = {k: {t: _prf(c) for t, c in v.items()} for k, v in per_type.items()}
    if return_preds:
        return out, preds, golds
    return out


# ------------------------------------------------------- 高置信度数据挖掘
def collect_high_confidence(pred_rows: list[dict], sp_margin: float = 0.25,
                            o_margin: float = 0.15, require_crf_consistent: bool = True,
                            keep_negatives: float = 0.0, seed: int = 42):
    """逐 token 相对置信度过滤（沿用旧 ner `self_train._decode_entities_margin` 口径）。

    判据：
      1) 实体 span 内**所有 token** 的 margin ≥ sp_margin；
      2) O 侧 token 的 margin **中位数** ≥ o_margin；
      3) CRF 方案额外要求 Viterbi 路径 == 发射 argmax 路径（解码稳定）；
      4) 至多保留 `keep_negatives` 比例的「无实体」高置信句（作为负样本）。
    返回 (保留句, 统计)。
    """
    kept, stat = [], Counter()
    neg_candidates = []
    for r in pred_rows:
        stat["total"] += 1
        if require_crf_consistent and r.get("crf_consistent") is False:
            stat["drop_inconsistent"] += 1
            continue
        ents = token_tags_to_char_entities(r["offsets"], r["tags"])
        margins = r["margins"]
        if ents:
            # token 级 margin：按 token 下标展开实体区间
            bad = False
            for i, tg in enumerate(r["tags"]):
                if tg.startswith(("B-", "I-", "E-")):
                    if i >= len(margins) or margins[i] < sp_margin:
                        bad = True
                        break
            if bad:
                stat["drop_span_margin"] += 1
                continue
        o_ms = sorted(margins[i] for i, tg in enumerate(r["tags"]) if tg == "O")
        if o_ms:
            med = o_ms[len(o_ms) // 2]
            if med < o_margin:
                stat["drop_o_margin"] += 1
                continue
        row = {"text": r["text"], "entities": ents,
               "min_margin": round(min(margins) if margins else 0.0, 4)}
        if r.get("crf_score_per_token") is not None:
            row["crf_score_per_token"] = r["crf_score_per_token"]
        if ents:
            kept.append(row)
            stat["kept_pos"] += 1
        else:
            neg_candidates.append(row)
    if keep_negatives > 0 and neg_candidates:
        n_neg = int(len(kept) * keep_negatives)
        n_neg = min(n_neg, len(neg_candidates))
        random.Random(seed).shuffle(neg_candidates)
        kept.extend(neg_candidates[:n_neg])
        stat["kept_neg"] = n_neg
    return kept, dict(stat)
