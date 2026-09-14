"""教师 CE 多版本评估：用独立 dev + 黄金集选最优，防过拟合。

维度 1（主）：ce_dev_v2 381 条（未参与训练，query 与 train 零重叠）pair_acc
维度 2（辅）：gold_eval_clean 490 条 (query, gold_clause) 匹配打分均值/中位
"""
import json
import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEV = "data/phase2/ce_dev_v2.jsonl"
GOLD = "data/phase7/gold_eval_clean.jsonl"
CORPUS = "data/corpus/clauses.jsonl"
MODELS = [f"data/models/cross_v2_ep{ep}" for ep in [1, 2, 3, 4]]


def load_dev():
    rows = [json.loads(l) for l in open(DEV, encoding="utf-8") if l.strip()]
    dev_q = set()
    for l in open("data/phase2/ce_dev_v2.jsonl", encoding="utf-8"):
        if l.strip():
            dev_q.add(json.loads(l)["query"])
    return rows, dev_q


def load_gold():
    clauses = {}
    for l in open(CORPUS, encoding="utf-8"):
        if l.strip():
            c = json.loads(l)
            clauses[c["id"]] = c["text"]
    rows = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    return rows, clauses


def score_pairs(model, tok, pairs, batch=16):
    """pairs: list of (text_a, text_b) -> sigmoid logits list"""
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            chunk = pairs[i : i + batch]
            enc = tok(
                [p[0] for p in chunk], [p[1] for p in chunk],
                padding=True, truncation=True, max_length=128, return_tensors="pt",
            ).to("cuda")
            logits = model(**enc).logits.squeeze(-1)
            out.extend(float(x) for x in torch.sigmoid(logits))
    return out


def main():
    dev_rows, _ = load_dev()
    gold_rows, clauses = load_gold()
    gold_pairs = [(r["query"], clauses[r["clause_id"]]) for r in gold_rows]
    print(f"dev 判别样本: {len(dev_rows)} | gold 匹配样本: {len(gold_pairs)}")
    print("=" * 78)

    results = []
    for mp in MODELS:
        tok = AutoTokenizer.from_pretrained(mp)
        model = AutoModelForSequenceClassification.from_pretrained(mp).to("cuda")
        # 维度1: dev 判别
        pos = [(r["query"], r["doc"]) for r in dev_rows if r["label"] == 1]
        neg = [(r["query"], r["doc"]) for r in dev_rows if r["label"] == 0]
        sp = score_pairs(model, tok, pos)
        sn = score_pairs(model, tok, neg)
        tp = sum(1 for s in sp if s >= 0.5) / max(len(sp), 1)
        fp = sum(1 for s in sn if s >= 0.5) / max(len(sn), 1)
        acc = (sum(1 for s in sp if s >= 0.5) + sum(1 for s in sn if s < 0.5)) / len(dev_rows)
        # 维度2: gold 匹配打分
        sg = score_pairs(model, tok, gold_pairs)
        g_mean = sum(sg) / len(sg)
        g_med = sorted(sg)[len(sg) // 2]
        results.append((mp, acc, tp, fp, g_mean, g_med))
        print(f"{mp.split('/')[-1]:14s} dev_acc {acc:.4f} | gold_sig {g_mean:.4f} | "
              f"dev 正例命中 {tp:.3f} 负例误报 {fp:.3f} | gold p50 {g_med:.4f}")

    print("=" * 78)
    best = max(results, key=lambda x: x[1])
    print(f"最优: {best[0]} (dev_acc {best[1]:.4f})")
    with open("data/phase7/ce_version_eval.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"model": m, "dev_acc": a, "dev_pos_hit": t, "dev_neg_fp": f,
              "gold_sig_mean": g, "gold_sig_p50": med}
             for m, a, t, f, g, med in results],
            f, ensure_ascii=False, indent=2,
        )


if __name__ == "__main__":
    main()
