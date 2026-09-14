"""用已清洗判定对生成「蒸馏数据集」：query + pos/neg 文本 + 教师(CE)分数。

这是 P4 蒸馏的关键数据源：
- 蒸馏用 margin ranking loss 需要 **每个样本一个 margin**（用户要求：不设固定 margin，
  margin = 教师模型对 pos 与 neg 打分的差），所以必须由 CE 预打分。
- 输出 `distill_data.jsonl`：`{qid, query, pos_text, neg_text, s_pos, s_neg, margin}`
  其中 `margin = sigmoid(s_pos) - sigmoid(s_neg)`（将 CE logit 压到 [0,1]，
  与学生余弦差异同量级；sigmoid 差在语义上比原始 logit 差更稳）。

用法：
  python scripts/gen_distill_data.py \
    --judgments data/phase2/judgments.jsonl \
    --ce data/models/cross_p1 \
    --out data/phase7/distill_data.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import data_dir  # noqa: E402
from scripts.build_ce_data import load_pairs  # noqa: E402


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", default=str(data_dir("phase2", "judgments.jsonl")))
    ap.add_argument("--judge-files", nargs="*", default=[
        str(data_dir("phase2", "judge_A_samesource.jsonl")),
        str(data_dir("phase2", "judge_B_band.jsonl")),
    ])
    ap.add_argument("--ce", default=str(data_dir("models", "cross_p1")))
    ap.add_argument("--out", default=str(data_dir("phase7", "distill_data.jsonl")))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="softmax/sigmoid 温度缩放。教师过拟合后 sigmoid 差会饱和到 1.0，"
                         "蒸馏时用 T>1（如 3）恢复梯度区分度（T=3: sigmoid(s/T) 差）。")
    a = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    jud = [json.loads(l) for l in open(a.judgments, encoding="utf-8") if l.strip()]
    pool = load_pairs(a.judge_files)
    # 判定 difficulty 映射：(qid, cand_id) -> difficulty
    diff_map = {(j["qid"], j["cand_id"]): j.get("difficulty", "") for j in jud}

    # ⚠️ gold 泄漏过滤：负例/正例条款若本身是黄金评估集的正确答案，
    # 蒸馏 margin ranking 会"教模型打压 gold 答案"，直接损害评估 MRR。
    # 使用深度复核后的 gold_eval_clean（已剔除 10 条串规范假阳性）。
    gold_clauses: set[str] = set()
    gold_path = data_dir("phase7", "gold_eval_clean.jsonl")
    if Path(gold_path).exists():
        for line in open(gold_path, encoding="utf-8"):
            if line.strip():
                gold_clauses.add(json.loads(line)["clause_id"])

    recs = []
    n_dropped_gold = 0
    for j in jud:
        if j["verdict"] != "true_neg":
            continue
        r = pool.get((j["qid"], j["cand_id"]))
        if r:
            r = dict(r)                      # 拷贝，避免污染 pool
            r["difficulty"] = diff_map.get((j["qid"], j["cand_id"]), "")
            # 剔除 neg 或 pos 命中 gold 的样本
            if gold_clauses and (r.get("cand_id") in gold_clauses or r.get("pos_id") in gold_clauses):
                n_dropped_gold += 1
                continue
            recs.append(r)

    tok = AutoTokenizer.from_pretrained(a.ce)
    model = AutoModelForSequenceClassification.from_pretrained(a.ce).to(a.device)
    model.eval()

    import torch
    T = a.temperature
    out = []
    with torch.no_grad():
        for i, r in enumerate(recs, 1):
            q, p, n = r["query"], r["pos_text"], r["cand_text"]
            enc = tok([q, q], [p, n], padding=True, truncation=True,
                      max_length=a.max_len, return_tensors="pt").to(a.device)
            logits = model(**enc).logits.squeeze(-1)
            s_pos, s_neg = float(logits[0]), float(logits[1])
            out.append({
                "qid": r["qid"],
                "query": q,
                "pos_text": p,
                "neg_text": n,
                "s_pos": round(s_pos, 4),
                "s_neg": round(s_neg, 4),
                "margin": round(sigmoid(s_pos / T) - sigmoid(s_neg / T), 4),
                "difficulty": r.get("difficulty", ""),
            })
            if i % 50 == 0:
                print(f"  打分 {i}/{len(recs)}（margin 均值 "
                      f"{sum(o['margin'] for o in out)/len(out):.4f}）")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    margins = [o["margin"] for o in out]
    print(f"输出 {len(out)} 条 -> {a.out}（gold 泄漏剔除 {n_dropped_gold}）")
    print(f"margin 分布：min {min(margins):.4f} / p50 {sorted(margins)[len(margins)//2]:.4f} / "
          f"max {max(margins):.4f} / 负值 {sum(1 for m in margins if m <= 0)}（教师认为候选胜出正例，应剔除）")


if __name__ == "__main__":
    main()