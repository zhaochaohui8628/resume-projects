"""参数专项 Step2.5：**补全非「参数」类标签**，避免"部分标注"教坏模型。

问题：专项数据取自无标注池，只机械标了「参数」；句中出现的设备/工序/工程类型等若留空，
训练时会被当作负例 → 模型学会压制其他类型（学崩）。

做法：用当前最好模型 `s2_crf_stage2` 预测非「参数」实体，**只保留 span 内全部 token
的 margin ≥ --margin 的高置信结果**，与人工「参数」标签合并（「参数」优先，重叠丢弃模型输出）。
非「参数」标签仅作"别忘掉"的上下文约束，不作为本轮的评测对象。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.bert.engine import load_ckpt, predict_batch  # noqa: E402
from ner2.src.bert.decode import token_tags_to_char_entities  # noqa: E402
from ner2.src.common.paths import BASE_BERT, PHASE6  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ner2/models/s2_crf_stage2/model.pt")
    ap.add_argument("--in", dest="inp", default=os.path.join(PHASE6, "param_specialist.jsonl"))
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ck, _ = load_ckpt(args.ckpt, device=device, fallback_base=BASE_BERT)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_BERT)

    for suffix in ("", "_dev"):
        path = args.inp.replace(".jsonl", f"{suffix}.jsonl")
        if not os.path.exists(path):
            print(f"[fill] 跳过（缺 {path}）")
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        preds = predict_batch(model, tok, [r["text"] for r in rows],
                              max_len=ck.get("max_len", 96), batch=args.batch, device=device)
        stat = Counter()
        out = []
        for r, p in zip(rows, preds):
            mine = r["entities"]
            covered = [(e["start"], e["end"]) for e in mine]
            added = []
            for e in token_tags_to_char_entities(p["offsets"], p["tags"]):
                if e["type"] == "参数":
                    continue
                s, en = e["start"], e["end"]
                if any(s < y and x < en for x, y in covered):
                    continue
                # span 内全部 token 的 margin 必须达标
                idx = [i for i, (a, b) in enumerate(p["offsets"]) if a >= s and b <= en and b > a]
                if not idx:
                    continue
                if min(p["margins"][i] for i in idx) < args.margin:
                    stat["drop_low_margin"] += 1
                    continue
                added.append({"type": e["type"], "start": s, "end": en})
                covered.append((s, en))
                stat[f"add_{e['type']}"] += 1
            ents = sorted(mine + added, key=lambda x: x["start"])
            row = {"text": r["text"], "entities": ents, "src": "param_specialist_full",
                   "forms": r.get("forms", []), "nouns": r.get("nouns", [])}
            out.append(row)
        outp = args.inp.replace(".jsonl", f"_full{suffix}.jsonl")
        with open(outp, "w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[fill] {os.path.basename(path)} {len(rows)} 句 → {os.path.basename(outp)}："
              + json.dumps(dict(stat), ensure_ascii=False))


if __name__ == "__main__":
    main()
