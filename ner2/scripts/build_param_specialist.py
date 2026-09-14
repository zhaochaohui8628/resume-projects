"""参数专项 Step2：**机械生成**符合「参数口径 v2」的专项数据集（正确性可审计）。

v2 变更（对齐黄金测试集 gold_test 实证口径，2026-09-11）：
- 只标 A（裸指标量名，来自已核验词表 core_nouns）+ SPEC（纯规格值/强度等级/独立数值+单位）；
- **不再合并「量名+紧邻数值」与「量名+比较词+值」**——gold_test 实证为分离标注
  （`安全间距≥500mm` → `安全间距` 与 `500mm` 两个实体；`桩长60m` 只标 `桩长`）；
- 泛化量名（荷载/厚度/高度/深度/速度/间距/直径/系数/半径/压力/温度/沉降/轴力/力矩/
  标高/埋深/强度等级…）移入 reject_words：命中只作负样本判定，不标实体；
- reject 屏蔽从「先屏蔽再匹配」改为「core/SPEC 优先命中，reject 仅参与负样本判定」，
  避免 reject 子串（如 荷载⊂施工荷载）误屏蔽 core 词。

泄漏防护：池中句子与 gold_test / gold_consensus / silver_* / gold_eval_v2 逐句 md5 命中即剔除，
并对 gold_test 做子串包含守卫。最后按句子 hash 稳定划分 train/dev（dev 供早停，绝不碰测试集）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import (  # noqa: E402
    GOLD_CONSENSUS, GOLD_TEST, GOLD_TRAIN_PORTION, LEGACY_GOLD, PHASE1, PHASE6,
    SILVER_TRAIN, SILVER_VAL,
)

CMP_MULTI = re.compile(r"(?:不大于|不小于|不超过|不少于|不得大于|不得小于|不得低于|不宜大于|不宜小于)")
NEG_PATTERNS = [
    re.compile(r"\d{4}\s?年\s?\d{1,2}\s?月"),
    re.compile(r"(附图|附件|照片|表|图)\s?[0-9一二三四五六七八九十]"),
    re.compile(r"第[一二三四五六七八九十百\d]+[章节条]"),
    re.compile(r"(GB|JGJ|DG|CJJ|JG|SH|T)/?[A-Z]*\s?\d{2,5}[-—]\d{4}"),
    re.compile(r"〔\d{4}〕\s?\d{1,3}\s?号"),
    re.compile(r"[A-Z]{1,2}\d{1,2}\s?轴"),
    re.compile(r"\d\s?F\b"),
    re.compile(r"\d{1,2}:\d{2}"),
]


def _h(t: str) -> str:
    return hashlib.md5(t.strip().encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default=os.path.join(PHASE6, "param_lexicon.json"))
    ap.add_argument("--pool", default=os.path.join(PHASE1, "unlabeled_sentences.jsonl"))
    ap.add_argument("--out", default=os.path.join(PHASE6, "param_specialist.jsonl"))
    ap.add_argument("--dev-ratio", type=float, default=0.12)
    ap.add_argument("--max-per-noun", type=int, default=22)
    ap.add_argument("--max-neg", type=int, default=400)
    ap.add_argument("--max-total", type=int, default=2200)
    args = ap.parse_args()

    lex = json.load(open(args.lexicon, encoding="utf-8"))
    nouns = sorted(set(lex["core_nouns"]), key=len, reverse=True)
    rejects = sorted(set(lex["reject_words"]), key=len, reverse=True)
    spec_pats = [(p["name"], re.compile(p["regex"])) for p in lex["spec_value_patterns"]]

    banned = set()
    for p in (GOLD_TEST, GOLD_CONSENSUS, GOLD_TRAIN_PORTION, SILVER_TRAIN, SILVER_VAL, LEGACY_GOLD):
        if os.path.exists(p):
            banned |= {_h(json.loads(l)["text"]) for l in open(p, encoding="utf-8") if l.strip()}
    gold_test_blob = ""
    if os.path.exists(GOLD_TEST):
        gold_test_blob = "".join(json.loads(l)["text"] for l in open(GOLD_TEST, encoding="utf-8")
                                 if l.strip())

    pool, seen = [], set()
    for l in open(args.pool, encoding="utf-8"):
        if not l.strip():
            continue
        t = json.loads(l)["text"].strip()
        h = _h(t)
        if h in banned or h in seen or len(t) < 6:
            continue
        if t in gold_test_blob:                       # 测试集子串守卫
            continue
        seen.add(h)
        pool.append(t)
    print(f"[build] 去泄漏后候选池 {len(pool)} 句")

    per_noun = Counter()
    samples, stats = [], Counter()
    for text in pool:
        covered = [False] * len(text)
        ents = []
        # ---- A：指标量名（v2：不合并紧邻值） ----
        for n in nouns:
            s = 0
            while True:
                i = text.find(n, s)
                if i < 0:
                    break
                e = i + len(n)
                if any(covered[i:e]):                  # 已被更长量名/SPEC 占用
                    s = i + 1
                    continue
                ents.append({"type": "参数", "start": i, "end": e, "_form": "A", "_noun": n})
                for k in range(i, e):
                    covered[k] = True
                s = i + 1
        # ---- SPEC：径规格 / 强度等级值 / 抗渗等级 ----
        for name, pat in spec_pats:
            for mm in pat.finditer(text):
                if any(covered[mm.start():mm.end()]):
                    continue
                ents.append({"type": "参数", "start": mm.start(), "end": mm.end(),
                             "_form": f"SPEC:{name}", "_noun": "<spec>"})
                for k in range(mm.start(), mm.end()):
                    covered[k] = True
        # ---- 去重叠（长优先） ----
        ents.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
        kept = []
        for e in ents:
            if any(e["start"] < k["end"] and k["start"] < e["end"] for k in kept):
                continue
            kept.append(e)
        kept.sort(key=lambda x: x["start"])

        if kept:
            for e in kept:
                per_noun[e["_noun"]] += 1
                stats[f"form_{e['_form']}"] += 1
                stats[f"noun_{e['_noun']}"] += 1
            if per_noun[kept[0]["_noun"]] > args.max_per_noun and len(kept) == 1:
                stats["skip_over_cap"] += 1
                continue
            samples.append({"text": text,
                            "entities": [{"type": "参数", "start": e["start"], "end": e["end"]}
                                         for e in kept],
                            "src": "param_specialist",
                            "forms": [e["_form"] for e in kept],
                            "nouns": [e["_noun"] for e in kept]})
            stats["pos"] += 1
        else:
            # 负样本判定：reject 词（泛化量名/仅构词成分）或硬负模式
            is_neg = (any(rw in text for rw in rejects)
                      or any(p.search(text) for p in NEG_PATTERNS))
            if is_neg and stats["neg"] < args.max_neg:
                samples.append({"text": text, "entities": [], "src": "param_specialist",
                                "forms": [], "nouns": []})
                stats["neg"] += 1
    if args.max_total and len(samples) > args.max_total:
        samples = samples[:args.max_total]

    train, dev = [], []
    for s in samples:
        (dev if int(_h(s["text"])[:8], 16) % 100 < args.dev_ratio * 100 else train).append(s)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    for path, rows in ((args.out, train),
                       (args.out.replace(".jsonl", "_dev.jsonl"), dev)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"pos": stats["pos"], "neg": stats["neg"], "train": len(train), "dev": len(dev),
              "entities": sum(len(r["entities"]) for r in samples),
              "forms": {k[5:]: v for k, v in stats.items() if k.startswith("form_")},
              "distinct_nouns": len(nouns),
              "noun_hits": len([k for k in stats if k.startswith("noun_")]),
              "top_nouns": Counter({k[5:]: v for k, v in stats.items()
                                    if k.startswith("noun_")}).most_common(15),
              "pool_after_leak_guard": len(pool)}
    with open(os.path.join(PHASE6, "param_specialist_stats.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("[build] " + json.dumps({k: v for k, v in report.items() if k != "top_nouns"},
                                 ensure_ascii=False))
    print("[build] top 量名：" + "、".join(f"{w}({n})" for w, n in report["top_nouns"]))


if __name__ == "__main__":
    main()
