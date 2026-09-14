"""黄金集生成来源抽样：从 13661 父块分层挑出 1500+ 个"生成来源条款"。

## 目的

P7 黄金集（训练 1000+ / 评估 500）的撰写清单。每一条 = 一个**生成来源条款**，
后续由我逐条逆向写成自然语言问句（gold = 该条款，来源回溯，绝不从检索结果反挑）。

## 分层策略（配额 长尾 40% : 模糊 30% : 对抗 30%）

- **长尾**：条款数最少的规范（冷门/上海地标）、附录条款（part=附录）、正文低频条款。
  这类问句检验"冷门知识能否被召回"，是 P7 相对 val 的核心增量。
- **模糊**：正文常规条款中数值/条件密集者。撰写时把问句加工成"指代不明 / 信息不全 /
  口语化"（如"那个承台钢筋间距一般留多少？"），考验系统在信息不足时仍能定位条款。
- **边界对抗**：同规范相邻条款号配对（如 `4.2.3` vs `4.3.3`，检测对象不同但结构相似）、
  含"不少于/不应小于/不宜小于"数值卡点条款。撰写时制造"看起来对但适用条件不同"的干扰。

## 独立性铁律

- 排除 phase1 train/val 已用的 1829 个 clause_id（黄金集与训练/验证零重叠）；
- train / eval 两个子集互不重叠；
- 同一规范的条款尽量分散到不同 scenario，避免评估偏向某类。

## 产物

  data/phase7/source_list.jsonl   撰写清单（字段见 _rec 注释）
  data/phase7/source_stats.json   分层统计
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import data_dir  # noqa: E402

SEED = 20260911
N_TRAIN = 1000
N_EVAL = 500
# 配额：长尾 40% : 模糊 30% : 对抗 30%
SCEN_SHARE = {"长尾": 0.40, "模糊": 0.30, "对抗": 0.30}


def _rec(idx: int, c: dict, scenario: str, split: str,
         pair: str | None = None) -> dict:
    m = c["metadata"]
    return {
        "idx": idx,
        "clause_id": c["id"],
        "source": m["source"],
        "clause_no": m["clause_no"],
        "part": m.get("part", "正文"),
        "text_len": len(c["text"]),
        "scenario": scenario,
        "split": split,
        "pair_clause_id": pair,
    }


def main() -> None:
    rng = random.Random(SEED)
    used = {l.strip() for l in
            open(data_dir("phase7_used_phase1_ids.txt"), encoding="utf-8") if l.strip()}
    rows = [json.loads(l) for l in
            open(data_dir("corpus", "clauses.jsonl"), encoding="utf-8") if l.strip()]
    pool = [c for c in rows if c["id"] not in used]
    print(f"语料 {len(rows)} → 排除 phase1 已用 {len(used)} → 可用来源 {len(pool)}")

    # ---- 1. 分层候选 ----
    by_src: dict[str, list[dict]] = defaultdict(list)
    for c in pool:
        by_src[c["metadata"]["source"]].append(c)
    # 长尾：条款数最少的规范（冷门）
    src_counts = sorted(((len(v), s) for s, v in by_src.items()))
    cold_srcs = {s for _n, s in src_counts[:15]}
    appendix = [c for c in pool if c["metadata"].get("part") == "附录"]

    # 模糊：正文条款，数值/条件密集（数字出现 >=4 次，长度适中）
    def num_dense(c: dict) -> bool:
        t = c["text"]
        return (sum(1 for ch in t if ch.isdigit()) >= 4
                and 60 <= len(t) <= 1500)
    fuzzy_cand = [c for c in pool
                  if c["metadata"].get("part", "正文") == "正文" and num_dense(c)]

    # 对抗：同规范相邻条款号配对（同章同节、号相邻、长度适中）
    pair_cand: list[tuple[dict, dict]] = []
    for s, cs in by_src.items():
        by_no: dict[str, dict] = {}
        for c in cs:
            no = c["metadata"]["clause_no"]
            if no and no[0].isdigit():
                by_no[no] = c
        nos = sorted(by_no)
        for a, b in zip(nos, nos[1:]):
            if a.split(".")[:2] == b.split(".")[:2]:
                pa, pb = by_no[a], by_no[b]
                if 30 <= len(pa["text"]) <= 600 and 30 <= len(pb["text"]) <= 600:
                    pair_cand.append((pa, pb))
    print(f"分层候选：冷门规范 {len(cold_srcs)} 本 / 附录 {len(appendix)} / "
          f"模糊候选 {len(fuzzy_cand)} / 对抗配对 {len(pair_cand)}")

    # ---- 2. 分层取样（每类精确取到配额即停，不先全取再裁剪）----
    n_total = N_TRAIN + N_EVAL
    n_long = int(n_total * SCEN_SHARE["长尾"])
    n_fuzzy = int(n_total * SCEN_SHARE["模糊"])
    n_adv = n_total - n_long - n_fuzzy
    print(f"配额：长尾 {n_long} / 模糊 {n_fuzzy} / 对抗 {n_adv}")

    picked2: list[tuple[dict, str, str | None]] = []   # (clause, scenario, pair)
    used2: set[str] = set()

    def take2(c: dict, sc: str, pair: str | None = None) -> bool:
        if c["id"] in used2:
            return False
        used2.add(c["id"])
        picked2.append((c, sc, pair))
        return True

    # 长尾：附录优先但限 40% 且限长（附录大数值表格无法逆向成有意义的问句，排除），
    # 冷门规范正文（每本至少 2 条强制覆盖），其余随机补
    appendix_w = [c for c in appendix if len(c["text"]) <= 1500]
    n_tail_appendix = min(len(appendix_w), int(n_long * 0.40))
    rng.shuffle(appendix_w)
    for c in appendix_w[:n_tail_appendix]:
        take2(c, "长尾")
    # 冷门规范强制覆盖：每本至少 2 条正文条款
    cold_pool = {s: [c for c in by_src[s]
                     if c["metadata"].get("part", "正文") == "正文"
                     and len(c["text"]) <= 1500]
                 for s in cold_srcs}
    for s, cs in cold_pool.items():
        rng.shuffle(cs)
        need = min(2, len(cs))
        for c in cs[:need]:
            take2(c, "长尾")
    # 冷门规范剩余正文继续补
    for s, cs in cold_pool.items():
        for c in cs:
            if sum(1 for _c, sc, _p in picked2 if sc == "长尾") >= n_long:
                break
            take2(c, "长尾")
    rng.shuffle(pool)
    for c in pool:
        if sum(1 for _c, s, _p in picked2 if s == "长尾") >= n_long:
            break
        take2(c, "长尾")

    # 模糊：正文数值密集条款，精确取到配额
    rng.shuffle(fuzzy_cand)
    for c in fuzzy_cand:
        if sum(1 for _c, s, _p in picked2 if s == "模糊") >= n_fuzzy:
            break
        take2(c, "模糊")

    # 对抗：配对双取，精确取到配额
    rng.shuffle(pair_cand)
    for pa, pb in pair_cand:
        if sum(1 for _c, s, _p in picked2 if s == "对抗") >= n_adv:
            break
        take2(pa, "对抗", pb["id"])
        take2(pb, "对抗", pa["id"])

    # 汇总
    per_sc: dict[str, list[tuple[dict, str, str | None]]] = defaultdict(list)
    for c, sc, pair in picked2:
        per_sc[sc].append((c, sc, pair))
    final = picked2
    print(f"分层取样完成：长尾 {sum(1 for _c, s, _p in final if s=='长尾')} / "
          f"模糊 {sum(1 for _c, s, _p in final if s=='模糊')} / "
          f"对抗 {sum(1 for _c, s, _p in final if s=='对抗')} / 总 {len(final)}")

    # ---- 3. 分配 train/eval（互不重叠；eval 每场景按配额取，其余进 train）----
    by_sc: dict[str, list[tuple[dict, str, str | None]]] = defaultdict(list)
    for c, sc, pair in final:
        by_sc[sc].append((c, sc, pair))
    eval_final: list[tuple[dict, str, str | None]] = []
    train_final: list[tuple[dict, str, str | None]] = []
    for sc, arr in by_sc.items():
        cap = min(len(arr), int(N_EVAL * SCEN_SHARE[sc]))
        rng.shuffle(arr)
        eval_final.extend(arr[:cap])
        train_final.extend(arr[cap:])
    print(f"分配：eval {len(eval_final)} / train {len(train_final)}")

    # ---- 4. 落盘 ----
    out = data_dir("phase7")
    out.mkdir(parents=True, exist_ok=True)
    eval_ids = {(c["id"], sc) for c, sc, _p in eval_final}
    recs = []
    for i, (c, sc, pair) in enumerate(eval_final + train_final, 1):
        split = "eval" if (c["id"], sc) in eval_ids else "train"
        recs.append(_rec(i, c, sc, split, pair))
    with open(out / "source_list.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {
        "n_total": len(recs),
        "n_train": sum(1 for r in recs if r["split"] == "train"),
        "n_eval": sum(1 for r in recs if r["split"] == "eval"),
        "scenario": dict(Counter(r["scenario"] for r in recs)),
        "split_by_scenario": {
            s: dict(Counter(r["split"] for r in recs if r["scenario"] == s))
            for s in ("长尾", "模糊", "对抗")
        },
        "sources_covered": len({r["source"] for r in recs}),
        "appendix_covered": sum(1 for r in recs if r["part"] == "附录"),
        "seed": SEED,
    }
    with open(out / "source_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"落盘：{out}/source_list.jsonl（{len(recs)} 条）+ source_stats.json")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
