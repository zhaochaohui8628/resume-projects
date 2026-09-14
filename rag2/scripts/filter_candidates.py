"""Phase 2：把 12660 条难负例候选收敛成两批"可逐条判定"的清洗清单。

背景
----
`mine_negatives.py` 的产出是**按 query 分组的嵌套结构**（一行 = 一条 query + 它自己的
topN 候选），适合挖掘、不适合逐条判定：
  - 一条 query 的 30 个候选里，rank<5 的多是"一眼不相关"的易负例，判定价值低；
  - 同一本规范内的干扰项（限定条件/参数口径冲突）才是清洗的富矿，却和跨源候选混在一起；
  - 没有稳定可引用的 pair 主键，判定结果无法与候选回接。

本脚本做三件事：
  1. **对齐 qid**：按行序把 `phase1/train.jsonl` 与 `candidates.jsonl` 对齐（逐行校验
     query + pos_id 一致，不一致直接报错退出），给每条 query 分配稳定主键 `q0001..`；
  2. **拉平成 pair 级清单**：一行 = 一个 (query, 候选) 对，带齐判定所需的全部上下文
     （两边原文、来源、两路名次、融合分）+ 规范效力提示；
  3. **按优先级切两批**：
       judge_A_samesource.jsonl  同源候选（候选与正样本同一本规范）——第一优先级，
                                 限定条件/数值口径冲突几乎都在这里；
       judge_B_band.jsonl        跨源且 best_rank ∈ [5,20]——去掉 rank<5 的"一眼易负例"
                                 与 >20 的尾部，信息密度最高的一段。
     两批互斥且并集有界，避免同一对被判两次。

规范效力提示来自 `agent/data/rules/abolished_clauses.json`（住建部公告口径）：
命中的 pair 会带 `validity` 字段，清洗时优先按"①规范效力"直接判 true_neg，
不必再逐条比对文本。

用法：
  cd rag2
  python scripts/filter_candidates.py
  # 产物
  #   data/phase2/judge_A_samesource.jsonl
  #   data/phase2/judge_B_band.jsonl
  #   data/phase2/judge_stats.json

判定结果（下一步由人逐条写）落盘格式：
  {"qid":"q0001","cand_id":"...","verdict":"true_neg|false_neg","reason":"..."}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import PROJECT_ROOT, data_dir  # noqa: E402

# 跨源候选的 best_rank 频段（PROGRESS 第 5 节：去掉一眼易负例与尾部）
BAND_LO, BAND_HI = 5, 20

# 巨块阈值：全语料条款长度 p50=79 / p90=292 / p99=1256 字符。超过 1200 字的"条款"
# 几乎都是切分断链把整节乃至整本吞进来的伪影（最长的一条是 12.2 万字符 = 一整本规范）。
# 它们做 CE 负例有害无益：max_len 256 会把它截成一个随机前缀，模型只会学到
# "长文本=负例"这种伪特征，还让清洗成本暴涨（252 对占掉 A 批 76.9% 的文字量）。
MAX_CAND_LEN = 1200


def _norm_std(std_no: str) -> str:
    """规范编号归一：`JGJ 130-2011` / `JGJ130-2011` / `jgj130-2011` → `JGJ130-2011`。

    只吃掉编号内部的空格；`DG/TJ08-61-2018` 这类省市地标原样保留（去空格后比较）。
    """
    s = (std_no or "").strip().upper()
    s = re.sub(r"\s+", "", s)
    s = s.replace("—", "-").replace("–", "-").replace("－", "-").replace("‐", "-")
    return s


def _std_of_source(source: str) -> str:
    """从 clause_id 的来源段里取规范编号：`JGJ33-2012_建筑机械使用安全技术规程` → `JGJ33-2012`。"""
    head = (source or "").split("_", 1)[0].strip()
    return _norm_std(head)


def load_validity(path: Path) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    """读废止规则库 → (整本废止表, 条文废止表)。返回空表表示无此文件（不影响主流程）。"""
    if not path.exists():
        return {}, {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    full: dict[str, dict] = {}
    for it in raw.get("full_doc_abolished", []):
        full[_norm_std(it.get("std_no", ""))] = it
    clause: dict[tuple[str, str], dict] = {}
    for grp in raw.get("clause_abolished", []):
        by = grp.get("by_std_no", "")
        for it in grp.get("items", []):
            code = _norm_std(it.get("std_no", ""))
            for cno in it.get("clauses", []):
                clause[(code, str(cno).strip())] = {
                    "superseded_by": by,
                    "by_std_name": grp.get("by_std_name", ""),
                    "effective_date": grp.get("effective_date", ""),
                }
    return full, clause


def validity_hint(source: str, clause_no: str,
                  full: dict[str, dict], clause: dict[tuple[str, str], dict]) -> dict | None:
    code = _std_of_source(source)
    if code in full:
        it = full[code]
        return {"kind": "full_doc", "std_no": it.get("std_no", ""),
                "replaced_by": it.get("replaced_by", ""),
                "effective_date": it.get("effective_date", ""),
                "by_document": it.get("by_document", "")}
    hit = clause.get((code, str(clause_no).strip()))
    if hit:
        return {"kind": "clause", "std_no": code, **hit}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(data_dir("phase1", "train.jsonl")))
    ap.add_argument("--candidates", default=str(data_dir("phase2", "candidates.jsonl")))
    ap.add_argument("--rules", default=str(PROJECT_ROOT / "agent" / "data" / "rules"
                                          / "abolished_clauses.json"))
    ap.add_argument("--out-a", default=str(data_dir("phase2", "judge_A_samesource.jsonl")))
    ap.add_argument("--out-b", default=str(data_dir("phase2", "judge_B_band.jsonl")))
    ap.add_argument("--out-c", default=str(data_dir("phase2", "judge_C_toolong.jsonl")))
    ap.add_argument("--stats", default=str(data_dir("phase2", "judge_stats.json")))
    ap.add_argument("--band-lo", type=int, default=BAND_LO)
    ap.add_argument("--band-hi", type=int, default=BAND_HI)
    ap.add_argument("--max-cand-len", type=int, default=MAX_CAND_LEN,
                    help="候选文本超过该长度视为「巨块」（切分断链吞掉整节/整本的产物），"
                         "单独分流不进 CE 负例池")
    a = ap.parse_args()

    # ---------- 1) 逐行对齐 qid ----------
    qrows = [json.loads(l) for l in open(a.queries, encoding="utf-8") if l.strip()]
    crows = [json.loads(l) for l in open(a.candidates, encoding="utf-8") if l.strip()]
    if len(qrows) != len(crows):
        raise SystemExit(f"[FATAL] 行数不一致：queries {len(qrows)} vs candidates {len(crows)}"
                         "——candidates.jsonl 可能不是由当前 train.jsonl 挖出来的")
    for i, (q, c) in enumerate(zip(qrows, crows)):
        if q["query"] != c["query"] or q["pos_id"] != c["pos_id"]:
            raise SystemExit(f"[FATAL] 第 {i+1} 行对不齐：\n  train: {q['query'][:40]} | {q['pos_id']}\n"
                             f"  cand : {c['query'][:40]} | {c['pos_id']}")

    full_abol, clause_abol = load_validity(Path(a.rules))
    n_rules = len(full_abol) + len(clause_abol)

    # ---------- 2) 拉平成 pair 级 ----------
    rows_a: list[dict] = []
    rows_b: list[dict] = []
    rows_c: list[dict] = []
    seen: set[tuple[str, str]] = set()
    n_dup = 0
    for i, c in enumerate(crows, 1):
        qid = f"q{i:04d}"
        pos_source = c.get("pos_source", "") or c["pos_id"].split("::", 1)[0]
        for cd in c.get("candidates", []):
            key = (qid, cd["cand_id"])
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            best = min([x for x in (cd.get("bm25_rank"), cd.get("dense_rank")) if x],
                       default=10 ** 9)
            row = {
                # --- PROGRESS 指定的必备字段 ---
                "qid": qid,
                "query": c["query"],
                "pos_id": c["pos_id"],
                "pos_text": c.get("pos_text", ""),
                "cand_id": cd["cand_id"],
                "cand_text": cd.get("text", ""),
                "bm25_rank": cd.get("bm25_rank"),
                "dense_rank": cd.get("dense_rank"),
                "fused": cd.get("fused"),
                # --- 判定所需的附加上下文 ---
                "pair_key": f"{qid}|{cd['cand_id']}",
                "pos_source": pos_source,
                "cand_source": cd.get("source", ""),
                "cand_clause_no": cd.get("clause_no", ""),
                "dense_cos": cd.get("dense_cos"),
                "best_rank": None if best == 10 ** 9 else best,
                "same_source": cd.get("source", "") == pos_source,
                "recall_via": ("both" if cd.get("bm25_rank") and cd.get("dense_rank")
                               else "bm25" if cd.get("bm25_rank") else "dense"),
                "cand_len": len(cd.get("text", "")),
            }
            v = validity_hint(cd.get("source", ""), cd.get("clause_no", ""), full_abol, clause_abol)
            if v:
                row["validity"] = v
            if row["same_source"]:
                bucket_a = True
            elif best != 10 ** 9 and a.band_lo <= best <= a.band_hi:
                bucket_a = False
            else:
                continue
            # 巨块单独分流：既不做 CE 负例，也不占我的判定预算
            if row["cand_len"] > a.max_cand_len:
                row["exclude_reason"] = (f"cand_len={row['cand_len']}>{a.max_cand_len}"
                                        "，疑似切分断链的整节/整本块")
                rows_c.append(row)
            elif bucket_a:
                rows_a.append(row)
            else:
                rows_b.append(row)

    # 组内按 (qid, 融合分降序) 稳定排序，便于分批判定与断点续做
    for rows in (rows_a, rows_b, rows_c):
        rows.sort(key=lambda r: (r["qid"], -(r["fused"] or 0.0), r["cand_id"]))

    for path, rows in ((Path(a.out_a), rows_a), (Path(a.out_b), rows_b),
                       (Path(a.out_c), rows_c)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- 3) 统计 ----------
    def _stat(rows: list[dict]) -> dict:
        return {
            "pairs": len(rows),
            "queries": len({r["qid"] for r in rows}),
            "via": dict(Counter(r["recall_via"] for r in rows)),
            "with_validity_hint": sum(1 for r in rows if "validity" in r),
            "best_rank_p50": _pct([r["best_rank"] for r in rows if r["best_rank"]], 50),
            "cand_len_p50": _pct([r["cand_len"] for r in rows], 50),
            "cand_len_p90": _pct([r["cand_len"] for r in rows], 90),
            "total_chars": sum(r["cand_len"] for r in rows),
            "top_cand_sources": Counter(r["cand_source"] for r in rows).most_common(8),
            "pairs_per_query_p50": _pct(list(Counter(r["qid"] for r in rows).values()), 50),
        }

    stats = {
        "queries": len(qrows),
        "candidates_total": len(seen),
        "dup_pairs_dropped": n_dup,
        "band": [a.band_lo, a.band_hi],
        "max_cand_len": a.max_cand_len,
        "rules_loaded": n_rules,
        "A_samesource": _stat(rows_a),
        "B_band": _stat(rows_b),
        "C_toolong_excluded": _stat(rows_c),
        "batch_plan": {
            "batch_size": 600,
            "A_batches": -(-len(rows_a) // 600),
            "B_batches": -(-len(rows_b) // 600),
        },
    }
    Path(a.stats).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    clean = len(rows_a) + len(rows_b)
    print(f"对齐 {len(qrows)} 条 query（qid q0001..q{len(qrows):04d}），候选对 {len(seen)}"
          f"（丢弃重复 {n_dup}）")
    print(f"A 同源      : {len(rows_a):6d} 对 / {stats['A_samesource']['queries']:3d} 条 query"
          f"  → {a.out_a}")
    print(f"B 跨源 band : {len(rows_b):6d} 对 / {stats['B_band']['queries']:3d} 条 query"
          f"  best_rank∈[{a.band_lo},{a.band_hi}]  → {a.out_b}")
    print(f"C 巨块分流  : {len(rows_c):6d} 对（cand_len>{a.max_cand_len}）"
          f"  → {a.out_c}")
    print(f"合计待判定  : {clean:6d} 对"
          f"（{stats['batch_plan']['A_batches'] + stats['batch_plan']['B_batches']} 批 ×600）")
    print(f"效力提示命中: A {stats['A_samesource']['with_validity_hint']} 对 / "
          f"B {stats['B_band']['with_validity_hint']} 对（规则库 {n_rules} 条）")
    tot_a0 = stats['A_samesource']['total_chars'] + stats['B_band']['total_chars'] \
        + stats['C_toolong_excluded']['total_chars']
    kept = stats['A_samesource']['total_chars'] + stats['B_band']['total_chars']
    print(f"字符量      : 保留 {kept/1e6:.2f}M / 全部候选 {tot_a0/1e6:.2f}M"
          f"（巨块分流省掉 {100*(1-kept/max(1,tot_a0)):.0f}% 的判定阅读量）")
    print(f"统计 → {a.stats}")


def _pct(xs: list[int], q: int) -> int | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q / 100))]


if __name__ == "__main__":
    main()
