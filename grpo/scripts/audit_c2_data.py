"""C2/C3 危大阈值数据体检（只读诊断，不改数据）。

检查项：
  A 区间倒挂  每维三档采样区间的 lo/hi 是否 lo>hi
  B 档位越界  采样值是否落入与标签不符的档位（非危大/危大/超规模）
  C 哨兵误用  h=0.1（"本身即危大，无数值线"）是否被当成有数值线使用
  D 数值荒谬  采样值是否超出工程常识区间
  E C3 阈值   是否对"无数值危大线"的维度提问危大阈值

用法：
  python grpo/scripts/audit_c2_data.py
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GRPO = os.path.dirname(HERE)
ROOT = os.path.dirname(GRPO)
for p in (GRPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import gen_rl_data as G  # noqa: E402

# 工程常识区间（仅用于提示，不参与判分）: 维度参数名 -> (最小值, 最大值)
PLAUSIBLE = {
    "基坑开挖深度": (0.5, 40),
    "模板支架搭设高度": (1, 30),
    "模板支架跨度": (2, 40),
    "施工总荷载": (1, 60),
    "集中线荷载": (1, 80),
    "落地式钢管脚手架搭设高度": (3, 80),
    "悬挑式脚手架分段架体搭设高度": (3, 80),
    "附着式升降脚手架提升高度": (10, 400),
    "单件起吊重量": (1, 2000),
    "幕墙工程施工高度": (5, 400),
    "人工挖孔桩开挖深度": (3, 40),
    "钢结构安装跨度": (6, 120),
}

SENTINEL_H = 0.1      # "本身即危大，无数值危大线"哨兵


def band(val, h, s):
    if s is not None and val >= s:
        return "超规模"
    if h is not None and val >= h:
        return "危大"
    return "非危大"


def audit_dims():
    print("=" * 96)
    print("A. DANGER_DIMS 逐维体检（h=危大线 / s=超规模线；h=0.1 为「本身即危大」哨兵）")
    print("=" * 96)
    print(f"{'类别':<8}{'参数':<24}{'h':>7}{'s':>7}{'s/h':>7}  {'非危大档':>16}{'危大档':>18}"
          f"{'超规模档':>16}  状态")
    print("-" * 96)
    bad = []
    for key, pname, unit, h, s, multi in G.DANGER_DIMS:
        has_h = h and h >= 1
        vals = G.sample_values(h, s, pname=pname)
        by_band = {"非危大": [], "危大": [], "超规模": []}
        for v in vals:
            b = band(v, h, s)
            if b in by_band:
                by_band[b].append(v)
        def rng_txt(arr, tag):
            if not arr:
                return "—"
            return f"{min(arr)}~{max(arr)}"
        n_txt = rng_txt(by_band["非危大"], "非危大")
        h_txt = rng_txt(by_band["危大"], "危大")
        s_txt = rng_txt(by_band["超规模"], "超规模")
        if s is not None and by_band["危大"] and min(by_band["危大"]) >= s:
            bad.append((key, pname, "危大档越超规模线", ""))
        if not has_h and s is None:
            bad.append((key, pname, "h/s 皆无", ""))
        ratio = (s / h) if (s and h) else None
        print(f"{key:<8}{pname:<22}{h:>7}{str(s):>7}"
              f"{(f'{ratio:.2f}' if ratio else '—'):>7}  {n_txt:>16}{h_txt:>18}{s_txt:>16}")
    print()

    print("=" * 96)
    print("B. 采样值档位 / 物理合理性")
    print("=" * 96)
    for key, pname, unit, h, s, multi in G.DANGER_DIMS:
        vals = G.sample_values(h, s, pname=pname)
        has_h = h and h >= 1
        lo_p, hi_p = PLAUSIBLE.get(pname, (None, None))
        rows = []
        for v in vals:
            b = band(v, h, s)
            warn = ""
            if not has_h and b == "非危大":
                warn = "哨兵维度产生非危大样本(语义错误)"
                bad.append((key, pname, "哨兵维度产生非危大样本", f"val={v}"))
            if lo_p is not None and not (lo_p <= v <= hi_p):
                warn += f" 数值越常识[{lo_p},{hi_p}]"
                bad.append((key, pname, "数值超出工程常识", f"val={v}"))
            rows.append(f"{v}{unit}({b}){warn}")
        print(f"[{key}/{pname}] h={h} s={s}")
        print("   " + " | ".join(rows))
    print()

    print("=" * 96)
    print("C. C3 阈值题（危大线提问）合法性")
    print("=" * 96)
    for key, pname, unit, h, s, _m in G.DANGER_DIMS:
        if h and h >= 1:
            print(f"  问「危大工程」阈值 {pname} = {h}{unit}  ✓")
        elif s is not None:
            print(f"  不问危大阈值（哨兵 h={h}）{pname}，仅问超规模 = {s}{unit}  ✓")
        else:
            print(f"  !! {pname}: h={h}（无数值线）且 s=None → 无任何阈值题")
    for key, pname, unit, _h, s in G.THRESHOLD_ONLY:
        print(f"  THRESHOLD_ONLY  {pname} 超规模 = {s}{unit}")
    print()

    print("=" * 96)
    print("D. 问题汇总")
    print("=" * 96)
    if not bad:
        print("  无")
    else:
        seen = set()
        for key, pname, kind, detail in bad:
            sig = (key, pname, kind)
            if sig in seen:
                continue
            seen.add(sig)
            print(f"  [{kind}] {key} / {pname}  {detail}")
    return bad


def audit_generated(path):
    """对已生成的 jsonl 做抽样体检：C2 样本数值 vs 判定标签自洽性。"""
    if not os.path.exists(path):
        print(f"[skip] {path} 不存在")
        return
    total = c2 = 0
    probs = []
    dims = {(k, p): (h, s) for k, p, u, h, s, _m in G.DANGER_DIMS}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        total += 1
        jm = d.get("judge_meta") or {}
        if jm.get("check_dim") != "C2":
            continue
        c2 += 1
        q = d["query"]
        m = re.search(r"为([\d.]+)(米|kN/m²|kN/m|kN)（", q)
        if not m:
            continue
        val = float(m.group(1))
        unit = m.group(2)
        for (k, p), (h, s) in dims.items():
            if p in q:
                exp_band = band(val, h, s)
                expect = jm.get("expect_value", "")
                got_band = ("超规模" if "超规模" in expect else
                            "危大" if expect.startswith("危大") else "非危大")
                if exp_band != got_band:
                    probs.append((p, val, unit, exp_band, got_band, expect))
                break
    print(f"  {os.path.basename(path)}: 共 {total} 条，C2 {c2} 条，档位标签不一致 {len(probs)} 条")
    for p, v, u, eb, gb, exp in probs[:10]:
        print(f"    ! {p}={v}{u} 按阈值应为「{eb}」，标签为「{gb}」（{exp}）")


if __name__ == "__main__":
    bad = audit_dims()
    print()
    print("=" * 96)
    print("E. 已生成数据集体检")
    print("=" * 96)
    for name in ("train", "val"):
        audit_generated(os.path.join(ROOT, "data", "grpo", f"{name}.jsonl"))
    print(f"\n结构性问题 {len(bad)} 项")
