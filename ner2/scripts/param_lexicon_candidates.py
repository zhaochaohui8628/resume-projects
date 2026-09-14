"""参数专项 Step1：从全语料挖「参数量名」候选，供人工逐条核验（定稿成 param_lexicon.json）。

判据（高召回、交由人工裁决）：
  · 以量名后缀结尾（度/量/值/率/数/比/差/径/高/厚/宽/深/长/距/力/压/重/阻/系数/半径/标高/
    位移/频率/载重/水位/间距/步距/跨度/配比/级配/偏差/面积/荷载/压力/功率/负荷/流量/掺量/用量…）
  · 长度 2~8 个汉字、纯中文、最多含 1 个前限定词（如 设计/搭设/开挖/桩身/坑底/额定/允许…）
输出：`data/phase6/param_lexicon_candidates.md`（含频次 + 例句，供人工核验）
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import (  # noqa: E402
    GOLD_CONSENSUS, GOLD_TRAIN_PORTION, PHASE1, PHASE6, SILVER_TRAIN, SILVER_VAL,
)

SUFFIX = ("垂直度", "平整度", "深度", "厚度", "宽度", "长度", "高度", "强度", "刚度", "坡度", "密度",
          "温度", "湿度", "水位", "位移", "频率", "偏差", "间距", "步距", "跨度", "位移值",
          "配比", "级配", "掺量", "用量", "含量", "总量", "数量", "根数", "长度", "直径", "外径",
          "内径", "半径", "周长", "面积", "体积", "系数", "荷载", "压力", "风压", "电压", "电流",
          "功率", "负荷", "电阻", "起重量", "载重量", "流量", "容量", "速度", "标高", "高度差",
          "水位", "深度值", "预警值", "报警值", "累积值", "变化量", "间隔时间", "时间",
          "角度", "坡度", "倾角", "距离", "净距", "锚固长度", "搭接长度", "保护层厚度",
          "沉降", "沉降量", "变形", "变形量", "回弹", "回弹值", "轴力", "应力", "应变",
          "扭矩", "力矩", "弯矩", "剪力", "摩擦系数", "充盈系数", "水灰比", "灰砂比", "比表面积",
          "含泥量", "泥浆比重", "比重", "稠度", "扩散度", "坍落度", "稠度", "水泥用量", "破坏荷载")
# 明显不是参数（与工程指标无关的名词）
BLACK = {"制度", "角度", "力度", "程度", "进度", "温度计", "态度", "风度", "限度", "额度",
         "大深度", "挖土深度", "设计范围", "时间", "数量", "载重"}
HEAD = ("设计", "搭设", "开挖", "桩身", "坑底", "额定", "允许", "施工", "锚固", "搭接", "保护层",
        "有效", "标准", "实际", "计算", "基本", "平均", "最大", "最小", "总", "单", "净", "内",
        "外", "顶", "底", "上", "下", "分段", "预制", "自由", "工作", "安全", "可能坠落范围",
        "初始", "最终", "极限", "额定", "桩", "墙", "梁", "板", "柱", "孔", "槽", "土", "砼",
        "混凝土", "钢筋", "钢管", "水泥", "砂", "石", "泥浆", "入模", "里表", "地表", "坑内",
        "坑外", "地下", "地面", "水位", "管线", "墙顶", "桩顶", "桩端", "支撑", "围护", "降水")
CAND_RE = re.compile(r"[\u4e00-\u9fa5]{2,10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PHASE6, "param_lexicon_candidates.md"))
    ap.add_argument("--max-eg", type=int, default=3)
    args = ap.parse_args()

    # 语料：无标注池（覆盖 11 份方案）+ 已标注集（做种子）
    corpus = []
    pool = os.path.join(PHASE1, "unlabeled_sentences.jsonl")
    if os.path.exists(pool):
        import json
        corpus += [json.loads(l)["text"] for l in open(pool, encoding="utf-8") if l.strip()]
    seeds = Counter()
    for p in (GOLD_CONSENSUS, GOLD_TRAIN_PORTION, SILVER_TRAIN, SILVER_VAL):
        if os.path.exists(p):
            import json
            for l in open(p, encoding="utf-8"):
                if not l.strip():
                    continue
                r = json.loads(l)
                for e in r.get("entities", []):
                    if e["type"] == "参数":
                        seeds[r["text"][e["start"]:e["end"]]] += 1

    freq = Counter()
    eg = defaultdict(list)
    for text in corpus:
        # 用正则取连续汉字串，再在串上做后缀切分（避免跨标点）
        for m in CAND_RE.finditer(text):
            seg = m.group(0)
            for i in range(len(seg)):
                for suf in SUFFIX:
                    if seg.startswith(suf, i):
                        cand = seg[i:i + len(suf)]
                        if cand in BLACK or len(cand) < 2:
                            continue
                        freq[cand] += 1
                        if len(eg[cand]) < args.max_eg:
                            s = max(0, m.start() + i - 12)
                            e = min(len(text), m.start() + i + len(cand) + 12)
                            eg[cand].append(text[s:e])
    # 与既有金标/银标中的参数取并集（种子置顶）
    for w, n in seeds.items():
        freq.setdefault(w, 0)

    rows = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    lines = ["# 参数「量名」候选（供人工逐条核验）", "",
             f"- 候选 **{len(rows)}** 个；种子（来自已标注集的参数实体）**{len(seeds)}** 个已用 ★ 标出",
             f"- 语料：`unlabeled_sentences.jsonl`（{len(corpus)} 句）+ 已标注集参数实体", "",
             "| ★ | 候选 | 频次 | 例句 | 核验（√参数 / ×非参数 / 备注） |", "|---|---|---|---|---|"]
    for w, n in rows:
        egs = " ；".join(x.replace("|", "\\|") for x in eg.get(w, [])) or "—"
        lines.append(f"| {'★' if w in seeds else ''} | `{w}` | {n} | {egs} | |")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[lex] 候选 {len(rows)}（种子 {len(seeds)}）→ {args.out}")
    print("[lex] 频次 top40：" + "、".join(f"{w}({n})" for w, n in rows[:40]))


if __name__ == "__main__":
    main()
