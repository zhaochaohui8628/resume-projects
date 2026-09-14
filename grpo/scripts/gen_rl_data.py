"""从规则库 + 真实条款语料生成 GRPO 四维奖励数据集（真实 agent 场景模拟）。

与旧版（4 类固定答案题、4 档离散判分）的区别：
1. 每条样本 = 一次「真实 agent 调用」：待审方案片段(plan_excerpt) + 用户提问(user_query)
   + 模拟 RAG 检索证据(evidence，优先取 rag2 语料真实条款、缺失时规则合成)
   + 金标准 agent 输出(gold_response，含 cot_steps 推理链)。
2. judge_meta 携带四维判分所需全部标签：
   golden_basis（溯源金据）/ golden_cot（金推理链）/ golden_explanation
   + violation_expected（是否预期为合规漏洞，支撑漏报/误报矩阵）。
3. 含正负样本（violation_expected=False）：C1 现行规范、C2 低于阈值（非危大）、
   C4 方案齐全（无缺失），用于测量假阳性误报率。

四类任务（check_dim 对应 C1..C4）：
  version_abolished  C1 版本/废止判断   exact + alias（含 replaced_by/basis）
  danger_level       C2 危大分级         set_match（级别+论证，连续打分）
  threshold_value    C3 阈值数值问答      numeric（连续衰减）
  missing_section    C4 方案缺章判断      exact / set_match（多章组合）

扩充策略（2026-09-12，目标 train>1000 / val>200）：按「参数维度 × 自动采样值 ×
多问法模板 × 多工程场景」组合生成，而非重复模板凑数：
  C1 条文废止放开至 173 条全量 + 语料现行规范负样本
  C2 覆盖住建部令第37号 9 类危大、12 个参数维度，每维自动采样 8 个数值点（远离边界）
  C3 覆盖 22 组阈值（危大线 + 超规模线，含多参数维度）
  C4 缺 1/2/3 章组合 + 建办质〔2021〕48号九类细化要素 + 齐全负样本

数值口径取自 hazardous_work_types.json（与住建部令第37号附件1/2 核对一致）；
多参数维度的题目统一加「其他参数均不达危大判定线」限定语，保证单变量判定无歧义。

用法：
  python grpo/scripts/gen_rl_data.py [--scale 4] [--clause-cap 173] [--neg-std-cap 45]
        [--outdir data/grpo] [--corpus rag2/data/corpus/clauses.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRPO = os.path.dirname(HERE)                       # grpo/
ROOT = os.path.dirname(GRPO)                       # 项目根
for p in (GRPO,):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as C                                  # noqa: E402
from prompts import OPTIONS                         # noqa: E402
from reward import build_gold_response, score_breakdown, predict_violation, violation_expected  # noqa: E402

ALL_CHAPTERS = [c for c in OPTIONS["missing_section"] if c != "无缺失"]  # 9 章
RULES = os.path.join(ROOT, "agent", "data", "rules")

# ---------------- 真实条款语料（模拟 RAG 证据源） ----------------


class EvidenceBank:
    """加载 rag2 分块语料做证据检索；语料缺失/未命中时退化为规则合成。"""

    def __init__(self, corpus_path=None):
        self._clauses = []
        if corpus_path and os.path.exists(corpus_path):
            for line in open(corpus_path, encoding="utf-8"):
                if not line.strip():
                    continue
                d = json.loads(line)
                m = d.get("metadata") or {}
                self._clauses.append({
                    "source": m.get("source", "?"),
                    "clause_no": m.get("clause_no", ""),
                    "chapter": m.get("chapter", ""),
                    "text": self._clean(d.get("text", "")),
                })
        self.loaded = len(self._clauses) > 0

    @staticmethod
    def _clean(text, cap=180):
        t = re.sub(r"\s+", "", text or "")
        t = re.sub(r"^\d+(?:\.\d+)+", "", t)   # 去掉行首条款号
        return t[:cap]

    def find(self, source_kw, text_kw=None, limit=1):
        if not self.loaded:
            return []
        sk = re.sub(r"\s+", "", (source_kw or "")).lower()
        hits = [c for c in self._clauses if sk and sk in re.sub(r"\s+", "", c["source"]).lower()]
        if text_kw:
            hits = [c for c in hits if text_kw in c["text"]]
        return [{"source": c["source"], "clause_no": c["clause_no"], "text": c["text"]}
                for c in hits[:limit]]

    def sources(self):
        """语料内唯一规范来源（格式 "编号_名称"）。"""
        seen, out = set(), []
        for c in self._clauses:
            if c["source"] not in seen:
                seen.add(c["source"]); out.append(c["source"])
        return out

    @staticmethod
    def synth(source, clause_no, text):
        return [{"source": source, "clause_no": clause_no, "text": text}]


BANK = None


def evidence(source_kw, text_kw, source_synth, clause_synth, text_synth, limit=1):
    """统一取证据：真实条款优先，合成兜底。"""
    global BANK
    hits = BANK.find(source_kw, text_kw, limit) if BANK else []
    if hits:
        return hits
    return EvidenceBank.synth(source_synth, clause_synth, text_synth)


def _pretty_std(no: str) -> str:
    """语料源编号 JGJ-T46-2024 → JGJ/T 46-2024；GB55034-2022 → GB 55034-2022。"""
    if no.startswith("JGJ-T"):
        return "JGJ/T " + no[5:].lstrip("-")
    if no.startswith("DG-TJ"):
        return "DG/TJ " + no[5:].lstrip("-")
    m = re.match(r"^([A-Za-z]+)(\d.*)$", no)
    return f"{m.group(1)} {m.group(2)}" if m else no


# ---------------- 工程场景池（真实 agent 审查背景） ----------------

SCENES = {
    "基坑工程": ["某医院建设项目深基坑工程专项施工方案审查",
                 "某地铁车站附属结构基坑支护与开挖方案审查",
                 "某商业综合体地下室基坑工程专项方案审查"],
    "模板支撑": ["某会展中心高大模板支撑体系专项施工方案审查",
                 "某办公楼超限梁板模板支架专项方案审查",
                 "某体育场馆看台高支模专项施工方案审查"],
    "脚手架": ["某住宅小区外脚手架工程专项施工方案审查",
               "某超高层办公楼附着式升降脚手架方案审查",
               "某厂房悬挑式钢管脚手架专项施工方案审查"],
    "人工挖孔桩": ["某市政桥梁人工挖孔桩工程专项施工方案审查",
                   "某山地建筑人工挖孔灌注桩专项方案审查",
                   "某边坡治理工程抗滑桩（人工挖孔）方案审查"],
    "幕墙安装": ["某会展中心建筑幕墙安装工程专项施工方案审查",
                 "某超高层写字楼单元式幕墙安装方案审查",
                 "某商业裙楼石材幕墙施工专项方案审查"],
    "起重吊装": ["某钢结构厂房起重吊装工程专项施工方案审查",
                 "某大型设备非常规起重吊装专项方案审查",
                 "某桥梁预制构件吊装工程专项施工方案审查"],
    "钢结构安装": ["某会展中心大跨度钢结构安装专项施工方案审查",
                   "某体育馆空间网架结构安装方案审查",
                   "某工业厂房门式刚架钢结构安装方案审查"],
    "拆除": ["某旧厂区内构筑物拆除工程专项施工方案审查",
             "某医院改扩建工程毗邻建筑拆除方案审查"],
    "暗挖": ["某地铁区间盾构法暗挖工程专项施工方案审查",
             "某市政管廊顶管法施工专项方案审查"],
    "version": ["某工程专项施工方案编制依据合规性审查",
                "某危大工程专项方案引用规范有效性核查"],
    "section": ["某危大工程专项施工方案编制要素审查",
                "某专项施工方案章节完整性审查"],
}

HINT = {
    "基坑工程": "采用钻孔灌注桩+内支撑支护形式",
    "模板支撑": "采用盘扣式满堂支撑架",
    "脚手架": "采用钢管脚手架体系",
    "人工挖孔桩": "采用人工挖孔成桩工艺",
    "幕墙安装": "采用单元式幕墙+吊篮安装工艺",
    "起重吊装": "采用起重机械进行吊装作业",
    "钢结构安装": "采用分件吊装+高空拼装工艺",
    "拆除": "涉及毗邻建筑保护",
    "暗挖": "处于城市建成区",
}


def _scene(key, i):
    arr = SCENES.get(key) or SCENES["version"]
    return arr[i % len(arr)]


def plan_excerpt_of(key, desc, i=0):
    return f"本方案为{_scene(key, i)}。{desc}。"


def _pack(id_, task_type, check_dim, scene, excerpt, ev, q, judge_meta):
    """统一组装样本（gold_response 由 judge_meta 金据生成）。"""
    return {
        "id": id_, "task_type": task_type, "check_dim": check_dim,
        "agent_scenario": {"scene": scene, "plan_excerpt": excerpt, "evidence": ev,
                           "user_query": q},
        "query": f"【待审方案】{excerpt}\n【问题】{q}",
        "gold_response": json.loads(build_gold_response(judge_meta)),
        "judge_meta": judge_meta,
    }


# ---------------- 危大参数维度（C2 判定 + C3 阈值共用） ----------------
# (类别, 参数名, 单位, 危大线 h, 超规模线 s, 是否多参数维度)
# h=0.1 表示"该类型本身即属危大工程"（无数值危大线，见附件1对应条目）；s=None 表示无统一超规模线。
# 2026-09-12 修正（对照建办质〔2018〕31号附件1/2 全文）：
#   幕墙安装 附件1七(一) 无高度门槛（本身即危大）→ h=0.1；50m 是附件2七(一) 超规模线 → s=50。
#   悬挑/附着脚手架参数名对齐法规口径："分段架体搭设高度"/"提升高度"。
DANGER_DIMS = [
    ("基坑工程", "基坑开挖深度", "米", 3, 5, False),
    ("模板支撑", "模板支架搭设高度", "米", 5, 8, True),
    ("模板支撑", "模板支架跨度", "米", 10, 18, True),
    ("模板支撑", "施工总荷载", "kN/m²", 10, 15, True),
    ("模板支撑", "集中线荷载", "kN/m", 15, 20, True),
    ("脚手架", "落地式钢管脚手架搭设高度", "米", 24, 50, True),
    ("脚手架", "悬挑式脚手架分段架体搭设高度", "米", 0.1, 20, True),
    ("脚手架", "附着式升降脚手架提升高度", "米", 0.1, 150, True),
    ("起重吊装", "单件起吊重量", "kN", 10, 100, True),
    ("幕墙安装", "幕墙工程施工高度", "米", 0.1, 50, False),
    ("人工挖孔桩", "人工挖孔桩开挖深度", "米", 0.1, 16, False),
    ("钢结构安装", "钢结构安装跨度", "米", 0.1, 36, False),
]

# 各维度的工程合理取值下限（避免 h 哨兵维度采出"悬挑脚手架 1 米"等荒谬值）
VAL_FLOOR = {
    "基坑开挖深度": 1.0,
    "模板支架搭设高度": 2.0,
    "模板支架跨度": 3.0,
    "施工总荷载": 2.0,
    "集中线荷载": 3.0,
    "落地式钢管脚手架搭设高度": 5.0,
    "悬挑式脚手架分段架体搭设高度": 4.0,
    "附着式升降脚手架提升高度": 15.0,
    "单件起吊重量": 2.0,
    "幕墙工程施工高度": 6.0,
    "人工挖孔桩开挖深度": 3.0,
    "钢结构安装跨度": 6.0,
}

# 危大判定的前置条件（法规原文限定语，缺失会教模型"无条件判危大/非危大"）
PRECOND = {
    "单件起吊重量": "采用非常规起重设备、方法",
}

# 仅用于 C3 阈值问答的维度（无危大数值线，只有超规模线）
THRESHOLD_ONLY = [
    ("起重吊装", "起重机械搭设总高度", "米", None, 200),
    ("起重吊装", "起重机械搭设基础标高", "米", None, 200),
    ("起重吊装", "起重机械起重量", "kN", None, 300),
]

# 定性类（无数值阈值，工艺本身即危大）
QUALITATIVE = [
    ("拆除", "可能影响行人、交通、电力设施、通信设施或其他建(构)筑物结构安全的拆除作业",
     "危大工程，不需专家论证", "拆除作业本身即属危大工程（附件1），拆除类无统一超规模线"),
    ("暗挖", "采用矿山法、盾构法、顶管法施工的暗挖工程",
     "危大工程，不需专家论证", "暗挖工程（矿山法/盾构法/顶管法）本身即属危大工程"),
]

# 每类的证据检索配置（语料来源 kw / 文本 kw / 合成来源）
# 2026-09-12 修正：C2/C3 判定依据是住建部令第37号附件1/2 的**阈值条款**，该标准不在 rag2 语料
# （语料内仅 DG-TJ08-2077-2021 危大安全管理流程条款，无附件阈值），真实条款检索必然命中无关条款
# → C2/C3 证据**一律规则合成** 37 号令附件条款（语义精准、来源与 golden_basis 一致）。
# C1/C4 证据仍走真实条款（废除规范/编制指南本就该引原文），不受影响。
EVID_SPEC = {
    "基坑工程": (None, None, "住建部令第37号"),
    "模板支撑": (None, None, "住建部令第37号"),
    "脚手架": (None, None, "住建部令第37号"),
    "人工挖孔桩": (None, None, "住建部令第37号"),
    "幕墙安装": (None, None, "住建部令第37号"),
    "起重吊装": (None, None, "住建部令第37号"),
    "钢结构安装": (None, None, "住建部令第37号"),
    "拆除": (None, None, "住建部令第37号"),
    "暗挖": (None, None, "住建部令第37号"),
}

_LIMIT = "其他参数均不达危大判定线"

_TMPL_C2 = [
    "某工程{pname}为{val}{unit}（{lim}），按住建部令第37号及其附件判定：属于危大工程还是超规模危大工程？是否需要专家论证？",
    "{pname}达到{val}{unit}的专项工程（{lim}），请判定危大级别（危大工程/超规模危大工程/非危大）并说明是否需要专家论证。",
    "审查某专项施工方案：{pname}={val}{unit}（{lim}）。请定性该工程是否属于危大工程、是否需专家论证。",
    "按现行危大工程管理规定，{pname}{val}{unit}（{lim}）应判定为何种危大级别，专家论证要求如何？",
    "{pname}为{val}{unit}的分部分项工程（{lim}），请给出危大分级与专家论证结论。",
]

_TMPL_C3 = [
    "按住建部令第37号，{pname}达到多少{unit}即属于{level}？",
    "超过多少{unit}的{pname}应判定为{level}？",
    "请给出{pname}的{level}判定阈值（数值+单位）。",
    "{pname}的危大判定中，{level}的界限值是多少{unit}？",
    "按现行规定，{pname}达到什么数值需按{level}管理（{unit}）？",
    "审查专项方案需核对：{pname}的{level}阈值是多少{unit}？",
    "编制{level}判定表时需要填写：{pname}的起判值是多少{unit}？",
    "{pname}达到多少{unit}时，该分部分项工程应按{level}编制并管理？",
]

_LEVEL_CN = {"危大工程": "危大工程", "超规模危大工程": "超规模危大工程"}


def _lin(lo, hi, n):
    """区间等距采样（保留 1 位小数）。"""
    if n <= 0:
        return []
    if n == 1:
        return [round((lo + hi) / 2, 1)]
    step = (hi - lo) / (n - 1)
    return [round(lo + step * i, 1) for i in range(n)]


def sample_values(h, s, pname=None, n_non=4, n_haz=4, n_sup=5):
    """按危大线/超规模线自动采样数值点，各档均与边界留裕度（避免歧义样本）。

    区间定义（2026-09-12 修正，根治「危大档下界 > 上界」倒挂）：
      旧实现用两个**互相独立**的固定倍率——下界 1.2h、上界 0.85s。当 s/h < 1.2/0.85
      ≈ 1.412 时必有 1.2h > 0.85s（集中线荷载 15/20 → 18.0 > 17.0），只能退化成中点，
      危大档样本量从 n_haz 掉到 1。
      新实现改为**在 [h, s] 区间内按比例插值**：lo = h + 0.15(s−h)、hi = h + 0.85(s−h)。
      只要 s > h 就恒有 lo < hi，且天然满足「离两条边界各留 15% 裕度」的原始意图。
      h 为哨兵 0.1（本身即危大、无数值危大线）时下界取 0，区间退化为 [0.15s, 0.85s]，
      再与 VAL_FLOOR[pname] 取大（避免"悬挑脚手架 1 米"这类荒谬值）。
      超规模档收敛为 1.15s ~ 1.6s（旧 1.2s ~ 2.0s 对 s=150 会采出 300 米附着式脚手架）。
    """
    vmin = VAL_FLOOR.get(pname, 0.0)
    has_h = bool(h and h >= 1)
    base = float(h) if has_h else 0.0
    vals = []
    # 非危大档：[0.35h, 0.85h]，上界恒 < h（留 15% 裕度），下限抬到 vmin
    if has_h:
        lo = max(round(h * 0.35, 1), vmin)
        hi = round(h * 0.85, 1)
        if lo > hi:
            lo = hi = round((h * 0.35 + h * 0.85) / 2, 1)
        vals += _lin(lo, hi, n_non)
    if s is not None:
        # 危大档：区间内 15%/85% 插值，恒 lo < hi（只要 s > h）
        lo_h = max(round(base + 0.15 * (s - base), 1), vmin)
        hi_h = round(base + 0.85 * (s - base), 1)
        if lo_h > hi_h:                       # 极端窄区间兜底（理论上不触发）
            lo_h = hi_h = round((base + s) / 2, 1)
        vals += _lin(lo_h, hi_h, n_haz)
        # 超规模档：[1.15s, 1.6s]
        vals += _lin(max(round(s * 1.15, 1), vmin), round(s * 1.6, 1), n_sup)
    else:                                     # 无超规模线（如"本身即危大且无超规线"）
        lo_h = max(round(h * 1.15, 1), vmin) if has_h else vmin
        vals += _lin(lo_h, round(max(h, 1) * 1.8, 1), n_haz)
    out = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def expect_of(val, h, s):
    """→ (期望结论, violation_expected)。"""
    if s is not None and val >= s:
        return "超规模危大工程，需专家论证", True
    if h is not None and val >= h:
        return "危大工程，不需专家论证", True
    return "非危大，不需专家论证", False


def _c2_cot(pname, val, unit, h, s, expect):
    no_line = not (h and h >= 1)          # 本身即危大、无数值危大线（h=0.1 哨兵）
    if s is not None and val >= s:
        cmp_txt = f"达超规模线（{s}{unit}）"
    elif no_line:
        cmp_txt = "该分部分项工程列入附件1危大工程范围（无数值判定线）" + (
            f"，且未达超规模线（{s}{unit}）" if s is not None else "，且无统一超规模线")
    elif h is not None and val >= h:
        cmp_txt = f"达危大线（{h}{unit}）" + (f"但未达超规模线（{s}{unit}）" if s else "且无统一超规模线")
    else:
        cmp_txt = f"未达危大线（{h}{unit}）"
    return [
        f"识别参数：{pname}为{val}{unit}，对照住建部令第37号附件1危大工程判定线。",
        f"比对：{val}{unit} {cmp_txt}。",
        f"结论：{expect}，应按规定编制专项施工方案并执行相应管理要求。",
    ]


def _thr(v):
    return f"≥{v}" if v is not None else "—"


def _dim_evidence(key, h, s, pname):
    """C2/C3 证据：纯规则合成 37 号令附件条款（语料无附件阈值条款，真实检索必命中无关条款）。"""
    src_kw, txt_kw, synth_src = EVID_SPEC.get(key, (None, None, "住建部令第37号"))
    if h and h >= 1:
        head = f"《危险性较大的分部分项工程安全管理规定》附件1：{pname}达到{_thr(h)}属危大工程"
    else:
        head = (f"《危险性较大的分部分项工程安全管理规定》附件1：{pname}所属分部分项工程"
                f"列入危大工程范围（无数值判定线）")
    txt = head + (f"；附件2：{pname}达到{_thr(s)}属超规模危大工程。" if s is not None else "。")
    return EvidenceBank.synth(synth_src, "附件1", txt)


# ---------------- C1 ----------------
def _load_abolished():
    d = json.load(open(os.path.join(RULES, "abolished_clauses.json"), encoding="utf-8"))
    full = d["full_doc_abolished"]
    clause_items = []
    for grp in d.get("clause_abolished", []):
        for it in grp.get("items", []):
            for c in it.get("clauses", []):
                clause_items.append({"std_no": it["std_no"], "clause": c,
                                     "by": grp.get("by_std_no", ""),
                                     "by_name": grp.get("by_std_name", "")})
    return full, clause_items


_TMPL_C1_FULL = [
    "请核查本方案编制依据：引用《{std_name}》（{std_no}）是否合规？该标准是否已废止？若废止请指出替代规范编号。",
    "方案编制依据引用《{std_no}》。请审查该引用是否合规、该标准是否仍有效。",
    "审查专项施工方案编制依据：其中《{std_name}》（{std_no}）目前是否已被废止？",
    "本方案引用《{std_no}》作为编制依据。请判断该标准当前是否有效，并给出结论。",
]
_TMPL_C1_NEG = [
    "请审查该方案引用《{std_name}》（{std_no}）是否合规？该标准是否已被废止？",
    "方案编制依据列出《{std_name}》（{std_no}）。请核查其是否仍为现行有效标准。",
    "审查编制依据：《{std_no}》当前是否已被废止或失效？请给出结论。",
]
_TMPL_C1_CLAUSE = [
    "《{std_no}》第 {clause} 条现在是否已被废止？若废止请给出依据规范编号。",
    "本方案引用《{std_no}》第 {clause} 条。请核查该条文当前是否仍有效。",
]
_TMPL_C1_REPL = [
    "《{std_no}》废止后由哪份现行规范替代？请给出编号。",
    "审查编制依据：《{std_no}》已废止，其替代标准编号是什么？",
    "《{std_no}》不再适用后，应按哪份现行规范执行？请给出编号。",
]


def gen_version_abolished(scale=4, clause_cap=173, neg_std_cap=45):
    """C1：整本废止 yes / 替代 / 条文废止（全量） + 现行规范负样本。"""
    full, clause_items = _load_abolished()
    out, cid = [], 0

    # 1) 整本废止（violation_expected=True）
    for f in full:
        rep = (f.get("replaced_by") or "").strip()
        eff = f.get("effective_date", "")
        for k in range(min(scale, len(_TMPL_C1_FULL))):
            j = {"mode": "exact", "expect_value": "yes",
                 "alias": ["废止", "不再适用", "失效"],
                 "conclusion_options": OPTIONS["version_abolished"],
                 "golden_basis": [rep] if rep else [f.get("by_document", "")],
                 "golden_cot": [
                     f"核对现行规范库：{f['std_no']} 已被 {rep} 替代（{eff} 起实施）。",
                     "引用已废止标准属于编制依据不合规，违反现行通用规范编制要求。",
                     "结论：该引用不合规，应改用替代规范。",
                 ],
                 "golden_explanation": f"{f['std_no']} 已废止，应改用 {rep}",
                 "violation_expected": True,
                 "check_dim": "C1", "task_type": "version_abolished", "require_cot": True}
            ev = evidence("JGJ-T46-2024" if "JGJ 46-2005" in f["std_no"] else rep.split("+")[0].strip(),
                          "废止" if "JGJ 46-2005" in f["std_no"] else None,
                          rep, "前言", f"{rep}：自{eff}起实施，{f['std_no']}同时废止。")
            excerpt = plan_excerpt_of("version", f"编制依据引用《{f['std_name']}》（{f['std_no']}）", k)
            q = _TMPL_C1_FULL[k].format(std_name=f["std_name"], std_no=f["std_no"])
            out.append(_pack(f"C1-{k}-{cid}", "version_abolished", "C1", _scene("version", k),
                             excerpt, ev, q, j)); cid += 1
        # 2) 替代题（contains，无违规语义）
        if rep:
            for k in range(min(3, len(_TMPL_C1_REPL))):
                j = {"mode": "contains", "expect_value": rep, "alias": [],
                     "conclusion_options": [], "golden_basis": [rep],
                     "golden_cot": [f"查询现行规范库：{f['std_no']} 的替代规范为 {rep}。"],
                     "golden_explanation": f"替代规范为 {rep}",
                     "violation_expected": None,
                     "check_dim": "C1", "task_type": "version_abolished", "require_cot": True}
                ev = evidence("JGJ-T46-2024" if "JGJ 46-2005" in f["std_no"] else rep.split("+")[0].strip(),
                              None, rep, "前言", f"{rep}：自{eff}起实施，{f['std_no']}同时废止。")
                excerpt = plan_excerpt_of("version", f"原依据《{f['std_no']}》已废止", k)
                q = _TMPL_C1_REPL[k].format(std_no=f["std_no"])
                out.append(_pack(f"C1r-{cid}", "version_abolished", "C1", _scene("version", k),
                                 excerpt, ev, q, j)); cid += 1

    # 3) 条文废止（全量 clause_items，violation_expected=True）
    rng = random.Random(7)
    rng.shuffle(clause_items)
    for it in clause_items[:clause_cap]:
        for k in range(min(1, len(_TMPL_C1_CLAUSE))):   # 173 条全量，单问法即可（控制 C1 占比）
            j = {"mode": "exact", "expect_value": "yes", "alias": ["废止"],
                 "conclusion_options": OPTIONS["version_abolished"],
                 "golden_basis": [it["by"]],
                 "golden_cot": [
                     f"核对条文库：《{it['std_no']}》第 {it['clause']} 条已被 {it['by_name']} 全文强制条款覆盖。",
                     "GB550xx 系全文强制规范，被覆盖条文自施行日起废止。",
                     "结论：该条已废止，应依据现行通用规范执行。",
                 ],
                 "golden_explanation": f"该条已被 {it['by_name']} 废止",
                 "violation_expected": True,
                 "check_dim": "C1", "task_type": "version_abolished", "require_cot": True}
            ev = evidence(it["by"], None, it["by"], it["clause"],
                          f"{it['by_name']}：第 {it['clause']} 条相关要求已由本规范强制条文替代，"
                          f"{it['std_no']} 该条废止。")
            excerpt = plan_excerpt_of("version", f"编制依据含《{it['std_no']}》第 {it['clause']} 条", cid)
            q = _TMPL_C1_CLAUSE[k].format(std_no=it["std_no"], clause=it["clause"])
            out.append(_pack(f"C1c-{cid}", "version_abolished", "C1", _scene("version", cid),
                             excerpt, ev, q, j)); cid += 1

    # 4) 现行有效规范负样本（violation_expected=False）
    n_neg = 0
    for src in (BANK.sources() if BANK else []):
        if n_neg >= neg_std_cap:
            break
        if "_" not in src:
            continue
        raw_no, std_name = src.split("_", 1)
        std_no = _pretty_std(raw_no)
        if std_no in ("JGJ 46-2005",):
            continue                                   # 已整本废止，不作负样本
        for k in range(len(_TMPL_C1_NEG)):
            j = {"mode": "exact", "expect_value": "no", "alias": ["有效", "现行", "未废止"],
                 "conclusion_options": OPTIONS["version_abolished"],
                 "golden_basis": [std_no],
                 "golden_cot": [
                     f"核对现行规范库：{std_no} 为现行有效规范，未被整本废止。",
                     "该标准在现行规范体系中有效，引用合规。",
                     "结论：引用合规，可继续使用。",
                 ],
                 "golden_explanation": f"{std_no} 现行有效，引用合规",
                 "violation_expected": False,
                 "check_dim": "C1", "task_type": "version_abolished", "require_cot": True}
            ev = evidence(raw_no, None, std_no, "1.0.1",
                          f"{std_no}《{std_name}》：本规范为现行有效标准，适用于{std_name}相关工程。")
            excerpt = plan_excerpt_of("version", f"编制依据引用《{std_name}》（{std_no}）", n_neg)
            q = _TMPL_C1_NEG[k].format(std_name=std_name, std_no=std_no)
            out.append(_pack(f"C1n-{cid}", "version_abolished", "C1", _scene("version", n_neg),
                             excerpt, ev, q, j)); cid += 1
        n_neg += 1                                     # cap 计「规范本数」，每本出 3 种问法
    return out


# ---------------- C2：危大分级 ----------------
def gen_danger(scale=4):
    out, cid = [], 0
    for key, pname, unit, h, s, multi in DANGER_DIMS:
        pre = PRECOND.get(pname)
        lim = f"{pre}，且{_LIMIT}" if (pre and multi) else (
            pre or (_LIMIT if multi else "该参数为唯一判定参数"))
        for val in sample_values(h, s, pname=pname):
            expect, ve = expect_of(val, h, s)
            basis = ["住建部令第37号附件1"]
            if s is not None and val >= s:
                basis.append("住建部令第37号附件2")
            for k in range(min(scale, len(_TMPL_C2))):
                q = _TMPL_C2[k].format(pname=pname, val=val, unit=unit, lim=lim)
                j = {"mode": "set_match", "expect_value": expect, "alias": [],
                     "conclusion_options": OPTIONS["danger_level"],
                     "golden_basis": basis,
                     "golden_cot": _c2_cot(pname, val, unit, h, s, expect),
                     "golden_explanation": f"{pname} {val}{unit}，对应：{expect}",
                     "violation_expected": ve,
                     "check_dim": "C2", "task_type": "danger_level", "require_cot": True}
                ev = _dim_evidence(key, h, s, pname)
                idx = cid + k
                excerpt = plan_excerpt_of(key, f"{pname}为{val}{unit}，{HINT.get(key, '')}", idx)
                out.append(_pack(f"C2-{key}-{cid}-{k}", "danger_level", "C2", _scene(key, idx),
                                 excerpt, ev, q, j)); cid += 1
    # 定性类（工艺本身即危大）
    for key, desc, expect, why in QUALITATIVE:
        for k in range(min(scale, len(_TMPL_C2))):
            q = (f"某工程采用{desc}，按住建部令第37号判定：是否属于危大工程？是否需要专家论证？"
                 if k % 2 == 0 else
                 f"审查方案：本工程{desc}。请判定其危大级别（危大工程/超规模危大工程/非危大）"
                 f"并说明是否需要专家论证。")
            j = {"mode": "set_match", "expect_value": expect, "alias": [],
                 "conclusion_options": OPTIONS["danger_level"],
                 "golden_basis": ["住建部令第37号附件1"],
                 "golden_cot": [
                     f"识别工艺：本工程{desc}。",
                     f"对照住建部令第37号附件1：该类工程列入危大工程范围，{why}。",
                     f"结论：{expect}。",
                 ],
                 "golden_explanation": f"{key}：{why}",
                 "violation_expected": True,
                 "check_dim": "C2", "task_type": "danger_level", "require_cot": True}
            src_kw, txt_kw, synth_src = EVID_SPEC.get(key, (None, None, "住建部令第37号"))
            ev = EvidenceBank.synth(synth_src, "附件1",
                                    f"《危险性较大的分部分项工程安全管理规定》附件1：{desc}属于危大工程。")
            excerpt = plan_excerpt_of(key, f"本工程{desc}，{HINT.get(key, '')}", cid)
            out.append(_pack(f"C2q-{key}-{cid}", "danger_level", "C2", _scene(key, cid),
                             excerpt, ev, q, j)); cid += 1
    return out


# ---------------- C3：阈值问答 ----------------
def gen_threshold(scale=5):
    out, cid = [], 0
    groups = []                                        # (key, pname, unit, level, value, att)
    for key, pname, unit, h, s, _multi in DANGER_DIMS:
        if h and h >= 1:                               # 有实质危大数值线才问危大阈值
            groups.append((key, pname, unit, "危大工程", h, "附件1"))
        if s is not None:
            groups.append((key, pname, unit, "超规模危大工程", s, "附件2"))
    for key, pname, unit, _h, s in THRESHOLD_ONLY:
        groups.append((key, pname, unit, "超规模危大工程", s, "附件2"))

    for key, pname, unit, level, tv, att in groups:
        for k in range(len(_TMPL_C3)):      # 阈值题问法固定全用（与 scale 无关，控制 C3 占比）
            q = _TMPL_C3[k].format(pname=pname, unit=unit, level=_LEVEL_CN[level])
            j = {"mode": "numeric", "expect_value": str(tv), "tolerance": 0.1,
                 "conclusion_options": OPTIONS["threshold_value"],
                 "golden_basis": [f"住建部令第37号{att}"],
                 "golden_cot": [
                     f"定位危大判定标准：{pname}{level}阈值在住建部令第37号{att}。",
                     f"读取阈值：{pname}{level}为 {tv}{unit}。",
                 ],
                 "golden_explanation": f"{pname}{level}阈值为 {tv}{unit}",
                 "violation_expected": None,
                 "check_dim": "C3", "task_type": "threshold_value", "require_cot": True}
            src_kw, txt_kw, synth_src = EVID_SPEC.get(key, (None, None, "住建部令第37号"))
            ev = EvidenceBank.synth(synth_src, att,
                                    f"《危险性较大的分部分项工程安全管理规定》{att}：{pname}{level}阈值为 {tv}{unit}。")
            excerpt = plan_excerpt_of(key, f"审查人员需核对{pname}的{level}判定阈值", cid)
            out.append(_pack(f"C3-{key}-{cid}", "threshold_value", "C3", _scene(key, cid),
                             excerpt, ev, q, j)); cid += 1
    return out


# ---------------- C4：编制要素缺项 ----------------
_TMPL_C4 = [
    "某专项施工方案的目录包含：{chapters}。对照建办质〔2018〕31号第九条，该方案缺少以下哪一章？",
    "下列章节：{chapters}，来自某危大工程专项施工方案目录。按规定该方案还缺少哪一章必含内容？",
    "审查专项方案目录：{chapters}。按建办质〔2018〕31号第九条，尚缺哪些必含章节？",
    "该危大工程专项方案现有章节为：{chapters}。请指出缺失的必含章节。",
    "方案章节清单：{chapters}。请对照31号文第九条做章节完整性审查，列出缺失项。",
]
_TMPL_C4_DETAIL = [
    "按建办质〔2021〕48号编制指南，{ptype}专项施工方案还应包含细化要素。现有要素：{items}。请指出缺失的细化要素（候选：{cands}）。",
    "审查{ptype}专项方案：已编制细化要素 {items}。对照建办质〔2021〕48号，缺少哪一项细化要素（候选：{cands}）？",
    "{ptype}方案细化要素自查：现有 {items}，对照48号指南缺失哪项（候选：{cands}）？",
]
_TMPL_C4_OK = [
    "该专项施工方案目录已包含：{chapters}。对照建办质〔2018〕31号第九条，是否还有缺失章节？",
    "审查方案目录：{chapters}。按建办质〔2018〕31号第九条核查章节完整性，结论如何？",
    "本方案章节为：{chapters}。请核对九章必含内容是否齐全。",
]

_SEC31_TEXT = ("《危险性较大的分部分项工程专项施工方案编制指南》（建办质〔2018〕31号）第九条："
               "专项施工方案应当包括工程概况、编制依据、施工计划、施工工艺技术、施工安全保证措施、"
               "施工管理及作业人员配备和分工、验收要求、应急处置措施、计算书及相关施工图纸等内容。")


def gen_missing_section(scale=4, combo_max=30, tri_max=40):
    out, cid = [], 0
    rng = random.Random(11)

    # 1) 缺 1 章
    for miss in ALL_CHAPTERS:
        rest = [c for c in ALL_CHAPTERS if c != miss]
        for k in range(min(scale, len(_TMPL_C4))):
            shown = rest if k % 2 == 0 else [rest[-1]] + rest[:-1]
            j = {"mode": "exact", "expect_value": miss, "alias": [miss],
                 "conclusion_options": OPTIONS["missing_section"],
                 "golden_basis": ["建办质〔2018〕31号 第九条"],
                 "golden_cot": [
                     "对照建办质〔2018〕31号第九条：专项方案应包含工程概况、编制依据、施工计划、施工工艺技术、"
                     "施工安全保证措施、施工管理及作业人员配备和分工、验收要求、应急处置措施、计算书及相关施工图纸九章。",
                     f"比对目录：方案已含 {len(shown)} 章，缺少「{miss}」。",
                     f"结论：缺少章节为「{miss}」，属编制要素缺失。",
                 ],
                 "golden_explanation": f"缺少：{miss}",
                 "violation_expected": True,
                 "check_dim": "C4", "task_type": "missing_section", "require_cot": True}
            ev = EvidenceBank.synth("建办质〔2018〕31号", "第九条", _SEC31_TEXT)
            excerpt = plan_excerpt_of("section", f"方案目录：{('、'.join(shown))}", cid)
            q = _TMPL_C4[k].format(chapters="、".join(shown))
            out.append(_pack(f"C4-{cid}", "missing_section", "C4", _scene("section", cid),
                             excerpt, ev, q, j)); cid += 1

    # 2) 缺 2 章组合
    pairs = [(a, b) for i, a in enumerate(ALL_CHAPTERS) for b in ALL_CHAPTERS[i + 1:]]
    rng.shuffle(pairs)
    for a, b in pairs[:combo_max]:
        rest = [c for c in ALL_CHAPTERS if c not in (a, b)]
        for k in range(min(3, len(_TMPL_C4))):
            shown = rest if k == 0 else [rest[-1]] + rest[:-1]
            j = {"mode": "set_match", "expect_value": f"{a}、{b}", "alias": [],
                 "conclusion_options": [],           # 多章组合为自由文本，靠 set_match 判分
                 "golden_basis": ["建办质〔2018〕31号 第九条"],
                 "golden_cot": [
                     "对照建办质〔2018〕31号第九条九章清单，逐一核对方案目录。",
                     f"比对：方案已含 {len(shown)} 章，缺少「{a}」与「{b}」两章。",
                     f"结论：缺少章节为「{a}」「{b}」，属编制要素缺失。",
                 ],
                 "golden_explanation": f"缺少：{a}、{b}",
                 "violation_expected": True,
                 "check_dim": "C4", "task_type": "missing_section", "require_cot": True}
            ev = EvidenceBank.synth("建办质〔2018〕31号", "第九条", _SEC31_TEXT)
            excerpt = plan_excerpt_of("section", f"方案目录：{('、'.join(shown))}", cid)
            q = _TMPL_C4[k].format(chapters="、".join(shown))
            out.append(_pack(f"C4m2-{cid}", "missing_section", "C4", _scene("section", cid),
                             excerpt, ev, q, j)); cid += 1

    # 3) 缺 3 章组合
    tris = [(a, b, c) for i, a in enumerate(ALL_CHAPTERS)
            for j2, b in enumerate(ALL_CHAPTERS[i + 1:], i + 1)
            for c in ALL_CHAPTERS[j2 + 1:]]
    rng.shuffle(tris)
    for tri in tris[:tri_max]:
        rest = [c for c in ALL_CHAPTERS if c not in tri]
        j = {"mode": "set_match", "expect_value": "、".join(tri), "alias": [],
             "conclusion_options": [],
             "golden_basis": ["建办质〔2018〕31号 第九条"],
             "golden_cot": [
                 "对照建办质〔2018〕31号第九条九章清单，逐一核对方案目录。",
                 f"比对：方案已含 {len(rest)} 章，缺少「{'」「'.join(tri)}」三章。",
                 f"结论：缺少{'、'.join(tri)}，属编制要素缺失。",
             ],
             "golden_explanation": f"缺少：{'、'.join(tri)}",
             "violation_expected": True,
             "check_dim": "C4", "task_type": "missing_section", "require_cot": True}
        ev = EvidenceBank.synth("建办质〔2018〕31号", "第九条", _SEC31_TEXT)
        excerpt = plan_excerpt_of("section", f"方案目录：{('、'.join(rest))}", cid)
        q = _TMPL_C4[2].format(chapters="、".join(rest))
        out.append(_pack(f"C4m3-{cid}", "missing_section", "C4", _scene("section", cid),
                         excerpt, ev, q, j)); cid += 1

    # 4) 建办质〔2021〕48号 九类工程细化要素缺项
    detail = json.load(open(os.path.join(RULES, "required_sections.json"), encoding="utf-8"))
    for ptype, secs in (detail.get("per_type_sections") or {}).items():
        names = [s["name"] for s in secs]
        for miss in names:
            items = [n for n in names if n != miss]
            for k in range(len(_TMPL_C4_DETAIL)):
                j = {"mode": "exact", "expect_value": miss, "alias": [miss],
                     "conclusion_options": names + ["无缺失"],
                     "golden_basis": ["建办质〔2021〕48号"],
                     "golden_cot": [
                         f"对照建办质〔2021〕48号编制指南：{ptype}专项方案细化要素包括{'、'.join(names)}。",
                         f"比对：方案已含 {len(items)} 项，缺少「{miss}」。",
                         f"结论：缺少细化要素「{miss}」。",
                     ],
                     "golden_explanation": f"{ptype}缺少细化要素：{miss}",
                     "violation_expected": True,
                     "check_dim": "C4", "task_type": "missing_section", "require_cot": True}
                ev = EvidenceBank.synth("建办质〔2021〕48号", "编制指南",
                                        f"《危险性较大的分部分项工程专项施工方案编制指南》（建办质〔2021〕48号）："
                                        f"{ptype}专项施工方案应包含{'、'.join(names)}等细化要素。")
                excerpt = plan_excerpt_of("section", f"{ptype}方案已编制细化要素：{('、'.join(items))}", cid)
                q = _TMPL_C4_DETAIL[k].format(ptype=ptype, items="、".join(items),
                                              cands="、".join(names))
                out.append(_pack(f"C4d-{cid}", "missing_section", "C4", _scene("section", cid),
                                 excerpt, ev, q, j)); cid += 1

    # 5) 负样本：九章齐全（violation_expected=False）
    for k in range(min(3, len(_TMPL_C4_OK))):
        shown = ALL_CHAPTERS if k % 2 == 0 else [ALL_CHAPTERS[-1]] + ALL_CHAPTERS[:-1]
        j = {"mode": "exact", "expect_value": "无缺失", "alias": ["完整", "齐全", "无缺章"],
             "conclusion_options": OPTIONS["missing_section"],
             "golden_basis": ["建办质〔2018〕31号 第九条"],
             "golden_cot": [
                 "对照建办质〔2018〕31号第九条九章清单，逐一核对方案目录。",
                 f"方案目录已包含全部九章必含内容（{len(shown)} 项），无缺失。",
                 "结论：编制要素完整，无缺失章节。",
             ],
             "golden_explanation": "九章齐全，无缺失",
             "violation_expected": False,
             "check_dim": "C4", "task_type": "missing_section", "require_cot": True}
        ev = EvidenceBank.synth("建办质〔2018〕31号", "第九条", _SEC31_TEXT)
        excerpt = plan_excerpt_of("section", f"方案目录：{('、'.join(shown))}", cid)
        q = _TMPL_C4_OK[k].format(chapters="、".join(shown))
        out.append(_pack(f"C4n-{cid}", "missing_section", "C4", _scene("section", cid),
                         excerpt, ev, q, j)); cid += 1
    return out


# ---------------- 切分与主流程 ----------------

def split_by_source(items, val_ratio, rng):
    """按任务族隔离切分（val 与 train 无同族同源泄漏）。"""
    fam = defaultdict(list)
    for it in items:
        fam[re.sub(r"\d", "", it["id"].split("-")[0])].append(it)
    train, val = [], []
    for arr in fam.values():
        rng.shuffle(arr)
        nv = max(1, round(len(arr) * val_ratio))
        val.extend(arr[:nv]); train.extend(arr[nv:])
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=4, help="每个参数点的问法模板数（放大样本量）")
    ap.add_argument("--clause-cap", type=int, default=173, help="C1 条文废止题上限（默认全量）")
    ap.add_argument("--neg-std-cap", type=int, default=81, help="C1 现行规范负样本「规范本数」上限")
    ap.add_argument("--combo-max", type=int, default=36, help="C4 缺 2 章组合上限（全量 36）")
    ap.add_argument("--tri-max", type=int, default=84, help="C4 缺 3 章组合上限（全量 84）")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data", "grpo"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", default=C.CORPUS_CLAUSES,
                    help="rag2 条款语料路径；缺省时证据全部为规则合成")
    args = ap.parse_args()

    global BANK
    BANK = EvidenceBank(args.corpus)
    print(f"[evidence] 真实条款语料加载: {BANK.loaded}"
          f"（{'语料存在' if BANK.loaded else '缺失，退化为规则合成证据'}）")

    gen = (gen_version_abolished(args.scale, args.clause_cap, args.neg_std_cap)
           + gen_danger(args.scale)
           + gen_threshold(max(5, args.scale))
           + gen_missing_section(args.scale, args.combo_max, args.tri_max))

    # 去重（同 query 只保留一条）
    seen, uniq = set(), []
    for it in gen:
        if it["query"] in seen:
            continue
        seen.add(it["query"]); uniq.append(it)
    print(f"[dedup] {len(gen)} -> {len(uniq)} 条唯一样本")

    rng = random.Random(args.seed)
    train, val = split_by_source(uniq, args.val_ratio, rng)
    rng.shuffle(train); rng.shuffle(val)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"共生成 {len(uniq)} 条 | train {len(train)} / val {len(val)}")
    print("train 任务分布:", dict(Counter(i["task_type"] for i in train)))
    print("check_dim 分布(全量):", dict(Counter(i["check_dim"] for i in uniq)))
    print("violation 分布(全量):", dict(Counter(str(i["judge_meta"].get("violation_expected")) for i in uniq)))
    for name, arr in (("train", train), ("val", val)):
        p = os.path.join(args.outdir, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for it in arr:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"已写 {p}")

    # 判分器自检：金标准回答须四维全满分；漏报/误报口径自洽
    ok = total = 0
    for it in uniq:
        bd = score_breakdown(build_gold_response(it["judge_meta"]), it["judge_meta"])
        total += 1
        ok += int(bd["total"] == 1.0)
    print(f"[checker] 金标准回答四维满分率: {ok}/{total}")

    v_expect = [it for it in uniq if violation_expected(it["judge_meta"]) is not None]
    self_err = 0
    for it in v_expect:
        pred = predict_violation(it["judge_meta"], build_gold_response(it["judge_meta"]))
        if pred != it["judge_meta"].get("violation_expected"):
            self_err += 1
    print(f"[checker] 金标准漏报/误报标签自洽（不一致数）: {self_err}/{len(v_expect)}")

    if len(train) <= 1000 or len(val) <= 200:
        print(f"[warn] 未达目标 train>1000 / val>200（当前 {len(train)}/{len(val)}），"
              f"可调大 --scale / --clause-cap / --neg-std-cap")


if __name__ == "__main__":
    main()
