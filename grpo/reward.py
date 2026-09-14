"""四维连续奖励判分器（零模型，代码判分）——替代旧 4 档离散奖励。

动机：离散且跨度大的稀疏奖励（{1.0,0.7,0.3,0.0}）会使策略梯度更新剧烈震荡。
本模块将奖励拆解为四维连续评分，每维 ∈ [0,1]，加权求和形成平滑奖励曲线：

    format  格式合规分    —— JSON 语法正确性 + 必需字段 + conclusion 白名单（分级连续）
    cot     CoT 推理连贯分 —— 思维链结构（步数/衔接词/数值与依据引用），确定性代理
    basis   依据溯源精准分 —— 预测依据（规范编号/条款号）与 golden_basis 的命中率 − 捏造惩罚
    answer  核心判分结论分 —— 与旧 answer_ok 同源，但 set_match/numeric 改为连续打分

总分 = Σ w_i · score_i，权重见 config.REWARD_WEIGHTS（默认 format .15 / cot .20 /
basis .25 / answer .40，求和=1）。训练循环直接用连续总分作 reward，组内 z-score 优势
在 loss.py 结算；轨迹/评测同时记录四维分解，供多维度评测矩阵消费。

模型输出 schema（与 prompts.py 同源）：
    {"cot_steps": ["...", "..."], "conclusion": "...", "basis": ["规范 条款", ...],
     "explanation": "..."}
conclusion 白名单 / golden_basis / golden_cot 由 gen_rl_data.py 写入 judge_meta。
"""
from __future__ import annotations

import json
import re

import config as C

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_FW_DIGITS = str.maketrans("０１２３４５６７８９．－", "0123456789.-")
# CoT 衔接词（推理连贯的弱信号：由证据到结论的因果/归纳连接）
_COT_LINK = ("因此", "综上", "所以", "因为", "由于", "依据", "故", "则", "即",
             "对照", "比对", "换算", "代入", "据此", "由此")
_REQUIRED_FIELDS = ("conclusion", "basis", "explanation")


_CN_DIGIT = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_num(m) -> str:
    """把一段中文数字（一/十/十二/二十/二十三…）转成阿拉伯数字。"""
    t = m.group(0)
    if "十" in t:
        head, _, tail = t.partition("十")
        tens = _CN_DIGIT.get(head, 1) if head else 1
        ones = _CN_DIGIT.get(tail, 0) if tail else 0
        return str(tens * 10 + ones)
    return "".join(str(_CN_DIGIT[c]) for c in t if c in _CN_DIGIT)


_CN_NUM_RE = re.compile(r"[〇零一二两三四五六七八九十]{1,4}")


def _digits_norm(s: str) -> str:
    """数字归一：中文数字 -> 阿拉伯数字。

    ⚠️ 修复实测缺陷：金据写「建办质〔2018〕31号 **第九条**」而模型答
    「… **第9条**」，归一化后 `…第九条` != `…第9条` 且互不包含 →
    basis 被判 0，造成「basis 回退」的假象（C4m3-177 / C4m2-104 两例）。
    该转换对 gold/pred **对称应用**，故不会放宽标准（只是消除书写形式差异）。
    """
    return _CN_NUM_RE.sub(_cn_num, s or "")


def _norm(s: str) -> str:
    basic = re.sub(r"[\s，。；、：:（）()【】\[\]\"'“”‘’/\\·—–-]+", "", (s or "")).lower()
    return _digits_norm(basic)


def _extract_numeric(text: str):
    m = _NUM_RE.search((text or "").translate(_FW_DIGITS))
    return float(m.group()) if m else None


# ---------------- 响应解析 ----------------

def parse_response(response: str):
    """→ (fmt_ok, payload)。

    fmt_ok=True 且 payload 为 dict：JSON 合法且核心字段齐全（含可选 cot_steps）。
    否则 payload 为原始文本（供宽松 answer 抽取，模拟「格式乱但内容对」路径）。
    """
    if not response or not response.strip():
        return False, ""
    try:
        obj = json.loads(response)
        if not isinstance(obj, dict):
            raise ValueError
        if any(obj.get(k) is None for k in _REQUIRED_FIELDS):
            # 缺核心字段：可提取结论但格式不完整
            return False, str(obj.get("conclusion") or response)
        return True, obj
    except Exception:
        m = re.search(r"\{.*\}", response, re.DOTALL)   # 尝试抽取内嵌 JSON 片段
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and all(obj.get(k) is not None for k in _REQUIRED_FIELDS):
                    return True, obj
            except Exception:
                pass
        return False, response


def _pred_basis(obj_or_text) -> list:
    """从响应中抽取「预测依据」列表（JSON basis 或原文正则扫规范编号）。"""
    if isinstance(obj_or_text, dict):
        return [str(b) for b in (obj_or_text.get("basis") or [])]
    text = obj_or_text or ""
    found = []
    # 规范编号模式：GB 55034-2022 / JGJ/T 46-2024 / 建办质〔2018〕31号 / 住建部令第37号 等
    pat = re.compile(r"(?<![0-9A-Za-z])(?:(?:GB|JGJ|JG|DGJ|DG|T|CJJ|CECS)[/\s]*T?[\s]*-?\s*[\d]{2,4}(?:-[\d]{2,4})?|"
                     r"建办质[〔\[]?\d{4}[〕\]]?\d{1,2}号|住建部令第37号|建质[〔\[]?\d{4}[〕\]]?\d{1,2}号|GB550\d{2}[-—]\d{4})")
    for m in pat.finditer(text):
        token = m.group(0)
        if token not in found:
            found.append(token)
    return found


def _conclusion(obj_or_text) -> str:
    if isinstance(obj_or_text, dict):
        return str(obj_or_text.get("conclusion") or "")
    return obj_or_text or ""


# ---------------- 维度 1：格式合规分 ----------------

def format_score(response: str, judge_meta: dict) -> float:
    """JSON 语法正确性 + 必需字段 + conclusion 白名单（分级连续，非二元）。

    1.0  合法 JSON，核心字段齐全，conclusion ∈ 白名单（白名单为空视为放行）
    0.8  合法 JSON、字段齐全，但 conclusion 不在白名单内或缺 cot_steps(require_cot)
    0.6  合法 JSON，缺一个核心字段（conclusion/basis/explanation）
    0.4  非合法 JSON，但可抽取到内嵌 {…} JSON 对象
    0.2  非 JSON 的原始文本（非空）
    0.0  空响应
    """
    if not response or not response.strip():
        return 0.0
    try:
        obj = json.loads(response)
        is_dict = isinstance(obj, dict)
    except Exception:
        obj, is_dict = None, False
    if is_dict:
        if any(obj.get(k) is None for k in _REQUIRED_FIELDS):
            return 0.6
        opts = judge_meta.get("conclusion_options") or []
        require_cot = bool(judge_meta.get("require_cot", True))
        if require_cot and not obj.get("cot_steps"):
            return 0.8
        concl = _conclusion(obj)
        if opts and concl not in opts:
            return 0.8
        return 1.0
    # 非合法 JSON：尝试抽取内嵌 JSON 片段
    return 0.4 if re.search(r"\{.*\}", response, re.DOTALL) else 0.2


# ---------------- 维度 2：CoT 推理连贯分（确定性代理）----------------

def cot_score(response: str, judge_meta: dict, parse_cache=None) -> float:
    """思维链推理连贯度（0..1）。结构化代理信号，无 LLM 时也能判分：

    base 0.30   存在 cot_steps 列表
    +0.20       ≥2 步（推理有展开）
    +0.20       存在衔接词（因此/综上/依据…，证据→结论的因果连接）
    +0.30       步骤引用关键证据：命中期望值 / 问题数值 / golden_basis 规范编号
    （cap 1.0；judge_meta.golden_cot 存在时，与金链的逐段命中可再叠加 +0.15）
    """
    if parse_cache is None:
        fmt_ok, payload = parse_response(response)
        parse_cache = payload if fmt_ok else None
    obj = parse_cache if isinstance(parse_cache, dict) else None
    steps = obj.get("cot_steps") if obj else None
    if not steps or not isinstance(steps, list):
        return 0.0
    steps = [str(s) for s in steps if str(s).strip()]
    if not steps:
        return 0.0
    s = 0.30
    if len(steps) >= 2:
        s += 0.20
    joined = "".join(steps)
    # 金链全命中 → 推理链与参考链一致，直接满分（金标准自检/评测理想答案依赖此规则）
    gold = judge_meta.get("golden_cot") or []
    if gold:
        gold_norm = [_norm(str(g)) for g in gold]
        gold_hit = sum(1 for gn in gold_norm if gn and len(gn) >= 4 and gn in _norm(joined))
        if gold_hit / len(gold_norm) >= 1.0:
            return 1.0
    if any(w in joined for w in _COT_LINK):
        s += 0.20
    # 关键证据引用：期望值 / 依据编号（不用通用数字提取——年份/编号数字会误报）
    ref_hit = False
    exp = str(judge_meta.get("expect_value", ""))
    if exp and _norm(exp) and _norm(exp)[:3] in _norm(joined):
        ref_hit = True
    if not ref_hit:
        for g in (judge_meta.get("golden_basis") or []):
            gnorm = _norm(str(g))
            if len(gnorm) >= 3 and gnorm in _norm(joined):
                ref_hit = True
                break
    if ref_hit:
        s += 0.30
    # 与金链逐段语义命中（弱监督：金链片段在步骤中出现的比例）
    if gold:
        hit = sum(1 for gn in gold_norm if gn and len(gn) >= 4 and gn in _norm(joined))
        s += 0.15 * (hit / len(gold_norm))
    return min(1.0, s)


# ---------------- 维度 3：依据溯源精准分 ----------------

def basis_score(response: str, judge_meta: dict, parse_cache=None) -> float:
    """依据溯源精准度（0..1）。

    golden 非空：hit_ratio = 命中金据数 / 金据数（规范编号/条款号归一化相等或互相包含）
                false_ratio = 预测据中非命中数 / 预测据数（捏造惩罚）
                score = clamp(hit_ratio − 0.3·false_ratio)
    golden 为空：有依据得 0.5，无依据 0（不苛求，仅轻微鼓励）
    """
    if parse_cache is None:
        fmt_ok, payload = parse_response(response)
        parse_cache = payload if fmt_ok else None
    pred = _pred_basis(parse_cache if isinstance(parse_cache, dict) else (parse_cache or response))
    gold = [str(g) for g in (judge_meta.get("golden_basis") or [])]
    if not gold:
        return 0.5 if pred else 0.0

    def _match(gn, pn):
        return gn == pn or (len(gn) >= 3 and (gn in pn or pn in gn))

    gold_norm = [_norm(g) for g in gold if _norm(g)]
    pred_norm = [_norm(p) for p in pred if _norm(p)]
    if not gold_norm:
        return 0.5 if pred_norm else 0.0
    hit = sum(1 for gn in gold_norm if any(_match(gn, pn) for pn in pred_norm))
    hit_ratio = hit / len(gold_norm)
    matched_pred = sum(1 for pn in pred_norm if any(_match(gn, pn) for gn in gold_norm))
    false_ratio = (len(pred_norm) - matched_pred) / max(1, len(pred_norm))
    return max(0.0, min(1.0, hit_ratio - 0.3 * false_ratio))


# ---------------- 维度 4：核心判分结论分（连续版 answer_ok）----------------

def answer_ok(judge_meta: dict, text: str) -> bool:
    """旧判定保留：是否答对（布尔）。answer_score 的 mode=exact/contains 复用。"""
    mode = judge_meta.get("mode", "exact")
    expect = str(judge_meta.get("expect_value", "")).strip()
    alias = [str(a) for a in judge_meta.get("alias", []) or []]
    tol = judge_meta.get("tolerance")

    if mode == "boolean" or mode == "exact":
        n = _norm(expect)
        hay = _norm(text)
        if n and (hay == n or n in hay):
            return True
        return any((a := _norm(x)) and (hay == a or a in hay) for x in alias)
    if mode == "contains":
        hay = _norm(text)
        if expect and _norm(expect) in hay:
            return True
        return any(_norm(a) in hay for a in alias)
    if mode == "set_match":
        hay = _norm(text)
        for token in re.split(r"[,，;；、]", expect):
            token = token.strip()
            if not token:
                continue
            if _norm(token) not in hay and not any(_norm(a) in hay for a in alias):
                return False
        return True
    if mode == "numeric":
        got = _extract_numeric(text)
        if got is None:
            return False
        exp = _extract_numeric(expect)
        return exp is not None and abs(got - exp) <= (tol or 0.01)
    return False


def answer_score(response: str, judge_meta: dict, parse_cache=None) -> float:
    """核心判分结论分（连续 0..1）。mode 连续化：

    exact/boolean/contains —— 命中 1.0，未命中 0.0
    set_match            —— 期望项逐段命中比例（连续，旧版全或无 → 平滑）
    numeric              —— 1 − |got−exp|/tol 线性衰减（tol 缺省 = max(0.1·|exp|, 0.5)）
    结论缺失时回退到原文宽松匹配（保证「格式乱但答对」在 answer 维仍拿分）。
    """
    if parse_cache is None:
        fmt_ok, payload = parse_response(response)
        parse_cache = payload if fmt_ok else None
    text = _conclusion(parse_cache)
    mode = judge_meta.get("mode", "exact")
    expect = str(judge_meta.get("expect_value", "")).strip()

    if mode == "set_match":
        hay = _norm(text)
        tokens = [t.strip() for t in re.split(r"[,，;；、]", expect) if t.strip()]
        if not tokens:
            return 1.0
        hits = 0
        for t in tokens:
            tn = _norm(t)
            if tn and (tn in hay or any(_norm(a) in hay for a in (judge_meta.get("alias") or []))):
                hits += 1
        return hits / len(tokens)
    if mode == "numeric":
        got = _extract_numeric(text)
        if got is None:
            return 0.0
        exp = _extract_numeric(expect)
        if exp is None:
            return 0.0
        tol = judge_meta.get("tolerance") or max(0.1 * abs(exp), 0.5)
        return max(0.0, 1.0 - abs(got - exp) / tol)
    return 1.0 if answer_ok(judge_meta, text) else 0.0


# ---------------- 四维分解与总分 ----------------

def score_breakdown(response: str, judge_meta: dict) -> dict:
    """四维分解（含总分）。训练/评测/轨迹统一入口。"""
    fmt_ok, payload = parse_response(response)
    cache = payload if fmt_ok else response   # 解析失败回退原文（宽松匹配路径）
    fmt = format_score(response, judge_meta)
    cot = cot_score(response, judge_meta, cache)
    basis = basis_score(response, judge_meta, cache)
    ans = answer_score(response, judge_meta, cache)
    w = C.REWARD_WEIGHTS
    total = (w["format"] * fmt + w["cot"] * cot + w["basis"] * basis + w["answer"] * ans)
    return {"format": round(fmt, 4), "cot": round(cot, 4), "basis": round(basis, 4),
            "answer": round(ans, 4), "total": round(total, 4), "fmt_ok": fmt_ok}


def score_response(query: str, response: str, judge_meta: dict) -> float:
    """单条回答综合分（0..1 连续）。query 保留以兼容旧接口（未参与判分）。"""
    return score_breakdown(response, judge_meta)["total"]


def check_format(judge_meta: dict, obj_or_text, raw_response: str = "") -> bool:
    """格式合规（布尔，用于报告/兼容）：format 维 ≥ EVAL_FMT_OK_MIN 视为合规。"""
    return format_score(raw_response or (obj_or_text if isinstance(obj_or_text, str) else json.dumps(obj_or_text, ensure_ascii=False)),
                        judge_meta) >= C.EVAL_FMT_OK_MIN


# ---------------- 漏报 / 误报 判定 ----------------

def predict_violation(judge_meta: dict, response: str) -> bool | None:
    """模型是否「判违规」。None 表示该任务无违规语义（如 C3 数值问答）。

    按任务类型定义「结论 → 是否判违规」映射（与 judge_meta.violation_expected 对照）：
      version_abolished  —— conclusion 为 yes/废止 → 违规
      danger_level       —— 结论含「危大」（非「非危大」）→ 违规
      missing_section    —— conclusion 非空且 ≠「无缺失」→ 违规
      threshold_value    —— None（无违规语义，不计入漏报/误报）
    """
    task = judge_meta.get("task_type") or judge_meta.get("check_dim") or ""
    fmt_ok, payload = parse_response(response)
    concl = _conclusion(payload if fmt_ok else None) or (payload if not fmt_ok else "")
    cn = _norm(concl)
    if task == "version_abolished":
        if judge_meta.get("mode") == "contains":
            return None    # 替代编号题无违规语义
        return cn in ("yes",) or any(_norm(a) in cn for a in (judge_meta.get("alias") or []))
    if task == "danger_level":
        if "非危大" in concl:
            return False
        return "危大" in concl
    if task == "missing_section":
        if not concl or "无缺失" in concl:
            return False
        return True
    return None


def violation_expected(judge_meta: dict) -> bool | None:
    """金标准：本样本是否预期为合规漏洞（数据生成器写入）。None=无违规语义。"""
    return judge_meta.get("violation_expected")


def group_advantage(rewards) -> list:
    """组内 z-score（论文原版）：A = (r − mean(r)) / std(r)。纯函数入口，
    与 loss.py `_zscore` 口径一致（总体标准差 + clamp 防全同分除零）。"""
    rs = [float(r) for r in rewards]
    mean = sum(rs) / len(rs)
    std = (sum((r - mean) ** 2 for r in rs) / len(rs)) ** 0.5
    std = max(std, 1e-4)
    return [(r - mean) / std for r in rs]


def build_gold_response(judge_meta: dict) -> str:
    """由 judge_meta.golden_* 构造金标准响应 JSON（SFT 重建 / 数据自检 / 评测理想答案）。"""
    cot = judge_meta.get("golden_cot") or ["依据现行规定，给出确定结论。"]
    if isinstance(cot, str):
        cot = [cot]
    expect = judge_meta.get("expect_value", "")
    basis = judge_meta.get("golden_basis") or []
    expl = judge_meta.get("golden_explanation") or f"依据现行规定，答案为：{expect}"
    return json.dumps({
        "cot_steps": list(cot),
        "conclusion": expect,
        "basis": list(basis),
        "explanation": expl,
    }, ensure_ascii=False)


def rebuild_standard_json(judge_meta: dict, raw_answer: str) -> str | None:
    """高奖励但格式不完美的轨迹，SFT 语料重建：用判分金据拼合规 JSON。失败返回 None。"""
    try:
        return build_gold_response(judge_meta)
    except Exception:
        return None
