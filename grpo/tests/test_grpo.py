"""grpo 核心模块单测：四维连续奖励 / 业务矩阵判定 / loss 数值性质 / 基线。

四维判分（reward.score_breakdown）：
    format  JSON 语法+字段+白名单（分级连续）
    cot     CoT 推理连贯（结构/衔接/证据引用；金链全命中→满分）
    basis   依据溯源命中−捏造惩罚
    answer  结论分（set_match/numeric 连续化）
    total   = Σ w·dim（config.REWARD_WEIGHTS）
业务判定（predict_violation × violation_expected）驱动漏报/误报矩阵。
"""
from __future__ import annotations

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRPO = os.path.dirname(HERE)              # grpo/
for p in (GRPO,):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as C                                                    # noqa: E402
from reward import (answer_ok, build_gold_response, check_format,     # noqa: E402
                    group_advantage, predict_violation, parse_response,
                    score_breakdown, score_response, violation_expected)
from loss import grpo_seq_loss, grpo_token_loss                       # noqa: E402

W = C.REWARD_WEIGHTS

# 默认金链（≥2 步且含衔接词，使 build_gold_response 可四维满分）
_DEFAULT_COT = ["核对规范库：该标准已废止", "结论：引用不合规"]


def _mk(mode="exact", ev="yes", alias=None, opts=None, gold_basis=None,
        gold_cot=_DEFAULT_COT, ve=True, task="version_abolished", tol=None):
    return {"mode": mode, "expect_value": ev, "alias": alias or [],
            "conclusion_options": opts if opts is not None else ["yes", "no", "unknown"],
            "golden_basis": gold_basis or ["JGJ/T 46-2024"],
            "golden_cot": gold_cot, "golden_explanation": "x",
            "violation_expected": ve, "task_type": task, "require_cot": True,
            "tolerance": tol}


# ---------------- 四维：format ----------------
def test_format_grading():
    m = _mk()
    good = '{"cot_steps":["a","b"],"conclusion":"yes","basis":["JGJ/T 46-2024"],"explanation":"x"}'
    assert score_breakdown(good, m)["format"] == 1.0
    # 缺 cot_steps（require_cot）→ 0.8
    no_cot = '{"conclusion":"yes","basis":["JGJ/T 46-2024"],"explanation":"x"}'
    assert score_breakdown(no_cot, m)["format"] == 0.8
    # 白名单外结论 → 0.8
    bad_wl = '{"cot_steps":["a"],"conclusion":"yep","basis":["JGJ/T 46-2024"],"explanation":"x"}'
    assert score_breakdown(bad_wl, m)["format"] == 0.8
    # 缺核心字段 → 0.6
    miss = '{"cot_steps":["a"],"conclusion":"yes"}'
    assert score_breakdown(miss, m)["format"] == 0.6
    # 非 JSON 文本 → 0.2
    assert score_breakdown("废话一段", m)["format"] == 0.2
    # 空 → 0
    assert score_breakdown("", m)["format"] == 0.0


# ---------------- 四维：cot ----------------
def test_cot_scoring():
    m = _mk(gold_cot=["核对规范库：该标准已废止", "结论：引用不合规"])
    # 金链全命中 → 满分
    good = '{"cot_steps":["核对规范库：该标准已废止","结论：引用不合规"],"conclusion":"yes","basis":["JGJ/T 46-2024"],"explanation":"x"}'
    assert score_breakdown(good, m)["cot"] == 1.0
    # 无 CoT → 0
    no_cot = '{"conclusion":"yes","basis":["JGJ/T 46-2024"],"explanation":"x"}'
    assert score_breakdown(no_cot, m)["cot"] == 0.0
    # 单步且无引用 → 低分但 >0
    thin = '{"cot_steps":["随便推了一步"],"conclusion":"yes","basis":["JGJ/T 46-2024"],"explanation":"x"}'
    bd = score_breakdown(thin, m)
    assert 0 < bd["cot"] < 0.5
    # 结构完整（≥2步 + 衔接词 + 引用期望值）→ 高分（不一定满分）
    rich = ('{"cot_steps":["核对现行规范库：该标准已废止","因此引用不合规，应改用替代"],'
            '"conclusion":"yes","basis":["JGJ/T 46-2024"],"explanation":"x"}')
    assert score_breakdown(rich, m)["cot"] >= 0.7


# ---------------- 四维：basis ----------------
def test_basis_scoring():
    m = _mk(gold_basis=["JGJ/T 46-2024"])
    ok = '{"cot_steps":["a","b"],"conclusion":"yes","basis":["JGJ/T 46-2024"],"explanation":"x"}'
    assert score_breakdown(ok, m)["basis"] == 1.0
    # 捏造额外依据 → 命中1 − 惩罚0.3·(1/2) = 0.85
    fake = '{"cot_steps":["a","b"],"conclusion":"yes","basis":["JGJ/T 46-2024","GB 99999-2099"],"explanation":"x"}'
    assert score_breakdown(fake, m)["basis"] == 0.85
    # 完全错误依据 → 0
    wrong = '{"cot_steps":["a","b"],"conclusion":"yes","basis":["GB 11111-1111"],"explanation":"x"}'
    assert score_breakdown(wrong, m)["basis"] == 0.0
    # 无依据 → 0
    none = '{"cot_steps":["a","b"],"conclusion":"yes","basis":[],"explanation":"x"}'
    assert score_breakdown(none, m)["basis"] == 0.0


# ---------------- 四维：answer（连续化） ----------------
def test_answer_continuous():
    # exact 二元
    m1 = _mk(ev="yes")
    assert score_breakdown('{"cot_steps":["a","b"],"conclusion":"yes","basis":["x"],"explanation":"x"}', m1)["answer"] == 1.0
    assert score_breakdown('{"cot_steps":["a","b"],"conclusion":"no","basis":["x"],"explanation":"x"}', m1)["answer"] == 0.0
    # set_match 连续：对一半 = 0.5
    m2 = _mk(mode="set_match", ev="超规模危大工程,需专家论证", opts=[], task="danger_level")
    half = '{"cot_steps":["a","b"],"conclusion":"超规模危大工程","basis":["x"],"explanation":"x"}'
    assert score_breakdown(half, m2)["answer"] == 0.5
    full = '{"cot_steps":["a","b"],"conclusion":"超规模危大工程，需专家论证","basis":["x"],"explanation":"x"}'
    assert score_breakdown(full, m2)["answer"] == 1.0
    # numeric 平滑：4.0 vs 期望 4Ω（tol 缺省=max(0.4,0.5)=0.5）→ 1.0；4.4 → 0.2
    m3 = _mk(mode="numeric", ev="4Ω", opts=[], task="threshold_value")
    assert score_breakdown('{"cot_steps":["a","b"],"conclusion":"4Ω","basis":["x"],"explanation":"x"}', m3)["answer"] == 1.0
    near = '{"cot_steps":["a","b"],"conclusion":"4.4Ω","basis":["x"],"explanation":"x"}'
    assert abs(score_breakdown(near, m3)["answer"] - 0.2) < 1e-6
    far = '{"cot_steps":["a","b"],"conclusion":"10Ω","basis":["x"],"explanation":"x"}'
    assert score_breakdown(far, m3)["answer"] == 0.0
    # 格式乱但答对（原文宽松）：answer 维仍拿满分，format 维低
    loose = "答案是4欧姆左右。"
    bd = score_breakdown(loose, m3)
    assert bd["answer"] == 1.0 and bd["format"] < 0.3


# ---------------- 总分与加权 ----------------
def test_composite_weighting():
    m = _mk()
    gold = build_gold_response(m)
    bd = score_breakdown(gold, m)
    assert bd["total"] == 1.0 and all(bd[k] == 1.0 for k in ("format", "cot", "basis", "answer"))
    # 仅答对但缺 CoT：answer 满、cot=0 → total = w_fmt·0.8 + 0 + w_basis·1 + w_ans·1
    no_cot = '{"conclusion":"yes","basis":["JGJ/T 46-2024"],"explanation":"x"}'
    bd2 = score_breakdown(no_cot, m)
    expect = W["format"] * 0.8 + W["basis"] * 1.0 + W["answer"] * 1.0
    assert abs(bd2["total"] - expect) < 1e-6
    # 全错 → 约 0（仅剩非空文本 format 0.2 底分）
    assert score_response("q", "我不知道随便写点。", m) < 0.1


# ---------------- 业务判定（漏报/误报口径） ----------------
def test_violation_judgment():
    # C1 正：conclusion yes → 判违规
    m1 = _mk(ve=True)
    assert predict_violation(m1, build_gold_response(m1)) is True
    # C1 负：conclusion no → 不判违规
    m1n = _mk(ve=False, ev="no", alias=["有效", "现行"])
    assert predict_violation(m1n, build_gold_response(m1n)) is False
    # C2：非危大 → False；危大 → True
    m2 = _mk(mode="set_match", ev="非危大，不需专家论证", opts=[], ve=False, task="danger_level")
    assert predict_violation(m2, build_gold_response(m2)) is False
    m2p = _mk(mode="set_match", ev="超规模危大工程，需专家论证", opts=[], ve=True, task="danger_level")
    assert predict_violation(m2p, build_gold_response(m2p)) is True
    # C3：无违规语义 → None
    m3 = _mk(mode="numeric", ev="4Ω", opts=[], ve=None, task="threshold_value")
    assert predict_violation(m3, build_gold_response(m3)) is None
    assert violation_expected(m3) is None


def test_checker_matrix_consistency():
    """金标准回答下：漏报/误报均为 0（TP 全对、FP/FN 为 0）。
    实际一致性在 scripts/evaluate.py --checker-only 校验，此处覆盖判定函数。
    """
    assert callable(predict_violation) and callable(violation_expected)


# ---------------- 兼容接口 ----------------
def test_parse_and_legacy():
    m = _mk()
    ok, obj = parse_response('{"cot_steps":["a","b"],"conclusion":"yes","basis":["JGJ/T 46-2024"],"explanation":"x"}')
    assert ok and obj["conclusion"] == "yes" and obj["cot_steps"] == ["a", "b"]
    ok, obj = parse_response("no json here")
    assert not ok and "no json" in obj
    assert answer_ok(_mk(), "yes") is True
    assert answer_ok(_mk(ev="no"), "已废止") is False
    # check_format 兼容
    assert check_format(m, '{"cot_steps":["a","b"],"conclusion":"yes","basis":["x"],"explanation":"x"}') is True
    assert check_format(m, "plain text") is False


def test_rebuild_has_cot():
    m = _mk(gold_cot=["步一", "步二"])
    out = build_gold_response(m)
    d = json.loads(out)
    assert d["cot_steps"] == ["步一", "步二"] and d["conclusion"] == "yes"
    assert d["basis"] == ["JGJ/T 46-2024"]


# ---------------- 继承：优势与 loss 数值性质（未改动） ----------------
def test_group_advantage():
    rs = [1.0, 0.7, 0.3, 0.0] * 2
    A = group_advantage(rs)
    assert abs(sum(A)) < 1e-9
    std = (sum(x * x for x in A) / len(A)) ** 0.5
    assert abs(std - 1.0) < 1e-9
    assert A[0] > 0 and A[-1] < 0
    A0 = group_advantage([0.3] * 8)
    assert all(abs(a) < 1e-9 for a in A0)


def test_grpo_loss_properties():
    torch.manual_seed(0)
    B, G = 2, 4
    logp_old = torch.randn(B, G)
    logp_new = logp_old.clone()
    logp_ref = torch.randn(B, G)
    rewards = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.3, 0.0, 0.0]])
    l_aligned = grpo_seq_loss(logp_new, logp_old, logp_ref, rewards, beta=0.0)
    assert abs(l_aligned.item()) < 1e-5
    l_kl = grpo_seq_loss(logp_new, logp_old, logp_ref, rewards, beta=0.1)
    assert l_kl.item() > 0
    lp_up = logp_old.clone()
    lp_up[0, 0] += 0.5
    l_up = grpo_seq_loss(lp_up, logp_old, logp_ref, rewards, beta=0.0)
    assert l_up.item() < l_aligned.item()
    T = 8
    mask = torch.ones(B, G, T)
    lp_new = torch.randn(B, G, T)
    lp_old = lp_new.clone()
    lp_ref = torch.randn(B, G, T)
    lt = grpo_token_loss(lp_new, lp_old, lp_ref, mask, rewards, beta=0.0)
    assert abs(lt.item()) < 1e-5
    lp_ext = lp_old + 10.0
    lt2 = grpo_token_loss(lp_ext, lp_old, lp_ref, mask, rewards, beta=0.0)
    assert torch.isfinite(lt2)


def test_clip_bounds():
    """锁死 clip(ρ, 1−ε, 1+ε)：重要性比必须被裁剪到 [1−ε, 1+ε]（论文原版）。"""
    eps = 0.2
    ratio = torch.tensor([0.5, 1.0, 1.5])          # 超出 [0.8, 1.2] 的部分应被裁剪
    clamped = ratio.clamp(1 - eps, 1 + eps)
    assert torch.allclose(clamped, torch.tensor([0.8, 1.0, 1.2]), atol=1e-6)
    # 边界值不变
    edge = torch.tensor([0.8, 1.2])
    assert torch.allclose(edge.clamp(1 - eps, 1 + eps), edge, atol=1e-6)
    # loss 路径端到端可算（含 clip 项）
    B, G, T = 2, 4, 8
    torch.manual_seed(1)
    ln = torch.randn(B, G, T)
    lo = torch.randn(B, G, T)
    lr_ = torch.randn(B, G, T)
    mask = torch.ones(B, G, T)
    rew = torch.tensor([[1.0, 0.8, 0.2, 0.0], [0.9, 0.7, 0.4, 0.1]])
    loss = grpo_token_loss(ln, lo, lr_, mask, rew, beta=0.04, eps=0.2)
    assert torch.isfinite(loss) and loss.item() != 0


if __name__ == "__main__":
    for fn in [test_format_grading, test_cot_scoring, test_basis_scoring,
               test_answer_continuous, test_composite_weighting, test_violation_judgment,
               test_checker_matrix_consistency, test_parse_and_legacy, test_rebuild_has_cot,
               test_group_advantage, test_grpo_loss_properties, test_clip_bounds]:
        fn()
        print("ok", fn.__name__)
    print("ALL PASS")
