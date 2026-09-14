"""评估：四维连续奖励 + 多维业务对齐评测矩阵（要素②）。

替代旧版「平均分/正确率/格式率」单一报告。新报告结构：

A. 基础奖励（四维）：
   综合分均值 / format·cot·basis·answer 四维均值 / 答案正确率(answer≥阈值) /
   格式合规率(format≥阈值) / 满分率(total=1.0)

B. 业务对齐矩阵（仅 violation_expected 非空的样本，即真实施工审查口径）：
   TP/TN/FP/FN
   合规漏洞漏报率  = FN/(TP+FN)   （应判违规未判 —— 越低越好）
   假阳性误报率    = FP/(FP+TN)   （不应判违规误判 —— 越低越好）
   检出召回率      = TP/(TP+FN)
   溯源条文准确率  = mean(basis 维)（依据条文命中金据的比例）
   CoT 逻辑可解释性 = mean(cot 维)（确定性代理）；
                    --judge-llm 时由大模型裁判（DeepSeek）按国家标准复核推理步骤

两种模式：
   --checker-only 本机可跑：判分器自检（金标准四维满分 / 无意义=0 / 格式乱答对可达 /
                   缺 CoT 扣分 / 捏造依据扣分 / 漏报误报标签自洽）
   --model <路径> GPU：加载模型（含合并链任意阶段）对 val 逐条采样 1 次，输出 A+B 报告；
                   加 --judge-llm 启用 LLM 裁判评 CoT（需 DEEPSEEK_API_KEY）

用法：
   python grpo/scripts/evaluate.py --checker-only
   python grpo/scripts/evaluate.py --model data/models/qwen-grpo/final --temperature 0.0
   python grpo/scripts/evaluate.py --model <路径> --judge-llm   # CoT 用大模型裁判
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GRPO = os.path.dirname(HERE)
ROOT = os.path.dirname(GRPO)
for p in (GRPO,):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as C                          # noqa: E402
from prompts import make_system_prompt      # noqa: E402
from reward import (build_gold_response, score_breakdown,  # noqa: E402
                    predict_violation, violation_expected,
                    format_score, cot_score, basis_score, answer_score)


def load_val(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"[eval] val 集 {len(rows)} 条")
    return rows


def evidence_text_of(row) -> str:
    ev = row.get("agent_scenario", {}).get("evidence") or []
    return "\n".join(f"- [{e['source']}] {e['clause_no']}: {e['text']}" for e in ev)


# ---------------- 业务矩阵 ----------------

def report_matrix(bds, val, title, judge_llm=None):
    """bds: list of score_breakdown dict（与 val 一一对应）。输出 A+B 报告。"""
    n = len(bds)
    if n == 0:
        print(f"\n== {title} (n=0) ==")
        return
    avg = sum(b["total"] for b in bds) / n
    full = sum(1 for b in bds if b["total"] >= 1.0)
    ans_ok = sum(1 for b in bds if b["answer"] >= C.EVAL_ANSWER_OK_MIN)
    fmt_ok = sum(1 for b in bds if b["format"] >= C.EVAL_FMT_OK_MIN)
    dims = {k: sum(b[k] for b in bds) / n for k in ("format", "cot", "basis", "answer")}

    # 业务矩阵（仅违规语义样本）
    vp = [(it, b) for it, b in zip(val, bds)
          if violation_expected(it["judge_meta"]) is not None]
    tp = tn = fp = fn = 0
    for it, b in vp:
        exp = it["judge_meta"].get("violation_expected")
        pred = predict_violation(it["judge_meta"], it.get("_response", ""))
        if exp is True and pred is True:
            tp += 1
        elif exp is True and pred is not True:
            fn += 1
        elif exp is False and pred is True:
            fp += 1
        else:
            tn += 1
    miss_rate = fn / max(1, tp + fn)          # 漏报率
    fp_rate = fp / max(1, fp + tn)            # 误报率
    recall = tp / max(1, tp + fn)             # 检出召回率
    spec = tn / max(1, fp + tn)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * recall / max(1e-9, prec + recall)
    basis_acc = sum(b["basis"] for b in bds) / n          # 溯源条文准确率

    cot_mean = dims["cot"]
    if judge_llm is not None:
        cot_llm = [judge_llm(it, b) for it, b in zip(val, bds) if b["cot"] > 0]
        cot_mean = (sum(cot_llm) / max(1, len(cot_llm))) if cot_llm else cot_mean

    print(f"\n== {title} (n={n}) ==")
    print("— A 基础奖励（四维）—")
    print(f"综合分均值 {avg:.4f} | 满分率 {full/max(1,n):.4f} | 答案正确率 {ans_ok/max(1,n):.4f} | 格式合规率 {fmt_ok/max(1,n):.4f}")
    print(f"  维度均值  format={dims['format']:.4f}  cot={dims['cot']:.4f}  basis={dims['basis']:.4f}  answer={dims['answer']:.4f}")
    print("— B 业务对齐矩阵 —")
    print(f"样本(n={len(vp)}): TP {tp} / TN {tn} / FP {fp} / FN {fn}")
    print(f"合规漏洞漏报率 {miss_rate:.4f} | 假阳性误报率 {fp_rate:.4f}")
    print(f"检出召回率 {recall:.4f} | 特异度 {spec:.4f} | 精确率 {prec:.4f} | F1 {f1:.4f}")
    print(f"溯源条文准确率 {basis_acc:.4f} | CoT 逻辑可解释性 {cot_mean:.4f}"
          + ("（LLM 裁判）" if judge_llm is not None else "（确定性代理）"))


# ---------------- LLM 裁判（CoT 逻辑可解释性） ----------------

_JUDGE_PROMPT = """你是建筑施工安全规范审查的独立裁判。请按国家标准复核以下回答的推理链（CoT），
从三个维度打分（每项 0-10 分）：
1) 逻辑连贯性：推理步骤是否由证据到结论逐步收敛、无跳跃；
2) 依据合规性：引用的规范编号/条款号是否与问题匹配、结论是否符合现行国家标准口径；
3) 数值/事实正确性：推理中出现的数值、阈值、日期与事实是否与国标一致。
只输出 JSON：{{"logic": 0-10, "basis": 0-10, "fact": 0-10, "comment": "一句话"}}

【问题】{query}
【金标准结论】{gold_conclusion}
【金标准依据】{gold_basis}
【待评判回答】
{response}"""


def make_llm_judge():
    """返回 judge_llm(row, breakdown)->float 函数。DeepSeek 裁判，缺 Key 返回 None。"""
    sys.path.insert(0, os.path.join(ROOT, "agent", "src"))
    from llm.deepseek import DeepSeekLLM  # noqa: E402
    try:
        llm = DeepSeekLLM()
    except RuntimeError as e:
        print(f"[judge-llm] 未启用：{e}")
        return None

    def judge(row, bd):
        m = row["judge_meta"]
        resp = row.get("_response", "")
        prompt = _JUDGE_PROMPT.format(
            query=row.get("query", "")[:200],
            gold_conclusion=m.get("expect_value", ""),
            gold_basis="、".join(str(x) for x in (m.get("golden_basis") or [])),
            response=resp[:1200])
        try:
            out = llm.complete([{"role": "user", "content": prompt}], temperature=0.0)
        except Exception as e:
            print(f"[judge-llm] 调用失败 {row.get('id')}: {e}")
            return bd["cot"]
        m2 = re.search(r"\{.*\}", out, re.DOTALL)
        if not m2:
            m3 = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", out)
            return float(m3.group(1)) / 10.0 if m3 else bd["cot"]
        try:
            d = json.loads(m2.group(0))
            return (float(d.get("logic", 0)) + float(d.get("basis", 0)) + float(d.get("fact", 0))) / 30.0
        except Exception:
            return bd["cot"]

    return judge


# ---------------- 判分器自检（checker-only） ----------------

def checker_only(val):
    full_ok = zero = cot_penalty = basis_penalty = 0
    n = min(len(val), 300)
    for it in val[:n]:
        m = it["judge_meta"]
        if score_breakdown(build_gold_response(m), m)["total"] >= 1.0:
            full_ok += 1
        # 无意义回答 → 综合分 ≈ 0（<0.1；非空文本仍有 0.2×0.15 格式底分，非精确 0）
        if score_breakdown("我不知道，随便写点。", m)["total"] < 0.1:
            zero += 1
        # 缺 CoT：format 应降为 0.8（require_cot），total 下降
        no_cot = json.dumps({"conclusion": m["expect_value"],
                             "basis": m.get("golden_basis", []), "explanation": "x"}, ensure_ascii=False)
        if score_breakdown(no_cot, m)["format"] <= C.EVAL_FMT_OK_MIN:
            cot_penalty += 1
        # 捏造依据：basis 维应显著低于满分
        fake = json.dumps({"cot_steps": ["推理。", "结论。"], "conclusion": m["expect_value"],
                           "basis": ["GB 99999-2099", "JGJ 00000-0000"],
                           "explanation": "x"}, ensure_ascii=False)
        if score_breakdown(fake, m)["basis"] < 0.5:
            basis_penalty += 1
    # 漏报/误报标签自洽（金标准回答）
    v = [it for it in val if violation_expected(it["judge_meta"]) is not None]
    err = sum(1 for it in v
              if predict_violation(it["judge_meta"], build_gold_response(it["judge_meta"]))
              != it["judge_meta"].get("violation_expected"))
    print(f"[checker] 金标准四维满分 {full_ok}/{n} | 无意义回答=0 {zero}/{n} "
          f"| 缺CoT格式降级 {cot_penalty}/{n} | 捏造依据扣分 {basis_penalty}/{n}")
    print(f"[checker] 漏报/误报标签自洽（不一致） {err}/{len(v)}")


# ---------------- 模型评测 ----------------

def model_eval(model_path, val, temperature, device, max_new=192, judge_llm=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True).eval()

    bds = []
    for it in val:
        ev_text = evidence_text_of(it)
        msgs = [{"role": "system", "content": make_system_prompt(it["task_type"], ev_text)},
                {"role": "user", "content": it["query"]}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      tokenize=True, return_tensors="pt")
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(model.device)
        # 逐条前清缓存：推理峰值偶发 OOM（某条超长序列），清掉上一轮残留
        torch.cuda.empty_cache()
        text = ""
        for trial_mn in (max_new, 128, 64):      # OOM 自动降级：256->128->64，保证全量跑完
            try:
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=ids, do_sample=temperature > 0,
                        temperature=max(temperature, 1e-3),
                        top_p=0.95, max_new_tokens=trial_mn,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id)
                text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"      [warn] {it.get('id')} OOM@max_new={trial_mn}，降级重试", flush=True)
        torch.cuda.empty_cache()
        it["_response"] = text
        bds.append(score_breakdown(text, it["judge_meta"]))
        if len(bds) <= 3:
            print(f"  [{it['task_type']}] Q: {it['query'][:44]}...")
            print(f"      A: {text[:90]!r}  total={bds[-1]['total']}")
        if len(bds) % 20 == 0:
            print(f"  ... {len(bds)}/{len(val)}  均分 "
                  f"{sum(b['total'] for b in bds) / len(bds):.3f}", flush=True)
    report_matrix(bds, val, f"model {os.path.basename(model_path)}", judge_llm)

    # 落盘：逐条响应 + 四维分解（供跨模型对比与复算）
    out = os.environ.get("EVAL_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump([{"id": it.get("id"), "task_type": it.get("task_type"),
                        "response": it.get("_response"), "breakdown": b}
                       for it, b in zip(val, bds)], f, ensure_ascii=False, indent=2)
        print(f"[eval] 明细 -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=os.path.join(ROOT, "data", "grpo", "val.jsonl"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 条（0=全量）")
    ap.add_argument("--checker-only", action="store_true")
    ap.add_argument("--judge-llm", action="store_true",
                    help="CoT 可解释性改用大模型裁判（需 DEEPSEEK_API_KEY；默认确定性代理）")
    args = ap.parse_args()

    val = load_val(args.val)
    if args.limit:
        val = val[:args.limit]
        print(f"[eval] 截取前 {len(val)} 条（--limit）")
    judge_llm = make_llm_judge() if args.judge_llm else None
    if args.checker_only or args.model is None:
        checker_only(val)
        if args.model is None and not args.checker_only:
            print("提示：加 --model <权重路径> 跑模型评测")
        return
    model_eval(args.model, val, args.temperature, args.device,
               max_new=args.max_new_tokens, judge_llm=judge_llm)


if __name__ == "__main__":
    main()
