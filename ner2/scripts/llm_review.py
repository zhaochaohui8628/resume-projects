"""P2 LLM 二次校验与纠错（对远程监督弱标签逐条复核）。

⚠️ 本机默认**不实际运行**（用户硬约束：清洗/纠错由模型扮演 LLM 逐条手工执行，
产物 judgments.jsonl 格式与本脚本输出完全一致）。本脚本保留完整实现，便于
①未来在获得 API Key 后一键批量复核；②用 --mock 验证管道；③作为手工判定的 schema 参照。

用法（项目根）：
    # 模式A：真实调用 LLM（需 DEEPSEEK_API_KEY 与 llm client）
    python ner2/scripts/llm_review.py --weak ner2/data/phase1/weak.jsonl \
        --out ner2/data/phase2/judgments.jsonl \
        --llm agent.src.llm.deepseek:DeepSeekClient

    # 模式B：mock 判定（管道自检，勿用于正式数据）
    python ner2/scripts/llm_review.py --weak ner2/data/phase1/weak.jsonl --mock

输出 judgments.jsonl 每行：
{"qid","text","verdict":"keep|fix|drop","entities":[...],"issues":[...],"reason":"..."}
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ner2.src.common.paths import JUDGMENTS, WEAK, ROOT
from ner2.src.review.parser import normalize_judgment, parse_llm_json
from ner2.src.review.prompts import build_review_messages


def _load_client(spec: str):
    """'module.path:ClassName' -> 实例（实现 complete(messages)->str 即可）。"""
    mod_path, _, cls_name = spec.partition(":")
    if not cls_name:
        raise SystemExit("--llm 需 'module:Class' 形式，例如 agent.src.llm.deepseek:DeepSeekClient")
    sys.path.insert(0, ROOT)
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls()


def mock_judge(text: str, ents: list[dict]) -> dict:
    """确定性 mock：全部 keep（管道自检用，非真实判定）。"""
    return {"verdict": "keep", "entities": [
        {"type": e["type"], "start": e["start"], "end": e["end"]} for e in ents],
        "issues": [], "reason": "mock"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weak", default=WEAK)
    ap.add_argument("--out", default=JUDGMENTS)
    ap.add_argument("--llm", default="", help="module:Class 形式的 LLM client")
    ap.add_argument("--mock", action="store_true", help="确定性 mock 判定（管道自检）")
    ap.add_argument("--max-samples", type=int, default=0, help="0=全部")
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()

    if not os.path.exists(args.weak):
        raise SystemExit(f"[error] 缺弱标数据 {args.weak}，先跑 weak_label.py")
    if not args.mock and not args.llm:
        print("[info] 未提供 --llm：本机默认不调 LLM（硬约束），"
              "请由模型逐条手工判定，产物格式见脚本头注释。")
        print("[info] 管道自检请加 --mock。")
        return

    client = _load_client(args.llm) if args.llm else None
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = [json.loads(l) for l in open(args.weak, encoding="utf-8")]
    if args.max_samples:
        rows = rows[: args.max_samples]

    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if args.mock:
                obj = mock_judge(r["text"], r["entities"])
            else:
                msgs = build_review_messages(r["text"], r["entities"])
                raw = client.complete(msgs)
                obj = parse_llm_json(raw)
            j = normalize_judgment(r["text"], obj, r["qid"])
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
            if (i + 1) % args.batch == 0:
                print(f"  ...{i + 1}/{len(rows)}")
    print(f"-> {args.out}（{len(rows)} 条）")


if __name__ == "__main__":
    main()
