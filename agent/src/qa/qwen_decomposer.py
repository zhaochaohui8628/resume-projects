"""拆解器 LLM 增强 —— 本地微调 Qwen2.5-3B（grpo 产物 final）分解复合问句。

设计：
- **lazy 单例**：首次调用才加载模型（5.75G bf16），后续复用；显存/内存不足自动回落规则。
- **LLM 优先 + 规则兜底**：LLM 返回 JSON 拆解槽位；解析失败/不可用 → 回落 decompose() 纯规则。
- **低内存加载**：复用 grpo/scripts/load_qwen.py 的 load_qwen_lowmem（逐张量流式，规避 mmap OSError 1455）。
- **模型路径**：优先 data/models/qwen-grpo/final（grpo 微调合并产物），回退 Qwen2.5-3B-Instruct 基座。

拆解输出与 decompose() 同构：{category, subject, equipment, capacity, unit,
dimensions: [{key,label,enabled}], is_super_scale} —— 组装器无需改动。

⚠️ 显存占用：本机训练中（nvidia-smi ~7.3G/8.2G 已用）时加载必 OOM → 自动回落规则，
   训练结束释放显存后自动启用 LLM（无需改代码）。
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.dirname(HERE)                     # agent/src
WS = os.path.dirname(os.path.dirname(AGENT_SRC))      # 工作区根
GRPO_SCRIPTS = os.path.join(WS, "grpo", "scripts")
MODEL_FINAL = os.path.join(WS, "data", "models", "qwen-grpo", "final")
MODEL_BASE = os.path.join(WS, "data", "models", "Qwen2.5-3B-Instruct")

_llm = None           # (model, tokenizer) 单例
_llm_failed = False   # 一次性失败标记（避免反复尝试加载）

# 维度槽位（与 decompose.py 一致）
_DIMENSIONS = [
    {"key": "risks",      "label": "风险点",       "kw": ("风险", "危险源", "隐患")},
    {"key": "plan",       "label": "方案编制内容",  "kw": ("方案编制", "编制内容", "编制要求", "编制")},
    {"key": "controls",   "label": "风险管控清单",  "kw": ("管控", "控制措施", "预防措施", "清单")},
    {"key": "acceptance", "label": "验收节点",      "kw": ("验收", "检查节点", "验评")},
]

_SYSTEM = """你是施工方案合规审查系统的问句拆解器。把用户的复合问句拆成结构化槽位。
只输出 JSON，不要解释。JSON 格式：
{
  "category": "起重吊装|基坑工程|模板支撑|脚手架|拆除|暗挖|幕墙安装|人工挖孔桩|钢结构安装|null",
  "subject": "吊装/施工对象，无则 null",
  "equipment": "设备，无则 null",
  "capacity": 数值或 null,
  "unit": "t|kN|m|mm|null",
  "dimensions": ["risks", "plan", "controls", "acceptance"]  # 问句提到的维度子集
}
维度含义：risks=风险点/危险源；plan=方案编制内容/编制要求；controls=风险管控清单/控制措施；
acceptance=验收节点/检查验收。问句提到哪个就列哪个；都不明确则给全部四个。"""


def _load_model():
    """lazy 加载（单例）。返回 (model, tok)；失败返回 (None, None) 并置 _llm_failed。"""
    global _llm, _llm_failed
    if _llm is not None:
        return _llm
    if _llm_failed:
        return None, None
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        # 显存预检：用 mem_get_info 取真实可用物理显存（<7G 放弃；bf16 需 5.75G 空壳 + KV cache，
        # 且 WDDM 下分配有碎片开销，7G 阈值更稳）
        _ = torch.zeros(1, device="cuda")          # 触发上下文初始化
        _free, _total = torch.cuda.mem_get_info(0)
        free_gb = _free / 1024**3
        if free_gb < 7.0:
            print(f"[qwen_decomposer] 显存不足（free {free_gb:.1f}G < 7G），回落规则拆解", flush=True)
            _llm_failed = True
            return None, None
        if GRPO_SCRIPTS not in sys.path:
            sys.path.insert(0, GRPO_SCRIPTS)
        from load_qwen import load_qwen_lowmem
        model_dir = MODEL_FINAL if os.path.isdir(MODEL_FINAL) else MODEL_BASE
        print(f"[qwen_decomposer] 加载 {model_dir} ...", flush=True)
        model, tok = load_qwen_lowmem(model_dir, device="cuda")
        _llm = (model, tok)
        return _llm
    except Exception as e:
        print(f"[qwen_decomposer] LLM 加载失败，回落规则拆解：{type(e).__name__}: {e}", flush=True)
        _llm_failed = True
        return None, None


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def llm_decompose(query: str) -> dict | None:
    """LLM 拆解。返回与 decompose() 同构的槽位 dict；不可用/失败返回 None（调用方回落规则）。"""
    model, tok = _load_model()
    if model is None:
        return None
    try:
        import torch
        msgs = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": query}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, max_new_tokens=128, do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        out = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        d = _extract_json(out)
        if not d:
            return None
        # 归一化 category（LLM 可能给中文名，映射回 key）
        cat = d.get("category")
        if cat and cat not in ("null", None):
            _ALIAS = {"起重吊装及安装拆卸工程": "起重吊装", "深基坑工程": "基坑工程",
                      "模板工程及支撑体系": "模板支撑", "脚手架工程": "脚手架"}
            cat = _ALIAS.get(cat, cat)
        # 归一化 dimensions
        dim_keys = {x["key"] for x in _DIMENSIONS}
        dims = [x for x in (d.get("dimensions") or []) if x in dim_keys]
        if not dims:                       # LLM 没给维度 → 全开
            dims = [x["key"] for x in _DIMENSIONS]
        dims_out = [{"key": x["key"], "label": x["label"], "enabled": x["key"] in dims}
                    for x in _DIMENSIONS]
        cap = d.get("capacity")
        try:
            cap = float(cap) if cap not in (None, "null") else None
        except (TypeError, ValueError):
            cap = None
        unit = d.get("unit") if d.get("unit") not in (None, "null") else None
        return {
            "category": cat if cat not in (None, "null") else None,
            "subject": d.get("subject") if d.get("subject") not in (None, "null") else None,
            "equipment": d.get("equipment") if d.get("equipment") not in (None, "null") else None,
            "capacity": cap,
            "unit": unit,
            "dimensions": dims_out,
            "query": query,
            "llm": True,
        }
    except Exception as e:
        print(f"[qwen_decomposer] LLM 拆解失败，回落规则：{type(e).__name__}: {e}", flush=True)
        return None
