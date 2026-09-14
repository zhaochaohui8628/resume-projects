"""级联混合提取管道（Cascade Pipeline）。

架构（用户 2026-09-11 定稿，取代旧 ner 的 RuleNER/PromptNER/BertNER 三后端并存）：
    层1 确定性规则/词典（rule_layer）—— 高精度、零延迟、可解释
        · 规范编号（标准号/公文号正则）
        · 危大类别（封闭关键词类目）
        · 高置信封闭专有名词（设备型号、工程类型、工序、参数词典最长匹配）
    层2 微调轻量实体模型（model_layer）—— 兜底复杂上下文
        · 处理层1 未覆盖的开放词表 / 边界歧义 / 类型歧义 / 未登录词

    ⇒ 合并规则：层1 结果优先占用字符网格；层2 只在**未被层1 占用**的位置补充
      （杜绝两套逻辑互相打架，回归可解释、可调试）。

**已砍掉 PromptNER（纯 few-shot）**：维护成本高（第三套底层逻辑）、
施工领域未做指令调优 → 效果差且高延迟；其能力由 层1 词典 + 层2 微调模型 覆盖。

用得起的两档运行模式：
    CascadeExtractor(lexicon)                     # 纯规则（零依赖、可测、无 GPU）
    CascadeExtractor(lexicon, model_path=...)      # 规则 + 微调模型（P4 后）
"""
from __future__ import annotations

import json
import os

from ner2.src.common.types import TYPE_SET
from ner2.src.pipeline.model_layer import load_model_backend
from ner2.src.pipeline.rule_layer import RuleLayer


class CascadeExtractor:
    """规则/词典 + 微调模型 级联抽取器。"""

    def __init__(self, lexicon: dict, model_path: str | None = None,
                 device: str | None = None):
        self.rule = RuleLayer(lexicon)
        self.model = load_model_backend(model_path, device=device)

    # ---- 单句 ----
    def extract(self, text: str) -> list[dict]:
        """返回实体列表（字符偏移不重叠），每条含 layer/conf。"""
        ents = self.rule.extract(text)
        if self.model is not None:
            ents = self._merge(ents, self.model.extract(text))
        return sorted(ents, key=lambda x: (x["start"], x["end"] - x["start"]))

    # ---- 批量 ----
    def extract_batch(self, texts: list[str]) -> list[list[dict]]:
        return [self.extract(t) for t in texts]

    # ---- 合并：规则优先，模型补空缺 ----
    @staticmethod
    def _merge(rule_ents: list[dict], model_ents: list[dict]) -> list[dict]:
        used = [(e["start"], e["end"]) for e in rule_ents]
        out = list(rule_ents)
        for e in sorted(model_ents, key=lambda x: (x["start"], x["end"] - x["start"])):
            if e.get("type") not in TYPE_SET:
                continue
            s, en = e["start"], e["end"]
            if s >= en:
                continue
            # 与规则层重叠 -> 丢弃（规则优先，保证可解释）
            if any(s < ue and us < en for us, ue in used):
                continue
            used.append((s, en))
            out.append({"type": e["type"], "start": s, "end": en,
                        "layer": "model", "conf": e.get("conf", 0.9)})
        return out

    # ---- 便捷：从 json 词典构建 ----
    @classmethod
    def from_lexicon_file(cls, path: str, model_path: str | None = None):
        lex = json.load(open(path, encoding="utf-8"))
        return cls(lex, model_path=model_path)


def default_lexicon_path() -> str:
    """默认词典：ner2 合并词典优先，退回旧 ner 种子。"""
    from ner2.src.common.paths import LEXICON_DIR, LEGACY_LEXICON
    merged = os.path.join(LEXICON_DIR, "lexicon.json")
    return merged if os.path.exists(merged) else LEGACY_LEXICON
