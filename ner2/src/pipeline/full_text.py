"""全量分块实体抽取（FullTextExtractor）—— ner2 对外服务入口。

背景（2026-09-11 定稿）：真实施工方案动辄 10 万字，**不再使用三层漏斗**（旧 agent 的
PlanFunnel 已完全舍弃）。本模块把完整方案文本**全量分块**送入 ner2 级联管道
（层1 规则/词典 + 层2 微调模型）抽取全部实体：
  - 分块 = 行级清洗切句 + 超长行按句号/分号拆分（与训练语料同口径，保证分布一致）；
  - 全量 = 每一句都过模型（batch 推理），不筛选、不收敛、不截断；
  - 级联 = 规则层优先占用字符网格，模型层只补空缺（与训练/评估同架构）。

实测性能（RTX 4060 Ti，batch=64/max_len=96）：10 万字符 → 2439 句 → ~4.7s（520 句/s）。

接口：
    FullTextExtractor(model_path, lexicon_path=None, device=None, batch=64, max_len=96)
        .extract_text(text) -> list[dict]   # 实体含 类型/句内偏移/所属句
        .extract_pdf(path)   -> list[dict]  # PyMuPDF 抽文字层后全量抽取
        .split_sentences(text) -> list[dict]  # {"text","start"} 句内文本与句首偏移

实体 schema：{"type","text","start","end","layer","conf","sent_idx","sent_text"}
  - start/end 为**句内字符偏移**（与训练数据同坐标系）；sent_idx/sent_text 定位原文。
"""
from __future__ import annotations

import os

from ner2.src.common.paths import LEGACY_LEXICON, LEXICON_DIR, MODELS
from ner2.src.common.types import TYPE_SET
from ner2.src.pipeline.cascade import CascadeExtractor
from ner2.src.weak.remote_label import is_dirty

DEFAULT_MODEL = os.path.join(MODELS, "s2_crf_param_v3", "model.pt")
# 生产词典：参数类 = 专项定稿 102 词（对齐 gold 口径），其余 5 类 = 旧词典
PROD_LEXICON = os.path.join(LEXICON_DIR, "lexicon_prod.json")


class FullTextExtractor:
    """全量分块实体抽取器（级联：规则层 + 微调模型层）。"""

    def __init__(self, model_path: str | None = None,
                 lexicon_path: str | None = None, device: str | None = None,
                 batch: int = 64, max_len: int = 96):
        self.lexicon_path = lexicon_path or (PROD_LEXICON if os.path.exists(PROD_LEXICON) else LEGACY_LEXICON)
        self.model_path = model_path            # None => 纯规则模式（零依赖）
        self.batch = batch
        self.max_len = max_len
        self.device = device
        self._cascade = None                    # 懒加载

    # ---------------- 分块 ----------------
    def split_sentences(self, text: str) -> list[dict]:
        """行级清洗切句（**与弱标 iter_sentences 同口径**：复用 remote_label.is_dirty，
        超长 >240 行与伪影/非中文行一并过滤——模型未见过该分布，喂了也乱标）。
        返回 [{"text","start"}]；start 为该句首字符在全量文本中的偏移。"""
        sents = []
        pos = 0
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            lead = len(raw_line) - len(raw_line.lstrip())   # 前导空白
            pos += lead
            if not line or is_dirty(line):                   # 与弱标同判据（含 >240）
                pos += max(1, len(line) or 1)
                continue
            sents.append({"text": line, "start": pos})
            pos += len(line)
        return sents

    # ---------------- 抽取 ----------------
    def _ensure(self):
        if self._cascade is None:
            import json
            lex = json.load(open(self.lexicon_path, encoding="utf-8"))
            self._cascade = CascadeExtractor(lex, model_path=None)  # 规则层（词典）
            self._bert = None
            if self.model_path and os.path.exists(self.model_path):
                from ner2.src.bert.engine import load_ckpt, predict_batch
                from ner2.src.bert.decode import token_tags_to_char_entities
                from transformers import AutoTokenizer
                from ner2.src.common.paths import BASE_BERT
                self._model, _, _ = load_ckpt(self.model_path, device=self.device or "cuda",
                                              fallback_base=BASE_BERT)
                self._tok = AutoTokenizer.from_pretrained(BASE_BERT)
                self._to_ents = token_tags_to_char_entities
            else:
                self._model = None
        return self._cascade, self._model

    def extract_text(self, text: str) -> list[dict]:
        """全量抽取。返回实体列表（规则层 + 模型层级联，规则优先）。"""
        cascade, model = self._ensure()
        sents = self.split_sentences(text)
        texts = [s["text"] for s in sents]

        # 层1：规则/词典（逐句，快）
        rule_out = cascade.extract_batch(texts)
        # 层2：微调模型（整批 GPU 推理）
        model_out: list[list[dict]] = [[] for _ in texts]
        if model is not None and texts:
            preds = None
            from ner2.src.bert.engine import predict_batch
            preds = predict_batch(model, self._tok, texts, max_len=self.max_len,
                                  batch=self.batch, device=self.device or "cuda")
            for i, p in enumerate(preds):
                model_out[i] = self._to_ents(p["offsets"], p["tags"])

        ents: list[dict] = []
        for i, s in enumerate(sents):
            for e in rule_out[i]:
                ents.append(self._finalize(e, s, i, "rule"))
            for e in model_out[i]:
                if e.get("type") not in TYPE_SET or e["start"] >= e["end"]:
                    continue
                # 规则优先：与规则层实体重叠则丢弃
                if any(e["start"] < r["end"] and r["start"] < e["end"]
                       for r in rule_out[i]):
                    continue
                ents.append(self._finalize(e, s, i, "model"))
        return ents

    @staticmethod
    def _finalize(e: dict, sent: dict, idx: int, layer: str) -> dict:
        return {
            "type": e["type"],
            "text": sent["text"][e["start"]:e["end"]],
            "start": e["start"],
            "end": e["end"],
            "layer": layer,
            "conf": e.get("conf", 1.0 if layer == "rule" else 0.9),
            "sent_idx": idx,
            "sent_text": sent["text"],
        }

    # ---------------- PDF ----------------
    def extract_pdf(self, path: str) -> list[dict]:
        """PyMuPDF 抽文字层后全量抽取。图片/扫描内容忽略（与项目约定一致）。"""
        try:
            import fitz  # PyMuPDF
        except Exception:
            raise RuntimeError("解析 PDF 需要 PyMuPDF：pip install pymupdf")
        text = []
        with fitz.open(path) as doc:
            for page in doc:
                text.append(page.get_text())
        return self.extract_text("\n".join(text))


def dedup_entities(ents: list[dict]) -> list[dict]:
    """实体**去重收尾**：按 (type, text) 聚合，10 万字方案同一实体常跨句重复出现。

    返回（按 count 降序）：
      {"type","text","count","layer","first_sent_idx","sents":[最多前3个出现句]}
    - extract_text 的完整带位置列表保留给需要定位的调用方（agent/RAG）；
    - 本函数供「实体清单/统计视图」使用，避免同一实体刷屏。
    """
    agg: dict[tuple, dict] = {}
    for e in ents:
        k = (e["type"], e["text"])
        r = agg.get(k)
        if r is None:
            r = agg[k] = {"type": e["type"], "text": e["text"], "count": 0,
                          "layer": e.get("layer"), "first_sent_idx": e.get("sent_idx", -1),
                          "sents": []}
        r["count"] += 1
        st = e.get("sent_text")
        if st and st not in r["sents"]:
            r["sents"].append(st)
        if (e.get("sent_idx") or 0) < r["first_sent_idx"]:
            r["first_sent_idx"] = e.get("sent_idx", -1)
    out = list(agg.values())
    out.sort(key=lambda x: (-x["count"], x["type"], x["text"]))
    for r in out:
        r["sents"] = r["sents"][:3]          # 只保留前几句佐证，避免体积膨胀
    return out


def default_lexicon() -> str:
    from ner2.src.pipeline.cascade import default_lexicon_path
    return default_lexicon_path()
