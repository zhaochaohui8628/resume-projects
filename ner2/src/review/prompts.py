"""LLM 二次校验提示词构造（P2）。

角色：施工方案 NER 质检员。对远程监督（词典/正则）打出的弱标签逐条复核：
- 词典词串是否真的构成实体（语境/限定条件/否定/非专用语境）；
- 边界是否准确（词典只含词干时，金标风格是完整短语）；
- 类型是否歧义（同一词串命中多个类，如"基坑降水"=工程类型 vs 工序）；
- 是否漏标（明显实体但规则没打上——只补"高置信"的，宁缺勿滥）。

输出约束（parser.py 会做确定性兜底，这里只作提示）：
- 仅输出 JSON；
- type 限 6 类白名单；
- entities.start/end 为字符偏移（半开区间），text[start:end] 必须与原文逐字一致；
- issues 列出每条问题的实体级描述；reason 一句话总结。

注意：本模块只负责构造 messages，不负责调用 LLM——调用在 scripts/llm_review.py，
本机默认不实际调用（由模型扮演 LLM 逐条手工判定，产物格式与 scripts 输出一致）。
"""
from __future__ import annotations

import json

VALID_TYPES = ["工程类型", "工序", "设备", "参数", "规范编号", "危大类别"]

SYSTEM_PROMPT = """你是施工方案文本的 NER 质检员。给定一条句子和一组"规则弱标签"（由词典/正则远程监督自动生成），
你的任务是对每个弱标签做二次校验并纠错。

实体类型仅限 6 类：工程类型 / 工序 / 设备 / 参数 / 规范编号 / 危大类别。

判定规则：
1. keep：规则标签正确无误（类型 + 边界 + 语义都成立）。
2. fix：规则标签大致方向对，但需要修正——修正项包括：
   - 边界：只标了词干、漏了修饰/量词/单位（如"拆"应为"拆除"、漏"深度 12.0m"的"深度"）；
   - 类型：词串命中的词典类型与句中实际语义不符（如"基坑降水"此处指工程而非工序）；
   - 漏标：句中存在明显实体而规则未标（仅补高置信的）。
3. drop：弱标签错误——词串在句中不是实体（禁止性表述/泛指/非本义/数值残缺），应整条丢弃或剔除该实体。

输出要求：
- 只输出一个 JSON 对象，不要任何解释文字：
{"verdict": "keep|fix|drop", "entities": [{"type": "...", "start": 0, "end": 2}, ...], "issues": ["..."], "reason": "一句话"}
- entities 为复核后的最终实体列表（keep 时与输入一致；fix 时是修正后的；drop 时为空列表）；
- start/end 是字符偏移，text[start:end] 必须逐字等于实体原文；
- verdict 判定以实体整体计：全部正确=keep；需修正=fix；整体错误/无实体=drop。
"""


def _entities_for_prompt(entities: list[dict]) -> list[dict]:
    """只保留训练兼容字段给 LLM 看。"""
    return [{"type": e["type"], "start": e["start"], "end": e["end"]} for e in entities]


def build_review_messages(text: str, rule_entities: list[dict],
                          max_entities: int = 30) -> list[dict]:
    """构造 LLM 二次校验的 messages。"""
    ents = _entities_for_prompt(rule_entities)[:max_entities]
    payload = {"text": text, "rule_entities": ents}
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def build_batch_messages(samples: list[dict], max_per_batch: int = 10) -> list[list[dict]]:
    """批量构造：每条样本一个 user 消息，返回 [messages_1, messages_2, ...]。"""
    out = []
    for i in range(0, len(samples), max_per_batch):
        batch = samples[i:i + max_per_batch]
        content = json.dumps(
            [{"idx": j, "text": s["text"],
              "rule_entities": _entities_for_prompt(s["entities"])}
             for j, s in enumerate(batch)],
            ensure_ascii=False)
        out.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ])
    return out
