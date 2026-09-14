"""system prompt 构建（要素②）。训练采样与评估必须使用同一模板。

格式约束说明（与四维判分器 reward.py 同源）：
- JSON 纯输出（无前后缀文字、无 Markdown 代码块），字段：
      cot_steps  推理链（数组，先给证据→推演的因果步骤，再收敛结论）
      conclusion 最终结论
      basis      依据溯源（规范编号 + 条款号）
      explanation 一句话说明
- 检索证据由 gen_rl_data.py 模拟 RAG 写入 user 消息；system 可选注入「证据要求」。
- 取值白名单与 gen_rl_data.py 写入 judge_meta.conclusion_options 的常量同源
  （reward.format_score 校验白名单），保证 prompt 与奖励口径一致。
"""
from __future__ import annotations

# 取值白名单（生成器写入 judge_meta 时引用同一常量）
OPTIONS = {
    "version_abolished": ["yes", "no", "unknown"],
    "danger_level": [],               # 组合结论（级别+论证），用 set_match 判词
    "threshold_value": [],            # 数值+单位，自由文本
    "missing_section": ["工程概况", "编制依据", "施工计划", "施工工艺技术",
                        "施工安全保证措施", "施工管理及作业人员配备和分工",
                        "验收要求", "应急处置措施", "计算书及相关施工图纸",
                        "无缺失"],
}

_TASK_CN = {
    "version_abolished": "规范有效性判断",
    "danger_level": "危大工程分级",
    "threshold_value": "强条数值问答",
    "missing_section": "方案编制要素判断",
}

TEMPLATE = """你是建筑施工安全方案合规审查专家，熟悉现行规范（GB550xx 全文强制规范、JGJ 行业标准、上海地标）及其废止/替代关系，熟悉住建部令第37号危大工程目录与专项方案编制要求。

当前任务类型：{task_cn}。请模拟真实 agent 审核流程：先依据检索到的规范条文逐步推理（先列证据、再比对、最后收敛结论），再给出确定结论；无法确定时 conclusion 输出 "unknown"。

回答格式（必须严格输出单个 JSON 对象，不得包含任何前后缀文字、Markdown 代码块）：
{{"cot_steps": ["<推理步骤1>", "<推理步骤2>", ...], "conclusion": "<结论>", "basis": ["<依据规范编号/条款号>", ...], "explanation": "<一句话说明>"}}

要求：
- cot_steps：至少 2 步，体现「证据 → 比对 → 结论」的因果链，可引用检索证据中的数值/条款；
- basis：给出规范编号与条款号（溯源条文），不得捏造未提供的依据；
- conclusion 取值约定：
  - version_abolished：conclusion ∈ {{"yes","no","unknown"}}；yes 时 basis 给出替代规范编号。
  - danger_level：conclusion 必须同时给出级别（"危大工程"/"超规模危大工程"/"非危大"）与是否需专家论证（"需专家论证"/"不需专家论证"），如 "超规模危大工程，需专家论证"。
  - threshold_value：conclusion 为数值+单位（如 "4Ω"、"5m"）。
  - missing_section：conclusion ∈ 下列之一且为缺失章节名，或 "无缺失"：{section_options}。
违反上述格式、字段缺失、无推理链或结论不在取值约定内将直接扣分。"""


def make_system_prompt(task_type: str, evidence: str | None = None) -> str:
    """构造 system prompt。evidence 为「检索证据摘要」（数据集/评测注入，可选）。"""
    if task_type not in TEMPLATE_DEFS and task_type not in OPTIONS:
        raise ValueError(f"未知任务类型: {task_type}")
    body = TEMPLATE.format(
        task_cn=_TASK_CN.get(task_type, task_type),
        section_options="、".join(OPTIONS["missing_section"]),
    )
    if evidence:
        body += f"\n\n检索到的规范条文证据（仅可依据此证据作答）：\n{evidence}"
    return body


# 训练中 batch 内 query 可能跨类型 -> 生成器保证一条 query 单类型；
# 混合采样时用其类型模板。此处保留 defs 以便复用。
TEMPLATE_DEFS = set(OPTIONS.keys())
