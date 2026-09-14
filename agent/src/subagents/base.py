"""SubAgent 抽象基类与结果协议（v5）。

各 subagent 实现 run(query, plan, ctx) -> SubAgentResult；
ctx 含：rag（RagClient 单例）/ rag_rerank（bool）/ ner_crf（bool）/ memory_dir / verbose。
结果协议新增 trace（链路明细）/ rag_hits（检索溯源）/ markdown（渲染片段）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubAgentResult:
    """单个 subagent 输出协议（v5）。"""
    name: str                                    # subagent 名称
    title: str                                   # 给用户看的标题
    summary: str = ""                            # 一句话结论
    risks: list = field(default_factory=list)    # 风险清单（结构同 rules 层）
    entities: list = field(default_factory=list) # NER 实体
    basis: list = field(default_factory=list)    # 规范依据
    meta: dict = field(default_factory=dict)     # 自由扩展
    markdown: str = ""                           # 渲染好的 markdown 片段
    # ---- v5 新增 ----
    trace: list = field(default_factory=list)    # 链路步骤 [{step, kind, label, detail}]
    rag_hits: list = field(default_factory=list) # RAG 检索溯源 [{agent, rank, score, source, clause_no, text}]


class SubAgent:
    """SubAgent 基类：每个 subagent 必须实现 description/doc/run/to_markdown。

    description: 一行描述（system prompt 摘要用）
    doc:         完整说明（input/output/适用场景/边界）
    run:         实际执行，返回 SubAgentResult
    to_markdown: SubAgentResult → Markdown 片段
    """

    name: str = "base"
    title: str = "基类"
    description: str = ""
    doc: str = ""

    def run(self, query: str = "", plan: str = "", ctx: dict | None = None) -> SubAgentResult:
        raise NotImplementedError

    def to_markdown(self, r, brief: bool = True) -> str:
        return f"### {r.title}\n\n{r.summary}"