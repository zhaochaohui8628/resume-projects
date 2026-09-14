"""Skill 协议与注册表（rag/ner/rules 封装为可调用技能）——渐进式披露加载。

渐进式披露（progressive disclosure）：
  - 轻量清单：system prompt 中只给「name: 一行描述」（brief），模型据此决定用哪个工具；
  - 完整说明：仅当模型首次选择调用某 skill 时，才把它的完整参数用法（doc）注入下一轮
    上下文并标记已展开（记入 procedural 记忆），避免每轮都携带所有工具的详细文档。
"""
from __future__ import annotations


class Skill:
    """单个可调用技能。invoke 接收字符串参数，返回字符串观察。"""

    def __init__(self, name: str, description: str, doc: str, invoke):
        self.name = name
        self.description = description          # 单行，轻量披露用
        self.doc = doc                          # 完整说明（参数/用法/输出），调用时才披露
        self._invoke = invoke

    def __call__(self, argument: str) -> str:
        return self._invoke(argument)


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        self._skills[skill.name] = skill

    def names(self) -> list[str]:
        return list(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def brief(self) -> str:
        """轻量清单（system prompt 用）：一行一个。"""
        return "\n".join(f"- {s.name}: {s.description}" for s in self._skills.values())

    def __len__(self):
        return len(self._skills)


# ---------------------------------------------------------------- 项目技能
def _clip(text: str, n: int = 260) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n] + "…"


def make_rag_skill(rag_client, sink: list | None = None, use_rerank: bool | None = None) -> Skill:
    """rag skill：规范条文检索（来源=项目1 rag）。

    sink：可选的命中收集列表——每次检索把**结构化命中**（rank/score/rerank_score/
    source/clause_no/text/query）追加进去，供 UI 溯源面板展示（此前只返回文本，
    导致 ReAct 问答路径的"RAG 检索源"永远是空的）。
    use_rerank：精排开关（None=用客户端默认），按次传入 → 复用同一份模型实例也能控制开关。
    """
    def invoke(query: str) -> str:
        if rag_client is None or not rag_client.available():
            return "（rag 技能不可用：规范索引未就绪）"
        rows = rag_client.search(query, top_k=3, use_rerank=use_rerank)
        if not rows:
            return "（无检索结果）"
        parts = []
        for i, r in enumerate(rows, 1):
            md = r.get("metadata", {}) or {}
            src = md.get("source", "?")
            parts.append(f"[{src}] {_clip(r.get('text', ''), 220)}")
            if sink is not None:
                sink.append({"rank": i, "score": r.get("score", 0.0),
                             "rerank_score": r.get("rerank_score"),
                             "source": src, "clause_no": md.get("clause_no", ""),
                             "source_path": md.get("source_path", ""),
                             "text": (r.get("text", "") or "")[:200],
                             "query": query[:60]})
        return "\n".join(parts)

    return Skill(
        name="retrieve_standards",
        description="按一个合规问题/关键词检索施工规范条文（GB/JGJ/DG 等），返回带来源的条文片段。",
        doc=("参数：action_input 填自然语言查询或关键词串，如「基坑开挖深度超过5m是否需要专家论证」。\n"
             "返回：至多 3 条最相关规范条文片段，每条带 [规范编号] 前缀。用它们做合规判断的依据。"),
        invoke=invoke,
    )


def make_ner_skill(ner_client) -> Skill:
    """ner skill：实体抽取（来源=项目2 ner）。"""
    def invoke(text: str) -> str:
        if ner_client is None:
            return "（ner 技能不可用）"
        try:
            ents = ner_client.extract(text) or []
        except Exception as e:  # rule 后端不应失败，防御
            return f"（ner 技能异常：{e}）"
        if not ents:
            return "（未识别到实体）"
        lines = [f"{e.get('type', '?')}={e.get('text', '')}" for e in ents[:20]]
        return "；".join(lines)

    return Skill(
        name="extract_entities",
        description="从一段方案文本抽取结构化实体：工程类型/工序/设备/参数/规范编号/危大类别。",
        doc=("参数：action_input 填待抽取的方案文本片段（≤2000 字效果最佳）。\n"
             "返回：type=value 列表（最多 20 个）。实体可辅助定位方案引用了哪些规范/危大类型。"),
        invoke=invoke,
    )


def make_rules_skill(rules_checker) -> Skill:
    """rules skill：确定性规则自查（C1 废止引用 / C2 危大缺项 / C4 编制要素）。"""
    def invoke(text: str) -> str:
        out = rules_checker(text) if callable(rules_checker) else rules_checker.run_checks(text)
        risks = out.get("risks", [])
        if not risks:
            return "（规则检查未命中）"
        lines = [
            f"[{r.get('severity', '?')}] {r.get('check', '?')}: {_clip(r.get('title', ''), 80)}"
            for r in risks[:10]
        ]
        return "\n".join(lines)

    return Skill(
        name="compliance_check",
        description="对方案文本做确定性规则自查，返回高危/中危风险条目（过期规范引用、危大缺项、要素缺失）。",
        doc=("参数：action_input 填需要检查的方案文本片段。\n"
             "返回：至多 10 条 [级别] 检查项: 风险标题。用于定位需要进一步核实的合规问题。"),
        invoke=invoke,
    )


def build_compliance_skills(rules_checker=None, rag_client=None, ner_client=None,
                            rag_sink: list | None = None,
                            rag_rerank: bool | None = None) -> SkillRegistry:
    """组装项目技能注册表（依赖注入，缺项自动跳过该技能）。

    rag_sink：透传给 rag skill 的命中收集列表（UI 溯源用）。
    rag_rerank：透传给 rag skill 的精排开关（None=客户端默认）。
    """
    reg = SkillRegistry()
    if rules_checker is not None:
        reg.register(make_rules_skill(rules_checker))
    if ner_client is not None:
        reg.register(make_ner_skill(ner_client))
    if rag_client is not None:
        reg.register(make_rag_skill(rag_client, sink=rag_sink, use_rerank=rag_rerank))
    return reg
