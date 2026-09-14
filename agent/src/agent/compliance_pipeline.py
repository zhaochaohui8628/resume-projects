"""ReAct 审核编排（Agent 项目核心）。

主链路 run()：NER 抽取实体 → 规则引擎(C1/C2/C4) → 对高危项用 RAG 补「规范依据」→ 可选 LLM 生成自然语言描述。
react()：当提供 LLM 时，执行标准的 Thought-Action-Observation 循环（最多 max_steps 步，finish 即止）。

所有外部依赖（rag/ner/llm）缺省时自动降级，保证无 API Key、无规范 PDF 也能跑通主链路。
"""
from __future__ import annotations

import json
import re

REACT_SYSTEM = """你是施工方案合规审核助手。按 ReAct 范式工作：
每步输出 JSON：{"thought":"...","action":"retrieve|check_rules|finish","action_input":"..."}
- retrieve：用 action_input 作为查询去检索规范条文
- check_rules：用 action_input 作为文本做规则检查
- finish：审核结束
只输出 JSON，不要解释。"""


class ComplianceAgent:
    def __init__(self, rules_checker, ner_client=None, rag_client=None, llm=None, max_steps: int = 3):
        self.rc = rules_checker
        self.ner = ner_client
        self.rag = rag_client
        self.llm = llm
        self.max_steps = max_steps

    # ---------------- 主链路 ----------------
    def run(self, plan_text: str) -> dict:
        # 两线并行：
        #   线A 规则引擎（self.rc）：永远吃全量 plan_text，先跑 —— 确定性正则查表，零成本不漏检，
        #       产出 risks/detected_types（权威结论）。
        #   线B NER（self.ner）：ner2 **全量分块识别**完整方案（不做三层漏斗/收敛/截断）。
        # 输出口径：risks/detected_types = 权威结论；entities = 结构化旁证（ReAct/RAG/UI），不合并。
        rules_out = self.rc(plan_text) if callable(self.rc) else self.rc.run_checks(plan_text)
        detected = rules_out["detected_types"]
        entities = self.ner.extract(plan_text) if self.ner else []
        risks = rules_out["risks"]
        for r in risks:
            if self.rag and r["severity"] in ("HIGH", "MEDIUM"):
                basis = self.rag.search(self._basis_query(r), top_k=2)
                r["basis"] = [b.get("text", "") for b in basis]
            if self.llm:
                r["description_nl"] = self._nl(r)
        return {
            "entities": entities,
            "detected_types": detected,
            "risk_count": len(risks),
            "risks": risks,
            "summary": self._summary(risks, detected),
        }

    # ---------------- ReAct 循环 ----------------
    def react(self, plan_text: str, entities: list = None) -> list:
        if not self.llm:
            return [{"thought": "(无 LLM) 跳过 ReAct，使用确定性规则链路", "action": "skip", "observation": ""}]
        context = {"plan": plan_text, "entities": entities or [], "risks": []}
        trace = []
        for _ in range(self.max_steps):
            user = self._react_user(context)
            resp = self.llm.complete(
                [{"role": "system", "content": REACT_SYSTEM}, {"role": "user", "content": user}]
            )
            act = self._parse_action(resp)
            act["observation"] = self._exec(act, context)
            trace.append(act)
            if act.get("action") == "finish":
                break
        return trace

    # ---------------- 内部方法 ----------------
    def _basis_query(self, r: dict) -> str:
        return f"{r.get('title', '')} {r.get('detail', '')}"

    def _nl(self, r: dict) -> str:
        prompt = (
            "你是资深施工安全工程师，用一句中文说明该合规风险并给出修改建议：\n"
            f"风险：{r.get('title','')}\n细节：{r.get('detail','')}"
        )
        return self.llm.complete([{"role": "user", "content": prompt}])

    def _summary(self, risks: list, types: list) -> str:
        high = sum(1 for r in risks if r["severity"] == "HIGH")
        med = sum(1 for r in risks if r["severity"] == "MEDIUM")
        return (
            f"识别危大类型：{', '.join(types) or '无'}；"
            f"共 {len(risks)} 项风险（高危 {high} / 中危 {med}）。"
        )

    @staticmethod
    def _react_user(context: dict) -> str:
        ents = context.get("entities", [])
        ent_str = "; ".join(f"{e.get('type')}:{e.get('text')}" for e in ents[:10]) if ents else "（无实体）"
        return f"方案实体：{ent_str}\n已发现风险数：{len(context.get('risks', []))}\n请决定下一步 action。"

    @staticmethod
    def _parse_action(resp: str) -> dict:
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if not m:
            return {"thought": resp, "action": "finish", "action_input": ""}
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"thought": resp, "action": "finish", "action_input": ""}
        return {
            "thought": d.get("thought", ""),
            "action": d.get("action", "finish"),
            "action_input": d.get("action_input", ""),
        }

    def _exec(self, act: dict, context: dict) -> str:
        a = act.get("action")
        if a == "retrieve" and self.rag:
            res = self.rag.search(act.get("action_input", ""), top_k=2)
            return "; ".join(b.get("text", "")[:80] for b in res) or "无检索结果"
        if a == "check_rules":
            out = self.rc(act.get("action_input", "")) if callable(self.rc) else self.rc.run_checks(act.get("action_input", ""))
            context["risks"].extend(out["risks"])
            return f"规则检查命中 {out['risk_count']} 项"
        return "done"
