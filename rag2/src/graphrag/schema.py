"""GraphRAG 规范知识图谱 Schema 定义。

节点类型：
- Standard       规范（如 GB55032-2022 建筑与市政工程施工质量控制通用规范）
- Clause         条款（规范内的具体条文，clause_id = source::clause_no）
- HazardCategory 危大工程类别（如"深基坑工程""模板支撑工程"）
- Entity         关键实体（工序/设备/参数/材料，从条款抽取）

关系类型：
- (:Clause)-[:BELONGS_TO]->(:Standard)
- (:Clause)-[:REFERENCES]->(:Standard)     条款引用其他规范（如"应符合GB50007"）
- (:Clause)-[:REFERENCES_CLAUSE]->(:Clause) 条款引用具体条文（如"按附录B执行"）
- (:Standard)-[:SUPERSEDES]->(:Standard)    上位法/替代（如 JGJ94 被 GB550xx 替代）
- (:Standard)-[:CONFLICTS_WITH]->(:Standard) 数值/要求冲突（需人工确认）
- (:Standard)-[:HIERARCHY]->(:Standard)     上下位（强制性国标 > 行业标准 > 地标）
- (:Clause)-[:COVERS]->(:HazardCategory)    条款覆盖某危大类别
- (:Clause)-[:MENTIONS]->(:Entity)          条款涉及某实体
- (:HazardCategory)-[:REGULATED_BY]->(:Standard) 危大类别由哪些规范监管（多标准交叉）
"""
from __future__ import annotations

# 节点标签
STD = "Standard"
CLS = "Clause"
HAZ = "HazardCategory"
ENT = "Entity"

# 关系类型
BELONGS_TO = "BELONGS_TO"
REFERENCES = "REFERENCES"               # Clause -> Standard
REFERENCES_CLAUSE = "REFERENCES_CLAUSE"  # Clause -> Clause
SUPERSEDES = "SUPERSEDES"               # Standard -> Standard（废止/替代）
CONFLICTS_WITH = "CONFLICTS_WITH"       # Standard -> Standard
HIERARCHY = "HIERARCHY"                 # Standard -> Standard（上位法）
COVERS = "COVERS"                       # Clause -> HazardCategory
MENTIONS = "MENTIONS"                   # Clause -> Entity
REGULATED_BY = "REGULATED_BY"           # HazardCategory -> Standard

ALL_NODE_LABELS = [STD, CLS, HAZ, ENT]
ALL_REL_TYPES = [BELONGS_TO, REFERENCES, REFERENCES_CLAUSE, SUPERSEDES,
                 CONFLICTS_WITH, HIERARCHY, COVERS, MENTIONS, REGULATED_BY]
