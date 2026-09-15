# 图谱 demo v2 — 实体三元组清单（实体 -关系-> 实体）

来源：`graphrag/data/demo_graph_v2.json`（生成脚本 `graphrag/scripts/build_demo_graph_v2.py`）
共 **96 条三元组 / 13 类关系**。
新增部分（demo 多跳演示所需）= ALIAS_OF/HAS_METRIC/HAS_THRESHOLD/FOR_CATEGORY/TRIGGERS/SUPERSEDES/REFERENCES_CLAUSE = **42 条**；
既有骨架 = BELONGS_TO/REGULATED_BY/COVERS/MENTIONS/HIERARCHY/REFERENCES = 54 条。

---

## 新增 · 供 demo 多跳查询

### ALIAS_OF · 术语归一（用户口语词 → 规范词）· 6 条
```
高支模(术语)           -ALIAS_OF->  模板支撑工程
深基坑(术语)           -ALIAS_OF->  深基坑工程
排架(术语)             -ALIAS_OF->  模板支架
满堂架(术语)           -ALIAS_OF->  模板支架
塔吊(术语)             -ALIAS_OF->  起重吊装工程
爬架(术语)             -ALIAS_OF->  脚手架工程
```

### HAS_METRIC · 类别 → 量名（含单位）· 7 条
```
模板支撑工程   -HAS_METRIC->  搭设高度(模板支撑工程)    单位 m
深基坑工程     -HAS_METRIC->  开挖深度(深基坑工程)      单位 m
模板支撑工程   -HAS_METRIC->  施工总荷载(模板支撑工程)  单位 kN/m²
模板支撑工程   -HAS_METRIC->  集中线荷载(模板支撑工程)  单位 kN/m
起重吊装工程   -HAS_METRIC->  单件起吊重量(起重吊装工程) 单位 kN
脚手架工程     -HAS_METRIC->  搭设高度(脚手架工程)      单位 m
降水工程       -HAS_METRIC->  开挖深度(降水工程)        单位 m
```

### HAS_THRESHOLD · 量名 → 阈值（危大线 / 超规模线）· 10 条
```
搭设高度(模板支撑工程)   -HAS_THRESHOLD->  5m 危大线
搭设高度(模板支撑工程)   -HAS_THRESHOLD->  8m 超规模线
施工总荷载(模板支撑工程) -HAS_THRESHOLD->  10kN/m² 危大线
施工总荷载(模板支撑工程) -HAS_THRESHOLD->  15kN/m² 超规模线
开挖深度(深基坑工程)     -HAS_THRESHOLD->  3m 危大线
开挖深度(深基坑工程)     -HAS_THRESHOLD->  5m 超规模线
搭设高度(脚手架工程)     -HAS_THRESHOLD->  24m 危大线
搭设高度(脚手架工程)     -HAS_THRESHOLD->  50m 超规模线
单件起吊重量(起重吊装工程) -HAS_THRESHOLD-> 10kN 危大线
单件起吊重量(起重吊装工程) -HAS_THRESHOLD-> 100kN 超规模线
```

### FOR_CATEGORY · 阈值 → 类别（回指，便于反向遍历）· 10 条
```
5m 危大线@搭设高度    -FOR_CATEGORY->  模板支撑工程
8m 超规模线@搭设高度  -FOR_CATEGORY->  模板支撑工程
10kN/m² 危大线@施工总荷载 -FOR_CATEGORY-> 模板支撑工程
15kN/m² 超规模线@施工总荷载 -FOR_CATEGORY-> 模板支撑工程
3m 危大线@开挖深度    -FOR_CATEGORY->  深基坑工程
5m 超规模线@开挖深度  -FOR_CATEGORY->  深基坑工程
24m 危大线@搭设高度   -FOR_CATEGORY->  脚手架工程
50m 超规模线@搭设高度 -FOR_CATEGORY->  脚手架工程
10kN 危大线@单件起吊重量 -FOR_CATEGORY-> 起重吊装工程
100kN 超规模线@单件起吊重量 -FOR_CATEGORY-> 起重吊装工程
```

### TRIGGERS · 阈值 → 义务（触发的后果）· 16 条
```
5m 危大线@搭设高度      -TRIGGERS->  应编制专项施工方案
5m 危大线@搭设高度      -TRIGGERS->  方案应经审批
8m 超规模线@搭设高度    -TRIGGERS->  应组织专家论证
10kN/m² 危大线@施工总荷载 -TRIGGERS-> 应编制专项施工方案
10kN/m² 危大线@施工总荷载 -TRIGGERS-> 方案应经审批
15kN/m² 超规模线@施工总荷载 -TRIGGERS-> 应组织专家论证
3m 危大线@开挖深度      -TRIGGERS->  应编制专项施工方案
3m 危大线@开挖深度      -TRIGGERS->  方案应经审批
3m 危大线@开挖深度      -TRIGGERS->  应进行基坑监测
5m 超规模线@开挖深度    -TRIGGERS->  应组织专家论证
24m 危大线@搭设高度     -TRIGGERS->  应编制专项施工方案
24m 危大线@搭设高度     -TRIGGERS->  方案应经审批
50m 超规模线@搭设高度   -TRIGGERS->  应组织专家论证
10kN 危大线@单件起吊重量 -TRIGGERS->  应编制专项施工方案
10kN 危大线@单件起吊重量 -TRIGGERS->  方案应经审批
100kN 超规模线@单件起吊重量 -TRIGGERS-> 应组织专家论证
```

### SUPERSEDES · 规范废止/替代 · 2 条
```
GB55023-2022 施工脚手架通用规范 -SUPERSEDES-> JGJ130-2011 建筑施工扣件式钢管脚手架安全技术规范
GB55032-2022 建筑与市政工程施工质量控制通用规范 -SUPERSEDES-> JGJ46-2005 施工现场临时用电安全技术规范
```

### REFERENCES_CLAUSE · 条文/附录跳转 · 1 条（demo）
```
JGJ130-2011 6.2.4（架高超7m 连墙件两步三跨） -REFERENCES_CLAUSE-> JGJ130-2011 附录B（连墙件布置表）
```

---

## 既有骨架 · 复用 demo_graph.json

### BELONGS_TO · 条款 → 所属规范 · 8 条
```
JGJ311-2013 7.1.2 -BELONGS_TO-> JGJ311-2013 建筑深基坑工程施工安全技术规范
JGJ120-2012 3.3.1 -BELONGS_TO-> JGJ120-2012 建筑基坑支护技术规程
DG-TJ08-61-2018 2.1.2 -BELONGS_TO-> DG-TJ08-61-2018 基坑工程技术标准
DG-TJ08-2077-2021 8.4.7 -BELONGS_TO-> DG-TJ08-2077-2021 危险性较大的分部分项工程安全管理标准
GB51210-2016 3.2.3 -BELONGS_TO-> GB51210-2016 建筑施工脚手架安全技术统一标准
GB55023-2022 5.2.1 -BELONGS_TO-> GB55023-2022 施工脚手架通用规范
GB51004-2015 5.3.2 -BELONGS_TO-> GB51004-2015 建筑地基基础工程施工规范
JGJ130-2011 6.2.4 -BELONGS_TO-> JGJ130-2011 建筑施工扣件式钢管脚手架安全技术规范
```

### REGULATED_BY · 危大类别 → 监管规范（多标准交叉）· 12 条
```
深基坑工程   -REGULATED_BY-> JGJ311-2013 建筑深基坑工程施工安全技术规范
深基坑工程   -REGULATED_BY-> JGJ120-2012 建筑基坑支护技术规程
深基坑工程   -REGULATED_BY-> DG-TJ08-61-2018 基坑工程技术标准
深基坑工程   -REGULATED_BY-> DG-TJ08-2077-2021 危大工程安全管理标准
模板支撑工程 -REGULATED_BY-> GB51210-2016 建筑施工脚手架安全技术统一标准
模板支撑工程 -REGULATED_BY-> GB55023-2022 施工脚手架通用规范
模板支撑工程 -REGULATED_BY-> JGJ130-2011 建筑施工扣件式钢管脚手架安全技术规范
模板支撑工程 -REGULATED_BY-> DG-TJ08-2077-2021 危大工程安全管理标准
起重吊装工程 -REGULATED_BY-> DG-TJ08-2077-2021 危大工程安全管理标准
脚手架工程   -REGULATED_BY-> GB51210-2016 建筑施工脚手架安全技术统一标准
脚手架工程   -REGULATED_BY-> GB55023-2022 施工脚手架通用规范
脚手架工程   -REGULATED_BY-> JGJ130-2011 建筑施工扣件式钢管脚手架安全技术规范
```

### COVERS · 条款 → 覆盖危大类别 · 8 条
```
JGJ311-2013 7.1.2 -COVERS-> 深基坑工程
JGJ120-2012 3.3.1 -COVERS-> 深基坑工程
DG-TJ08-61-2018 2.1.2 -COVERS-> 深基坑工程
DG-TJ08-2077-2021 8.4.7 -COVERS-> 模板支撑工程
GB51210-2016 3.2.3 -COVERS-> 模板支撑工程
GB55023-2022 5.2.1 -COVERS-> 模板支撑工程
JGJ130-2011 6.2.4 -COVERS-> 脚手架工程
GB51004-2015 5.3.2 -COVERS-> 深基坑工程
```

### MENTIONS · 条款 → 涉及实体 · 10 条
```
JGJ311-2013 7.1.2 -MENTIONS-> 降排水
JGJ120-2012 3.3.1 -MENTIONS-> 支护结构
JGJ120-2012 3.3.1 -MENTIONS-> 基坑开挖
DG-TJ08-61-2018 2.1.2 -MENTIONS-> 支护结构
DG-TJ08-61-2018 2.1.2 -MENTIONS-> 降排水
DG-TJ08-61-2018 2.1.2 -MENTIONS-> 基坑开挖
GB55023-2022 5.2.1 -MENTIONS-> 模板支架
GB55023-2022 5.2.1 -MENTIONS-> 连墙件
JGJ130-2011 6.2.4 -MENTIONS-> 连墙件
GB51210-2016 3.2.3 -MENTIONS-> 模板支架
```

### HIERARCHY · 规范上下位 · 4 条
```
GB55023-2022 施工脚手架通用规范 -HIERARCHY-> GB51210-2016 建筑施工脚手架安全技术统一标准
GB51210-2016 建筑施工脚手架安全技术统一标准 -HIERARCHY-> JGJ130-2011 建筑施工扣件式钢管脚手架安全技术规范
JGJ120-2012 建筑基坑支护技术规程 -HIERARCHY-> DG-TJ08-61-2018 基坑工程技术标准
JGJ311-2013 建筑深基坑工程施工安全技术规范 -HIERARCHY-> DG-TJ08-61-2018 基坑工程技术标准
```

### REFERENCES · 条款引用其他规范 · 2 条
```
JGJ120-2012 3.3.1 -REFERENCES-> GB51004-2015 建筑地基基础工程施工规范
JGJ311-2013 7.1.2 -REFERENCES-> GB51004-2015 建筑地基基础工程施工规范
```
