# Neo4j 真实数据库复测结论

- 测试时间：2026-11-25T17:34:51.849242+08:00（Asia/Shanghai）。
- 测试方式：宿主机直连，使用项目配置；只读查询，无模拟数据库、无数据写入。
- 未执行模型生成或 GraphRAG 回答质量评测。

## 结论

**Neo4j 连接与本次测试范围内的多跳查询验证通过。**
共 **31 项通过、0 项失败**。
当前实库为 **535 个节点、838 条关系**；该统计是当前数据库快照，不能直接替代历史评测环境的规模。

**判定：连接、查询链路、来源字段覆盖与图谱连通性验证均已通过（种子数据未经过执业医师人工复核，仅限工程演示与测试）。**

## 数据库与数据资产

- 连接：`127.0.0.1:7687`。
- 服务：`[{"name": "Neo4j Kernel", "versions": ["5.26.32"], "edition": "community"}]`。
- 索引 10 个，约束 8 个。
- 自环关系 0 条，孤立节点 0 个。
- 有来源字段的节点：535/535。
- 有来源字段的关系：838/838。
- 来源字段存在不等于来源准确或通过医学审核；历史记录缺失时仍应显示 unknown。

| 节点类型 | 数量 |
|---|---:|
| BodyPart | 46 |
| Department | 16 |
| Disease | 40 |
| Examination | 62 |
| Food | 77 |
| Medication | 74 |
| Precaution | 101 |
| Symptom | 119 |

| 关系类型 | 数量 |
|---|---:|
| AFFECTS_BODY_PART | 75 |
| AVOIDS_FOOD | 71 |
| BELONGS_TO_DEPARTMENT | 40 |
| HANDLES_DISEASE | 40 |
| HAS_PRECAUTION | 121 |
| HAS_SIDE_EFFECT | 3 |
| HAS_SYMPTOM | 161 |
| INTERACTS_WITH | 2 |
| NEEDS_EXAMINATION | 89 |
| RECOMMENDS_FOOD | 54 |
| TREATED_BY_MEDICATION | 91 |
| TREATS_DISEASE | 91 |

## 当前状态与说明

1. **来源字段覆盖**：节点 535/535、关系 838/838 均已写入来源元数据（当前为 `demo_curated_seed` 演示种子来源，`medical_reviewed=false`，不代表临床人工审核）。
2. **孤立节点**：共 0 个（全图连通，无孤立节点）。
3. **规模口径**：本次为 40 种疾病、535 节点、838 关系，可支撑“500+ 节点、800+ 关系”的当前规模描述。此前文档的 121 种疾病、2,307 节点、3,636 关系未在本次连接的数据库中复现，需对应其他数据版本的统计证据。
4. **服务器通知**：复测期间捕获 0 条 Neo4j 通知日志（主要来自查询中对兼容旧字段名 `provenance_json` 的可选回退读取）。

## 功能验证

| 检查 | 结果 | 证据摘要 |
|---|---|---|
| Bolt 连接与认证 | 通过 | {"elapsed_ms": 15.736} |
| 知识图谱非空 | 通过 | {"nodes": 535, "relationships": 838} |
| 现有索引状态 | 通过 | {"indexes": 10, "constraints": 8} |
| 三类测试锚点覆盖 | 通过 | [{"type": "Disease", "name": "高血压", "degree": 31}, {"type": "Medication", "name": "阿司匹林", "degree": 7}, {"type": "Symptom", "name": "乏力", "degree": 6}] |
| Disease 默认多跳查询 | 通过 | {"root": "高血压", "error": null, "triples": 20, "hop1": 10, "hop2": 10, "hop3": 0} |
| Disease 三元组真实方向 | 通过 | {"matched_in_database": 20, "returned": 20} |
| Disease 单跳 | 通过 | {"triples": 10, "applied_hops": 1, "error": null} |
| Disease 超额跳数截断 | 通过 | {"triples": 20, "applied_hops": 2, "error": null} |
| Disease 单跳分支限额 | 通过 | {"triples": 2, "applied_hops": 2, "error": null} |
| Disease 总量限额 | 通过 | {"triples": 3, "applied_hops": 2, "error": null} |
| Medication 默认多跳查询 | 通过 | {"root": "阿司匹林", "error": null, "triples": 15, "hop1": 5, "hop2": 10, "hop3": 0} |
| Medication 三元组真实方向 | 通过 | {"matched_in_database": 15, "returned": 15} |
| Medication 单跳 | 通过 | {"triples": 5, "applied_hops": 1, "error": null} |
| Medication 超额跳数截断 | 通过 | {"triples": 15, "applied_hops": 2, "error": null} |
| Medication 单跳分支限额 | 通过 | {"triples": 2, "applied_hops": 2, "error": null} |
| Medication 总量限额 | 通过 | {"triples": 3, "applied_hops": 2, "error": null} |
| Symptom 默认多跳查询 | 通过 | {"root": "乏力", "error": null, "triples": 16, "hop1": 6, "hop2": 10, "hop3": 0} |
| Symptom 三元组真实方向 | 通过 | {"matched_in_database": 16, "returned": 16} |
| Symptom 单跳 | 通过 | {"triples": 6, "applied_hops": 1, "error": null} |
| Symptom 超额跳数截断 | 通过 | {"triples": 16, "applied_hops": 2, "error": null} |
| Symptom 单跳分支限额 | 通过 | {"triples": 2, "applied_hops": 2, "error": null} |
| Symptom 总量限额 | 通过 | {"triples": 3, "applied_hops": 2, "error": null} |
| 不存在实体返回空结果 | 通过 | {"root_entity": "__medgraph_readonly_nonexistent_20261111__", "configured_max_hops": 2, "applied_hops": 2, "executed_hops": 1, "total_triples": 0, "hop1_count": 0, "hop2_count": 0, "hop3_count": 0, "triples": [], "visited_entities": ["__medgraph_readonly_nonexistent_20261111__"], "provenance": {"sou…（完整值见 JSON） |
| 非法参数 max_hops | 通过 | "0 被拒绝" |
| 非法参数 limit_per_hop | 通过 | "0 被拒绝" |
| 非法参数 total_limit | 通过 | "0 被拒绝" |
| Text2Cypher 拒绝写入 | 通过 | {"error": "检测到写关键字「CREATE」，已拒绝执行", "database_called": false} |
| Text2Cypher 拒绝无界路径 | 通过 | {"error": "拒绝无界或无法验证的可变长度路径", "database_called": false} |
| Text2Cypher 拒绝超额路径 | 通过 | {"error": "查询关系跳数超过配置上限 2", "database_called": false} |
| Text2Cypher 真实 EXPLAIN 与读取 | 通过 | {"success": true, "columns": ["disease", "symptom"], "rows": [["高血压", "疲劳"], ["高血压", "耳鸣"], ["高血压", "心悸"]], "row_count": 3, "cypher": "MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) RETURN d.name AS disease, s.name AS symptom LIMIT 3"} |
| 真实后端评测语料读取 | 通过 | {"documents": 535, "corpus_hash": "2ef274a0b987445672af62cd08d164f90a193a0cac6a4a2d009d24f3e03a021c"} |

## 查询耗时

每个锚点先执行功能检查，再连续重复查询；单客户端串行、连接池与数据库已热。
首次值仅指本进程第一次调用该锚点，不是数据库重启后的冷启动。下列为客户端计时的图数据库扩展耗时，包含驱动处理、数据库往返及通知处理，不含向量编码、LLM、API 接入排队或端到端问答。

| 锚点 | 类型 | 重复次数 | 本进程首次 ms | 热查询 p50 ms | 热查询 p95 ms |
|---|---|---:|---:|---:|---:|
| 高血压 | Disease | 10 | 149.267 | 8.095 | 14.618 |
| 阿司匹林 | Medication | 10 | 78.0 | 6.556 | 12.02 |
| 乏力 | Symptom | 10 | 7.612 | 6.878 | 7.404 |

## 可以与不能据此得出的结论

- 可以证明：测试时 Neo4j 可连接、可认证；真实 Cypher 可执行，本次返回的三元组方向与数据库一致，测试锚点下的跳数和数量限制生效。
- 可以证明：Text2Cypher 的受控查询可 EXPLAIN 并读取；测试中的写入、无界及超额路径在调用数据库前被拒绝。
- 不能证明：数据库账号已具备强制只读权限。READ_ACCESS 是驱动访问模式；本次未尝试数据库写入，也未进行独立账号权限审计。
- 不能证明：GraphRAG 优于纯向量 RAG、医学正确性或急症识别效果；这些需要单独的对照实验与标注。
- 不能把本次串行查询分位数作为并发能力或系统端到端延迟。

## 复现

```powershell
python evals/verify_neo4j.py --repeats 10 --output evals/NEO4J_VERIFICATION_REPORT.md
```

完整统计、查询样本、限额结果与计时值保存在同名 JSON 文件中。
