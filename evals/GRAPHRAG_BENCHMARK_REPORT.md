# GraphRAG 对照实验：等待真实后端复测

原报告已撤回。旧实现使用字符二元组重合检索和内存字典关系扩展，
不能代表 BGE-M3/FAISS 与 Neo4j 的线上对照。旧“正确率”和“支持率”
为词面重合代理指标，也不能证明医学事实正确或降低幻觉。
旧报告中的 69.2% → 76.9%、22.2% → 29.7% 不再作为项目效果或简历依据。

修订后的执行器使用真实 FAISS IndexFlatIP 与 BGE-M3，将同一 Neo4j
快照的节点属性和完整关联事实构建为富文本基线；图谱组使用同一召回器的
相同数量锚点，再通过线上受控 Neo4j 扩展。两组使用同一模型、提示词、
证据字符上限和科室诊断排序器，不经过 ReAct 或答案缓存。

运行前需准备本地 BGE-M3、Neo4j 数据和可用模型配置，并冻结图谱写入：

```powershell
python evals/graphrag_benchmark.py --limit 3 --output evals/graphrag_smoke.md
python evals/graphrag_benchmark.py --output evals/GRAPHRAG_BENCHMARK_REPORT.md
```

相邻 JSON 报告保存每条答案、证据、哈希、后端、分母及错误；模型失败不再替换为模板输出。
医学事实正确率和陈述支持率须人工标注后才会显示，未标注时为 null。
人工标注以 `样本ID:vector` / `样本ID:graph` 为键，包含 `reviewer`、
`answer_hash`、`context_hash`；正确性需 `factual_correct` 和 `correctness_basis`；
支持率需 `claims_complete: true` 和完整 `claims` 数组，每项有 `claim`、
`verdict`（supported/contradicted/insufficient），supported 还需原文 `support_quote`。
完成标注后，对保存的 JSON 离线评分，无需重新生成答案或调用数据库：

```powershell
python evals/graphrag_benchmark.py --score-report evals/graphrag_smoke.json --judgments evals/review.json --output evals/graphrag_reviewed.md
```

哈希不匹配会报错，不得将旧答案标注套用到重生成答案。

当前尚未生成真实后端的新效果数字。科室 Hit@3 属于检索诊断，
不能直接称为线上分诊 Agent 提升；不自动输出“显著提升”或“降低幻觉”结论。
