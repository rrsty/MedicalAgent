# TDD — 智慧问诊Agent系统 技术设计文档

版本：1.0.0 ｜ 上游文档：PRD.md ｜ 配套验收器：`scripts/check_contracts.py`

---

# 第 0 章 工程规范（R1–R8）—— 后续所有生成指令的「宪法」

> 每条规范同时满足：① 结构性约束（写法上不可能违规）；② 机器化验收（可执行命令）。
> 固定指令开头：`【工程规范前置】严格遵守 TDD 第 0 章规范 R1–R8；生成后运行自检，全绿才算完成。`

## R1 单一事实来源

- 约束：节点类型（8 种）、关系类型（12 种）只允许定义在 `src/knowledge_graph/schema.py`
  的 `NodeType` / `RelationType`；意图类型只允许定义在 `src/agents/state.py` 的 `IntentType`；
  路径与文件名只允许定义在 `config/paths.py`；抽取提示词只允许定义在
  `src/extraction/disease_extractor.py` 的 `EXTRACTION_INSTRUCTION`。其他模块一律 import，
  禁止字符串字面量重复定义。
- 验收：`python scripts/check_contracts.py`（R1 项：静态扫描枚举值赋值越界即 FAIL）。

## R2 契约先行

- 约束：按依赖顺序生成——schema.py → extraction → knowledge_graph → vector_store → agents →
  api → frontend；扩展功能先改契约（schema/models），再改下游。
- 验收：`python scripts/check_contracts.py` 全绿（含 import 链静态检查）+ 模块导入冒烟
  `python -c "import src.agents.graph"`。

## R3 前后端字段契约

- 约束：API 字段只允许在 `src/api/models.py` 定义一次；前端 `frontend/src/api/index.js` 与
  `ChatView.vue` 按同一张表消费，不得私增字段。
- 契约表（ConsultationResponse）：`answer, intent, symptoms, severity, departments,
  medications[{name,purpose,side_effects,reminder}], advice, disclaimers, warnings,
  linked_entities[{input_entity,matched_entity,type,similarity}], trace, duration_ms, session_id`。
- ConsultationRequest：`query`（1~1000 字）、`session_id`（**必填**，6~64 位 `[A-Za-z0-9_-]`，
  客户端生成 uuid——禁止共享 "default" 会话，医疗对话隐私红线）。
- 验收：`python scripts/check_contracts.py`（R3 项：AST 解析 models.py 逐字段核对 + 前端引用检查）。

## R4 索引契约

- 约束：嵌入模型 BGE-M3、维度 1024、索引 IndexIVFFlat + METRIC_INNER_PRODUCT、向量归一化——
  只允许 `config/settings.py` 定义一次（`embedding_dim` 等）。换模型 = 改契约 → 重建索引。
- 验收：`python scripts/check_contracts.py --r4`（settings 1024 == 模型 config.json hidden_size
  == 索引映射 dim）。

## R5 数据即产物

- 约束：契约变更后**只做 clear + build 重建**（`graph_builder.py clear && build`、
  `build_index.py`），禁止手写迁移补丁；数据文件不入库（.gitignore）。
- 验收：`python scripts/check_contracts.py`（R5 项：FAISS 向量数 == 实体映射记录数）；
  关系端点表/节点属性表与枚举自洽。

## R6 单向关系

- 约束：一个事实只建一个方向的关系，反查用 Cypher 反向遍历。
  两个例外（真实建反向边）：`TREATED_BY_MEDICATION ↔ TREATS_DISEASE`、
  `BELONGS_TO_DEPARTMENT ↔ HANDLES_DISEASE`。
- 验收：graph_builder 落库统计 + `graph_validator.py` 示例正反查均命中。

## R7 模块自检

- 约束：每完成一个模块立即跑对应验收，全绿才进入下一步：

| 模块 | 自检命令 |
|---|---|
| 契约 | `python scripts/check_contracts.py` |
| 数据采集 | `python crawler/data_processor.py`（统计输出） |
| 抽取 | `python scripts/run_extraction.py --no-llm --limit 3`（冒烟） |
| 图谱 | `python src/knowledge_graph/graph_validator.py`（非 0 退出码） |
| 索引 | `python scripts/build_index.py`（维度校验内置） |
| Agent | `python -m src.agents.graph --ask "头痛挂什么科"` |
| API/集成 | `python scripts/integration_test.py`（非 0 退出码） |
| 评估 | `python evals/runner.py --limit 12` |

## R8 黄金冒烟

- 约束：系统可用 = 三绿：契约检查（R1/R3/R4/R5）+ 集成测试（`integration_test.py`）+
  评估门禁（`evals/runner.py` 三硬门禁）。
- 验收：三条命令退出码全 0。

## 命名公约

- Python：snake_case；类 PascalCase；枚举值 Relation 用 SCREAMING_SNAKE，NodeType 用 PascalCase；
- 前端：组件 PascalCase，组合式函数/变量 camelCase；
- 数据文件名只在 `config/paths.py` 定义（如 `KG_ENTITIES_FILE`）；
- LangGraph 节点名 = Agent 文件名（`intent_classifier`…），中文名只在 `NODE_LABELS_ZH` 映射。

---

# 第 1 章 总体架构

```
┌─────────────── 用户 ───────────────┐
│  Vue3 + Element Plus（dist/ 静态）   │
└──────────────┬─────────────────────┘
               │ HTTP / SSE
┌──────────────▼─────────────────────┐
│  FastAPI（单进程：API + 静态托管）    │
│  /api/consult · /consult/stream     │
│  /api/health · /stats · /convers.*  │
├────────────────────────────────────-┤
│  LangGraph 状态图（9 Agent）         │
│  意图→症状→科室/用药/知识→安全→融合   │
├───────────┬────────────┬────────────┤
│ Neo4j     │ FAISS+     │ Redis      │
│ 知识图谱   │ BGE-M3     │ 热缓存      │
│ (5.26)    │ 向量检索    │ MySQL 历史  │
└───────────┴────────────┴────────────┘
离线管线：crawler → data_processor → LLM抽取 → graph_builder → build_index → enrich
```

## 1.1 运行时问答流程（按意图路由的 DAG，医疗分支强制汇聚安全检查）

```
START → 意图识别(LLM)
  ├─ ⚠ 急症强制升级门（纯规则：query 命中 EMERGENCY_SYMPTOM_KEYWORDS 一律 → 安全检查，堵住
  │   general 分支「寒暄式回答急症」的缺口；词表单一来源 safety_checker）
  ├─ appointment/medication → 症状检测(LLM+最近3轮上下文) ─┬─ appointment → 科室推荐(Neo4j→LLM→默认) → 就医指导 → 安全检查
  │                                          ├─ medication·有症状 → 用药建议(图谱候选+LLM过滤) → 安全检查
  │                                          ├─ medication·无症状 → 医学知识
  │                                          └─ 急症关键词 → 安全检查
  ├─ knowledge → 医学知识(grounded-ReAct: 强制检索FAISS+Neo4j → 实体链接 → 可选ReAct多跳/Text2Cypher) → 安全检查
  ├─ emergency → 安全检查（急症警告）
  └─ general → 通用对话(LLM) → END
安全检查(纯规则) → 答案融合(LLM+确定性兜底) → END
```

## 1.2 数据流

- 离线构建：爬虫（预定义疾病列表，robots 合规，断点续爬）→ 清洗（JSON+XLSX）→
  LLM 结构化抽取（SimpleLLMOutput）→ entities/relations.json → MERGE 建图 → BGE-M3 编码建 FAISS；
- 在线问答：query + 历史（Redis→MySQL 回填，短期记忆=最近 12 轮/24 条消息）→ 状态图
  （history 注入意图分类/症状检测/医学知识检索与生成，做指代消解）→ 检索/查询 → 生成 →
  安全检查 → 融合 → 响应 + 双写历史（MySQL 先，Redis 后，缓存可缺不可幻）。

# 第 2 章 逐模块设计（要点）

| 模块 | 职责 | 关键设计 |
|---|---|---|
| config/ | 全局配置与路径 | pydantic_settings 读 .env；paths 自动建目录 |
| src/common | logger/llm/utils | LLM 单例；get_extract_llm 云端↔vLLM 切换 |
| crawler/ | 采集与清洗 | 预定义疾病表；robots 检查；随机 UA + 3-5s 延时；断点续爬 |
| src/extraction | LLM 结构化抽取 | Pydantic 契约；云端 json_mode 与 vLLM 裸调用双链路；占位符回声过滤 |
| src/knowledge_graph | Schema/建图/查询/校验/Text2Cypher | MERGE 幂等；每 500 条事务；只读五层防护 |
| src/vector_store | 语义检索/实体链接 | IndexIVFFlat(nlist=100,nprobe=10)；阈值 0.5/0.85；进程级单例 + 启动期 warmup（防 Windows 原生崩溃） |
| src/agents | 9 Agent + 状态图 | 抽取/分类纯 LLM；安全纯规则；事实回答强制检索；结构化决策三级回退 |
| src/api | REST+SSE | Semaphore 并发限制；超时；SPA 兜底；启动预加载 |
| src/history | 冷热分层 | Redis LPUSH+LTRIM 12 轮+TTL24h；MySQL 两表 utf8mb4；三级命中读取 |
| evals | 评估 harness | 双模式；规则+金标+RAGAS+LLM 裁判；三门禁一票否决 |

# 第 3 章 接口设计

见 R3 契约表与 `src/api/models.py`。SSE 事件协议：

| event | data |
|---|---|
| progress | `{"node","label","facts":[...]}`（每节点完成一帧） |
| answer | 完整 ConsultationResponse JSON（含 linked_entities） |
| error | `{"message"}` |
| done | `[DONE]` |

# 第 4 章 知识图谱 Schema

节点（8 种，name 唯一约束）：Disease(name,description,icd_code)、Symptom(name,severity)、
Medication(name,category,side_effects,contraindications)、Department(name,floor)、
BodyPart(name)、Examination(name,type)、Food(name,type)、Precaution(content)。

关系（12 种）：HAS_SYMPTOM(D→S)、MAY_INDICATE(S→D)、BELONGS_TO_DEPARTMENT(D→Dep)、
HANDLES_DISEASE(Dep→D)†、TREATED_BY_MEDICATION(D→M)、TREATS_DISEASE(M→D)†、
HAS_SIDE_EFFECT(M→S)、INTERACTS_WITH(M→M)、NEEDS_EXAMINATION(D→E)、
AFFECTS_BODY_PART(D→B)、RECOMMENDS_FOOD(D→F)、HAS_PRECAUTION(D→P)。
（† 为 R6 例外，成对真实建边。）

# 第 5 章 自检结论

1. R1–R8 齐全，每条附验收命令 ✅
2. 接口设计字段与 R3 契约表一致 ✅
3. 全文命名无不一致（节点名/枚举/文件名单一来源）✅
