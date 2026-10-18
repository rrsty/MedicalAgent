"""Text2Cypher：自然语言 → 只读 Cypher → Neo4j 执行（演进新增）。

只读五层防护（绝不允许写库，绝不抛异常——失败返回友好错误文本）：
  ① 只读事务：驱动层 default_access_mode=READ_ACCESS
  ② 写关键字拒绝：CREATE / MERGE / DELETE / SET / REMOVE / DETACH / DROP / LOAD CSV ...
  ③ EXPLAIN 校验：语法错误在执行前暴露
  ④ 无 LIMIT 自动追加：防全表扫描
  ⑤ 记录数截断 + 审计日志
"""
import re

from neo4j import READ_ACCESS

from config.settings import settings
from src.common.llm import get_llm
from src.common.logger import setup_logger
from src.knowledge_graph.schema import generate_llm_schema_text
from src.knowledge_graph.schema_manager import get_driver

logger = setup_logger("kg.text2cypher", "neo4j.log")

MAX_ROWS = 20

# ② 写关键字黑名单（词边界匹配，避免误伤 RETURN 子句里的单词）
_WRITE_PATTERN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b",
    re.IGNORECASE,
)

_GENERATE_PROMPT = """你是 Neo4j Cypher 专家。根据下面的知识图谱 Schema，把用户的自然语言问题转换为**一条只读 Cypher 查询**。

{schema}

规则：
1. 只生成 MATCH/OPTIONAL MATCH/WHERE/RETURN/WITH/ORDER BY/LIMIT 组成的只读查询
2. 绝不生成 CREATE/MERGE/DELETE/SET 等任何写操作
3. RETURN 只返回属性值，不要返回整个节点
4. 字符串匹配优先用精确匹配，必要时可用 CONTAINS
5. 只输出 Cypher 语句本身，不要任何解释和代码块标记

用户问题：{question}"""


def _generate_cypher(question: str) -> str:
    llm = get_llm(temperature=0)
    resp = llm.invoke(_GENERATE_PROMPT.format(schema=generate_llm_schema_text(), question=question))
    text = resp.content.strip()
    # 剥离代码块围栏，取第一行看起来像 Cypher 的内容
    text = re.sub(r"```(?:cypher)?", "", text).strip()
    if ";" in text:
        text = text.split(";", 1)[0]
    return text.strip()


def _validate_readonly(cypher: str) -> str | None:
    """② 写关键字拒绝；返回拒绝原因或 None。"""
    match = _WRITE_PATTERN.search(cypher)
    if match:
        return f"检测到写关键字「{match.group(1)}」，已拒绝执行"


def _ensure_limit(cypher: str) -> str:
    """④ 无 LIMIT 自动追加。"""
    if re.search(r"\bLIMIT\s+\d+\b", cypher, re.IGNORECASE):
        return cypher
    return f"{cypher}\nLIMIT {MAX_ROWS}"


def run_readonly_cypher(cypher: str, max_rows: int = MAX_ROWS) -> dict:
    """执行只读 Cypher（五层防护），返回 {success, columns, rows, row_count, cypher, error}。"""
    cypher = cypher.strip().rstrip(";")
    try:
        # ② 写关键字拒绝
        rejected = _validate_readonly(cypher)
        if rejected:
            logger.warning("[Text2Cypher 审计] 拒绝: %s | %s", rejected, cypher[:200])
            return {"success": False, "error": rejected, "rows": []}

        cypher = _ensure_limit(cypher)  # ④

        driver = get_driver()
        with driver.session(default_access_mode=READ_ACCESS) as session:  # ① 只读事务
            session.run(f"EXPLAIN {cypher}")  # ③ EXPLAIN 校验语法

            result = session.run(cypher)
            keys = list(result.keys())
            rows = []
            for record in result:  # ⑤ 记录数截断
                rows.append([record[k] for k in keys])
                if len(rows) >= max_rows:
                    break

        logger.info("[Text2Cypher 审计] 执行成功 (%d 行): %s", len(rows), cypher[:200])
        return {
            "success": True,
            "columns": keys,
            "rows": rows,
            "row_count": len(rows),
            "cypher": cypher,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Text2Cypher 审计] 执行失败: %s | %s", exc, cypher[:200])
        return {"success": False, "error": f"Cypher 执行失败: {exc}", "rows": []}


def natural_language_query(question: str) -> dict:
    """自然语言 → Text2Cypher → 执行 的完整入口（settings.text2cypher_enabled 开关）。"""
    if not settings.text2cypher_enabled:
        return {"success": False, "error": "Text2Cypher 未启用", "rows": []}
    try:
        cypher = _generate_cypher(question)
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"Cypher 生成失败: {exc}", "rows": []}
    return run_readonly_cypher(cypher)
