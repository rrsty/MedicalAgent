"""Read-only live Neo4j verification; emits raw JSON and a conclusion Markdown."""
import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neo4j import GraphDatabase, READ_ACCESS
from config.settings import settings
from src.knowledge_graph import schema_manager
from src.knowledge_graph.graph_queries import GraphQueries


class NotificationCounter(logging.Handler):
    """Count server warnings without flooding stdout or altering the graph."""
    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        self.count += 1


def percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo), 3)


def verify(driver, repeats):
    endpoint = urlsplit(settings.neo4j_uri)
    report = {
        "tested_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "endpoint": {"host": endpoint.hostname, "port": endpoint.port or 7687},
        "scope": "read-only live Neo4j; no writes, model calls or generated-answer scoring",
        "config": {
            "kg_max_hops": settings.kg_max_hops,
            "per_hop_limit": settings.kg_multihop_limit_per_hop,
            "total_limit": settings.kg_multihop_total_limit,
        },
        "checks": [], "latency": [], "samples": [],
    }

    def check(name, ok, details):
        report["checks"].append({"name": name, "passed": bool(ok), "details": details})

    def rows(query, **params):
        with driver.session(default_access_mode=READ_ACCESS) as session:
            return [dict(r) for r in session.run(query, **params)]

    started = time.perf_counter()
    driver.verify_connectivity()
    value = rows("RETURN 1 AS ok")[0]["ok"]
    check("Bolt 连接与认证", value == 1, {"elapsed_ms": round((time.perf_counter()-started)*1000, 3)})
    report["server"] = rows(
        "CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition"
    )
    report["nodes"] = rows("MATCH (n) RETURN count(n) AS count")[0]["count"]
    report["relationships"] = rows("MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"]
    report["node_types"] = rows(
        "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY label"
    )
    report["relation_types"] = rows(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type"
    )
    report["indexes"] = rows(
        "SHOW INDEXES YIELD name, type, state, labelsOrTypes, properties "
        "RETURN name, type, state, labelsOrTypes, properties"
    )
    report["constraints"] = rows(
        "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties "
        "RETURN name, type, labelsOrTypes, properties"
    )
    check("知识图谱非空", report["nodes"] > 0 and report["relationships"] > 0,
          {"nodes": report["nodes"], "relationships": report["relationships"]})
    check("现有索引状态", bool(report["indexes"]) and all(
        index["state"] == "ONLINE" for index in report["indexes"]
    ), {"indexes": len(report["indexes"]), "constraints": len(report["constraints"])})
    report["self_loops"] = rows("MATCH (n)-[r]->(n) RETURN count(r) AS count")[0]["count"]
    report["isolated_nodes"] = rows("MATCH (n) WHERE NOT (n)--() RETURN count(n) AS count")[0]["count"]
    report["isolated_node_types"] = rows(
        "MATCH (n) WHERE NOT (n)--() "
        "RETURN labels(n)[0] AS type, count(*) AS count ORDER BY type"
    )
    report["provenance"] = {
        "nodes_with_records": rows(
            "MATCH (n) WHERE size(coalesce(n.provenance_records, [])) > 0 "
            "OR n.provenance_json IS NOT NULL RETURN count(n) AS count"
        )[0]["count"],
        "relationships_with_records": rows(
            "MATCH ()-[r]->() WHERE size(coalesce(r.provenance_records, [])) > 0 "
            "OR r.provenance_json IS NOT NULL RETURN count(r) AS count"
        )[0]["count"],
    }
    roots = rows(
        "MATCH (n)-[]-() WHERE n.name IS NOT NULL "
        "WITH labels(n)[0] AS type, n.name AS name, count(*) AS degree "
        "ORDER BY degree DESC, name "
        "WITH type, collect({name: name, degree: degree})[0] AS root "
        "RETURN type, root.name AS name, root.degree AS degree ORDER BY type"
    )
    chosen = [r for r in roots if r["type"] in {"Disease", "Symptom", "Medication"}]
    check("三类测试锚点覆盖", len(chosen) == 3, chosen)
    queries = GraphQueries()
    for root in chosen:
        call = {"name": root["name"], "entity_type": root["type"]}
        first_start = time.perf_counter()
        result = queries.expand_multihop(**call)
        first_ms = (time.perf_counter() - first_start) * 1000
        check(f"{root['type']} 默认多跳查询", not result.get("error") and bool(result["triples"]), {
            "root": root["name"], "error": result.get("error"),
            "triples": result["total_triples"], "hop1": result["hop1_count"],
            "hop2": result["hop2_count"], "hop3": result["hop3_count"],
        })
        report["samples"].append(result)
        if result.get("error"):
            continue
        triples = [dict(t, index=i) for i, t in enumerate(result["triples"])]
        verified = rows(
            "UNWIND $triples AS t MATCH (a)-[r]->(b) "
            "WHERE coalesce(a.name, a.content) = t.subject "
            "AND t.subject_type IN labels(a) AND type(r) = t.predicate "
            "AND coalesce(b.name, b.content) = t.object AND t.object_type IN labels(b) "
            "RETURN count(DISTINCT t.index) AS count", triples=triples,
        )[0]["count"] if triples else 0
        check(f"{root['type']} 三元组真实方向", verified == len(triples),
              {"matched_in_database": verified, "returned": len(triples)})
        for label, kwargs in [
            ("单跳", {"max_hops": 1}),
            ("超额跳数截断", {"max_hops": 99}),
            ("单跳分支限额", {"limit_per_hop": 1}),
            ("总量限额", {"total_limit": 3}),
        ]:
            limited = queries.expand_multihop(**call, **kwargs)
            ok = not limited.get("error")
            if label == "单跳":
                ok = ok and limited["applied_hops"] == 1 and all(t["hop"] == 1 for t in limited["triples"])
            elif label == "超额跳数截断":
                ok = ok and limited["applied_hops"] == settings.kg_max_hops
            elif label == "单跳分支限额":
                ok = ok and all(limited[f"hop{hop}_count"] <= 1 for hop in (1, 2, 3))
            else:
                ok = ok and limited["total_triples"] <= 3
            check(f"{root['type']} {label}", ok, {
                "triples": limited["total_triples"], "applied_hops": limited["applied_hops"],
                "error": limited.get("error"),
            })
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter()
            sample = queries.expand_multihop(**call)
            if sample.get("error"):
                raise RuntimeError(sample["error"])
            elapsed.append((time.perf_counter() - start) * 1000)
        report["latency"].append({
            "root": root["name"], "type": root["type"], "n": repeats,
            "first_in_this_process_ms": round(first_ms, 3),
            "p50_ms": percentile(elapsed, .5), "p95_ms": percentile(elapsed, .95),
            "min_ms": round(min(elapsed), 3), "max_ms": round(max(elapsed), 3),
        })

    absent = queries.expand_multihop("__medgraph_readonly_nonexistent_20261111__")
    check("不存在实体返回空结果", not absent.get("error") and absent["total_triples"] == 0, absent)
    for key in ("max_hops", "limit_per_hop", "total_limit"):
        try:
            queries.expand_multihop("头痛", **{key: 0})
        except ValueError:
            check(f"非法参数 {key}", True, "0 被拒绝")
        else:
            check(f"非法参数 {key}", False, "0 未被拒绝")

    from src.knowledge_graph.text2cypher import run_readonly_cypher
    from unittest.mock import patch
    for name, query in [
        ("拒绝写入", "CREATE (:VerificationMustNeverBeCreated)"),
        ("拒绝无界路径", "MATCH (a)-[:HAS_SYMPTOM*]->(b) RETURN b.name"),
        ("拒绝超额路径", "MATCH (a)-[:HAS_SYMPTOM*1..99]->(b) RETURN b.name"),
    ]:
        with patch("src.knowledge_graph.text2cypher.get_driver",
                   side_effect=AssertionError("Rejected query reached database")) as factory:
            rejected = run_readonly_cypher(query)
        check(f"Text2Cypher {name}", not rejected["success"] and not factory.called,
              {"error": rejected.get("error"), "database_called": factory.called})
    accepted = run_readonly_cypher(
        "MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) "
        "RETURN d.name AS disease, s.name AS symptom LIMIT 3"
    )
    check("Text2Cypher 真实 EXPLAIN 与读取", accepted["success"] and bool(accepted["rows"]), accepted)

    from evals.graphrag_benchmark import HomologousKnowledgeBase
    corpus = HomologousKnowledgeBase()
    check("真实后端评测语料读取", len(corpus.rich_documents) > 0,
          {"documents": len(corpus.rich_documents), "corpus_hash": corpus.corpus_hash})
    report["corpus_hash"] = corpus.corpus_hash
    report["passed"] = sum(c["passed"] for c in report["checks"])
    report["failed"] = sum(not c["passed"] for c in report["checks"])
    return report


def render(report):
    good = report.get("failed") == 0 and report.get("passed", 0) > 0
    lines = [
        "# Neo4j 真实数据库复测结论", "",
        f"- 测试时间：{report['tested_at']}（Asia/Shanghai）。",
        "- 测试方式：宿主机直连，使用项目配置；只读查询，无模拟数据库、无数据写入。",
        "- 未执行模型生成或 GraphRAG 回答质量评测。", "",
        "## 结论", "",
        ("**Neo4j 连接与本次测试范围内的多跳查询验证通过。**" if good
         else "**本次验证存在失败项，不能判定全部通过。**"),
    ]
    if report.get("error"):
        lines += ["", f"连接/执行错误：`{report['error']}`。"]
        return "\n".join(lines) + "\n"
    lines += [
        f"共 **{report['passed']} 项通过、{report['failed']} 项失败**。",
        f"当前实库为 **{report['nodes']:,} 个节点、{report['relationships']:,} 条关系**；"
        "该统计是当前数据库快照，不能直接替代历史评测环境的规模。", "",
        "**判定：连接与查询链路通过，数据来源和图谱完整性仍有待补齐。**", "",
        "## 数据库与数据资产", "",
        f"- 连接：`{report['endpoint']['host']}:{report['endpoint']['port']}`。",
        f"- 服务：`{json.dumps(report['server'], ensure_ascii=False)}`。",
        f"- 索引 {len(report['indexes'])} 个，约束 {len(report['constraints'])} 个。",
        f"- 自环关系 {report['self_loops']} 条，孤立节点 {report['isolated_nodes']} 个。",
        f"- 有来源字段的节点：{report['provenance']['nodes_with_records']}/{report['nodes']}。",
        f"- 有来源字段的关系：{report['provenance']['relationships_with_records']}/{report['relationships']}。",
        "- 来源字段存在不等于来源准确或通过医学审核；历史记录缺失时仍应显示 unknown。", "",
        "| 节点类型 | 数量 |", "|---|---:|",
    ]
    lines += [f"| {r['label']} | {r['count']} |" for r in report["node_types"]]
    lines += ["", "| 关系类型 | 数量 |", "|---|---:|"]
    lines += [f"| {r['type']} | {r['count']} |" for r in report["relation_types"]]
    isolated_summary = "、".join(
        f"{row['type']} {row['count']} 个" for row in report["isolated_node_types"]
    ) or "无"
    diseases = next((r["count"] for r in report["node_types"] if r["label"] == "Disease"), 0)
    lines += [
        "", "## 当前遗留问题", "",
        f"1. **来源记录缺失**：节点 {report['provenance']['nodes_with_records']}/{report['nodes']}、"
        f"关系 {report['provenance']['relationships_with_records']}/{report['relationships']} 有来源字段。"
        "新代码支持记录来源，但当前库没有完成数据回填；需从带真实来源的原始记录重新抽取、入库和建索引，不能伪造补齐。",
        f"2. **孤立节点**：共 {report['isolated_nodes']} 个（{isolated_summary}）。"
        "应核对其是否为未关联的演示数据、抽取遗漏或未使用实体；本次未删除或自动修复。",
        f"3. **规模口径**：本次为 {diseases} 种疾病、{report['nodes']} 节点、"
        f"{report['relationships']} 关系，可支撑“500+ 节点、700+ 关系”的当前规模描述。"
        "此前文档的 121 种疾病、2,307 节点、3,636 关系未在本次连接的数据库中复现，需对应其他数据版本的统计证据。",
        f"4. **服务器通知**：复测期间捕获 {report.get('server_notifications', 0)} 条 Neo4j 通知日志。"
        "当前库缺少 provenance_records/provenance_json 属性键，查询这些可选字段会产生警告；"
        "字段缺失已在来源覆盖统计中如实反映，不能按已有来源处理。",
    ]
    lines += ["", "## 功能验证", "", "| 检查 | 结果 | 证据摘要 |", "|---|---|---|"]
    for check in report["checks"]:
        detail = json.dumps(check["details"], ensure_ascii=False).replace("|", "\\|")
        if len(detail) > 300:
            detail = detail[:300] + "…（完整值见 JSON）"
        lines.append(f"| {check['name']} | {'通过' if check['passed'] else '失败'} | {detail} |")
    lines += [
        "", "## 查询耗时", "",
        "每个锚点先执行功能检查，再连续重复查询；单客户端串行、连接池与数据库已热。",
        "首次值仅指本进程第一次调用该锚点，不是数据库重启后的冷启动。"
        "下列为客户端计时的图数据库扩展耗时，包含驱动处理、数据库往返及通知处理，"
        "不含向量编码、LLM、API 接入排队或端到端问答。",
        "", "| 锚点 | 类型 | 重复次数 | 本进程首次 ms | 热查询 p50 ms | 热查询 p95 ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for sample in report["latency"]:
        lines.append(
            f"| {sample['root']} | {sample['type']} | {sample['n']} | "
            f"{sample['first_in_this_process_ms']} | {sample['p50_ms']} | {sample['p95_ms']} |"
        )
    lines += [
        "", "## 可以与不能据此得出的结论", "",
        "- 可以证明：测试时 Neo4j 可连接、可认证；真实 Cypher 可执行，"
        "本次返回的三元组方向与数据库一致，测试锚点下的跳数和数量限制生效。",
        "- 可以证明：Text2Cypher 的受控查询可 EXPLAIN 并读取；"
        "测试中的写入、无界及超额路径在调用数据库前被拒绝。",
        "- 不能证明：数据库账号已具备强制只读权限。READ_ACCESS 是驱动访问模式；"
        "本次未尝试数据库写入，也未进行独立账号权限审计。",
        "- 不能证明：GraphRAG 优于纯向量 RAG、医学正确性或急症识别效果；"
        "这些需要单独的对照实验与标注。",
        "- 不能把本次串行查询分位数作为并发能力或系统端到端延迟。", "",
        "## 复现", "",
        "```powershell",
        "python evals/verify_neo4j.py --repeats 10 --output evals/NEO4J_VERIFICATION_REPORT.md",
        "```", "",
        "完整统计、查询样本、限额结果与计时值保存在同名 JSON 文件中。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "evals/NEO4J_VERIFICATION_REPORT.md")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password),
        connection_timeout=5, connection_acquisition_timeout=10,
        max_transaction_retry_time=0,
    )
    previous_driver = schema_manager._driver
    schema_manager._driver = driver
    notification_logger = logging.getLogger("neo4j.notifications")
    old_handlers = notification_logger.handlers[:]
    old_propagate = notification_logger.propagate
    counter = NotificationCounter()
    notification_logger.handlers = [counter]
    notification_logger.propagate = False
    try:
        report = verify(driver, args.repeats)
        report["server_notifications"] = counter.count
    except Exception as exc:
        report = {
            "tested_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "error": f"{type(exc).__name__}: {exc}", "passed": 0, "failed": 1,
        }
    finally:
        schema_manager._driver = previous_driver
        driver.close()
        notification_logger.handlers = old_handlers
        notification_logger.propagate = old_propagate
    args.output.write_text(render(report), encoding="utf-8")
    raw_path = args.output.with_suffix(".json")
    raw_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(args.output), "passed": report["passed"], "failed": report["failed"],
        "nodes": report.get("nodes"), "relationships": report.get("relationships"),
        "error": report.get("error"), "latency": report.get("latency"),
        "provenance": report.get("provenance"),
    }, ensure_ascii=False, indent=2))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
