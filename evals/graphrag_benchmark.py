"""Real FAISS/BGE-M3 versus FAISS + bounded Neo4j expansion.

No lexical retrieval substitute, seed-data substitute, or model-failure generator.
Medical correctness and claim support require explicit human annotations bound to
answer/context hashes. Missing annotations remain null, never invented scores.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.common.provenance import decode_provenance, format_context_with_provenance
from src.knowledge_graph.graph_queries import GraphQueries

PROMPT = "仅依据给定证据回答医疗科普问题。证据不足时说明无法确认；不提供诊断或处方。"


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class HomologousKnowledgeBase:
    """One Neo4j snapshot supplies BOTH rich-text corpus and relation facts."""
    def __init__(self, nodes=None, edges=None):
        if nodes is None:
            from neo4j import READ_ACCESS
            from src.knowledge_graph.schema_manager import get_driver

            def read_snapshot(tx):
                ns = [dict(r) for r in tx.run(
                    "MATCH (n) WHERE n.name IS NOT NULL OR n.content IS NOT NULL "
                    "RETURN elementId(n) AS id, labels(n)[0] AS type, properties(n) AS props "
                    "ORDER BY id"
                )]
                es = [dict(r) for r in tx.run(
                    "MATCH (a)-[r]->(b) RETURN elementId(a) AS source, "
                    "elementId(b) AS target, type(r) AS relation ORDER BY source, relation, target"
                )]
                return ns, es

            with get_driver().session(default_access_mode=READ_ACCESS) as session:
                nodes, edges = session.execute_read(read_snapshot)
        self.nodes = nodes
        self.edges = edges or []
        self.rich_documents = self._documents()
        if not self.rich_documents:
            raise RuntimeError("Neo4j corpus is empty; populate the graph before benchmarking")
        self.corpus_hash = text_hash(json.dumps(self.rich_documents, ensure_ascii=False, sort_keys=True))

    def _documents(self):
        named = {n["id"]: n for n in self.nodes}
        documents = []
        for node in self.nodes:
            props = node["props"]
            name = props.get("name") or props.get("content")
            if not name:
                continue
            facts = []
            for edge in self.edges:
                if node["id"] not in (edge["source"], edge["target"]):
                    continue
                a, b = named.get(edge["source"]), named.get(edge["target"])
                if not a or not b:
                    continue
                a_name = a["props"].get("name") or a["props"].get("content")
                b_name = b["props"].get("name") or b["props"].get("content")
                facts.append(f"({a_name})-[:{edge['relation']}]->({b_name})")
            attributes = {k: v for k, v in props.items()
                          if k not in ("provenance_json", "provenance_records")}
            content = f"{name}（{node['type']}）\n" + json.dumps(attributes, ensure_ascii=False, sort_keys=True)
            content += "\n" + "\n".join(sorted(set(facts)))
            documents.append({
                "id": node["id"], "name": name, "type": node["type"],
                "content": format_context_with_provenance(
                    content, props.get("provenance_records", props.get("provenance_json"))
                ),
            })
        return documents


class PureVectorRetriever:
    """Normalized embeddings searched by a real FAISS IndexFlatIP."""
    def __init__(self, kb, encoder=None):
        import faiss
        import numpy as np

        if encoder is None:
            from src.vector_store.search import get_shared_retriever
            encoder = get_shared_retriever().model
        self.encoder = encoder
        self.documents = kb.rich_documents
        vectors = np.asarray(encoder.encode(
            [d["content"] for d in self.documents], normalize_embeddings=True
        ), dtype="float32")
        if vectors.ndim != 2 or not len(vectors):
            raise ValueError("Invalid embedding matrix")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query, top_k=3):
        import numpy as np
        start = time.perf_counter()
        vector = np.asarray(self.encoder.encode([query], normalize_embeddings=True), dtype="float32")
        scores, ids = self.index.search(vector, min(top_k, len(self.documents)))
        docs = [dict(self.documents[int(i)], score=float(score))
                for score, i in zip(scores[0], ids[0]) if i >= 0]
        return docs, (time.perf_counter() - start) * 1000


class GraphRAGRetriever:
    def __init__(self, kb, vector_retriever, queries=None):
        self.vector_retriever = vector_retriever
        self.queries = queries or GraphQueries()

    def search(self, query, top_k=3):
        start = time.perf_counter()
        anchors, _ = self.vector_retriever.search(query, top_k=top_k)
        facts, triples = [], []
        for anchor in anchors:
            expansion = self.queries.expand_multihop(anchor["name"], entity_type=anchor["type"])
            if expansion.get("error"):
                raise RuntimeError(expansion["error"])
            for triple in expansion["triples"]:
                triples.append(triple)
                facts.append(format_context_with_provenance(
                    f"({triple['subject']})-[:{triple['predicate']}]->({triple['object']})",
                    triple.get("provenance"),
                ))
        # Both arms use the same anchors and rich documents; only expansion differs.
        contexts = list(dict.fromkeys(facts)) + [d["content"] for d in anchors]
        return contexts, (time.perf_counter() - start) * 1000, triples


class EvidenceBudgetManager:
    def __init__(self, max_chars=800):
        if max_chars < 1:
            raise ValueError("Evidence budget must be positive")
        self.max_chars = max_chars

    def apply(self, text):
        return text.strip()[:self.max_chars]


def triage_department_ranker(context, all_departments):
    # Same deterministic ranker for both arms. This is a retrieval diagnostic,
    # NOT a measurement of the production department agent.
    scores = [(context.count(dep), dep) for dep in all_departments]
    return [dep for count, dep in sorted(scores, key=lambda x: (-x[0], x[1])) if count][:3]


def generate_answer_with_llm(query, context, reference=""):
    from src.common.llm import get_llm
    start = time.perf_counter()
    answer = get_llm(temperature=0.0).invoke([
        ("system", PROMPT), ("human", f"检索证据：\n{context}\n用户问题：{query}"),
    ]).content
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Model returned an empty/non-text answer")
    return answer.strip(), (time.perf_counter() - start) * 1000


def compute_metrics(answer, context, reference, gold_deps, top3_deps, annotation=None):
    """Human-reviewed claim support; no substring or n-gram medical scoring."""
    accuracy, faithfulness = None, None
    if annotation is not None:
        if annotation.get("answer_hash") != text_hash(answer) or annotation.get("context_hash") != text_hash(context):
            raise ValueError("Annotation hashes do not match the evaluated answer/evidence")
        if not annotation.get("reviewer"):
            raise ValueError("Annotations must identify their reviewer")
        value = annotation.get("factual_correct")
        if value is not None:
            if not isinstance(value, bool) or not annotation.get("correctness_basis"):
                raise ValueError("Correctness requires a boolean and a recorded review basis")
            accuracy = float(value)
        if annotation.get("claims_complete") is True:
            claims = annotation.get("claims", [])
            for claim in claims:
                if not claim.get("claim") or claim.get("verdict") not in {"supported", "contradicted", "insufficient"}:
                    raise ValueError("Invalid claim review")
                if claim["verdict"] == "supported":
                    quote = claim.get("support_quote", "")
                    if not quote or quote not in context:
                        raise ValueError("Supported claims require a verbatim evidence quote")
            if claims:
                faithfulness = sum(c["verdict"] == "supported" for c in claims) / len(claims)
    hit = any(dep in top3_deps for dep in gold_deps) if gold_deps else None
    return accuracy, faithfulness, hit


class GraphRAGBenchmarkSuite:
    def __init__(self, evidence_budget=800, kb=None, encoder=None, queries=None,
                 generator=None, annotations=None):
        self.kb = kb or HomologousKnowledgeBase()
        self.vector_retriever = PureVectorRetriever(self.kb, encoder=encoder)
        self.graph_retriever = GraphRAGRetriever(self.kb, self.vector_retriever, queries=queries)
        self.budget_mgr = EvidenceBudgetManager(evidence_budget)
        self.generator = generator or generate_answer_with_llm
        self.annotations = annotations or {}
        self.all_departments = sorted({d["name"] for d in self.kb.rich_documents if d["type"] == "Department"})

    def load_dataset(self, limit=None):
        cases = [json.loads(line) for line in (PROJECT_ROOT / "evals/dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        cases = [c for c in cases if set(c.get("tags", [])) & {"knowledge", "medication", "appointment"}]
        return cases[:limit] if limit else cases

    def run_case(self, case):
        gold = case.get("gold", {})
        row = {"id": case["id"], "query": case["query"], "tags": case.get("tags", []),
               "gold_departments": gold.get("departments_acceptable", [])}
        # Alternate order to reduce systematic warmup/rate-limit ordering bias.
        variants = ("vector", "graph") if int(text_hash(case["id"]), 16) % 2 else ("graph", "vector")
        for variant in variants:
            arm = {"status": "error", "factual_accuracy": None, "faithfulness": None,
                   "hit_at_3": False if row["gold_departments"] else None,
                   "generation_backend": "configured_llm", "top3_departments": []}
            started = time.perf_counter()
            try:
                if variant == "vector":
                    docs, retrieval_ms = self.vector_retriever.search(case["query"])
                    context = "\n\n".join(d["content"] for d in docs)
                else:
                    contexts, retrieval_ms, _ = self.graph_retriever.search(case["query"])
                    context = "\n\n".join(contexts)
                context = self.budget_mgr.apply(context)
                answer, generation_ms = self.generator(case["query"], context)
                deps = triage_department_ranker(context, self.all_departments)
                annotation = self.annotations.get(f"{case['id']}:{variant}")
                accuracy, faithfulness, hit = compute_metrics(
                    answer, context, gold.get("reference_answer", ""),
                    row["gold_departments"], deps, annotation,
                )
                arm.update(status="ok", context=context, answer=answer,
                           answer_hash=text_hash(answer), context_hash=text_hash(context),
                           retrieval_ms=retrieval_ms, generation_ms=generation_ms,
                           factual_accuracy=accuracy, faithfulness=faithfulness,
                           hit_at_3=hit, top3_departments=deps)
            except Exception as exc:
                arm["error"] = f"{type(exc).__name__}: {exc}"
            arm["total_ms"] = (time.perf_counter() - started) * 1000
            row[variant] = arm
        return row

    def run_benchmark(self, limit=None, cases=None):
        rows = [self.run_case(c) for c in (cases if cases is not None else self.load_dataset(limit))]
        return {
            "created_at": datetime.now(timezone.utc).isoformat(), "total_samples": len(rows),
            "corpus_hash": self.kb.corpus_hash, "embedding_model": settings.embedding_model_path,
            "generation_model": settings.model_name, "temperature": 0,
            "evidence_budget_chars": self.budget_mgr.max_chars, "prompt_hash": text_hash(PROMPT),
            "kg_max_hops": settings.kg_max_hops,
            "backends": {"vector": "BGE-M3 + FAISS IndexFlatIP (rich documents)", "graph": "same FAISS anchors + live bounded Neo4j expansion"},
            "vector_baseline": aggregate_samples(rows, "vector"), "graphrag": aggregate_samples(rows, "graph"), "samples": rows,
        }


def aggregate_samples(rows, variant):
    arms = [r[variant] for r in rows]
    def mean(field):
        values = [a[field] for a in arms if a.get(field) is not None]
        return sum(values) / len(values) if values else None
    return {
        "successful": sum(a["status"] == "ok" for a in arms),
        "errors": sum(a["status"] != "ok" for a in arms),
        "factual_accuracy": mean("factual_accuracy"), "faithfulness": mean("faithfulness"),
        "accuracy_reviewed": sum(a.get("factual_accuracy") is not None for a in arms),
        "faithfulness_reviewed": sum(a.get("faithfulness") is not None for a in arms),
        "triage_hit3": mean("hit_at_3"),
        "triage_samples": sum(a.get("hit_at_3") is not None for a in arms),
        "retrieval_ms": mean("retrieval_ms"), "total_ms": mean("total_ms"),
    }

def score_saved_report(summary, annotations):
    """Apply reviewed labels to saved answers without rerunning any model."""
    known = {f"{r['id']}:{v}" for r in summary["samples"] for v in ("vector", "graph")}
    if set(annotations) - known:
        raise ValueError("Annotations contain unknown sample/variant keys")
    for row in summary["samples"]:
        for variant in ("vector", "graph"):
            arm = row[variant]
            annotation = annotations.get(f"{row['id']}:{variant}")
            if arm["status"] != "ok":
                if annotation is not None:
                    raise ValueError("Cannot score a failed model/retrieval result")
                continue
            accuracy, faithfulness, hit = compute_metrics(
                arm["answer"], arm["context"], "", row["gold_departments"],
                arm["top3_departments"], annotation,
            )
            arm.update(factual_accuracy=accuracy, faithfulness=faithfulness, hit_at_3=hit)
    summary["vector_baseline"] = aggregate_samples(summary["samples"], "vector")
    summary["graphrag"] = aggregate_samples(summary["samples"], "graph")
    summary["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def generate_markdown_report(summary, output_path):
    def display(value, percent=False):
        if value is None:
            return "待人工标注 / 无可用结果"
        return f"{value * 100:.1f}%" if percent else f"{value:.3f}"
    lines = ["# GraphRAG 真实后端对照实验", "", f"运行时间：{summary['created_at']}",
             f"样本数：{summary['total_samples']}；语料哈希：`{summary['corpus_hash']}`。", "",
             "两组使用同源完整关系文本、相同向量锚点数量、模型、提示词和证据字符上限；关闭 ReAct 和答案缓存。", "",
             "| 指标 | FAISS 富文本 | FAISS + Neo4j |", "|---|---:|---:|"]
    for name, field, percent in [
        ("人工审核事实正确率", "factual_accuracy", True),
        ("人工审核陈述支持率", "faithfulness", True),
        ("检索诊断科室 Hit@3", "triage_hit3", True),
        ("成功样本平均检索耗时 ms", "retrieval_ms", False),
        ("全部请求平均耗时 ms（含失败）", "total_ms", False),
        ("失败样本数", "errors", False),
        ("正确率已审核样本数", "accuracy_reviewed", False),
        ("支持率已审核样本数", "faithfulness_reviewed", False),
        ("分诊样本数（失败计未命中）", "triage_samples", False),
    ]:
        a, b = summary["vector_baseline"].get(field), summary["graphrag"].get(field)
        lines.append(f"| {name} | {display(a, percent)} | {display(b, percent)} |")
    lines += ["", "本实验的科室排序器为统一检索诊断规则，不等同于线上分诊 Agent。",
              "缺失医学标注的指标保持空值；不得把词面重合度解释为医学正确性。",
              "模型/检索失败显式记录，禁止静默使用替代生成器。温度为零不保证跨运行完全一致。",
              "测试期间须冻结图谱内容；语料哈希记录初始快照，不代表临床审核。",
              "不自动生成提升、显著性或降低幻觉结论；需结合逐条标注、分母及配对分析解释结果。"]
    text = "\n".join(lines) + "\n"
    Path(output_path).write_text(text, encoding="utf-8")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--score-report", type=Path, help="对已有JSON报告离线应用人工标注，不调用模型或数据库")
    parser.add_argument("--judgments", type=Path, help="人工标注JSON，以sample_id:variant为键")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evals/GRAPHRAG_BENCHMARK_REPORT.md")
    args = parser.parse_args()
    annotations = json.loads(args.judgments.read_text(encoding="utf-8")) if args.judgments else {}
    if args.score_report:
        if not args.judgments:
            parser.error("--score-report requires --judgments")
        summary = score_saved_report(
            json.loads(args.score_report.read_text(encoding="utf-8")), annotations
        )
    else:
        suite = GraphRAGBenchmarkSuite(evidence_budget=args.budget, annotations=annotations)
        summary = suite.run_benchmark(limit=args.limit)
    args.output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_markdown_report(summary, args.output)
    print(f"Report: {args.output}; inspect raw samples and annotation coverage before making claims")
    if summary["vector_baseline"]["errors"] or summary["graphrag"]["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
