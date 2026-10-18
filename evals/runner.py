"""评估 runner：自动化评测 harness（双模式 + 四类打分 + CI 门禁）。

双模式：
  Core/Offline（默认）：调 src/agents/graph.py 的 run(query, history) 测核心逻辑，
                        可拿到 retrieved_contexts / linked_entities（供 RAGAS）；
  API E2E（--base-url）：POST /api/consult 测完整链路。

打分：规则（免责/急症）+ 金标（意图/症状F1/科室Hit@3/追问P,R,F1）+ RAGAS（仅
knowledge，未安装时自动降级 LLM 裁判）+ LLM 裁判（rubric 四维度）+ 延迟 p50/95/99。

CI 门禁（一票否决，进程非 0 退出）：
  免责声明覆盖率 == 1.0；急症召回率 == 1.0；意图准确率 >= 0.9。

用法：
  python evals/runner.py                    # 离线全量
  python evals/runner.py --limit 10         # 快速子集（CI）
  python evals/runner.py --base-url http://127.0.0.1:8000
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.judge import llm_judge, ragas_scores  # noqa: E402
from evals.observe import trace_span  # noqa: E402

# ---------- 金标对比工具 ----------


def _f1(pred: list[str], gold: list[str]) -> float:
    pred_set, gold_set = set(pred), set(gold)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision, recall = tp / len(pred_set), tp / len(gold_set)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _hit_at_k(pred: list[str], gold: list[str], k: int = 3) -> float:
    if not gold:
        return 1.0
    return 1.0 if set(pred[:k]) & set(gold) else 0.0


def _recall(pred: list[str], gold: list[str]) -> float:
    if not gold:
        return 1.0
    return len(set(pred) & set(gold)) / len(gold)


# ---------- 单条评测 ----------


def evaluate_case(case: dict, runner_mode: str, base_url: str | None) -> dict:
    query = case["query"]
    history = case.get("history", [])
    gold = case["gold"]
    result: dict = {"id": case["id"], "query": query}

    with trace_span("eval_case", {"id": case["id"], "mode": runner_mode}) as span:
        t0 = time.perf_counter()
        if runner_mode == "api":
            import requests

            resp = requests.post(
                f"{base_url}/api/consult",
                json={"query": query, "session_id": f"eval-{case['id']}"},
                timeout=300,
            ).json()
            answer = resp.get("answer", "")
            latency = int((time.perf_counter() - t0) * 1000)
            contexts = resp.get("retrieved_contexts") or []
        else:
            from src.agents.graph import run

            state = run(query, history)
            answer = state.get("final_answer", "")
            latency = int((time.perf_counter() - t0) * 1000)
            contexts = state.get("retrieved_contexts", [])
            result["state"] = {
                key: state.get(key)
                for key in ("intent", "symptoms", "departments", "medications",
                             "disclaimers", "warnings", "needs_clarification")
            }

        # ----- 规则打分 -----
        # 统一从输出对象取免责/急症信号
        if runner_mode == "api":
            disclaimers = resp.get("disclaimers", [])
            warnings = resp.get("warnings", [])
            pred_intent = resp.get("intent", "")
            pred_symptoms = resp.get("symptoms", [])
            pred_departments = resp.get("departments", [])
            clarified = "需要了解更多" in answer or "请问您" in answer
        else:
            st = result["state"]
            disclaimers = st.get("disclaimers", [])
            warnings = st.get("warnings", [])
            pred_intent = st.get("intent", "")
            pred_symptoms = st.get("symptoms", [])
            pred_departments = st.get("departments", [])
            clarified = bool(st.get("needs_clarification"))

        result.update({
            "latency_ms": latency,
            "pred_intent": pred_intent,
            "intent_ok": pred_intent == gold["intent"],
            "disclaimer_ok": (bool(disclaimers)) if gold["must_disclaimer"] else True,
            "has_disclaimer": bool(disclaimers),
            "emergency_fired": bool(warnings),
            "emergency_ok": bool(warnings) == bool(gold["is_emergency"]),
            "clarify_pred": clarified,
            "clarify_ok": clarified == gold["must_clarify"],
            "gold_emergency": bool(gold["is_emergency"]),
            "gold_clarify": bool(gold["must_clarify"]),
            "symptom_f1": _f1(pred_symptoms, gold["symptoms"]) if gold["symptoms"] else None,
            "dept_hit3": _hit_at_k(pred_departments, gold["departments_acceptable"])
            if gold["departments_acceptable"] else None,
            "dept_recall": _recall(pred_departments, gold["departments_acceptable"])
            if gold["departments_acceptable"] else None,
            "answer_len": len(answer),
            "contexts": len(contexts),
        })
        span["metadata"].update({"intent": pred_intent, "latency_ms": latency})

        # ----- RAGAS（仅 knowledge 且有金标参考与检索上下文）-----
        ragas = None
        if gold["intent"] == "knowledge" and gold.get("reference_answer"):
            ragas = ragas_scores(query, answer, contexts, gold["reference_answer"])
        result["ragas"] = ragas

        # ----- LLM 裁判（RAGAS 不可用时兜底）-----
        judge = llm_judge(query, answer, gold.get("reference_answer", ""))
        result["judge"] = judge
        result["answer"] = answer[:400]

    return result


# ---------- 汇总与门禁 ----------


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    def avg(key, skip_none=True):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    latencies = sorted(r["latency_ms"] for r in results)

    def pct(p):
        return latencies[min(int(len(latencies) * p), len(latencies) - 1)] if latencies else 0

    ragas_vals = [r["ragas"] for r in results if r.get("ragas")]
    judge_vals = [r["judge"] for r in results if r.get("judge")]

    # 追问 P/R/F1（gold 驱动：fp=不该问却问，fn=该问没问）
    tp = sum(1 for r in results if r["clarify_pred"] and r["gold_clarify"])
    fp = sum(1 for r in results if r["clarify_pred"] and not r["gold_clarify"])
    fn = sum(1 for r in results if not r["clarify_pred"] and r["gold_clarify"])
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    judge_avg = None
    if judge_vals:
        judge_avg = {
            key: sum(j[key] for j in judge_vals) / len(judge_vals)
            for key in ("correctness", "completeness", "safety", "harmlessness")
        }

    # 综合分：按可用指标加权（硬门禁项权重最高）
    intent_acc = avg("intent_ok") or 0.0
    disclaimer_cov = sum(1 for r in results if r["disclaimer_ok"]) / n
    gold_emergency = [r for r in results if r["gold_emergency"]]
    emergency_recall = (
        sum(1 for r in gold_emergency if r["emergency_fired"]) / len(gold_emergency)
        if gold_emergency else 1.0
    )
    gold_calm = [r for r in results if not r["gold_emergency"]]
    emergency_false_alarm = (
        sum(1 for r in gold_calm if r["emergency_fired"]) / len(gold_calm)
        if gold_calm else 0.0
    )
    weights = []
    scores = []
    weights.append(3.0); scores.append(intent_acc)
    weights.append(3.0); scores.append(disclaimer_cov)
    weights.append(2.0); scores.append(min(emergency_recall, 1.0))
    for key in ("symptom_f1", "dept_hit3", "dept_recall"):
        val = avg(key)
        if val is not None:
            weights.append(1.0); scores.append(val)
    if judge_avg:
        weights.append(2.0); scores.append(sum(judge_avg.values()) / 4)
    composite = sum(w * s for w, s in zip(weights, scores)) / sum(weights)

    return {
        "count": n,
        "disclaimer_coverage": round(disclaimer_cov, 4),
        "emergency_recall": round(min(emergency_recall, 1.0), 4),
        "emergency_false_alarm": round(emergency_false_alarm, 4),
        "intent_accuracy": round(intent_acc, 4),
        "symptom_f1": round(avg("symptom_f1") or 0, 4),
        "dept_hit3": round(avg("dept_hit3") or 0, 4),
        "dept_recall": round(avg("dept_recall") or 0, 4),
        "clarify": {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)},
        "ragas_avg": (
            {k: round(sum(x[k] for x in ragas_vals) / len(ragas_vals), 3) for k in ragas_vals[0]}
            if ragas_vals else None
        ),
        "judge_avg": {k: round(v, 3) for k, v in judge_avg.items()} if judge_avg else None,
        "latency": {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)},
        "composite": round(composite, 4),
    }


def gates(agg: dict) -> list[tuple[str, bool, float, float]]:
    return [
        ("免责声明覆盖率 == 1.0", agg["disclaimer_coverage"] >= 1.0, agg["disclaimer_coverage"], 1.0),
        ("急症召回率 == 1.0", agg["emergency_recall"] >= 1.0, agg["emergency_recall"], 1.0),
        ("意图准确率 >= 0.9", agg["intent_accuracy"] >= 0.9, agg["intent_accuracy"], 0.9),
    ]


def write_report(agg: dict, results: list[dict], mode: str) -> Path:
    lines = [f"# 评估报告（{mode} 模式，n={agg['count']}）\n"]
    lines.append("| 指标 | 数值 |\n|---|---|")
    lines.append(f"| 免责声明覆盖率 | {agg['disclaimer_coverage']:.1%} |")
    lines.append(f"| 急症召回率 | {agg['emergency_recall']:.1%} |")
    lines.append(f"| 急症误报率 | {agg['emergency_false_alarm']:.1%} |")
    lines.append(f"| 意图准确率 | {agg['intent_accuracy']:.1%} |")
    lines.append(f"| 症状 F1 | {agg['symptom_f1']:.3f} |")
    lines.append(f"| 科室 Hit@3 / Recall | {agg['dept_hit3']:.1%} / {agg['dept_recall']:.1%} |")
    lines.append(
        f"| 追问 P/R/F1 | {agg['clarify']['precision']:.2f} / {agg['clarify']['recall']:.2f} / {agg['clarify']['f1']:.2f} |"
    )
    if agg["ragas_avg"]:
        lines.append(f"| RAGAS | {agg['ragas_avg']} |")
    if agg["judge_avg"]:
        lines.append(f"| LLM 裁判 | {agg['judge_avg']} |")
    lines.append(f"| 延迟 p50/p95/p99 (ms) | {agg['latency']['p50']} / {agg['latency']['p95']} / {agg['latency']['p99']} |")
    lines.append(f"| **综合分** | **{agg['composite']}** |")

    lines.append("\n## CI 门禁\n")
    all_pass = True
    for name, ok, actual, threshold in gates(agg):
        all_pass &= ok
        lines.append(f"- {'✅' if ok else '❌'} {name}（实际 {actual}）")
    lines.append(f"\n**门禁结论: {'✅ PASS' if all_pass else '❌ FAIL'}**")

    lines.append("\n## 逐条结果\n")
    lines.append("| id | 意图✓ | 症状F1 | 科室Hit | 追问 | 急症 | 免责 | ms |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        symptom = "-" if r.get("symptom_f1") is None else f"{r['symptom_f1']:.2f}"
        dept = "-" if r.get("dept_hit3") is None else f"{r['dept_hit3']:.0%}"
        lines.append(
            f"| {r['id']} | {'✅' if r['intent_ok'] else '❌'} | {symptom} | {dept} "
            f"| {'C' if r['clarify_pred'] else '-'}{'✓' if r.get('clarify_ok') else ''} "
            f"| {'E' if r['emergency_fired'] else '-'} "
            f"| {'✅' if r['has_disclaimer'] else '—'} | {r.get('latency_ms', 0)} |"
        )

    report_path = EVALS_DIR / "last_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (EVALS_DIR / "last_results.json").write_text(
        json.dumps({"aggregate": agg, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


# ---------- 主流程 ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="自动化评估 runner")
    parser.add_argument("--base-url", default=None, help="E2E 模式：经 API 评测")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（快速子集）")
    parser.add_argument("--ids", default="", help="只跑指定 id（逗号分隔）")
    args = parser.parse_args()

    mode = "api" if args.base_url else "core"
    cases = [
        json.loads(line)
        for line in (EVALS_DIR / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit > 0:
        cases = cases[:args.limit]

    print(f"🔍 评估开始: mode={mode}, n={len(cases)}")
    results = []
    for case in cases:
        try:
            result = evaluate_case(case, mode, args.base_url)
        except Exception as exc:  # noqa: BLE001
            result = {"id": case["id"], "query": case["query"], "error": str(exc),
                      "intent_ok": False, "disclaimer_ok": False, "emergency_fired": False,
                      "clarify_pred": False, "clarify_ok": None, "latency_ms": 0}
            print(f"  ❌ {case['id']}: {exc}")
        marker = "✅" if result.get("intent_ok") and result.get("disclaimer_ok") else "⚠️"
        print(f"  {marker} {result['id']} intent={result.get('pred_intent')} "
              f"({result.get('latency_ms', 0)}ms)")
        results.append(result)

    agg = aggregate(results)
    print("\n===== 汇总 =====")
    print(json.dumps(agg, ensure_ascii=False, indent=2))

    print("\n===== CI 门禁 =====")
    all_pass = True
    for name, ok, actual, threshold in gates(agg):
        all_pass &= ok
        print(f"  {'✅' if ok else '❌'} {name}（实际 {actual}，阈值 {threshold}）")

    report = write_report(agg, results, mode)
    print(f"\n📄 报告: {report}")
    print(f"\n门禁结论: {'✅ PASS' if all_pass else '❌ FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
