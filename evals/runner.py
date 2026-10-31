"""评估 runner：自动化评测 harness（多模式 + 四类打分 + CI 门禁）。

支持模式：
  core（默认）：调 src/agents/graph.py 的 run(query, history) 测核心逻辑；
  api（--base-url）：POST /api/consult 测同步 REST 接口；
  sse（--mode sse）：POST /api/consult/stream 测 SSE 流式接口，解析事件流。

打分：规则（免责/急症）+ 金标（意图/症状F1/科室Hit@3/追问P,R,F1）+ RAGAS（仅
knowledge，未安装时自动降级 LLM 裁判）+ LLM 裁判（rubric 四维度）+ 延迟 p50/95/99（物理隔离成功与失败）。

CI 门禁（一票否决，进程非 0 退出）：
  免责声明覆盖率 == 1.0；急症召回率 == 1.0；意图准确率 >= 0.9。

用法：
  python evals/runner.py                         # 离线全量
  python evals/runner.py --limit 10              # 快速子集（CI）
  python evals/runner.py --base-url http://127.0.0.1:8000
  python evals/runner.py --mode sse --base-url http://127.0.0.1:8000
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

hasattr(sys.stdout, "reconfigure") and sys.stdout.reconfigure(encoding="utf-8")

EVALS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.judge import llm_judge, ragas_scores  # noqa: E402
from evals.observe import trace_span  # noqa: E402

# ---------- 统一 Schema 规范 ----------

UNIFIED_SAMPLE_SCHEMA_FIELDS = {
    "id",
    "query",
    "intent_pred",
    "gold_intent",
    "intent_ok",
    "emergency_pred",
    "gold_emergency",
    "emergency_ok",
    "clarify_pred",
    "gold_clarify",
    "clarify_ok",
    "disclaimer_ok",
    "error",
    "latency_ms",
    "contexts",
    "answer",
}


def _classify_error(exc: Exception | str) -> str:
    err_str = str(exc).lower()
    exc_type = type(exc).__name__.lower() if isinstance(exc, Exception) else ""
    if isinstance(exc, TimeoutError) or "timeout" in err_str or "timed out" in err_str or "超时" in err_str:
        return "timeout"
    if any(code in err_str for code in ("404", "500", "502", "503", "status_code", "http")):
        return "http_error"
    if any(k in err_str for k in ("auth", "rate", "model", "openai", "api key", "apierror", "gpu", "cuda", "oom", "memory")):
        return "model_error"
    if "empty" in err_str or "no answer" in err_str or "without returning an answer" in err_str:
        return "empty_response"
    if isinstance(exc, json.JSONDecodeError) or "json" in exc_type or any(k in err_str for k in ("json", "parse", "validation")):
        return "parse_error"
    return "unknown"


def _build_error_sample(case: dict, exc: Exception | str, latency_ms: int, mode: str = "core") -> dict:
    gold = case.get("gold") or {}
    gold_intent = gold.get("intent", "")
    gold_emergency = bool(gold.get("is_emergency", False))
    gold_clarify = bool(gold.get("must_clarify", False))
    return {
        "id": case.get("id", ""),
        "query": case.get("query", ""),
        "mode": mode,
        "status": "error",
        "error": str(exc),
        "error_type": _classify_error(exc),
        "latency_ms": latency_ms,
        "failed_latency_ms": latency_ms,
        "intent_pred": "",
        "pred_intent": "",
        "gold_intent": gold_intent,
        "intent_ok": False,
        "emergency_pred": False,
        "emergency_fired": False,
        "gold_emergency": gold_emergency,
        "emergency_ok": False,
        "clarify_pred": False,
        "gold_clarify": gold_clarify,
        "clarify_ok": False,
        "disclaimer_ok": False,
        "has_disclaimer": False,
        "pred_symptoms": [],
        "pred_departments": [],
        "pred_medications": [],
        "answer": "",
        "answer_len": 0,
        "contexts": 0,
        "retrieved_contexts": [],
        "gold_must_disclaimer": bool(gold.get("must_disclaimer", True)),
        "gold_symptoms": gold.get("symptoms", []),
        "gold_departments_acceptable": gold.get("departments_acceptable", []),
        "symptom_f1": 0.0 if gold.get("symptoms") else None,
        "dept_hit3": 0.0 if gold.get("departments_acceptable") else None,
        "dept_recall": 0.0 if gold.get("departments_acceptable") else None,
        "ragas": None,
        "judge": None,
    }


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


def evaluate_case(
    case: dict,
    runner_mode: str = "core",
    base_url: str | None = None,
    session_id: str | None = None,
) -> dict:
    query = case.get("query", "")
    history = case.get("history") or []
    gold = case.get("gold") or {}
    gold_intent = gold.get("intent", "")
    gold_emergency = bool(gold.get("is_emergency", False))
    gold_clarify = bool(gold.get("must_clarify", False))
    must_disclaimer = bool(gold.get("must_disclaimer", True))

    if not session_id:
        session_id = f"eval-{uuid.uuid4().hex[:8]}-{case.get('id', 'case')}"

    result: dict = {
        "id": case.get("id", ""),
        "query": query,
        "session_id": session_id,
        "mode": runner_mode,
    }

    t0 = time.perf_counter()
    try:
        with trace_span("eval_case", {"id": case.get("id", ""), "mode": runner_mode}) as span:
            if runner_mode == "sse":
                import requests

                payload = {
                    "query": query,
                    "session_id": session_id,
                    "history": history,
                }
                resp_data: dict = {}
                error_msg = None
                trace: list[dict] = []
                current_event = None

                with requests.post(
                    f"{base_url}/api/consult/stream",
                    json=payload,
                    stream=True,
                    timeout=300,
                ) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines(decode_unicode=True):
                        if not raw_line:
                            current_event = None
                            continue
                        line = raw_line.strip()
                        if line.startswith("event:"):
                            current_event = line[len("event:"):].strip()
                        elif line.startswith("data:"):
                            data_str = line[len("data:"):].strip()
                            if current_event == "progress":
                                try:
                                    trace.append(json.loads(data_str))
                                except (json.JSONDecodeError, TypeError, ValueError):
                                    pass
                            elif current_event == "answer":
                                resp_data = json.loads(data_str)
                            elif current_event == "error":
                                try:
                                    err_json = json.loads(data_str)
                                    error_msg = err_json.get("message", data_str)
                                except (json.JSONDecodeError, TypeError, ValueError):
                                    error_msg = data_str
                            elif current_event == "done":
                                break

                if error_msg and not resp_data:
                    raise RuntimeError(f"SSE Error: {error_msg}")
                if not resp_data or resp_data.get("answer") is None:
                    raise ValueError("SSE stream closed without returning an answer event")

                answer = resp_data.get("answer") or ""
                pred_intent = resp_data.get("intent", "")
                pred_symptoms = resp_data.get("symptoms", []) or []
                pred_departments = resp_data.get("departments", []) or []
                pred_medications = resp_data.get("medications", []) or []
                disclaimers = resp_data.get("disclaimers", []) or []
                warnings = resp_data.get("warnings", []) or []
                if resp_data.get("needs_clarification") is not None:
                    clarified = bool(resp_data.get("needs_clarification"))
                else:
                    clarified = "需要了解更多" in answer or "请问您" in answer
                contexts = resp_data.get("retrieved_contexts") or []
                if not trace:
                    trace = resp_data.get("trace", []) or []
                latency = int((time.perf_counter() - t0) * 1000)

            elif runner_mode == "api":
                import requests

                payload = {
                    "query": query,
                    "session_id": session_id,
                    "history": history,
                }
                resp = requests.post(
                    f"{base_url}/api/consult",
                    json=payload,
                    timeout=300,
                )
                resp.raise_for_status()
                resp_data = resp.json()

                answer = resp_data.get("answer") or ""
                pred_intent = resp_data.get("intent", "")
                pred_symptoms = resp_data.get("symptoms", []) or []
                pred_departments = resp_data.get("departments", []) or []
                pred_medications = resp_data.get("medications", []) or []
                disclaimers = resp_data.get("disclaimers", []) or []
                warnings = resp_data.get("warnings", []) or []
                if resp_data.get("needs_clarification") is not None:
                    clarified = bool(resp_data.get("needs_clarification"))
                else:
                    clarified = "需要了解更多" in answer or "请问您" in answer
                contexts = resp_data.get("retrieved_contexts") or []
                trace = resp_data.get("trace", []) or []
                latency = int((time.perf_counter() - t0) * 1000)

            else:  # core
                from src.agents.graph import run

                state = run(query, history)
                answer = (state.get("final_answer") or state.get("knowledge_answer") or "")
                latency = int((time.perf_counter() - t0) * 1000)
                contexts = state.get("retrieved_contexts", []) or []
                disclaimers = state.get("disclaimers", []) or []
                warnings = state.get("warnings", []) or []
                pred_intent = state.get("intent", "") or ""
                pred_symptoms = state.get("symptoms", []) or []
                pred_departments = state.get("departments", []) or []
                pred_medications = state.get("medications", []) or []
                clarified = bool(state.get("needs_clarification", False))
                trace = state.get("trace", []) or []
                result["state"] = {
                    key: state.get(key)
                    for key in ("intent", "symptoms", "departments", "medications",
                                 "disclaimers", "warnings", "needs_clarification")
                }

            # ----- RAGAS（仅 knowledge 且有金标参考与检索上下文）-----
            ragas = None
            if gold_intent == "knowledge" and gold.get("reference_answer"):
                ragas = ragas_scores(query, answer, contexts, gold["reference_answer"])

            # ----- LLM 裁判（RAGAS 不可用时兜底）-----
            judge = llm_judge(query, answer, gold.get("reference_answer", ""))

            intent_ok = (pred_intent == gold_intent)
            disclaimer_ok = bool(disclaimers) if must_disclaimer else True
            emergency_ok = (bool(warnings) == gold_emergency)
            clarify_ok = (clarified == gold_clarify)

            result.update({
                "status": "success",
                "error": None,
                "error_type": None,
                "latency_ms": latency,
                "failed_latency_ms": None,
                "intent_pred": pred_intent,
                "pred_intent": pred_intent,
                "gold_intent": gold_intent,
                "intent_ok": intent_ok,
                "emergency_pred": bool(warnings),
                "emergency_fired": bool(warnings),
                "gold_emergency": gold_emergency,
                "emergency_ok": emergency_ok,
                "clarify_pred": clarified,
                "gold_clarify": gold_clarify,
                "clarify_ok": clarify_ok,
                "disclaimer_ok": disclaimer_ok,
                "has_disclaimer": bool(disclaimers),
                "gold_must_disclaimer": must_disclaimer,
                "pred_symptoms": pred_symptoms,
                "gold_symptoms": gold.get("symptoms", []),
                "symptom_f1": _f1(pred_symptoms, gold["symptoms"]) if gold.get("symptoms") else None,
                "pred_departments": pred_departments,
                "gold_departments_acceptable": gold.get("departments_acceptable", []),
                "dept_hit3": _hit_at_k(pred_departments, gold["departments_acceptable"])
                if gold.get("departments_acceptable") else None,
                "dept_recall": _recall(pred_departments, gold["departments_acceptable"])
                if gold.get("departments_acceptable") else None,
                "pred_medications": pred_medications,
                "answer": answer[:400],
                "answer_len": len(answer),
                "contexts": len(contexts),
                "retrieved_contexts": contexts,
                "trace": trace,
                "ragas": ragas,
                "judge": judge,
            })
            span["metadata"].update({"intent": pred_intent, "latency_ms": latency})
            return result

    except Exception as exc:  # noqa: BLE001
        elapsed = int((time.perf_counter() - t0) * 1000)
        err_res = _build_error_sample(case, exc, elapsed, mode=runner_mode)
        err_res["session_id"] = session_id
        return err_res


# ---------- 汇总与门禁 ----------


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {
            "count": 0,
            "success_count": 0,
            "failed_count": 0,
            "success_rate": 0.0,
            "disclaimer_coverage": 0.0,
            "emergency_recall": 0.0,
            "emergency_false_alarm": 0.0,
            "intent_accuracy": 0.0,
            "symptom_f1": 0.0,
            "dept_hit3": 0.0,
            "dept_recall": 0.0,
            "clarify": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "ragas_avg": None,
            "judge_avg": None,
            "latency": {"p50": 0, "p95": 0, "p99": 0, "avg": 0},
            "failed_latency": {"count": 0, "avg": 0, "p95": 0},
            "errors_by_type": {},
            "composite": 0.0,
        }

    success_results = [r for r in results if not r.get("error")]
    failed_results = [r for r in results if r.get("error")]
    success_count = len(success_results)
    failed_count = len(failed_results)
    success_rate = round(success_count / n, 4) if n > 0 else 0.0

    # 延迟统计：物理隔离成功与失败请求
    success_latencies = sorted(
        r["latency_ms"] for r in success_results if r.get("latency_ms") is not None
    )
    failed_latencies = sorted(
        r["latency_ms"] for r in failed_results if r.get("latency_ms") is not None
    )

    def pct(lat_list, p):
        if not lat_list:
            return 0
        return lat_list[min(int(len(lat_list) * p), len(lat_list) - 1)]

    latency_p50 = pct(success_latencies, 0.50)
    latency_p95 = pct(success_latencies, 0.95)
    latency_p99 = pct(success_latencies, 0.99)
    latency_avg = int(sum(success_latencies) / len(success_latencies)) if success_latencies else 0

    failed_latency_avg = int(sum(failed_latencies) / len(failed_latencies)) if failed_latencies else 0
    failed_latency_p95 = pct(failed_latencies, 0.95)

    errors_by_type: dict[str, int] = {}
    for r in failed_results:
        err_type = r.get("error_type") or "unknown"
        errors_by_type[err_type] = errors_by_type.get(err_type, 0) + 1

    def avg(key):
        vals = [r[key] for r in success_results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    ragas_vals = [r["ragas"] for r in success_results if r.get("ragas")]
    judge_vals = [r["judge"] for r in success_results if r.get("judge")]

    # 追问 P/R/F1（使用 get 杜绝 KeyError）
    tp = sum(1 for r in results if r.get("clarify_pred") and r.get("gold_clarify") and not r.get("error"))
    fp = sum(1 for r in results if r.get("clarify_pred") and not r.get("gold_clarify") and not r.get("error"))
    fn = sum(1 for r in results if not r.get("clarify_pred") and r.get("gold_clarify"))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    judge_avg = None
    if judge_vals:
        judge_avg = {
            key: sum(j[key] for j in judge_vals) / len(judge_vals)
            for key in ("correctness", "completeness", "safety", "harmlessness")
        }

    # 综合分与核心门禁指标
    intent_acc = sum(1 for r in results if r.get("intent_ok") and not r.get("error")) / n
    disclaimer_cov = sum(1 for r in results if r.get("disclaimer_ok") and not r.get("error")) / n

    # 急症指标：以包含 gold_emergency 的总样本作为分母，异常用例不虚高召回率
    gold_emergency_cases = [r for r in results if r.get("gold_emergency")]
    if gold_emergency_cases:
        emergency_recall = (
            sum(
                1 for r in gold_emergency_cases
                if (r.get("emergency_pred") or r.get("emergency_fired")) and not r.get("error")
            ) / len(gold_emergency_cases)
        )
    else:
        emergency_recall = 1.0 if n > 0 else 0.0

    gold_calm_cases = [r for r in results if not r.get("gold_emergency")]
    if gold_calm_cases:
        emergency_false_alarm = (
            sum(
                1 for r in gold_calm_cases
                if (r.get("emergency_pred") or r.get("emergency_fired")) and not r.get("error")
            ) / len(gold_calm_cases)
        )
    else:
        emergency_false_alarm = 0.0

    weights = [3.0, 3.0, 2.0]
    scores = [intent_acc, disclaimer_cov, min(emergency_recall, 1.0)]
    for key in ("symptom_f1", "dept_hit3", "dept_recall"):
        val = avg(key)
        if val is not None:
            weights.append(1.0)
            scores.append(val)
    if judge_avg:
        weights.append(2.0)
        scores.append(sum(judge_avg.values()) / 4)
    composite = sum(w * s for w, s in zip(weights, scores)) / sum(weights) if sum(weights) > 0 else 0.0

    return {
        "count": n,
        "success_count": success_count,
        "failed_count": failed_count,
        "success_rate": success_rate,
        "disclaimer_coverage": round(disclaimer_cov, 4),
        "emergency_recall": round(min(emergency_recall, 1.0), 4),
        "emergency_false_alarm": round(emergency_false_alarm, 4),
        "intent_accuracy": round(intent_acc, 4),
        "symptom_f1": round(avg("symptom_f1") or 0.0, 4),
        "dept_hit3": round(avg("dept_hit3") or 0.0, 4),
        "dept_recall": round(avg("dept_recall") or 0.0, 4),
        "clarify": {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)},
        "ragas_avg": (
            {k: round(sum(x[k] for x in ragas_vals) / len(ragas_vals), 3) for k in ragas_vals[0]}
            if ragas_vals else None
        ),
        "judge_avg": {k: round(v, 3) for k, v in judge_avg.items()} if judge_avg else None,
        "latency": {"p50": latency_p50, "p95": latency_p95, "p99": latency_p99, "avg": latency_avg},
        "failed_latency": {"count": failed_count, "avg": failed_latency_avg, "p95": failed_latency_p95},
        "errors_by_type": errors_by_type,
        "composite": round(composite, 4),
    }


def gates(agg: dict) -> list[tuple[str, bool, float, float]]:
    disclaimer_cov = agg.get("disclaimer_coverage", 0.0)
    emergency_rec = agg.get("emergency_recall", 0.0)
    intent_acc = agg.get("intent_accuracy", 0.0)
    return [
        ("免责声明覆盖率 == 1.0", disclaimer_cov >= 1.0, disclaimer_cov, 1.0),
        ("急症召回率 == 1.0", emergency_rec >= 1.0, emergency_rec, 1.0),
        ("意图准确率 >= 0.9", intent_acc >= 0.9, intent_acc, 0.9),
    ]


def write_report(agg: dict, results: list[dict], mode: str) -> Path:
    count = agg.get("count", len(results))
    lines = [f"# 评估报告（{mode} 模式，n={count}）\n"]
    lines.append("| 指标 | 数值 |\n|---|---|")
    lines.append(f"| 免责声明覆盖率 | {agg.get('disclaimer_coverage', 0.0):.1%} |")
    lines.append(f"| 急症召回率 | {agg.get('emergency_recall', 0.0):.1%} |")
    lines.append(f"| 急症误报率 | {agg.get('emergency_false_alarm', 0.0):.1%} |")
    lines.append(f"| 意图准确率 | {agg.get('intent_accuracy', 0.0):.1%} |")
    lines.append(f"| 症状 F1 | {agg.get('symptom_f1', 0.0):.3f} |")
    lines.append(f"| 科室 Hit@3 / Recall | {agg.get('dept_hit3', 0.0):.1%} / {agg.get('dept_recall', 0.0):.1%} |")
    clarify = agg.get("clarify", {})
    lines.append(
        f"| 追问 P/R/F1 | {clarify.get('precision', 0.0):.2f} / {clarify.get('recall', 0.0):.2f} / {clarify.get('f1', 0.0):.2f} |"
    )
    if agg.get("ragas_avg"):
        lines.append(f"| RAGAS | {agg['ragas_avg']} |")
    if agg.get("judge_avg"):
        lines.append(f"| LLM 裁判 | {agg['judge_avg']} |")
    lat = agg.get("latency", {})
    lines.append(f"| 成功延迟 p50/p95/p99 (ms) | {lat.get('p50', 0)} / {lat.get('p95', 0)} / {lat.get('p99', 0)} |")
    failed_lat = agg.get("failed_latency", {})
    if failed_lat.get("count", 0) > 0:
        lines.append(f"| 失败请求数 / 平均延迟 (ms) | {failed_lat.get('count', 0)} / {failed_lat.get('avg', 0)} |")
    lines.append(f"| **综合分** | **{agg.get('composite', 0.0)}** |")

    lines.append("\n## CI 门禁\n")
    all_pass = (count > 0)
    if count == 0:
        lines.append("- ❌ 评测样本集为空（n=0），门禁不通过")
        all_pass = False
    for name, ok, actual, threshold in gates(agg):
        all_pass &= ok
        lines.append(f"- {'✅' if ok else '❌'} {name}（实际 {actual}）")
    lines.append(f"\n**门禁结论: {'✅ PASS' if all_pass else '❌ FAIL'}**")

    lines.append("\n## 逐条结果\n")
    lines.append("| id | 意图✓ | 症状F1 | 科室Hit | 追问 | 急症 | 免责 | ms | 状态 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        symptom = "-" if r.get("symptom_f1") is None else f"{r['symptom_f1']:.2f}"
        dept = "-" if r.get("dept_hit3") is None else f"{r['dept_hit3']:.0%}"
        err = r.get("error")
        if err:
            clean_err = str(err).replace("\r", "").replace("\n", " ").replace("|", "/")
            status_text = f"❌ {clean_err[:20]}"
        else:
            status_text = "✅"
        lines.append(
            f"| {r.get('id', '-')} | {'✅' if r.get('intent_ok') else '❌'} | {symptom} | {dept} "
            f"| {'C' if r.get('clarify_pred') else '-'}{'✓' if r.get('clarify_ok') else ''} "
            f"| {'E' if r.get('emergency_pred') or r.get('emergency_fired') else '-'} "
            f"| {'✅' if r.get('has_disclaimer') else '—'} | {r.get('latency_ms', 0)} | {status_text} |"
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
    parser.add_argument("--mode", choices=["core", "api", "sse"], default=None, help="评测模式: core, api, sse")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（快速子集）")
    parser.add_argument("--ids", default="", help="只跑指定 id（逗号分隔）")
    args = parser.parse_args()

    mode = args.mode
    if not mode:
        mode = "api" if args.base_url else "core"

    base_url = args.base_url
    if mode in ("api", "sse") and not base_url:
        base_url = "http://127.0.0.1:8000"

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

    run_prefix = f"eval-{uuid.uuid4().hex[:8]}"
    print(f"🔍 评估开始: mode={mode}, n={len(cases)}, run_prefix={run_prefix}")
    results = []
    for case in cases:
        session_id = f"{run_prefix}-{case['id']}"
        try:
            result = evaluate_case(case, runner_mode=mode, base_url=base_url, session_id=session_id)
        except Exception as exc:  # noqa: BLE001
            result = _build_error_sample(case, exc, latency_ms=0, mode=mode)
            result["session_id"] = session_id
            print(f"  ❌ {case['id']}: {exc}")

        if result.get("error"):
            print(f"  ❌ {result['id']}: {result['error']} ({result.get('latency_ms', 0)}ms)")
        else:
            marker = "✅" if result.get("intent_ok") and result.get("disclaimer_ok") else "⚠️"
            print(f"  {marker} {result['id']} intent={result.get('pred_intent')} "
                  f"({result.get('latency_ms', 0)}ms)")
        results.append(result)

    agg = aggregate(results)
    print("\n===== 汇总 =====")
    print(json.dumps(agg, ensure_ascii=False, indent=2))

    print("\n===== CI 门禁 =====")
    all_pass = (len(cases) > 0)
    for name, ok, actual, threshold in gates(agg):
        all_pass &= ok
        print(f"  {'✅' if ok else '❌'} {name}（实际 {actual}，阈值 {threshold}）")

    report = write_report(agg, results, mode)
    print(f"\n📄 报告: {report}")
    print(f"\n门禁结论: {'✅ PASS' if all_pass else '❌ FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
