"""Evals Runner 深度对抗与边界用例测试。

覆盖 5 大对抗目标：
1. 空数据集：aggregate([])、gates()、write_report() 异常免疫
2. 100% 失败数据集：超时/连接错误下延迟隔离、错误分类、门禁拒绝
3. 金标字段缺失：case 缺 gold、gold 为空字典、缺 is_emergency/must_clarify
4. 延迟物理隔离：0ms 瞬间失败与 300s 超时用例不污染成功请求 p50/p95/avg
5. 混合成败样本：准确率、急症召回率、误报率及追问 P/R 分母不失真
6. SSE 事件流异常：进度事件、报错事件、流意外中断与非法 JSON
"""
import json
from unittest.mock import MagicMock, patch

from evals.runner import (
    UNIFIED_SAMPLE_SCHEMA_FIELDS,
    _build_error_sample,
    _classify_error,
    aggregate,
    evaluate_case,
    gates,
    write_report,
)


def test_target1_empty_dataset_handling(tmp_path):
    """Target 1: 空数据集边界测试。"""
    agg = aggregate([])
    assert agg["count"] == 0
    assert agg["success_count"] == 0
    assert agg["failed_count"] == 0
    assert agg["success_rate"] == 0.0
    assert agg["disclaimer_coverage"] == 0.0
    assert agg["emergency_recall"] == 0.0
    assert agg["intent_accuracy"] == 0.0
    assert agg["latency"] == {"p50": 0, "p95": 0, "p99": 0, "avg": 0}
    assert agg["failed_latency"] == {"count": 0, "avg": 0, "p95": 0}
    assert agg["clarify"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # 门禁判定：全部拒绝
    gate_results = gates(agg)
    assert len(gate_results) == 3
    for name, ok, actual, threshold in gate_results:
        assert ok is False

    # 报告生成测试
    with patch("evals.runner.EVALS_DIR", tmp_path):
        report_path = write_report(agg, [], "core")
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "❌ 评测样本集为空（n=0），门禁不通过" in content
        assert "门禁结论: ❌ FAIL" in content
        assert (tmp_path / "last_results.json").exists()


def test_target2_hundred_percent_error_dataset():
    """Target 2: 100% 失败数据集（全量超时/连接错误）。"""
    cases = [
        ({"id": "e1", "query": "q1", "gold": {"intent": "emergency", "is_emergency": True}}, TimeoutError("timed out"), 30000),
        ({"id": "a1", "query": "q2", "gold": {"intent": "appointment", "must_clarify": True}}, ConnectionError("http 502"), 150),
        ({"id": "m1", "query": "q3", "gold": {"intent": "medication"}}, RuntimeError("openai auth failed"), 50),
        ({"id": "e2", "query": "q4", "gold": {"intent": "emergency", "is_emergency": True}}, ValueError("json parse error"), 0),
        ({"id": "k1", "query": "q5", "gold": {"intent": "knowledge"}}, TimeoutError("request timeout"), 60000),
    ]
    results = [_build_error_sample(c, exc, lat) for c, exc, lat in cases]

    agg = aggregate(results)
    assert agg["count"] == 5
    assert agg["success_count"] == 0
    assert agg["failed_count"] == 5
    assert agg["success_rate"] == 0.0

    # 成功延迟必须为 0，不能包含任何失败请求的耗时
    assert agg["latency"] == {"p50": 0, "p95": 0, "p99": 0, "avg": 0}

    # 失败延迟正常统计
    assert agg["failed_latency"]["count"] == 5
    assert agg["failed_latency"]["avg"] == int((30000 + 150 + 50 + 0 + 60000) / 5)
    assert agg["failed_latency"]["p95"] == 60000

    # 错误归类正确
    assert agg["errors_by_type"] == {
        "timeout": 2,
        "http_error": 1,
        "model_error": 1,
        "parse_error": 1,
    }

    # 核心指标与门禁
    assert agg["intent_accuracy"] == 0.0
    assert agg["disclaimer_coverage"] == 0.0
    # 2 个 gold_emergency 均失败，急症召回率严格为 0.0（不虚高）
    assert agg["emergency_recall"] == 0.0

    gate_results = gates(agg)
    assert all(ok is False for _, ok, _, _ in gate_results)


def test_target3_missing_gold_fields():
    """Target 3: 输入样本缺失 gold / 字段缺失鲁棒性测试。"""
    # 3.1 完全没有 gold 键
    case_no_gold = {"id": "c1", "query": "你好"}
    err_sample = _build_error_sample(case_no_gold, "network down", 20)
    assert UNIFIED_SAMPLE_SCHEMA_FIELDS.issubset(err_sample.keys())
    assert err_sample["gold_emergency"] is False
    assert err_sample["gold_clarify"] is False
    assert err_sample["gold_intent"] == ""

    # 3.2 gold 为空字典
    case_empty_gold = {"id": "c2", "query": "头痛", "gold": {}}
    err_sample2 = _build_error_sample(case_empty_gold, "timeout", 100)
    assert UNIFIED_SAMPLE_SCHEMA_FIELDS.issubset(err_sample2.keys())
    assert err_sample2["gold_emergency"] is False
    assert err_sample2["gold_clarify"] is False

    # 3.3 aggregate 直接接收缺失全部金标字段的原始字典（KeyError 免疫）
    raw_results = [
        {"id": "r1", "query": "q1"},
        {"id": "r2", "query": "q2", "error": "timeout"},
        {"id": "r3", "query": "q3", "intent_ok": True, "disclaimer_ok": True, "latency_ms": 120},
    ]
    agg = aggregate(raw_results)
    assert agg["count"] == 3
    assert agg["success_count"] == 2
    assert agg["failed_count"] == 1
    assert agg["intent_accuracy"] == round(1 / 3, 4)
    assert agg["disclaimer_coverage"] == round(1 / 3, 4)


def test_target4_latency_physical_isolation():
    """Target 4: 极值与 0ms 失败样本物理隔离验证。"""
    results = [
        # 5 个成功样本：100, 200, 300, 400, 500 ms
        {"id": "s1", "error": None, "latency_ms": 100, "intent_ok": True, "disclaimer_ok": True},
        {"id": "s2", "error": None, "latency_ms": 200, "intent_ok": True, "disclaimer_ok": True},
        {"id": "s3", "error": None, "latency_ms": 300, "intent_ok": True, "disclaimer_ok": True},
        {"id": "s4", "error": None, "latency_ms": 400, "intent_ok": True, "disclaimer_ok": True},
        {"id": "s5", "error": None, "latency_ms": 500, "intent_ok": True, "disclaimer_ok": True},
        # 4 个异常失败样本：包含 0ms 瞬间失败、300s 超时、600s 挂死、None
        {"id": "f1", "error": "quick reset", "latency_ms": 0},
        {"id": "f2", "error": "quick reset", "latency_ms": 0},
        {"id": "f3", "error": "timeout 300s", "latency_ms": 300000},
        {"id": "f4", "error": "hung 600s", "latency_ms": 600000},
        {"id": "f5", "error": "no latency reported", "latency_ms": None},
    ]

    agg = aggregate(results)

    # 成功延迟严格反映 100~500 的分布
    assert agg["latency"]["avg"] == 300
    assert agg["latency"]["p50"] == 300
    assert agg["latency"]["p95"] == 500
    assert agg["latency"]["p99"] == 500

    # 失败延迟独立统计
    assert agg["failed_latency"]["count"] == 5
    assert agg["failed_latency"]["avg"] == int((0 + 0 + 300000 + 600000) / 4)
    assert agg["failed_latency"]["p95"] == 600000


def test_target5_mixed_success_failure_denominators():
    """Target 5: 混合样本各指标分母保真测试。"""
    # 构造 10 条样本：4 条急症（1成功召回，1成功漏报，2报错失败）
    # 6 条非急症（1误报急症，3正常安抚，2报错失败）
    results = [
        # 急症组 (total=4)
        {"id": "e1", "error": None, "gold_emergency": True, "emergency_pred": True, "intent_ok": True, "disclaimer_ok": True, "gold_clarify": False, "clarify_pred": False},
        {"id": "e2", "error": None, "gold_emergency": True, "emergency_pred": False, "intent_ok": True, "disclaimer_ok": True, "gold_clarify": False, "clarify_pred": False},
        _build_error_sample({"id": "e3", "gold": {"is_emergency": True}}, TimeoutError("timeout"), 30000),
        _build_error_sample({"id": "e4", "gold": {"is_emergency": True}}, RuntimeError("http 500"), 200),
        # 平和组 (total=6)
        {"id": "c1", "error": None, "gold_emergency": False, "emergency_pred": True, "intent_ok": True, "disclaimer_ok": True, "gold_clarify": True, "clarify_pred": True},
        {"id": "c2", "error": None, "gold_emergency": False, "emergency_pred": False, "intent_ok": False, "disclaimer_ok": True, "gold_clarify": True, "clarify_pred": False},
        {"id": "c3", "error": None, "gold_emergency": False, "emergency_pred": False, "intent_ok": True, "disclaimer_ok": False, "gold_clarify": False, "clarify_pred": True},
        {"id": "c4", "error": None, "gold_emergency": False, "emergency_pred": False, "intent_ok": True, "disclaimer_ok": True, "gold_clarify": False, "clarify_pred": False},
        _build_error_sample({"id": "c5", "gold": {"is_emergency": False, "must_clarify": True}}, TimeoutError("timeout"), 30000),
        _build_error_sample({"id": "c6", "gold": {"is_emergency": False, "must_clarify": False}}, ConnectionError("reset"), 0),
    ]

    agg = aggregate(results)

    assert agg["count"] == 10
    assert agg["success_count"] == 6
    assert agg["failed_count"] == 4
    assert agg["success_rate"] == 0.6

    # 急症召回率：分母包含失败的 2 例急症（总共 4 例），仅 1 例成功召回 -> 1/4 = 0.25
    assert agg["emergency_recall"] == 0.25

    # 急症误报率：分母为 6 例平和用例，1 例误报 -> 1/6 = 0.1667
    assert agg["emergency_false_alarm"] == 0.1667

    # 意图准确率：分母为 10（包括失败用例），成功且正确的 5 例 -> 5/10 = 0.50
    assert agg["intent_accuracy"] == 0.50

    # 免责覆盖率：分母为 10，成功且有免责的 5 例 -> 5/10 = 0.50
    assert agg["disclaimer_coverage"] == 0.50

    # 追问指标：gold_clarify 为 True 的有 c1 (TP), c2 (FN), c5 (FN)；FP 为 c3
    # tp=1, fp=1, fn=2 -> precision = 1/2 = 0.5, recall = 1/3 = 0.333
    assert agg["clarify"]["precision"] == 0.5
    assert agg["clarify"]["recall"] == 0.333


def test_sse_stream_parsing_and_error_handling():
    """SSE 流式解析与错误处理对抗测试。"""
    case = {"id": "sse_case", "query": "测试SSE", "gold": {"intent": "general"}}

    # 1. 服务端返回 error 事件
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            "event: error",
            'data: {"message": "model inference error: GPU out of memory"}',
            "",
            "event: done",
            "data: {}",
        ]
        mock_post.return_value.__enter__.return_value = mock_resp
        res = evaluate_case(case, runner_mode="sse", base_url="http://mock")
        assert res["status"] == "error"
        assert "GPU out of memory" in res["error"]
        assert res["error_type"] == "model_error"

    # 2. 流过早中断（未发送 answer 事件）
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            "event: progress",
            'data: {"step": "symptom_extract"}',
        ]
        mock_post.return_value.__enter__.return_value = mock_resp
        res = evaluate_case(case, runner_mode="sse", base_url="http://mock")
        assert res["status"] == "error"
        assert "without returning an answer" in res["error"]
        assert res["error_type"] == "empty_response"

    # 3. answer 数据包含损坏的 JSON（验证被捕获且有错误信息）
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            "event: answer",
            "data: {invalid_json_format",
        ]
        mock_post.return_value.__enter__.return_value = mock_resp
        res = evaluate_case(case, runner_mode="sse", base_url="http://mock")
        assert res["status"] == "error"
        assert "Expecting property name" in res["error"]
        assert res["error_type"] == "parse_error"

    # 4. 正常 SSE 流：包含 progress、answer、needs_clarification 与 trace 收集
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            "event: progress",
            'data: {"node": "intent", "time": 10}',
            "",
            "event: answer",
            'data: {"answer": "测试回答", "intent": "general", "needs_clarification": false, "disclaimers": ["免责声明"]}',
            "",
            "event: done",
            "data: {}",
        ]
        mock_post.return_value.__enter__.return_value = mock_resp
        res = evaluate_case(case, runner_mode="sse", base_url="http://mock")
        assert res["status"] == "success"
        assert res["intent_pred"] == "general"
        assert res["has_disclaimer"] is True
        assert res["clarify_pred"] is False
        assert len(res["trace"]) == 1
        assert res["trace"][0]["node"] == "intent"


def test_error_classification_heuristics():
    """测试错误类型启发式规则。"""
    assert _classify_error(TimeoutError("Read timed out")) == "timeout"
    assert _classify_error("HTTP 502 Bad Gateway") == "http_error"
    assert _classify_error("openai.error.AuthenticationError: invalid api key") == "model_error"
    assert _classify_error(RuntimeError("GPU out of memory")) == "model_error"
    assert _classify_error("CUDA error: out of memory") == "model_error"
    assert _classify_error("Empty answer from service") == "empty_response"
    assert _classify_error(ValueError("SSE stream closed without returning an answer event")) == "empty_response"
    assert _classify_error("JSONDecodeError: Expecting value") == "parse_error"
    try:
        json.loads("{invalid_json: 123")
    except json.JSONDecodeError as exc:
        assert _classify_error(exc) == "parse_error"
    assert _classify_error("Unknown system fault") == "unknown"


# ---------- 对抗发现的边界漏洞实测 ----------

def test_vulnerability_gold_is_none_reproduction():
    """验证漏洞 1 免疫: case 中显式指定 'gold': None 不会导致 AttributeError。"""
    case_null_gold = {"id": "null_g", "query": "测试", "gold": None}
    sample = _build_error_sample(case_null_gold, "error", 10)
    assert sample["gold_emergency"] is False
    assert sample["gold_clarify"] is False
    assert sample["gold_intent"] == ""
    assert UNIFIED_SAMPLE_SCHEMA_FIELDS.issubset(sample.keys())


def test_vulnerability_write_report_non_string_error(tmp_path):
    """验证漏洞 2 免疫: r['error'] 为非字符串时 write_report 正常格式化且不抛出 TypeError。"""
    results = [{"id": "r_err", "error": 500, "latency_ms": 10}]
    agg = aggregate(results)
    with patch("evals.runner.EVALS_DIR", tmp_path):
        report_path = write_report(agg, results, "core")
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "❌ 500" in content
