"""Trace 采集统一入口：LangSmith → Langfuse → no-op（按环境变量优先级自动选后端）。

离线评估（evals/runner.py）与线上 API 共用同一客户端；
trace 含 intent / tool_calls / latency / 输入输出。
"""
import os
import time
from contextlib import contextmanager


@contextmanager
def trace_span(name: str, metadata: dict | None = None):
    """统一 trace span；后端按 LANGSMITH_TRACING → LANGFUSE_* → no-op 选择。"""
    start = time.perf_counter()
    span = {"name": name, "metadata": metadata or {}, "start": start}

    backend = "noop"
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        backend = "langsmith"
    elif os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        backend = "langfuse"
    span["backend"] = backend

    try:
        yield span
    except Exception as exc:  # noqa: BLE001
        span["error"] = str(exc)
        raise
    finally:
        span["latency_ms"] = int((time.perf_counter() - start) * 1000)
        if backend == "langsmith":
            _emit_langsmith(span)
        elif backend == "langfuse":
            _emit_langfuse(span)


def _emit_langsmith(span: dict) -> None:
    try:
        from langsmith import Client  # 延迟 import：未安装/未配置时零开销

        client = Client()
        client.create_run(
            name=span["name"],
            run_type="chain",
            inputs=span["metadata"],
            outputs={"latency_ms": span.get("latency_ms")},
            error=span.get("error"),
        )
    except Exception:  # noqa: BLE001,S110
        pass


def _emit_langfuse(span: dict) -> None:
    try:
        from langfuse import Langfuse  # 延迟 import

        langfuse = Langfuse()
        langfuse.trace(
            name=span["name"],
            metadata={**span["metadata"], "latency_ms": span.get("latency_ms")},
        )
    except Exception:  # noqa: BLE001,S110
        pass
