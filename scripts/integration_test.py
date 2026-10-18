"""系统集成测试（API 级端到端验收，标准库 + requests，可进 CI）。

前置：服务已通过 `python scripts/start_api.py` 启动（--base-url 可覆盖，仅允许
http(s) 协议 + 私网/环回地址——本测试的目标是本地/内网服务）。
检查链：契约 → 健康与就绪 → 图谱统计 → 前端托管 → 五类意图问答（免责声明红线）。
任何失败 → 进程非 0 退出。
"""
import argparse
import ipaddress
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

results: list[tuple[str, bool, str]] = []


def validate_base_url(base_url: str) -> str | None:
    """SSRF 防护：仅允许 http(s) 且目标为环回/私网地址（本测试只打本地/内网服务）。"""
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        return f"协议不允许: {parsed.scheme}"
    host = parsed.hostname or ""
    try:
        ip = ipaddress.ip_address(host)
        if not (ip.is_loopback or ip.is_private):
            return f"目标地址非私网/环回: {host}"
    except ValueError:
        if host not in ("localhost",):
            return f"目标主机必须是 localhost 或 IP 私网地址: {host}"
    return None


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"{'✅' if ok else '❌'} {name} | {detail}")
    return ok


def run(base_url: str) -> int:
    print(f"🔍 集成测试目标: {base_url}\n")

    # 0. 契约门禁（R1/R3/R4/R5，先于一切端到端检查）
    contract = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "check_contracts.py")],
        capture_output=True, text=True,
    )
    check("契约门禁 check_contracts.py", contract.returncode == 0,
          contract.stdout.strip().splitlines()[-1] if contract.stdout else "")

    # 1. 健康与就绪
    try:
        health = requests.get(f"{base_url}/api/health", timeout=10).json()
        check("GET /api/health status=healthy", health.get("status") == "healthy", str(health.get("status")))
        check("neo4j_connected", health.get("neo4j_connected") is True)
        check("vector_index_loaded", health.get("vector_index_loaded") is True)
        check("agent_system_ready", health.get("agent_system_ready") is True)
    except Exception as exc:  # noqa: BLE001
        check("服务在线", False, f"{type(exc).__name__}: {exc}")
        print("\n💡 请先启动服务: python scripts/start_api.py")
        return 1

    # 2. 图谱统计
    stats = requests.get(f"{base_url}/api/stats", timeout=10).json()
    check("图谱节点 > 0", stats.get("total_nodes", 0) > 0, f"nodes={stats.get('total_nodes')}")
    check("图谱关系 > 0", stats.get("total_relations", 0) > 0, f"rels={stats.get('total_relations')}")

    # 3. 单进程前端托管
    try:
        index = requests.get(base_url, timeout=10)
        check("GET / 前端页面", index.status_code == 200 and 'id="app"' in index.text,
              f"status={index.status_code}")
    except Exception as exc:  # noqa: BLE001
        check("GET / 前端页面", False, str(exc))
    docs = requests.get(f"{base_url}/docs", timeout=10)
    check("GET /docs Swagger 未被覆盖", docs.status_code == 200, f"status={docs.status_code}")

    # 4. 五类意图问答验收
    # 免责断言按架构语义区分：general 直达 END 不经安全检查（PRD US7），
    # 其余医疗意图必须带免责声明（红线）。general 在此豁免，急症由 warnings 断言。
    cases = [
        ("我最近头痛、头晕，应该挂什么科？", "appointment", "departments 非空", True),
        ("高血压吃什么药好？", "medication", "answer 非空", True),
        ("什么是糖尿病？", "knowledge", "answer 含 Markdown", True),
        ("我突然胸痛、呼吸困难", "emergency", "warnings 非空", True),
        ("你好", "general", "answer 非空", False),
    ]
    for i, (query, expect_intent, extra_desc, need_disclaimer) in enumerate(cases):
        try:
            resp = requests.post(
                f"{base_url}/api/consult",
                json={"query": query, "session_id": f"integration-test-{i}"},
                timeout=200,
            ).json()
            ok = (
                bool(resp.get("answer"))
                and resp.get("intent") == expect_intent
                and "duration_ms" in resp
                and (bool(resp.get("disclaimers")) or not need_disclaimer)
            )
            extra = {
                "departments 非空": bool(resp.get("departments")),
                "answer 非空": bool(resp.get("answer")),
                "answer 含 Markdown": ("**" in resp.get("answer", "") or "\n-" in resp.get("answer", "")),
                "warnings 非空": bool(resp.get("warnings")),
            }[extra_desc]
            check(
                f"[{expect_intent}] {query[:18]}…",
                ok and extra,
                f"intent={resp.get('intent')} {extra_desc}={extra} "
                f"disclaimers={len(resp.get('disclaimers', []))} {resp.get('duration_ms')}ms",
            )
        except Exception as exc:  # noqa: BLE001
            check(f"[{expect_intent}] {query[:18]}…", False, str(exc))

    # 5. SSE 流式冒烟
    try:
        with requests.post(
            f"{base_url}/api/consult/stream",
            json={"query": "头疼应该挂什么科", "session_id": "integration-stream"},
            timeout=200, stream=True,
        ) as resp:
            events = []
            for line in resp.iter_lines(decode_unicode=True):
                if line.startswith("event: "):
                    events.append(line.split(": ", 1)[1])
                if line == "data: [DONE]":
                    break
        check(
            "SSE 流完整（progress/answer/done）",
            resp.status_code == 200 and "answer" in events and "done" in events,
            f"events={events}",
        )
    except Exception as exc:  # noqa: BLE001
        check("SSE 流完整", False, str(exc))

    # 6. 对话历史接口冒烟
    try:
        convos = requests.get(f"{base_url}/api/conversations", timeout=10).json()
        check("GET /api/conversations", isinstance(convos, list), f"{len(convos)} 条")
    except Exception as exc:  # noqa: BLE001
        check("GET /api/conversations", False, str(exc))

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n📊 集成测试通过率: {passed}/{len(results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="系统集成测试")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    error = validate_base_url(args.base_url)
    if error:
        print(f"❌ base-url 校验失败: {error}")
        sys.exit(2)
    sys.exit(run(args.base_url.rstrip("/")))
