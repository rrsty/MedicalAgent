"""一键启动编排：中间件 → 数据 → 索引 → 前端 → 服务 → 浏览器。

由根目录「一键启动.bat」调用（也可直接 python scripts/quick_start.py）。
幂等设计：已就绪的环节自动跳过，只补缺失的部分。

安全约定：
- Python 步骤（种子/建图/建索引）全部进程内直接调用，不起子进程；
- 外部命令仅 docker / npm 两类，调用点内联字面量参数列表 + shell=False；
- 启动 Docker Desktop 使用 os.startfile（ShellExecute，无 shell 拼接）。

用法：
    python scripts/quick_start.py               # 全流程 + 自动打开浏览器
    python scripts/quick_start.py --port 8000   # 指定端口
    python scripts/quick_start.py --no-browser  # 不开浏览器
"""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DOCKER_DESKTOP_CANDIDATES = [
    Path(r"C:\Program Files\Docker\Docker\Docker Desktop.exe"),
    Path(r"C:\Program Files\Docker\Docker\frontend\Docker Desktop.exe"),
    Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe")),
    Path(r"C:\Users\rsty\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe"),
]

SERVICES = ("neo4j", "redis", "mysql")


def step(title: str) -> None:
    print(f"\n{'=' * 60}\n🚀 {title}\n{'=' * 60}", flush=True)


# ---------- 1. Docker 中间件 ----------

def ensure_docker(timeout: int = 180) -> None:
    step("步骤 1/4：Docker 中间件（Neo4j / Redis / MySQL）")
    if _docker_ok():
        print("  ✅ Docker 守护进程已就绪")
    else:
        print("  ⏳ Docker 未运行，正在启动 Docker Desktop ...")
        exe = next((p for p in DOCKER_DESKTOP_CANDIDATES if p.exists()), None)
        if exe is None:
            raise SystemExit("❌ 未找到 Docker Desktop，请手动启动 Docker 后重试")
        os.startfile(str(exe))  # noqa: S606（ShellExecute 打开已验证的固定安装路径）
        _poll(_docker_ok, timeout, "Docker 守护进程")

    subprocess.run(["docker", "compose", "up", "-d"], cwd=PROJECT_ROOT,
                   shell=False, check=True)
    _poll(_services_healthy, 150, "中间件（neo4j/redis/mysql）")


def _docker_ok() -> bool:
    try:
        return subprocess.run(
            ["docker", "ps"], capture_output=True, timeout=10, shell=False
        ).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _services_healthy() -> bool:
    try:
        out = subprocess.run(
            ["docker", "compose", "ps", "--format", "{{.Name}} {{.Status}}"],
            capture_output=True, text=True, timeout=15, cwd=PROJECT_ROOT, shell=False,
        ).stdout
        return all(
            any(line.startswith(name) and ("Up" in line or "healthy" in line)
                for line in out.splitlines())
            for name in SERVICES
        )
    except Exception:  # noqa: BLE001
        return False


def _poll(fn, timeout: int, what: str) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if fn():
            print(f"  ✅ {what} 就绪")
            return
        time.sleep(5)
    raise SystemExit(f"❌ 等待 {what} 超时（{timeout}s），请检查 docker compose ps")


# ---------- 2. 数据与索引（进程内直接调用，幂等）----------

def ensure_data() -> None:
    step("步骤 2/4：数据链路（种子/图谱/向量索引，缺失才构建）")
    from config.paths import INDEX_FILE, KG_ENTITIES_FILE, PROCESSED_DISEASES_JSON

    if INDEX_FILE.exists():
        print("  ✅ FAISS 索引已存在，跳过数据构建")
        return

    # 实体数据：种子（仅当处理数据与实体都缺失时，避免覆盖真实管线产物）
    if not KG_ENTITIES_FILE.exists():
        if not PROCESSED_DISEASES_JSON.exists():
            print("  📦 生成种子演示数据 ...")
            from scripts.seed_demo_data import build_seed

            build_seed()
        else:
            raise SystemExit(
                "❌ 存在清洗数据但缺 entities.json——请先运行抽取:\n"
                "     .venv/Scripts/python scripts/run_extraction.py"
            )
    else:
        print("  ✅ 实体数据已存在")

    # Neo4j 建图（仅在图为空时执行；MERGE 幂等）
    from src.knowledge_graph.graph_queries import GraphQueries
    from src.knowledge_graph.schema_manager import close_driver, verify_connectivity

    if not verify_connectivity():
        raise SystemExit("❌ Neo4j 无法连接，请检查 docker compose ps")

    stats = GraphQueries().get_stats()
    close_driver()
    if stats.get("total_nodes", 0) == 0:
        print("  🧱 Neo4j 为空，构建知识图谱 ...")
        from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
        from src.common.utils import read_json
        from config.paths import KG_ENTITIES_FILE as _E, KG_RELATIONS_FILE as _R

        builder = KnowledgeGraphBuilder()
        builder.build_all(
            read_json(_E).get("entities", []),
            read_json(_R).get("relations", []),
        )
        builder_close = close_driver
        builder_close()
        print("  ✅ 知识图谱构建完成")
    else:
        print(f"  ✅ 知识图谱已存在（{stats['total_nodes']} 节点），跳过建图")

    # FAISS 索引（依赖 models/bge-m3；模型缺失时自动从 HuggingFace 拉取）
    from config.settings import settings

    model_dir = Path(settings.embedding_model_path)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    if not (model_dir / "config.json").exists():
        _download_bge_m3(model_dir)
    if not (model_dir / "config.json").exists():
        print(f"  ⚠️ BGE-M3 模型仍缺失（{model_dir}），跳过索引构建；"
              f"服务可启动但向量检索不可用。可手动执行: "
              f"huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3")
        return
    print("  🔨 构建 FAISS 向量索引 ...")
    from scripts.build_index import main as build_index_main

    build_index_main()
    print("  ✅ 向量索引构建完成")


def _download_bge_m3(model_dir: Path) -> None:
    """自动下载 BGE-M3（约 2.2GB）：先走官方 HF，失败自动切国内镜像。"""
    import huggingface_hub

    patterns = [
        "*.json", "*.txt", "pytorch_model.bin", "1_Pooling/*",
        "sentencepiece.bpe.model", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "modules.json", "configuration.json",
        "colbert_linear.pt", "sparse_linear.pt",
    ]
    for label, endpoint in (("HuggingFace 官方", None), ("国内镜像 hf-mirror.com", "https://hf-mirror.com")):
        try:
            print(f"  📥 自动下载 BGE-M3（{label}，约 2.2GB，视网速 5-20 分钟）...", flush=True)
            kwargs = dict(repo_id="BAAI/bge-m3", local_dir=str(model_dir),
                          allow_patterns=patterns, max_workers=4)
            if endpoint:
                kwargs["endpoint"] = endpoint
            huggingface_hub.snapshot_download(**kwargs)
            if (model_dir / "config.json").exists():
                print("  ✅ 模型下载完成")
                return
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ {label} 下载失败: {str(exc)[:120]}")


# ---------- 3. 前端 ----------

def ensure_frontend() -> None:
    step("步骤 3/4：前端（dist 缺失才构建）")
    if (PROJECT_ROOT / "frontend" / "dist" / "index.html").exists():
        print("  ✅ 前端已构建，跳过")
        return
    if not (PROJECT_ROOT / "frontend" / "package.json").exists():
        print("  ⚠️ frontend/ 不存在，跳过（仅 API 模式）")
        return
    fe = PROJECT_ROOT / "frontend"
    npm_cmd = shutil.which("npm") or "npm"
    if not (fe / "node_modules").exists():
        print("  📦 安装前端依赖 (npm install) ...")
        subprocess.run([npm_cmd, "install", "--no-audit", "--no-fund"],
                       cwd=fe, shell=sys.platform == "win32", check=True)
    print("  🔨 构建前端 (npm run build) ...")
    subprocess.run([npm_cmd, "run", "build"], cwd=fe, shell=sys.platform == "win32", check=True)


# ---------- 4. 启动服务 ----------

def start_server(port: int, open_browser: bool) -> None:
    step(f"步骤 4/4：启动服务 http://localhost:{port}")

    # 占位 Key 醒目警告：此状态下所有 LLM 调用失败，意图一律降级为 general（通用引导回答）
    from config.settings import settings

    if not settings.model_api_key or settings.model_api_key == "your_api_key_here":
        print("\n" + "!" * 60)
        print("!! ⚠️  MODEL_API_KEY 未配置（仍是占位符）！")
        print("!! 所有大模型调用将失败，任何提问都只会得到通用引导回答。")
        print("!! 请编辑项目根目录 .env，填入真实阿里云 DashScope API Key：")
        print("!!   MODEL_API_KEY=sk-xxxxxxxx（百炼控制台 https://bailian.console.aliyun.com 获取）")
        print("!! 填好后重新运行本脚本即可。")
        print("!" * 60 + "\n", flush=True)

    if open_browser:
        threading.Thread(target=_open_browser_when_ready, args=(port,), daemon=True).start()

    # 崩溃防护关键：主线程预加载嵌入检索栈（防原生崩溃），再起 uvicorn
    import faulthandler

    faulthandler.enable()
    from src.vector_store.search import preload_retrieval_stack

    print("  ⏳ 预加载 BGE-M3 嵌入检索栈 ...", flush=True)
    retriever = preload_retrieval_stack()
    print(f"  ✅ 向量检索栈就绪: index={'✅' if retriever and retriever.is_ready else '❌ 缺索引'}")

    import uvicorn

    print(f"\n🎉 就绪！浏览器访问 http://localhost:{port}（Swagger: /docs，Ctrl+C 停止）\n", flush=True)
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port)


def _open_browser_when_ready(port: int) -> None:
    import urllib.request

    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=3) as resp:
                if resp.status == 200:
                    webbrowser.open(f"http://localhost:{port}")
                    return
        except Exception:  # noqa: BLE001
            time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="一键启动智慧问诊Agent系统")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    print("🏥 智慧问诊Agent系统 · 一键启动")
    ensure_docker()
    ensure_data()
    ensure_frontend()
    start_server(args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
