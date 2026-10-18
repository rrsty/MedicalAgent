"""一键启动整合服务（生产模式：单进程同时服务 API + 前端静态文件）。

用法：
    python scripts/start_api.py             # 缺前端自动构建（若 npm 可用）
    python scripts/start_api.py --build     # 显式先构建前端
    python scripts/start_api.py --reload    # 开发热重载
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DIST_DIR = PROJECT_ROOT / "frontend" / "dist"


def build_frontend() -> bool:
    """构建 Vue 前端（npm run build）。"""
    if not (PROJECT_ROOT / "frontend" / "package.json").exists():
        print("⚠️ frontend/ 不存在，跳过前端构建")
        return False
    print("[BUILD] 构建前端 (npm run build)...")
    try:
        subprocess.run(["npm", "run", "build"], cwd=PROJECT_ROOT / "frontend", check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"⚠️ 前端构建失败（不影响 API）: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="启动智慧问诊Agent系统整合服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--build", action="store_true", help="强制先构建前端")
    parser.add_argument("--reload", action="store_true", help="开发热重载")
    parser.add_argument("--skip-build-check", action="store_true", help="跳过前端构建检查")
    args = parser.parse_args()

    import faulthandler

    faulthandler.enable()  # 原生崩溃时可从 stderr 读到 traceback

    # 防崩溃关键：嵌入检索栈必须在 uvicorn.run 之前、主线程预热
    # （Windows 上请求线程首次 import sentence_transformers 会触发 arrow.dll 访问违例）
    print("[PRELOAD] 正在预加载向量检索栈（BGE-M3 嵌入模型）...")
    from src.vector_store.search import preload_retrieval_stack

    retriever = preload_retrieval_stack()
    status = "OK" if retriever and retriever.is_ready else "索引缺失（请先 scripts/build_index.py）"
    print(f"[PRELOAD] 预加载完成: {status}")

    # 占位 Key 醒目警告：此状态下所有 LLM 调用失败，意图一律降级为 general
    from config.settings import settings

    if not settings.model_api_key or settings.model_api_key == "your_api_key_here":
        print("\n" + "!" * 60)
        print("!! ⚠️  MODEL_API_KEY 未配置（仍是占位符）！")
        print("!! 所有大模型调用将失败，任何提问都只会得到通用引导回答。")
        print("!! 请编辑 .env 填入真实 DashScope Key 后重启本服务。")
        print("!" * 60 + "\n", flush=True)

    if args.build or (not DIST_DIR.exists() and not args.skip_build_check):
        build_frontend()

    import uvicorn

    print(f"\n🚀 启动服务: http://localhost:{args.port}  （Swagger: /docs）")
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
