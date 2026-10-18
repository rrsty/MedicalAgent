"""FastAPI 应用入口：生命周期初始化 + CORS + 静态托管（单进程部署）。

启动顺序（崩溃复盘教训）：
  start_api.py 主线程 preload_retrieval_stack()（防 Windows 原生崩溃）
  → lifespan 初始化 Agent 图（复用共享检索器）
  → 注册 /api 路由 → 最后挂载静态文件（顺序不能反，否则 /api 被覆盖）
"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.paths import FRONTEND_DIST_DIR
from config.settings import settings
from src.agents.graph import NODE_LABELS_ZH, app as graph_app
from src.api.routes import init_router, router
from src.common.logger import setup_logger
from src.vector_store.search import get_shared_retriever, preload_retrieval_stack

logger = setup_logger("api.main", "api.log")

_state: dict = {"graph_app": None, "node_labels": NODE_LABELS_ZH}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 预加载向量检索栈（幂等：start_api.py 已预热则直接复用共享单例）
    retriever = preload_retrieval_stack() or get_shared_retriever()
    logger.info("[LOAD] 向量检索栈就绪: index=%s", retriever.is_ready)

    # 2. Agent 图常驻（模块级单例）
    _state["graph_app"] = graph_app
    logger.info("[LOAD] Agent 系统就绪（9 节点状态图）")

    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    init_router(_state, semaphore)

    yield
    _state["graph_app"] = None


app = FastAPI(
    title="智慧问诊Agent系统 API",
    version="1.0.0",
    description="基于知识图谱 + 向量检索 + LangGraph 多 Agent 的智慧问诊系统",
    lifespan=lifespan,
)

# CORS：开发模式前端(:5173)跨域访问；生产模式同源托管不依赖 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1) 先注册 API 路由
app.include_router(router)


# 2) SPA 兜底（必须晚于 /api 路由；html=True 保证刷新页面返回 index.html）
@app.get("/", include_in_schema=False)
async def index():
    index_html = FRONTEND_DIST_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return {
        "message": "智慧问诊Agent系统 API 运行中（前端未构建）",
        "hint": "运行 cd frontend && npm install && npm run build 后重启即可看到界面",
        "docs": "/docs",
    }


if (FRONTEND_DIST_DIR / "assets").exists():
    # /assets 显式托管（MIME/缓存正确）
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST_DIR / "assets"), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    """history 模式刷新兜底：非 /api 路径全部回 index.html。"""
    candidate = FRONTEND_DIST_DIR / full_path
    if full_path and candidate.is_file() and candidate.resolve().is_relative_to(
        FRONTEND_DIST_DIR.resolve()
    ):
        return FileResponse(candidate)
    index_html = FRONTEND_DIST_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return {"message": "页面不存在", "path": full_path}


if __name__ == "__main__":
    import faulthandler

    faulthandler.enable()
    preload_retrieval_stack()
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
