"""环境验证脚本：Python 版本 / 依赖包 / .env / Docker 中间件 / 模型 / API / 配置模块。

用法：python validate_env.py
"""
import importlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("🔍 开始环境验证...\n")
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "", hint: str = "") -> None:
    results.append((name, ok))
    print(f"{'✅' if ok else '❌'} {name}" + (f": {detail}" if detail else ""))
    if not ok and hint:
        print(f"   💡 {hint}")


# 1. Python 版本
v = sys.version_info
check("Python 版本 3.11", v.major == 3 and v.minor == 11, f"{v.major}.{v.minor}.{v.micro}",
      "conda create -n smc_project python=3.11 -y")

# 2. 必需包
required = [
    "langchain", "langchain_openai", "langchain_core", "langgraph", "neo4j",
    "faiss", "sentence_transformers", "requests", "bs4", "pandas", "openpyxl",
    "fastapi", "uvicorn", "pydantic", "pydantic_settings", "dotenv",
    "redis", "mysql.connector", "tqdm",
]
missing = []
for pkg in required:
    try:
        importlib.import_module(pkg)
    except Exception:  # noqa: BLE001
        missing.append(pkg)
check("必需包已安装", not missing, ", ".join(missing) if missing else f"{len(required)} 个全部就绪",
      "pip install -r requirements.txt")

# 3. .env 配置
env_file = PROJECT_ROOT / ".env"
env_ok = env_file.exists()
check(".env 配置文件存在", env_ok, str(env_file), "cp .env.example .env 并填入 MODEL_API_KEY")
if env_ok:
    content = env_file.read_text(encoding="utf-8")
    for key in ("MODEL_BASE_URL", "MODEL_API_KEY", "MODEL_NAME", "NEO4J_URI", "EMBEDDING_MODEL_PATH"):
        present = f"{key}=" in content
        check(f".env 含 {key}", present)
    key_filled = "your_api_key_here" not in content
    check("MODEL_API_KEY 已填真实值", key_filled, hint="编辑 .env 替换 your_api_key_here")

# 4. 配置模块可加载
try:
    from config.paths import PROJECT_ROOT as PR  # noqa: F401
    from config.settings import settings  # noqa: F401

    check("基础配置模块（config.settings / config.paths）", True)
except Exception as exc:  # noqa: BLE001
    check("基础配置模块", False, str(exc), "pip install pydantic-settings")

# 5. 公共工具模块
try:
    from src.common.llm import get_llm  # noqa: F401
    from src.common.logger import setup_logger  # noqa: F401
    from src.common.utils import clean_text  # noqa: F401

    check("公共工具模块（logger / llm / utils）", True)
except Exception as exc:  # noqa: BLE001
    check("公共工具模块", False, str(exc))

# 6. Docker 中间件（Neo4j / Redis / MySQL）
try:
    ps = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}:{{.Status}}"],
        capture_output=True, text=True, timeout=15,
    )
    running = ps.stdout if ps.returncode == 0 else ""
    for name in ("neo4j", "redis", "mysql"):
        up = any(line.startswith(name) and ("Up" in line or "healthy" in line)
                 for line in running.splitlines())
        check(f"{name} 容器运行中", up, hint=f"docker compose up -d（或 docker start {name}）")
except FileNotFoundError:
    for name in ("neo4j", "redis", "mysql"):
        check(f"{name} 容器运行中", False, detail="docker 不可用", hint="安装 Docker Desktop")
except subprocess.TimeoutExpired:
    print("⚠️ docker ps 超时")

# 7. BGE-M3 模型文件
from config.settings import settings  # noqa: E402

model_dir = Path(settings.embedding_model_path)
if not model_dir.is_absolute():
    model_dir = PROJECT_ROOT / model_dir
model_files_exist = (model_dir / "config.json").exists()
check("BGE-M3 模型文件存在", model_files_exist, str(model_dir),
      "huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3 "
      "（或从已有设备拷贝）")

# 8. DashScope API 可访问
try:
    from dotenv import dotenv_values

    env = dotenv_values(env_file)
    import requests as requests_lib

    api_key = env.get("MODEL_API_KEY", "")
    if api_key and api_key != "your_api_key_here":
        resp = requests_lib.get(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}, timeout=10,
        )
        check("DashScope API 可访问", resp.status_code == 200, f"HTTP {resp.status_code}",
              "检查 MODEL_API_KEY 是否有效")
    else:
        check("DashScope API 可访问", False, detail="API Key 未配置",
              hint="编辑 .env 填入 MODEL_API_KEY")
except Exception as exc:  # noqa: BLE001
    check("DashScope API 可访问", False, str(exc))

# 9. 向量索引
index_file = PROJECT_ROOT / "data" / "indexes" / "faiss.index"
check("FAISS 索引已构建", index_file.exists(), str(index_file),
      "python scripts/build_index.py")

# 10. 图谱数据
kg_file = PROJECT_ROOT / "data" / "knowledge_graph" / "entities.json"
check("知识图谱实体数据存在", kg_file.exists(), str(kg_file),
      "python scripts/run_extraction.py --no-llm（或先爬虫再抽取）")

passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"\n📊 验证结果: {passed}/{total} 通过")
if passed == total:
    print("✅ 环境就绪，可以开始开发！")
else:
    print("💡 请按上方提示修复失败项后重新验证")
sys.exit(0 if passed == total else 1)
