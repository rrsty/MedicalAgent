"""契约验收器（TDD 第 0 章 R1–R8 配套）：退出码 0 = 全绿，非 0 = 存在违规。

用法：
    python scripts/check_contracts.py          # 全量检查（R1/R3/R4/R5/命名一致性）
    python scripts/check_contracts.py --r4     # 只跑 R4 索引一致性

检查项：
  R1 单一事实来源：NodeType/RelationType/IntentType 枚举值禁止在 schema.py/state.py
     之外以字符串字面量重复定义（路由/查询必须 import 枚举）；
  R3 契约先行：API 请求/响应字段与 TDD 契约表一致（models.py 逐字段核对）；
  R4 索引契约：嵌入维度（settings == BGE-M3 config.json）== 索引映射 dim；
  R5 数据即产物：FAISS 向量数 == 实体映射记录数（不一致 = 数据过期，需 clear+build 重建）。
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

hasattr(sys.stdout, "reconfigure") and sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from src.knowledge_graph.schema import (  # noqa: E402
    NODE_PROPERTIES,
    RELATION_ENDPOINTS,
    NodeType,
    RelationType,
)
from src.agents.state import IntentType  # noqa: E402

failures: list[str] = []
passes: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (passes if ok else failures).append(f"{name}{(': ' + detail) if detail else ''}")
    print(f"{'✅' if ok else '❌'} {name}" + (f" | {detail}" if detail else ""))


# ---------- R1 单一事实来源 ----------
CANONICAL_FILES = {"schema.py", "state.py"}


def check_r1() -> None:
    enum_values = (
        {t.value for t in NodeType}
        | {r.value for r in RelationType}
        | {i.value for i in IntentType}
    )
    # 允许出现在合法定义文件之外的形式：import 引用 + 比较使用；违规形式 = 赋值定义字符串字面量
    pattern = re.compile(
        r"^\s*(?:[A-Z_]+\s*(?::\s*\w+)?\s*=\s*['\"](\w+)['\"])",
    )
    offenders: list[str] = []
    scan_roots = [PROJECT_ROOT / d for d in ("src", "crawler", "scripts", "evals")]
    fstring_enum = re.compile(
        r"f(['\"]).*?\{[^}]*(NodeType|RelationType|IntentType)(\.\w+)*[^}]*\}.*?"
    )
    for root in scan_roots:
        for py_file in root.rglob("*.py"):
            if py_file.name in CANONICAL_FILES or py_file.name == "__init__.py":
                continue
            for lineno, line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), 1):
                match = pattern.match(line)
                if match and match.group(1) in enum_values:
                    offenders.append(f"{py_file.relative_to(PROJECT_ROOT)}:{lineno} {line.strip()}")
                # f-string 直接插值枚举成员会渲染成 NodeType.XXX（真实踩过的坑），一律拦截
                if fstring_enum.search(line):
                    offenders.append(
                        f"{py_file.relative_to(PROJECT_ROOT)}:{lineno} f-string插值枚举（应使用 .value）: {line.strip()}"
                    )
    check("R1 单一事实来源", not offenders, "; ".join(offenders[:5]) if offenders else "枚举仅定义于 schema.py/state.py，扫描 src/crawler/scripts/evals")


# ---------- R3 契约先行（API 字段）----------
EXPECTED_REQUEST = {"query", "session_id"}
EXPECTED_RESPONSE = {
    "answer", "intent", "symptoms", "severity", "departments", "medications",
    "advice", "disclaimers", "warnings", "linked_entities", "trace",
    "duration_ms", "session_id",
}


def check_r3() -> None:
    models_src = (PROJECT_ROOT / "src" / "api" / "models.py").read_text(encoding="utf-8")
    tree = ast.parse(models_src)
    model_fields: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            fields = {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
            model_fields[node.name] = fields

    check("R3 ConsultationRequest 字段", EXPECTED_REQUEST <= model_fields.get("ConsultationRequest", set()),
          str(sorted(model_fields.get("ConsultationRequest", set()))))
    check("R3 ConsultationResponse 字段", EXPECTED_RESPONSE <= model_fields.get("ConsultationResponse", set()),
          str(sorted(model_fields.get("ConsultationResponse", set()))))

    # 前端按契约消费（在 frontend/src 全目录中检查关键字段引用）
    frontend_dir = PROJECT_ROOT / "frontend" / "src"
    if frontend_dir.exists():
        frontend_src = "\n".join(
            p.read_text(encoding="utf-8")
            for p in list(frontend_dir.rglob("*.js")) + list(frontend_dir.rglob("*.vue"))
        )
        missing = [f for f in ("answer", "intent", "departments", "disclaimers",
                               "warnings", "duration_ms") if f not in frontend_src]
        check("R3 前端字段消费", not missing,
              f"缺失: {missing}" if missing else "前端消费契约字段")
    else:
        check("R3 前端字段消费", False, "frontend/src 不存在")


# ---------- R4 索引契约 ----------
def check_r4() -> None:
    check("R4 settings 嵌入维度 = 1024", settings.embedding_dim == 1024, str(settings.embedding_dim))

    model_config = PROJECT_ROOT / settings.embedding_model_path / "config.json"
    if model_config.exists():
        try:
            hidden = json.loads(model_config.read_text(encoding="utf-8")).get("hidden_size")
            check("R4 BGE-M3 hidden_size = 1024", hidden == settings.embedding_dim, f"model={hidden}")
        except Exception as exc:  # noqa: BLE001
            check("R4 BGE-M3 config 读取", False, str(exc))
    else:
        # 模型不入库（2.2GB），CI/新克隆环境跳过；本地模型存在性由 validate_env.py 把关
        print(f"⏭️  R4 BGE-M3 模型不存在（跳过）: {model_config}")

    mapping_file = PROJECT_ROOT / "data" / "indexes" / "entities.json"
    if mapping_file.exists():
        try:
            dim = json.loads(mapping_file.read_text(encoding="utf-8")).get("dim")
            check("R4 索引映射维度一致", dim == settings.embedding_dim, f"index={dim}")
        except Exception as exc:  # noqa: BLE001
            check("R4 索引映射读取", False, str(exc))
    else:
        print("⏭️  R4 索引映射未构建（跳过，运行 scripts/build_index.py 后再验）")


# ---------- R5 数据即产物 ----------
def check_r5() -> None:
    index_file = PROJECT_ROOT / "data" / "indexes" / "faiss.index"
    mapping_file = PROJECT_ROOT / "data" / "indexes" / "entities.json"
    if index_file.exists() and mapping_file.exists():
        import faiss

        import numpy as np

        index = faiss.deserialize_index(np.frombuffer(index_file.read_bytes(), dtype="uint8"))
        mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
        count = len(mapping.get("entities", []))
        check("R5 FAISS 向量数 == 映射记录数", index.ntotal == count,
              f"index={index.ntotal}, mapping={count}（不一致请 clear + build 重建）")
    else:
        print("⏭️  R5 索引未构建（跳过）")

    # 契约枚举与关系端点表自洽
    bad = [r.value for r in RelationType if r.value not in RELATION_ENDPOINTS]
    check("R5 关系端点表完整", not bad, str(bad))
    bad_nodes = [n.value for n in NodeType if n.value not in NODE_PROPERTIES]
    check("R5 节点属性表完整", not bad_nodes, str(bad_nodes))


def main() -> None:
    parser = argparse.ArgumentParser(description="契约验收器（TDD 第 0 章）")
    parser.add_argument("--r4", action="store_true", help="只跑 R4 索引一致性")
    args = parser.parse_args()

    print("🔍 契约验收开始...\n")
    if args.r4:
        check_r4()
    else:
        check_r1()
        print()
        check_r3()
        print()
        check_r4()
        print()
        check_r5()

    print(f"\n📊 验收结果: {len(passes)} 通过 / {len(failures)} 失败")
    if failures:
        print("❌ 存在违规，禁止进入下一步（R7）")
        sys.exit(1)
    print("✅ 全绿")


if __name__ == "__main__":
    main()
