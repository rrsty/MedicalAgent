"""路径配置：pathlib 集中管理所有项目路径，避免硬编码（R1 单一事实来源）。

修改一处全局生效；自动创建必要目录；跨平台（Windows/Mac/Linux）。
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ----- 数据目录 -----
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_KG_DIR = DATA_DIR / "knowledge_graph"
DATA_INDEXES_DIR = DATA_DIR / "indexes"
DATA_FINETUNE_DIR = DATA_DIR / "finetune"

# ----- 模型与前端 -----
MODELS_DIR = PROJECT_ROOT / "models"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"

# ----- 日志 -----
LOGS_DIR = PROJECT_ROOT / "logs"

# ----- 常用数据文件（文件名只在此定义一次）-----
RAW_DISEASES_FILE = DATA_RAW_DIR / "diseases.json"
PROCESSED_DISEASES_JSON = DATA_PROCESSED_DIR / "diseases.json"
PROCESSED_DISEASES_XLSX = DATA_PROCESSED_DIR / "diseases.xlsx"
KG_ENTITIES_FILE = DATA_KG_DIR / "entities.json"
KG_RELATIONS_FILE = DATA_KG_DIR / "relations.json"
INDEX_FILE = DATA_INDEXES_DIR / "faiss.index"
INDEX_ENTITY_MAPPING_FILE = DATA_INDEXES_DIR / "entities.json"
FINETUNE_DATASET_FILE = DATA_FINETUNE_DIR / "ft_extract.json"

# ----- 评估 -----
EVALS_DIR = PROJECT_ROOT / "evals"
EVAL_DATASET_FILE = EVALS_DIR / "dataset.jsonl"
EVAL_REPORT_FILE = EVALS_DIR / "last_report.md"


def _ensure_dirs() -> None:
    for d in (
        DATA_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_KG_DIR,
        DATA_INDEXES_DIR, DATA_FINETUNE_DIR, MODELS_DIR, LOGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


_ensure_dirs()
