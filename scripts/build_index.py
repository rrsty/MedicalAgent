"""FAISS 索引构建：BGE-M3 编码实体 → IndexIVFFlat（R4 索引契约）。

用法：python scripts/build_index.py
输出：data/indexes/faiss.index + data/indexes/entities.json（索引ID→实体映射）
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np

from config.paths import (
    INDEX_ENTITY_MAPPING_FILE,
    INDEX_FILE,
    KG_ENTITIES_FILE,
    PROJECT_ROOT,
)
from config.settings import settings
from src.common.logger import setup_logger
from src.common.utils import read_json, write_json

logger = setup_logger("scripts.build_index", "vector.log")

# 实体文本 = 名称 + 类型标注（与讲座实测口径一致：口语词→规范实体相似度可达 0.9+，
# 支撑 0.85 实体链接阈值；描述事实由 Neo4j expand_entity 补充，不进向量）
# R4 索引契约：BGE-M3 / 1024 维 / IndexIVFFlat / METRIC_INNER_PRODUCT / 归一化


def entity_text(entity: dict) -> str:
    return f"{entity.get('name', '')}（{entity.get('type', '')}）"


def main() -> None:
    t0 = time.perf_counter()

    # 1. 加载 BGE-M3（模型路径从 .env 读取）
    from sentence_transformers import SentenceTransformer

    model_path = settings.embedding_model_path
    if not Path(model_path).is_absolute():
        model_path = str(PROJECT_ROOT / model_path)
    print(f"[1/4] 加载 BGE-M3: {model_path}")
    model = SentenceTransformer(model_path)

    # 2. 读取实体
    entities = read_json(KG_ENTITIES_FILE).get("entities", [])
    if not entities:
        print("❌ 实体数据为空，请先运行抽取与（可选）建图")
        sys.exit(1)
    print(f"[2/4] 读取实体 {len(entities)} 条，开始编码...")

    texts = [entity_text(e) for e in entities]
    vectors = np.asarray(
        model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True),
        dtype="float32",
    )
    dim = vectors.shape[1]
    if dim != settings.embedding_dim:
        print(f"❌ R4 违规：向量维度 {dim} ≠ 契约维度 {settings.embedding_dim}（索引必须与模型一致）")
        sys.exit(1)

    # 3. 构建 IndexIVFFlat（企业级主流；向量过少时自动退回 Flat 精确索引）
    print(f"[3/4] 构建 FAISS 索引（dim={dim}, nlist={settings.faiss_nlist}）...")
    nlist = min(settings.faiss_nlist, max(1, len(vectors) // 39))
    if len(vectors) >= nlist * 39:
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)  # IVF 需先训练聚类中心
        index.add(vectors)
        index.nprobe = settings.faiss_nprobe
        index_type = f"IndexIVFFlat(nlist={nlist}, nprobe={settings.faiss_nprobe})"
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        index_type = "IndexFlatIP（数据量小，精确搜索）"

    # 4. 保存
    print("[4/4] 保存索引与映射...")
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_bytes(faiss.serialize_index(index).tobytes())
    write_json(
        INDEX_ENTITY_MAPPING_FILE,
        {
            "entities": [
                {
                    "name": e.get("name", ""),
                    "type": e.get("type", ""),
                    "description": e.get("description", ""),
                    "provenance": e.get("provenance", []),
                }
                for e in entities
            ],
            "index_type": index_type,
            "embedding_model": settings.embedding_model_path,
            "dim": dim,
        },
    )

    seconds = time.perf_counter() - t0
    print("\n===== 索引构建完成 =====")
    print(f"向量数量: {index.ntotal} | 类型: {index_type} | 维度: {dim}")
    print(f"耗时: {seconds:.1f}s")
    print(f"索引 → {INDEX_FILE}")
    print(f"映射 → {INDEX_ENTITY_MAPPING_FILE}")


if __name__ == "__main__":
    main()
