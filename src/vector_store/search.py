"""语义检索：BGE-M3（1024维）+ FAISS IndexIVFFlat + 实体链接。

关键设计（崩溃复盘沉淀）：
- 重型原生依赖（torch / pyarrow DLL）的首次 import 一律发生在**主线程启动期**——
  warmup() / preload_embedding_stack() 由 start_api.py 在 uvicorn.run 之前调用；
  请求路径只用 get_shared_retriever() 拿进程级单例，绝不在请求线程里做「第一次」。
- sentence_transformers 延迟 import：模块可在未装 torch 的环境安全 import（离线测试用）。

R4 索引契约：BGE-M3 = 1024 维 / IndexIVFFlat / METRIC_INNER_PRODUCT / 向量已归一化。
"""
import threading

import faiss
import numpy as np

from config.paths import INDEX_ENTITY_MAPPING_FILE, INDEX_FILE
from config.settings import settings
from src.common.logger import setup_logger
from src.common.utils import read_json
from src.knowledge_graph.graph_queries import GraphQueries

logger = setup_logger("vector.search", "vector.log")


def _resolve_model_path() -> str:
    path = settings.embedding_model_path
    from pathlib import Path

    p = Path(path)
    if not p.is_absolute():
        from config.paths import PROJECT_ROOT

        p = PROJECT_ROOT / p
    return str(p)


class VectorRetriever:
    """语义检索器：加载 BGE-M3 与 FAISS 索引，提供 search / link_entities / hybrid_search。"""

    def __init__(self):
        self._model = None
        self._index = None
        self._entities: list[dict] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """加载 FAISS 索引与实体映射（快）；嵌入模型延迟到首次 encode 加载（慢）。"""
        if not INDEX_FILE.exists():
            logger.warning("FAISS 索引不存在（%s），请先运行 scripts/build_index.py", INDEX_FILE)
            return
        self._index = faiss.read_index(str(INDEX_FILE))
        self._index.nprobe = settings.faiss_nprobe
        if INDEX_ENTITY_MAPPING_FILE.exists():
            data = read_json(INDEX_ENTITY_MAPPING_FILE)
            self._entities = data.get("entities", [])
        logger.info(
            "FAISS index loaded: %d vectors (nlist=%s, nprobe=%s)",
            self._index.ntotal, getattr(self._index, "nlist", "-"), self._index.nprobe,
        )
        logger.info("Entity mapping loaded: %d entities", len(self._entities))

    @property
    def model(self):
        """BGE-M3 模型（首次访问时加载；正常情况已被启动期 warmup 预热）。"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("[MODEL] Loading BGE-M3 from %s ...", settings.embedding_model_path)
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(_resolve_model_path())
                    logger.info("[MODEL] BGE-M3 loaded")
        return self._model

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def index_size(self) -> int:
        """索引中的向量数（API 层用公开属性，不摸 _index 私有成员）。"""
        return self._index.ntotal if self._index is not None else 0

    # ---------- 核心能力 ----------
    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """语义搜索：返回 [{name, type, description, score}]，低于阈值过滤。"""
        if not self.is_ready:
            return []
        k = top_k or settings.retrieval_top_k
        k = min(k, self._index.ntotal)
        try:
            query_vec = self._encode([query])
            scores, indices = self._index.search(query_vec, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._entities):
                    continue
                if score < settings.retrieval_score_threshold:
                    continue
                entity = dict(self._entities[int(idx)])
                entity["score"] = round(float(score), 4)
                results.append(entity)
            return results
        except Exception as exc:  # noqa: BLE001
            logger.error("语义搜索失败: %s", exc)
            return []

    def link_entities(self, terms: list[str], threshold: float | None = None) -> list[dict]:
        """实体链接：口语词 → 图谱规范实体（高精度，阈值默认 0.85）。

        返回 [{input_entity, matched_entity, type, similarity}]；
        失败返回空列表，绝不阻断主问答流程。
        """
        if not self.is_ready or not terms:
            return []
        th = threshold if threshold is not None else settings.entity_link_threshold
        try:
            term_vecs = self._encode(list(terms))
            sims, idxs = self._index.search(term_vecs, 1)
            linked = []
            for term, sim, idx in zip(terms, sims[:, 0], idxs[:, 0]):
                if idx < 0 or idx >= len(self._entities):
                    continue
                if sim < th:
                    continue
                entity = self._entities[int(idx)]
                linked.append({
                    "input_entity": term,
                    "matched_entity": entity["name"],
                    "type": entity.get("type", ""),
                    "similarity": round(float(sim), 4),
                })
            return linked
        except Exception as exc:  # noqa: BLE001
            logger.warning("实体链接失败（不阻断主流程）: %s", exc)
            return []

    def hybrid_search(self, query: str, top_k: int | None = None) -> dict:
        """混合检索：向量语义召回 + Neo4j 图谱精确扩展。

        返回 {entities: [...], contexts: [...]}；
        contexts 为可直接喂给 LLM 的文本片段（RAGAS 的 retrieved_contexts）。
        """
        entities = self.search(query, top_k)
        contexts: list[str] = []
        queries = GraphQueries()

        seen = set()
        for entity in entities:
            key = (entity.get("type"), entity.get("name"))
            if key in seen:
                continue
            seen.add(key)
            desc = (entity.get("description") or "").strip()
            if desc:
                contexts.append(f"{entity['name']}（{entity['type']}）：{desc}")

            facts = queries.expand_entity(entity.get("name", ""))
            fact_map = facts.get("facts", {})
            if facts.get("type") == "Disease":
                parts = []
                if fact_map.get("symptoms"):
                    parts.append("常见症状: " + "、".join(fact_map["symptoms"][:8]))
                if fact_map.get("departments"):
                    parts.append("就诊科室: " + "、".join(fact_map["departments"][:4]))
                if fact_map.get("medications"):
                    parts.append("常用药物: " + "、".join(fact_map["medications"][:8]))
                if fact_map.get("examinations"):
                    parts.append("相关检查: " + "、".join(fact_map["examinations"][:6]))
                if parts:
                    contexts.append(f"{entity['name']}（疾病）：{ '；'.join(parts) }")
            elif facts.get("type") == "Symptom" and fact_map.get("diseases"):
                contexts.append(
                    f"{entity['name']}（症状）可能提示: " + "、".join(fact_map["diseases"][:8])
                )
            elif facts.get("type") == "Medication":
                parts = []
                if fact_map.get("category"):
                    parts.append(f"类别: {fact_map['category']}")
                if fact_map.get("side_effects"):
                    effects = fact_map["side_effects"]
                    if isinstance(effects, str):
                        effects = [effects]
                    parts.append("副作用: " + "、".join(effects[:6]))
                if fact_map.get("contraindications"):
                    cons = fact_map["contraindications"]
                    if isinstance(cons, str):
                        cons = [cons]
                    parts.append("禁忌: " + "、".join(cons[:6]))
                if fact_map.get("treats"):
                    parts.append("用于治疗: " + "、".join(fact_map["treats"][:6]))
                if parts:
                    contexts.append(f"{entity['name']}（药物）：{ '；'.join(parts) }")

        return {"entities": entities, "contexts": contexts}

    # ---------- 启动期预热（防 Windows arrow.dll 原生崩溃）----------
    def warmup(self) -> None:
        """主线程加载 BGE-M3 并跑通一次 encode + FAISS 检索。"""
        logger.info("[WARMUP] 预热嵌入检索栈（主线程）...")
        vec = self._encode(["预热"])
        if self.is_ready:
            self._index.search(vec, 1)
        logger.info("[WARMUP] 嵌入检索栈就绪（维度=%d）", vec.shape[1])


# ----- 进程级单例 -----
_shared_retriever: VectorRetriever | None = None
_singleton_lock = threading.Lock()


def get_shared_retriever() -> VectorRetriever:
    """进程级单例（索引与模型全进程只加载一次）。"""
    global _shared_retriever
    if _shared_retriever is None:
        with _singleton_lock:
            if _shared_retriever is None:
                _shared_retriever = VectorRetriever()
    return _shared_retriever


def preload_retrieval_stack() -> VectorRetriever | None:
    """启动预热入口：必须在 uvicorn.run 之前、主线程调用。"""
    try:
        retriever = get_shared_retriever()
        if retriever.is_ready:
            retriever.warmup()
        return retriever
    except Exception as exc:  # noqa: BLE001
        logger.error("向量检索栈预热失败: %s", exc)
        return None
