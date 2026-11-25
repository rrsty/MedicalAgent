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
from src.common.provenance import format_context_with_provenance
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
        # Native Windows FAISS file APIs cannot reliably open Unicode paths.
        self._index = faiss.deserialize_index(np.frombuffer(INDEX_FILE.read_bytes(), dtype="uint8"))
        # IndexFlatIP（小数据集回退）无 nprobe 属性，必须防护以免启动崩溃
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = settings.faiss_nprobe
        if INDEX_ENTITY_MAPPING_FILE.exists():
            data = read_json(INDEX_ENTITY_MAPPING_FILE)
            self._entities = data.get("entities", [])
        if self._index.d != settings.embedding_dim:
            logger.error(
                "R4 维度不一致: faiss.d=%d != settings.embedding_dim=%d",
                self._index.d, settings.embedding_dim,
            )
        if self._index.ntotal != len(self._entities):
            logger.warning(
                "R5 索引/映射数量不一致: index.ntotal=%d != mapping=%d（请重跑 scripts/build_index.py）",
                self._index.ntotal, len(self._entities),
            )
        logger.info(
            "FAISS index loaded: %d vectors (nlist=%s, nprobe=%s)",
            self._index.ntotal,
            getattr(self._index, "nlist", "-"),
            getattr(self._index, "nprobe", "-"),
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
        unique_terms = list(dict.fromkeys(t.strip() for t in terms if str(t or "").strip()))
        if not unique_terms:
            return []
        th = threshold if threshold is not None else settings.entity_link_threshold
        try:
            term_vecs = self._encode(unique_terms)
            sims, idxs = self._index.search(term_vecs, 1)
            by_matched: dict[str, dict] = {}
            for term, sim, idx in zip(unique_terms, sims[:, 0], idxs[:, 0]):
                if idx < 0 or idx >= len(self._entities):
                    continue
                if sim < th:
                    continue
                entity = self._entities[int(idx)]
                matched_name = entity["name"]
                score = round(float(sim), 4)
                prev = by_matched.get(matched_name)
                if prev is None or score > prev["similarity"]:
                    by_matched[matched_name] = {
                        "input_entity": term,
                        "matched_entity": matched_name,
                        "type": entity.get("type", ""),
                        "similarity": score,
                    }
            return list(by_matched.values())
        except Exception as exc:  # noqa: BLE001
            logger.warning("实体链接失败（不阻断主流程）: %s", exc)
            return []

    def hybrid_search(self, query: str, top_k: int | None = None) -> dict:
        """混合检索：向量语义召回 + Neo4j 图谱精确扩展。

        上下文排序策略：先输出各召回实体的紧凑结构化摘要与描述，再补充去重后的多跳三元组，
        防止第 1 个实体的海量零散三元组占满上下文窗口配额（max_items=10）而挤掉其余实体。
        """
        entities = self.search(query, top_k)
        summary_contexts: list[str] = []
        triple_contexts: list[str] = []
        queries = GraphQueries()

        seen = set()
        seen_triples: set[tuple] = set()
        for entity in entities:
            etype = entity.get("type")
            ename = entity.get("name", "")
            key = (etype, ename)
            if key in seen or not ename:
                continue
            seen.add(key)

            # 1) 实体摘要与结构化一跳全览（高信息密度，优先进入 LLM 上下文窗口）
            desc = (entity.get("description") or "").strip()
            if desc:
                summary_contexts.append(format_context_with_provenance(
                    f"{ename}（{etype}）：{desc}", entity.get("provenance")
                ))

            facts = queries.expand_entity(ename, entity_type=etype)
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
                if fact_map.get("precautions"):
                    parts.append("注意事项: " + "、".join(fact_map["precautions"][:6]))
                if fact_map.get("recommended_foods"):
                    parts.append("推荐饮食: " + "、".join(fact_map["recommended_foods"][:6]))
                if fact_map.get("avoid_foods"):
                    parts.append("忌口/禁忌食物: " + "、".join(fact_map["avoid_foods"][:6]))
                if parts:
                    summary_contexts.append(format_context_with_provenance(
                        f"{ename}（疾病）：{'；'.join(parts)}", entity.get("provenance")
                    ))
            elif facts.get("type") == "Symptom" and fact_map.get("diseases"):
                summary_contexts.append(format_context_with_provenance(
                    f"{ename}（症状）可能提示: " + "、".join(fact_map["diseases"][:8]),
                    entity.get("provenance"),
                ))
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
                if fact_map.get("interactions"):
                    parts.append("已知相互作用药物: " + "、".join(fact_map["interactions"][:6]))
                if parts:
                    summary_contexts.append(format_context_with_provenance(
                        f"{ename}（药物）：{'；'.join(parts)}", entity.get("provenance")
                    ))

            # 2) 受控多跳三元组（每实体保留前 4 条去重高价值关系，兼顾多实体覆盖）
            expansion = queries.expand_multihop(ename, entity_type=etype)
            added_for_entity = 0
            for triple in expansion.get("triples", []):
                tkey = (triple.get("subject"), triple.get("predicate"), triple.get("object"))
                if tkey in seen_triples:
                    continue
                seen_triples.add(tkey)
                triple_contexts.append(format_context_with_provenance(
                    f"({triple['subject']})-[:{triple['predicate']}]->({triple['object']})",
                    triple.get("provenance"),
                ))
                added_for_entity += 1
                if added_for_entity >= 4:
                    break

        contexts = list(dict.fromkeys([*summary_contexts, *triple_contexts]))
        contexts = [c if "[数据源:" in c else format_context_with_provenance(c)
                    for c in contexts]
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
