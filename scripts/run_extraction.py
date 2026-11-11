"""批量实体抽取：data/processed/diseases.json → entities.json + relations.json。

用法：
    python scripts/run_extraction.py                 # LLM 抽取（断点续传）
    python scripts/run_extraction.py --no-llm        # 字典转换模式（测试/离线）
    python scripts/run_extraction.py --limit 10      # 限量
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from config.paths import (
    KG_ENTITIES_FILE,
    KG_RELATIONS_FILE,
    PROCESSED_DISEASES_JSON,
)
from src.common.logger import setup_logger
from src.common.utils import read_json, write_json
from src.common.provenance import merge_provenance_records, normalize_record_provenance
from src.extraction.disease_extractor import DiseaseExtractor, build_input_text
from src.extraction.schemas import SimpleLLMOutput

logger = setup_logger("scripts.extraction", "extraction.log")

CHECKPOINT_FILE = Path(__file__).resolve().parents[1] / "data" / "knowledge_graph" / "_extraction_checkpoint.json"


# ---------- 序列化（entities.json / relations.json 契约）----------
def serialize(result: SimpleLLMOutput, provenance=None) -> tuple[list[dict], list[dict]]:
    entities: list[dict] = []
    relations: list[dict] = []

    disease_name = result.disease.name.strip()
    entities.append({
        "type": "Disease",
        "name": disease_name,
        "description": result.disease.description or "",
        "icd_code": result.disease.icd_code,
    })
    for symptom in result.symptoms:
        entities.append({"type": "Symptom", "name": symptom, "severity": ""})
        relations.append({
            "type": "HAS_SYMPTOM",
            "data": {"disease_name": disease_name, "symptom_name": symptom},
        })
    for medication in result.medications:
        entities.append({"type": "Medication", "name": medication})
        relations.append({
            "type": "TREATED_BY_MEDICATION",
            "data": {"disease_name": disease_name, "drug_name": medication},
        })
    for department in result.departments:
        entities.append({"type": "Department", "name": department})
        relations.append({
            "type": "BELONGS_TO_DEPARTMENT",
            "data": {"disease_name": disease_name, "department_name": department},
        })
    for examination in result.examinations:
        entities.append({"type": "Examination", "name": examination})
        relations.append({
            "type": "NEEDS_EXAMINATION",
            "data": {"disease_name": disease_name, "examination_name": examination},
        })
    for body_part in result.body_parts:
        entities.append({"type": "BodyPart", "name": body_part})
        relations.append({
            "type": "AFFECTS_BODY_PART",
            "data": {"disease_name": disease_name, "body_part_name": body_part},
        })
    for item in [*entities, *relations]:
        item["provenance"] = merge_provenance_records(provenance)
    return entities, relations


def no_llm_convert(disease: dict) -> SimpleLLMOutput:
    """--no-llm 字典转换模式：直接用爬取字段（treatment 不解析）。"""
    return SimpleLLMOutput(
        disease={
            "name": disease.get("name", ""),
            "description": disease.get("description", "") or None,
            "icd_code": None,
        },
        symptoms=disease.get("symptoms", []) or [],
        medications=disease.get("medications", []) or [],
        departments=[disease["department"]] if disease.get("department") else [],
        examinations=[],
        treatments=[],
        body_parts=[],
    ).clean()


def main() -> None:
    parser = argparse.ArgumentParser(description="批量实体抽取")
    parser.add_argument("--no-llm", action="store_true", help="字典转换模式（不用 LLM）")
    parser.add_argument("--limit", type=int, default=0, help="处理条数上限（0=全部）")
    args = parser.parse_args()

    diseases = read_json(PROCESSED_DISEASES_JSON).get("diseases", [])
    if args.limit > 0:
        diseases = diseases[:args.limit]
    logger.info("待抽取疾病 %d 条（no_llm=%s）", len(diseases), args.no_llm)

    # 断点续传：跳过已处理的疾病名
    done: dict[str, dict] = {}
    if CHECKPOINT_FILE.exists():
        try:
            done = read_json(CHECKPOINT_FILE).get("done", {})
        except Exception:  # noqa: BLE001
            done = {}

    all_entities: list[dict] = []
    all_relations: list[dict] = []
    failed: list[str] = []
    fallback_count = 0  # vLLM 失败 → 云端兜底次数（单独计数，不掩盖微调模型失败率）

    extractor = None if args.no_llm else DiseaseExtractor()

    for disease in tqdm(diseases, desc="实体抽取"):
        name = disease.get("name", "")
        if not name:
            continue
        if name in done:
            # Old checkpoints may predate provenance. Reattach the recorded source
            # rather than silently dropping it on resumed runs.
            for item in done[name]["entities"] + done[name]["relations"]:
                item["provenance"] = merge_provenance_records(
                    item.get("provenance"), normalize_record_provenance(disease)
                )
            all_entities.extend(done[name]["entities"])
            all_relations.extend(done[name]["relations"])
            continue

        try:
            if args.no_llm:
                result = no_llm_convert(disease)
            else:
                try:
                    result = extractor.extract_disease(disease)
                except RuntimeError as exc:
                    # vLLM 分支失败 → 云端主 LLM 兜底（保覆盖率 + 单独计数）
                    if extractor.use_vllm:
                        logger.warning("vLLM 抽取失败，云端兜底: %s | %s", name, exc)
                        fallback_count += 1
                        cloud = DiseaseExtractor.__new__(DiseaseExtractor)
                        from src.common.llm import get_llm

                        cloud.llm = get_llm(0)
                        cloud.use_vllm = False
                        result = cloud.extract_disease(disease)
                    else:
                        raise

            entities, relations = serialize(result, normalize_record_provenance(disease))
            all_entities.extend(entities)
            all_relations.extend(relations)
            done[name] = {"entities": entities, "relations": relations}
            # 每条写回 checkpoint（中断可续）
            write_json(CHECKPOINT_FILE, {"done": done})
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            logger.error("抽取失败 %s: %s", name, exc)

    # 去重实体（同名同类型只留一个，后者属性合并）
    merged: dict[tuple[str, str], dict] = {}
    for entity in all_entities:
        key = (entity["type"], entity["name"])
        if key in merged:
            merged[key]["provenance"] = merge_provenance_records(
                merged[key].get("provenance"), entity.get("provenance")
            )
            for k, v in entity.items():
                if v and not merged[key].get(k):
                    merged[key][k] = v
        else:
            merged[key] = dict(entity)

    write_json(KG_ENTITIES_FILE, {"entities": list(merged.values())})
    write_json(KG_RELATIONS_FILE, {"relations": all_relations})

    # 统计
    entity_stats: dict[str, int] = {}
    for entity in merged.values():
        entity_stats[entity["type"]] = entity_stats.get(entity["type"], 0) + 1
    relation_stats: dict[str, int] = {}
    for rel in all_relations:
        relation_stats[rel["type"]] = relation_stats.get(rel["type"], 0) + 1

    print("\n===== 抽取结果统计 =====")
    print(f"疾病处理: {len(diseases)} | 实体: {len(merged)}（按类型: {entity_stats}）")
    print(f"关系: {len(all_relations)}（按类型: {relation_stats}）")
    print(f"失败: {len(failed)} {failed[:10]}")
    if fallback_count:
        print(f"vLLM 失败 → 云端兜底: {fallback_count} 次")
    print(f"实体 → {KG_ENTITIES_FILE}")
    print(f"关系 → {KG_RELATIONS_FILE}")


if __name__ == "__main__":
    main()
