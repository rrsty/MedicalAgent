"""微调数据集生成：项目真实数据 → LlamaFactory alpaca 格式。

对齐目标：instruction 与 src/extraction/disease_extractor.py 的 EXTRACTION_INSTRUCTION
逐字一致（R1 单一来源：直接 import），output 为 SimpleLLMOutput schema 的 JSON 字符串，
使微调后的模型可无缝接回 DiseaseExtractor（不改项目代码）。

质量过滤：无关系跳过；五列表全空跳过；输入>1200/输出>1500 跳过；实体接地率<0.5 跳过。

用法：python scripts/generate_finetune_data.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import (
    DATA_FINETUNE_DIR,
    PROCESSED_DISEASES_JSON,
    KG_RELATIONS_FILE,
)
from src.common.logger import setup_logger
from src.common.utils import read_json
from src.extraction.disease_extractor import (
    EXTRACTION_INSTRUCTION,
    build_input_text,
)

logger = setup_logger("scripts.finetune", "finetune.log")

# 关系类型 → (relations.json data 里的尾字段, 输出列表字段)
REL_FIELD = {
    "HAS_SYMPTOM": ("symptom_name", "symptoms"),
    "TREATED_BY_MEDICATION": ("drug_name", "medications"),
    "BELONGS_TO_DEPARTMENT": ("department_name", "departments"),
    "NEEDS_EXAMINATION": ("examination_name", "examinations"),
    "AFFECTS_BODY_PART": ("body_part_name", "body_parts"),
}
LIST_FIELDS = ("symptoms", "medications", "departments", "examinations", "body_parts")

MAX_IN, MAX_OUT, MIN_GROUND, DESC_LEN = 1200, 1500, 0.5, 100


def build_answer(disease: dict, rels: list[dict]) -> dict:
    """金标答案：五个列表字段来自该疾病在图谱中对应类型的关系（去重、保序）。"""
    name = str(disease.get("name", "")).strip()
    description = str(disease.get("description", "") or "").strip()[:DESC_LEN]

    answer = {
        "disease": {"name": name, "description": description or None, "icd_code": None},
        "symptoms": [], "medications": [], "departments": [],
        "examinations": [], "treatments": [], "body_parts": [],
    }
    seen = {f: set() for f in LIST_FIELDS}
    for rel in rels:
        mapping = REL_FIELD.get(rel.get("type"))
        if not mapping:
            continue
        field_name, out_key = mapping
        value = str(rel.get("data", {}).get(field_name, "")).strip()
        if value and value not in seen[out_key]:
            seen[out_key].add(value)
            answer[out_key].append(value)
    return answer


def main() -> None:
    diseases = read_json(PROCESSED_DISEASES_JSON).get("diseases", [])
    relations = read_json(KG_RELATIONS_FILE).get("relations", [])

    rel_by_disease = defaultdict(list)
    for item in relations:
        disease_name = item.get("data", {}).get("disease_name")
        if disease_name:
            rel_by_disease[disease_name].append(item)

    dataset = []
    skip_stats: dict[str, int] = defaultdict(int)

    for disease in diseases:
        name = str(disease.get("name", "")).strip()
        rels = rel_by_disease.get(name, [])
        if not rels:
            skip_stats["无关系"] += 1
            continue

        input_text = build_input_text(disease)
        if len(input_text) > MAX_IN:
            skip_stats["输入过长"] += 1
            continue

        answer = build_answer(disease, rels)
        if not any(answer[f] for f in LIST_FIELDS):
            skip_stats["无有效关系"] += 1
            continue

        output_text = json.dumps(answer, ensure_ascii=False)
        if len(output_text) > MAX_OUT:
            skip_stats["输出过长"] += 1
            continue

        gold_names = [n for f in LIST_FIELDS for n in answer[f]]
        grounding = (
            sum(1 for n in gold_names if n in input_text) / len(gold_names)
            if gold_names else 1.0
        )
        if grounding < MIN_GROUND:
            skip_stats["接地率低"] += 1
            continue

        dataset.append({
            "instruction": EXTRACTION_INSTRUCTION,  # R1：与 DiseaseExtractor 逐字一致
            "input": input_text,
            "output": output_text,
        })

    out_path = DATA_FINETUNE_DIR / "ft_extract.json"
    DATA_FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"共生成 {len(dataset)} 条 alpaca 样本 -> {out_path}")
    print("跳过统计:", dict(skip_stats))
    if dataset:
        demo = dict(dataset[0])
        demo["instruction"] = demo["instruction"][:50] + " ...(与 DiseaseExtractor 提示词一致)"
        demo["input"] = demo["input"][:120] + " ..."
        demo["output"] = demo["output"][:300] + " ..."
        print("\n示例样本（截断展示）：")
        print(json.dumps(demo, ensure_ascii=False, indent=2))
        print("\n下一步（LlamaFactory）：")
        print('  1. 把 ft_extract.json 放入 LlamaFactory/data/，在 dataset_info.json 登记:')
        print('     {"ft_extract": {"file_name": "ft_extract.json"}}')
        print("  2. llamafactory-cli webui → Train: LoRA + SFT, qwen 模板, cutoff 2048")


if __name__ == "__main__":
    main()
