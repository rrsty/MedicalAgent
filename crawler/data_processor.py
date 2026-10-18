"""数据清洗与保存：原始爬取数据 → 结构化数据（JSON 程序用 + XLSX 人工查看）。

用法：python crawler/data_processor.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logger import setup_logger
from src.common.utils import clean_text

logger = setup_logger("crawler.processor", "crawler.log")

RAW_FILE = Path(__file__).resolve().parents[1] / "data" / "raw" / "diseases.json"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# 科室名称标准化（别名 → 标准名）
DEPARTMENT_NORMALIZE = {
    "心内科": "心血管内科", "心血管科": "心血管内科", "心外科": "心血管外科",
    "呼吸内科": "呼吸内科", "呼吸科": "呼吸内科",
    "消化内科": "消化内科", "消化科": "消化内科", "胃肠科": "消化内科",
    "内分泌科": "内分泌科", "内分泌": "内分泌科",
    "神经内科": "神经内科", "神经科": "神经内科",
    "肾内科": "肾内科", "泌尿外科": "泌尿外科", "泌尿科": "泌尿外科",
    "骨科": "骨科", "皮肤科": "皮肤科", "皮肤性病科": "皮肤科",
    "妇科": "妇产科", "产科": "妇产科", "妇产科": "妇产科",
    "儿科": "儿科", "小儿科": "儿科",
    "眼科": "眼科", "耳鼻喉科": "耳鼻喉科", "耳鼻咽喉科": "耳鼻喉科",
    "口腔科": "口腔科", "精神科": "精神心理科", "心理科": "精神心理科",
    "肿瘤科": "肿瘤科", "血液科": "血液科", "风湿免疫科": "风湿免疫科",
}


def normalize_department(name: str) -> str:
    name = clean_text(name)
    return DEPARTMENT_NORMALIZE.get(name, name)


def clean_record(record: dict) -> dict | None:
    """单条清洗：去空值/去重/标准化/文本清理。无效记录返回 None。"""
    name = clean_text(record.get("name"))
    if not name:
        return None

    symptoms_raw = record.get("symptoms") or []
    if isinstance(symptoms_raw, str):
        symptoms_raw = [s for s in symptoms_raw.replace("、", ",").split(",")]
    symptoms = []
    for s in symptoms_raw:
        s = clean_text(str(s))
        if s and s not in symptoms:
            symptoms.append(s)

    medications_raw = record.get("medications") or []
    if isinstance(medications_raw, str):
        medications_raw = [m for m in medications_raw.replace("、", ",").split(",")]
    medications = []
    for m in medications_raw:
        m = clean_text(str(m))
        if m and m not in medications:
            medications.append(m)

    return {
        "name": name,
        "description": clean_text(record.get("description"))[:2000],
        "symptoms": symptoms,
        "treatment": clean_text(record.get("treatment"))[:3000],
        "department": normalize_department(record.get("department", "")),
        "medications": medications,
        "precautions": clean_text(record.get("precautions"))[:1000],
        "url": record.get("url", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="数据清洗与保存")
    parser.add_argument("--input", default=str(RAW_FILE), help="原始数据文件")
    args = parser.parse_args()

    import json

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("原始数据不存在: %s，请先运行 python crawler/medical_crawler.py", input_path)
        sys.exit(1)

    raw = json.loads(input_path.read_text(encoding="utf-8")).get("diseases", [])
    logger.info("读取原始数据 %d 条", len(raw))

    cleaned: dict[str, dict] = {}
    skipped = 0
    for record in raw:
        item = clean_record(record)
        if item is None:
            skipped += 1
            continue
        cleaned[item["name"]] = item  # 按名称去重

    diseases = list(cleaned.values())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # JSON（程序使用）
    out_json = PROCESSED_DIR / "diseases.json"
    out_json.write_text(
        json.dumps({"diseases": diseases}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # XLSX（人工查看）
    try:
        import pandas as pd

        df = pd.DataFrame(diseases)
        df["symptoms"] = df["symptoms"].apply(lambda x: "、".join(x))
        df["medications"] = df["medications"].apply(lambda x: "、".join(x))
        df.to_excel(PROCESSED_DIR / "diseases.xlsx", index=False)
        xlsx_ok = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("XLSX 导出失败（openpyxl 未安装？）: %s", exc)
        xlsx_ok = False

    # 统计
    total = len(diseases)
    field_coverage = {
        field: round(sum(1 for d in diseases if d.get(field)) / total, 3) if total else 0
        for field in ("description", "symptoms", "treatment", "department", "medications")
    }
    print("\n===== 清洗结果统计 =====")
    print(f"原始: {len(raw)} 条 | 清洗后: {total} 条 | 跳过/去重: {len(raw) - total + skipped}")
    print(f"字段覆盖率: {field_coverage}")
    print(f"JSON: {out_json}")
    print(f"XLSX: {PROCESSED_DIR / 'diseases.xlsx'} ({'OK' if xlsx_ok else 'SKIP'})")
    low = {k: v for k, v in field_coverage.items() if v < 0.5}
    if low:
        print(f"⚠️ 以下字段覆盖率低于 50%（LLM 抽取时将从治疗文本补齐）: {low}")


if __name__ == "__main__":
    main()
