"""药物属性增强 pass（属性补齐与调试修复）。

对 entities.json 中 type=Medication 的去重药名各调 LLM 一次，补 category /
side_effects / contraindications（不编造：不知道返回空）。
- 幂等：已有 category 的药名跳过（断点续跑）；
- 增量写回 entities.json；
- 尽力 MERGE 更新 Neo4j（不可达只警告，不中断）。

用法：python scripts/enrich_medications.py
"""
import sys
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

from src.common.utils import coerce_str_list

from config.paths import KG_ENTITIES_FILE
from src.common.llm import get_llm
from src.common.logger import setup_logger
from src.common.utils import read_json, write_json

logger = setup_logger("scripts.enrich", "extraction.log")


class MedicationProfile(BaseModel):
    """药物属性（不编造：模型不知道的字段返回空/None）。"""

    category: str | None = Field(default=None, description="药物类别（如钙通道阻滞剂）")
    side_effects: list[str] = Field(default_factory=list, description="常见副作用")
    contraindications: list[str] = Field(default_factory=list, description="禁忌症")

    @field_validator("side_effects", "contraindications", mode="before")
    @classmethod
    def _coerce(cls, v):
        return coerce_str_list(v)

    PLACEHOLDERS: ClassVar[set[str]] = {"副作用1", "副作用2", "禁忌1", "药物类别"}


def enrich() -> None:
    data = read_json(KG_ENTITIES_FILE)
    entities = data.get("entities", [])
    meds = {
        e["name"]: e
        for e in entities
        if e.get("type") in ("Medication", "Drug") and e.get("name")
    }
    todo = [name for name, e in meds.items() if not e.get("category")]
    print(f"药物总数 {len(meds)}，待增强 {len(todo)}")

    if not todo:
        print("✅ 全部已增强（幂等跳过）")
        return

    llm = get_llm(temperature=0).with_structured_output(
        MedicationProfile, method="json_mode"
    )
    filled = 0

    for name in tqdm(todo, desc="药物增强"):
        try:
            profile: MedicationProfile = llm.invoke([
                (
                    "system",
                    "你是药学专家。给出药物的通用分类、常见副作用、禁忌症。"
                    "不确定的信息宁可不填，绝不编造（医疗安全红线）。以 JSON 格式输出。",
                ),
                ("human", name),
            ])
            entity = meds[name]
            entity["category"] = profile.category or ""
            entity["side_effects"] = [
                s for s in profile.side_effects
                if s and s not in MedicationProfile.PLACEHOLDERS
            ]
            entity["contraindications"] = [
                s for s in profile.contraindications
                if s and s not in MedicationProfile.PLACEHOLDERS
            ]
            filled += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("增强失败 %s: %s", name, exc)

        # 每批增量写回（中断可续）
        write_json(KG_ENTITIES_FILE, {"entities": entities})

    # 尽力 MERGE 更新 Neo4j（不可达只警告）
    try:
        from src.knowledge_graph.graph_queries import GraphQueries
        from src.knowledge_graph.schema_manager import get_driver, verify_connectivity

        if verify_connectivity():
            with get_driver().session() as session:
                for name, entity in meds.items():
                    if not entity.get("category"):
                        continue
                    side_effects = entity.get("side_effects") or []
                    cons = entity.get("contraindications") or []
                    session.run(
                        """
                        MATCH (m:Medication {name: $name})
                        SET m.category = $category,
                            m.side_effects = coalesce(m.side_effects, $side_effects),
                            m.contraindications = coalesce(m.contraindications, $cons)
                        """,
                        name=name, category=entity["category"],
                        side_effects=side_effects if isinstance(side_effects, list) else [side_effects],
                        cons=cons if isinstance(cons, list) else [cons],
                    )
            print("✅ Neo4j Medication 节点已同步更新")
        else:
            print("⚠️ Neo4j 不可达，entities.json 已增强；Neo4j 启动后重跑即可同步")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Neo4j 同步失败（不中断）: %s", exc)

    # 填充率
    with_category = sum(1 for e in meds.values() if e.get("category"))
    print(f"\n===== 增强结果 =====")
    print(f"本次填充: {filled} | category 填充率: {with_category}/{len(meds)} "
          f"({with_category / max(len(meds), 1):.0%})")


if __name__ == "__main__":
    enrich()
