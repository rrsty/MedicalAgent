"""抽取 Pydantic Schema —— LLM 结构化输出的「契约」（with_structured_output 配套）。

两套模型：
- SimpleLLMOutput：疾病抽取主链路输出（与微调数据集/ DiseaseExtractor 逐字对齐）；
- DiseaseEntity / SymptomEntity / MedicationEntity / DepartmentEntity：
  按实体类型分离的细分模型（symptom / drug / department 抽取器使用）。
"""
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from src.common.utils import coerce_str_list


class DiseaseInfo(BaseModel):
    """疾病实体（SimpleLLMOutput 的 disease 字段）。"""

    name: str = Field(description="疾病名称（标准医学术语）")
    description: str | None = Field(default=None, description="疾病描述")
    icd_code: str | None = Field(default=None, description="ICD编码（如知道）或 null")


class SimpleLLMOutput(BaseModel):
    """疾病抽取结果 —— 与 DiseaseExtractor 提示词、微调数据集输出 schema 完全一致。"""

    disease: DiseaseInfo = Field(description="疾病实体")
    symptoms: list[str] = Field(default_factory=list, description="症状列表")
    medications: list[str] = Field(default_factory=list, description="药物列表（通用名）")
    departments: list[str] = Field(default_factory=list, description="就诊科室列表")
    examinations: list[str] = Field(default_factory=list, description="检查项目列表")
    treatments: list[str] = Field(default_factory=list, description="治疗方案列表")
    body_parts: list[str] = Field(default_factory=list, description="受影响身体部位列表")

    @field_validator(
        "symptoms", "medications", "departments", "examinations",
        "treatments", "body_parts", mode="before",
    )
    @classmethod
    def _coerce_lists(cls, v):
        # DeepSeek/vLLM 偶发返回对象数组，统一收敛为字符串列表
        return coerce_str_list(v)

    # 模板占位符回声（LLM 照抄 schema 示例时过滤；ClassVar 不参与字段校验）
    PLACEHOLDER_ECHOES: ClassVar[set[str]] = {
        "症状1", "症状2", "药物1", "药物2", "科室1", "检查1", "检查2",
        "治疗方案1", "身体部位1", "疾病名称", "疾病描述",
    }

    def clean(self) -> "SimpleLLMOutput":
        """过滤占位符回声与空串（统一后处理，云端/vLLM 两条链共用）。"""
        def _clean_list(items: list[str]) -> list[str]:
            return [
                item.strip() for item in items
                if item and item.strip() and item.strip() not in self.PLACEHOLDER_ECHOES
            ]

        self.symptoms = _clean_list(self.symptoms)
        self.medications = _clean_list(self.medications)
        self.departments = _clean_list(self.departments)
        self.examinations = _clean_list(self.examinations)
        self.treatments = _clean_list(self.treatments)
        self.body_parts = _clean_list(self.body_parts)
        return self


class SymptomEntity(BaseModel):
    """症状实体。"""

    name: str = Field(description="症状名称（标准医学术语）")
    severity: str | None = Field(default=None, description="严重程度（轻度/中度/重度）")


class MedicationEntity(BaseModel):
    """药物实体。"""

    name: str = Field(description="药物通用名")
    category: str | None = Field(default=None, description="药物类别")
    side_effects: list[str] = Field(default_factory=list, description="副作用列表")
    contraindications: list[str] = Field(default_factory=list, description="禁忌症列表")


class DepartmentEntity(BaseModel):
    """科室实体。"""

    name: str = Field(description="科室名称（标准科室分类）")
    floor: str | None = Field(default=None, description="楼层位置（如知道）")


class SymptomExtractionResult(BaseModel):
    """症状抽取器输出：症状 + 可能提示的疾病。"""

    symptoms: list[SymptomEntity] = Field(default_factory=list, description="症状列表")
    possible_diseases: list[str] = Field(default_factory=list, description="可能相关的疾病")


class DrugExtractionResult(BaseModel):
    """药物抽取器输出。"""

    medications: list[MedicationEntity] = Field(default_factory=list, description="药物列表")
    treats_diseases: list[str] = Field(default_factory=list, description="适应症疾病")


class DepartmentExtractionResult(BaseModel):
    """科室抽取器输出。"""

    departments: list[DepartmentEntity] = Field(default_factory=list, description="科室列表")
    handles_diseases: list[str] = Field(default_factory=list, description="诊治疾病")
