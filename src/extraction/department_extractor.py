"""科室实体抽取器：标准科室名称 + 诊治疾病。"""
from src.common.llm import get_llm
from src.common.logger import setup_logger
from src.extraction.schemas import DepartmentExtractionResult

logger = setup_logger("extraction.department", "extraction.log")

SYSTEM_PROMPT = """你是一个专业的医疗信息抽取助手。请从文本中提取医院科室信息。

要求：
1. 科室名称使用标准科室分类（如"心血管内科"而非"心内科"）
2. 一级科室与二级科室都保留（如"内科"与"心血管内科"）
3. 如能推断，列出该科室诊治的疾病
4. 没有相关信息时返回空列表

以 JSON 格式输出结果"""


class DepartmentExtractor:
    """科室实体抽取器。"""

    def __init__(self, temperature: float = 0):
        self.llm = get_llm(temperature).with_structured_output(
            DepartmentExtractionResult, method="json_mode"
        )

    def extract(self, text: str) -> DepartmentExtractionResult:
        if not text or not text.strip():
            return DepartmentExtractionResult()
        try:
            return self.llm.invoke([
                ("system", SYSTEM_PROMPT),
                ("human", text),
            ])
        except Exception as exc:  # noqa: BLE001
            logger.error("科室抽取失败: %s", exc)
            return DepartmentExtractionResult()
