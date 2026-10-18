"""症状实体抽取器：纯 LLM 结构化输出，单一职责（同义词归一属检索层职责）。"""
from src.common.llm import get_llm
from src.common.logger import setup_logger
from src.extraction.schemas import SymptomExtractionResult

logger = setup_logger("extraction.symptom", "extraction.log")

SYSTEM_PROMPT = """你是一个专业的医疗信息抽取助手。请从用户描述中提取症状信息。

要求：
1. 症状名称使用标准医学术语（如"头痛"而非"脑袋疼"）
2. 提取所有提到的症状，包括隐含的
3. 如果能判断，给出每个症状的严重程度（轻度/中度/重度）
4. 如果能推断，列出可能相关的疾病
5. 没有相关信息时返回空列表

以 JSON 格式输出结果"""


class SymptomExtractor:
    """症状实体抽取器。"""

    def __init__(self, temperature: float = 0):
        self.llm = get_llm(temperature).with_structured_output(
            SymptomExtractionResult, method="json_mode"
        )

    def extract(self, text: str) -> SymptomExtractionResult:
        if not text or not text.strip():
            return SymptomExtractionResult()
        try:
            return self.llm.invoke([
                ("system", SYSTEM_PROMPT),
                ("human", f"用户描述：{text}"),
            ])
        except Exception as exc:  # noqa: BLE001
            logger.error("症状抽取失败: %s", exc)
            return SymptomExtractionResult()
