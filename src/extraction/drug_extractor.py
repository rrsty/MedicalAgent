"""药物实体抽取器：名称/类别/副作用/禁忌 + 适应症疾病。

不编造原则：模型不知道的属性返回空（医疗安全红线）。
"""
from src.common.llm import get_llm
from src.common.logger import setup_logger
from src.extraction.schemas import DrugExtractionResult

logger = setup_logger("extraction.drug", "extraction.log")

SYSTEM_PROMPT = """你是一个专业的药学信息抽取助手。请从文本中提取药物信息。

要求：
1. 药物名称使用通用名（如"氨氯地平"而非"络活喜"）
2. 类别使用标准药物分类（如"钙通道阻滞剂"）
3. 副作用、禁忌症必须是文本中明确提及或公认的医学常识
4. 不确定的信息宁可不填，绝不编造
5. 没有相关信息时返回空列表

以 JSON 格式输出结果"""


class DrugExtractor:
    """药物实体抽取器。"""

    def __init__(self, temperature: float = 0):
        self.llm = get_llm(temperature).with_structured_output(
            DrugExtractionResult, method="json_mode"
        )

    def extract(self, text: str) -> DrugExtractionResult:
        if not text or not text.strip():
            return DrugExtractionResult()
        try:
            return self.llm.invoke([
                ("system", SYSTEM_PROMPT),
                ("human", text),
            ])
        except Exception as exc:  # noqa: BLE001
            logger.error("药物抽取失败: %s", exc)
            return DrugExtractionResult()
