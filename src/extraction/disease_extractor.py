"""疾病实体/关系抽取器 —— 抽取主链路。

双后端链路（实测踩坑总结，勿合并成一条）：
- 云端分支：双消息提示词 + with_structured_output(json_mode)；
- vLLM 微调模型分支：完整 instruction（6 条要求 + 输出 schema）作为**单条 system**、
  疾病文本作为**单条 user**，裸调用后手工解析（不用 json_mode——引导解码会诱发失控生成）。

提示词原文与 scripts/generate_finetune_data.py 共用（R1：只在此定义一次）。
"""
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from src.common.llm import get_extract_llm
from src.common.logger import setup_logger
from src.common.utils import parse_llm_json
from src.extraction.schemas import SimpleLLMOutput

logger = setup_logger("extraction.disease", "extraction.log")

# ===== 抽取提示词（微调数据集 instruction 与此逐字一致，勿改动）=====
EXTRACTION_INSTRUCTION = """你是一个专业的医疗信息抽取助手。请从以下文本中提取疾病及其相关信息。

要求：
1. 疾病名称使用标准医学术语
2. 症状列表要完整，包含所有提到的症状
3. 科室名称使用标准科室名称（如"心血管内科"而非"心内科"）
4. 药物名称使用通用名（如"氨氯地平"而非商品名）
5. 如果没有相关信息，对应字段返回空列表
6. 严格按照 JSON 格式输出

请以 JSON 格式输出，包含以下字段：
{
    "disease": {
        "name": "疾病名称",
        "description": "疾病描述",
        "icd_code": "ICD编码（如知道）或 null"
    },
    "symptoms": ["症状1", "症状2"],
    "medications": ["药物1", "药物2"],
    "departments": ["科室1"],
    "examinations": ["检查1", "检查2"],
    "treatments": ["治疗方案1"],
    "body_parts": ["身体部位1"]
}"""

# 要求部分（云端分支的 system 消息；instruction 去掉 JSON schema 段）
_INSTRUCTION_REQUIREMENTS = EXTRACTION_INSTRUCTION.split("请以 JSON 格式输出")[0].strip()
# JSON schema 段（云端分支拼进 user 消息）
_OUTPUT_SCHEMA_TEXT = "请以 JSON 格式输出，包含以下字段：" + EXTRACTION_INSTRUCTION.split(
    "请以 JSON 格式输出", 1
)[1]

MAX_RETRIES = 3


def build_input_text(disease: dict) -> str:
    """把疾病字段拼成推理输入文本（与 run_extraction / 微调数据生成完全一致，勿改动）。"""
    parts = []
    name = str(disease.get("name", "") or "").strip()
    if name:
        parts.append(f"疾病名称：{name}")

    description = str(disease.get("description", "") or "").strip()
    if description:
        parts.append(f"疾病描述：{description}")

    symptoms = disease.get("symptoms") or []
    symptoms = [str(s).strip() for s in symptoms if str(s).strip()]
    if symptoms:
        parts.append(f"症状：{', '.join(symptoms)}")

    treatment = str(disease.get("treatment", "") or "").strip()
    if treatment:
        if len(treatment) > 1500:
            treatment = treatment[:1500] + "..."
        parts.append(f"治疗方法：{treatment}")

    department = str(disease.get("department", "") or "").strip()
    if department:
        parts.append(f"就诊科室：{department}")

    medications = disease.get("medications") or []
    medications = [str(m).strip() for m in medications if str(m).strip()]
    if medications:
        parts.append(f"常用药物：{', '.join(medications)}")

    return "\n".join(parts)


class DiseaseExtractor:
    """疾病实体抽取器（云端 / vLLM 微调模型双链路，自动按 .env 切换）。"""

    def __init__(self, temperature: float = 0):
        # temperature 传 0：vLLM 分支温度非零易失控生成（实测）
        self.llm = get_extract_llm(temperature=temperature)
        self.use_vllm = bool(settings.extraction_model_base_url)

    # ---------- vLLM 分支：单 system + 单 user，裸调用 + 手工解析 ----------
    def _extract_via_vllm(self, text: str) -> SimpleLLMOutput:
        resp = self.llm.invoke([
            SystemMessage(content=EXTRACTION_INSTRUCTION),
            HumanMessage(content=text),
        ])
        data = parse_llm_json(resp.content)
        if data is None:
            raise ValueError(f"vLLM 输出无法解析为 JSON: {resp.content[:200]}")
        return SimpleLLMOutput.model_validate(data)

    # ---------- 云端分支：双消息 + json_mode 结构化输出 ----------
    def _extract_via_cloud(self, text: str) -> SimpleLLMOutput:
        structured_llm = self.llm.with_structured_output(SimpleLLMOutput, method="json_mode")
        return structured_llm.invoke([
            SystemMessage(content=_INSTRUCTION_REQUIREMENTS),
            HumanMessage(content=f"{text}\n\n{_OUTPUT_SCHEMA_TEXT}"),
        ])

    def extract(self, text: str) -> SimpleLLMOutput:
        """抽取（带重试），返回清洗后的 SimpleLLMOutput。"""
        if not text or not text.strip():
            raise ValueError("抽取输入文本为空")

        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if self.use_vllm:
                    result = self._extract_via_vllm(text)
                else:
                    result = self._extract_via_cloud(text)
                return result.clean()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("抽取第 %d/%d 次失败: %s", attempt, MAX_RETRIES, exc)

        raise RuntimeError(f"抽取失败（已重试 {MAX_RETRIES} 次）: {last_exc}")

    def extract_disease(self, disease: dict) -> SimpleLLMOutput:
        """便捷方法：直接传疾病 dict（自动拼接输入文本）。"""
        return self.extract(build_input_text(disease))
