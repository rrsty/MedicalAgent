"""症状检测 Agent：纯 LLM 结构化输出，单一职责（同义词归一属检索层实体链接职责）。"""
from pydantic import BaseModel, Field, field_validator

from src.agents.safety_checker import EMERGENCY_SYMPTOM_KEYWORDS
from src.common.utils import coerce_str_list
from src.agents.state import AgentState, Severity
from src.common.llm import get_llm
from src.common.logger import setup_logger

logger = setup_logger("agent.symptom", "agents.log")


class SymptomDetection(BaseModel):
    """症状检测结果（结构化输出契约）。"""

    symptoms: list[str] = Field(default_factory=list, description="提取的症状列表（标准术语）")
    severity: str = Field(default="low", description="严重程度: low/medium/high/emergency")
    has_emergency_keyword: bool = Field(default=False, description="是否包含急症关键词")

    @field_validator("symptoms", mode="before")
    @classmethod
    def _coerce_symptoms(cls, v):
        # DeepSeek 偶发返回 [{"name": "头痛", ...}] 对象数组，归一为字符串列表
        return coerce_str_list(v)


SYSTEM_PROMPT = """你是一个症状分析专家。从用户描述中提取症状信息。

要求：
1. 症状使用标准医学术语（如"头痛"而非"脑袋疼"）
2. 提取当前描述中提到的症状。注意：
   - 只有当前描述使用明确代词（如"它"、"这个病"、"那"）指代上文，或者直接承接上文未完的症状追问（如"那应该挂什么科"）时，才从上下文中补全省略的症状；
   - 若当前描述已明确指出了具体的疾病诊断（例如"我有糖尿病"、"高血压怎么吃药"），切勿把上文无关的历史症状（如之前提到的头晕、恶心）盲目继承过来！
3. 综合判断严重程度：low（轻微）/ medium（影响生活）/ high（明显痛苦）/ emergency（急症）
4. 检测急症关键词：胸痛、呼吸困难、意识模糊等
5. 当前描述与上下文都没有症状时返回空列表

以 JSON 格式输出结果

示例：
1. 用户描述：我最近头痛、头晕，应该挂什么科？ → {"symptoms": ["头痛", "头晕"], "severity": "low", "has_emergency_keyword": false}
2. 用户描述：胃疼了一周了 → {"symptoms": ["胃痛", "腹痛"], "severity": "medium", "has_emergency_keyword": false}
3. 用户描述：我有糖尿病，我应该吃什么药？ → {"symptoms": [], "severity": "low", "has_emergency_keyword": false}
4. 用户描述：那应该挂什么科？（上下文：我最近总是头晕） → {"symptoms": ["头晕"], "severity": "low", "has_emergency_keyword": false}"""


def symptom_detector(state: AgentState) -> dict:
    """症状检测节点：LLM 结构化输出（含多轮上下文）+ 急症关键词规则兜底。"""
    query = state.get("query", "")
    history = state.get("history", [])

    keyword_hit = any(kw in query for kw in EMERGENCY_SYMPTOM_KEYWORDS)
    try:
        # 多轮上下文：最近 3 轮（6 条消息），供指代消解与症状补全（PRD US9 / t01）
        context_lines = [
            f"{'用户' if t.get('role') == 'user' else '助手'}：{t.get('content', '')[:200]}"
            for t in history[-6:]
        ]
        context_block = f"最近对话：\n" + "\n".join(context_lines) + "\n\n" if context_lines else ""
        def _detect(human_text: str) -> SymptomDetection:
            return get_llm(temperature=0.1).with_structured_output(
                SymptomDetection, method="json_mode"
            ).invoke([
                ("system", SYSTEM_PROMPT),
                ("human", human_text),
            ])

        human_input = f"{context_block}当前用户描述：{query}"
        detection = _detect(human_input)
        symptoms = [s.strip() for s in detection.symptoms if s.strip()]

        # LLM 偶发漏提（thinking 模式过保守）：能走到这里的 query 意图必为
        # appointment/medication，空结果先按 few-shot 加强重试一次再接受
        if not symptoms:
            logger.info("症状检测为空，few-shot 加强重试")
            retry_text = (
                f"{context_block}请从下面这段话里提取全部**明确的身体症状**"
                f"（如疼痛、发热、出血、麻木等），没有症状才返回空列表：\n{query}"
            )
            detection = _detect(retry_text)
            symptoms = [s.strip() for s in detection.symptoms if s.strip()]

        severity = detection.severity
        if detection.has_emergency_keyword or keyword_hit:
            severity = Severity.EMERGENCY.value
    except Exception as exc:  # noqa: BLE001
        logger.warning("症状检测 LLM 失败: %s", exc)
        symptoms, severity = [], Severity.LOW.value

    logger.info("症状检测: symptoms=%s severity=%s", symptoms, severity)
    return {"symptoms": symptoms, "severity": severity}
