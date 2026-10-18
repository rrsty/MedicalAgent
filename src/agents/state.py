"""LangGraph 全局状态 —— 所有 Agent 共享的「黑板」（R1：意图类型只在此定义）。

列表字段分两类：
- Annotated[list, operator.add]：多节点写入、需要**累加**（免责声明/警告/检索结果/实体链接）；
- 普通字段：单节点写入、直接覆盖（意图/症状/科室/最终回答）。
"""
import operator
from enum import Enum
from typing import Annotated, TypedDict


class IntentType(str, Enum):
    """意图类型（R1 单一事实来源，路由与评估统一 import）。"""

    APPOINTMENT = "appointment"
    MEDICATION = "medication"
    KNOWLEDGE = "knowledge"
    EMERGENCY = "emergency"
    GENERAL = "general"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EMERGENCY = "emergency"


class AgentState(TypedDict, total=False):
    # 输入
    query: str
    history: list  # 多轮对话历史 [{role, content}]，API 层注入（Redis 热缓存 → MySQL 回填）

    # 意图识别 Agent 输出
    intent: str
    confidence: float

    # 症状检测 Agent 输出
    symptoms: list[str]
    severity: str

    # 科室推荐 Agent 输出
    departments: list[str]
    department_sources: list[str]  # 每个科室的推荐依据（graph/llm/default）

    # 用药建议 Agent 输出
    medications: list[dict]  # [{name, purpose, side_effects, reminder}]

    # 医学知识 Agent 输出（grounded-ReAct）
    knowledge_answer: str

    # 就医指导 Agent 输出
    advice: list[str]

    # 累加字段（Annotated + operator.add）
    disclaimers: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    retrieved_entities: Annotated[list[dict], operator.add]
    retrieved_contexts: Annotated[list[str], operator.add]  # 供 RAGAS
    linked_entities: Annotated[list[dict], operator.add]

    # 追问标记（信息不足时置位，照常走到答案融合输出追问话术）
    needs_clarification: bool
    clarification_question: str

    # 输出
    final_answer: str

    # 调试观测（Agent 执行链路 / 工具调用轨迹）
    trace: Annotated[list[dict], operator.add]
