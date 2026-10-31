"""API 请求/响应模型 —— R3 前后端字段契约（前后端按同一张表开发，字段只在此定义）。

契约表（前端 src/api/index.js 与 ChatView 按此消费）：
  ConsultationRequest  : query, session_id
  ConsultationResponse : answer, intent, symptoms, severity, departments, medications,
                         advice, disclaimers, warnings, linked_entities, trace,
                         duration_ms, session_id
  HealthResponse       : status, neo4j_connected, vector_index_loaded, agent_system_ready, version
  StatsResponse        : total_nodes, total_relations, graph_density, node_stats,
                         vector_index_size, agent_ready
  ConversationSummary  : id, title, created_at, updated_at
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ConsultationRequest(BaseModel):
    query: str = Field(..., description="用户提问", min_length=1, max_length=1000)
    # 必填（无默认值）：医疗对话不允许匿名客户端共享 "default" 会话（隐私红线）
    session_id: str = Field(
        ..., description="会话ID（== conversation_id，客户端生成 uuid）",
        min_length=6, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$",
    )
    request_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None


class MedicationItem(BaseModel):
    name: str = Field(..., description="药物名称")
    purpose: str = Field(default="", description="用途")
    side_effects: str = Field(default="", description="常见副作用")
    reminder: str = Field(default="", description="用药提醒")


class LinkedEntity(BaseModel):
    input_entity: str = Field(..., description="用户原始表述")
    matched_entity: str = Field(..., description="图谱规范实体")
    type: str = Field(default="", description="实体类型")
    similarity: float = Field(default=0.0, description="相似度")


class ConsultationResponse(BaseModel):
    answer: str = Field(..., description="最终回答（Markdown）")
    intent: str = Field(default="general", description="意图类型")
    symptoms: list[str] = Field(default_factory=list, description="检测到的症状")
    severity: str = Field(default="", description="严重程度")
    departments: list[str] = Field(default_factory=list, description="推荐科室")
    medications: list[MedicationItem] = Field(default_factory=list, description="药物信息")
    advice: list[str] = Field(default_factory=list, description="就医建议")
    disclaimers: list[str] = Field(default_factory=list, description="免责声明")
    warnings: list[str] = Field(default_factory=list, description="急症警告")
    linked_entities: list[LinkedEntity] = Field(default_factory=list, description="实体链接（知识定位）")
    trace: list[dict] = Field(default_factory=list, description="Agent 执行链路（调试观测）")
    duration_ms: int = Field(default=0, description="处理耗时（毫秒）")
    session_id: str = Field(default="default", description="会话ID")
    needs_clarification: Optional[bool] = False
    retrieved_contexts: Optional[List[str]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = Field(default="unknown", description="服务状态")
    neo4j_connected: bool = Field(default=False, description="Neo4j 连接状态")
    vector_index_loaded: bool = Field(default=False, description="向量索引加载状态")
    agent_system_ready: bool = Field(default=False, description="Agent 系统就绪状态")
    version: str = Field(default="1.0.0", description="API 版本")


class StatsResponse(BaseModel):
    total_nodes: int = Field(default=0, description="图谱节点总数")
    total_relations: int = Field(default=0, description="图谱关系总数")
    graph_density: float = Field(default=0.0, description="图谱密度（关系/节点）")
    node_stats: dict = Field(default_factory=dict, description="节点按类型分布")
    vector_index_size: int = Field(default=0, description="向量索引规模")
    agent_ready: bool = Field(default=False, description="Agent 系统就绪")


class ConversationSummary(BaseModel):
    id: str = Field(..., description="会话ID")
    title: str = Field(default="新对话", description="标题（首问截断24字）")
    created_at: str = Field(default="", description="创建时间")
    updated_at: str = Field(default="", description="更新时间")


class MessageItem(BaseModel):
    role: str = Field(..., description="user / assistant")
    content: str = Field(..., description="消息内容")
    intent: str = Field(default="", description="意图")
    created_at: str = Field(default="", description="时间")
