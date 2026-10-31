"""全局配置：从 .env 读取，类型安全（pydantic_settings），导出全局单例 settings。

所有模块统一 `from config.settings import settings`，禁止散落读取 os.environ（R1）。
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ===== LLM（主模型：全部在线 Agent）=====
    model_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="LLM API 地址（OpenAI 兼容模式）",
    )
    model_api_key: str = Field(default="", description="LLM API Key")
    model_name: str = Field(default="qwen-plus", description="主模型名称")
    model_max_output_tokens: int = Field(
        default=2048,
        ge=128,
        le=8192,
        description="主模型单次最大输出 Token，防止失控生成污染下一轮上下文",
    )

    # ===== 抽取模型（可切换到自部署 vLLM 微调模型；base_url 留空=回退主 LLM）=====
    extraction_model_base_url: str = Field(default="", description="抽取专用 LLM 地址；留空回退主模型")
    extraction_model_api_key: str = Field(default="", description="抽取专用 LLM API Key")
    extraction_model_name: str = Field(default="", description="抽取专用模型名（须与 vLLM served-model-name 一致）")

    # ===== Neo4j =====
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j Bolt 地址")
    neo4j_user: str = Field(default="neo4j", description="Neo4j 用户名")
    neo4j_password: str = Field(default="12345678", description="Neo4j 密码")

    # ===== Redis / MySQL（对话历史：热缓存 + 冷存储）=====
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis 连接地址")
    mysql_host: str = Field(default="localhost", description="MySQL 主机")
    mysql_port: int = Field(default=3306, description="MySQL 端口")
    mysql_user: str = Field(default="root", description="MySQL 用户名")
    mysql_password: str = Field(default="123456", description="MySQL 密码")
    mysql_database: str = Field(default="smc_history", description="MySQL 库名（对话历史）")

    # ===== 向量检索（R4 索引契约：BGE-M3 = 1024 维，IndexIVFFlat，METRIC_INNER_PRODUCT）=====
    embedding_model_path: str = Field(default="models/bge-m3", description="BGE-M3 本地模型路径")
    embedding_dim: int = Field(default=1024, description="嵌入向量维度（必须与模型一致）")
    faiss_nlist: int = Field(default=100, description="IVF 聚类中心数")
    faiss_nprobe: int = Field(default=10, description="IVF 搜索探测簇数")
    retrieval_top_k: int = Field(default=5, description="语义检索 Top-K")
    retrieval_score_threshold: float = Field(default=0.5, description="语义检索相似度阈值（低于过滤）")
    entity_link_threshold: float = Field(
        default=0.80,
        description="实体链接高精度阈值（设计默认 0.85；基于随项目 BGE-M3 实测口语词相似度 0.82~0.87，校准为 0.80）",
    )

    # ===== Agent 行为开关 =====
    llm_disable_thinking: bool = Field(
        default=True,
        description="关闭 DeepSeek thinking 模式（实测单次调用 2.5x 提速；抽取/分类/融合类任务无需思考链）",
    )
    enable_react_layer: bool = Field(default=True, description="医学知识 Agent 可选 ReAct 多跳层")
    text2cypher_enabled: bool = Field(default=True, description="Text2Cypher 只读图谱查询工具开关")
    api_timeout_seconds: int = Field(
        default=180,
        description="单次问诊处理超时（秒）。ReAct 多跳链路耗时可达分钟级，30s 会误杀合法请求",
    )
    max_concurrent_requests: int = Field(default=8, description="并发问诊请求上限（Semaphore）")

    # ===== 对话历史 =====
    history_window: int = Field(default=12, description="短期记忆保留最近 N 轮")
    history_ttl_hours: int = Field(default=24, description="Redis 热缓存 TTL（小时）")

    # ===== 日志 =====
    log_level: str = Field(default="INFO", description="日志级别")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
