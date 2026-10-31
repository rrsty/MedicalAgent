"""LLM 客户端封装：单例模式，避免重复创建连接（节省时间与费用）。

- get_llm()：主模型，全部在线 Agent 使用
- get_extract_llm()：抽取专用——EXTRACTION_MODEL_BASE_URL 非空时指向自部署
  vLLM 微调模型，留空回退 get_llm()；独立缓存，日志标明当前走「云端」还是「本地 vLLM」。
"""
from functools import lru_cache

from langchain_openai import ChatOpenAI

from config.settings import settings
from src.common.logger import setup_logger

logger = setup_logger("common.llm", "llm.log")


@lru_cache
def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """返回主 LLM 实例（按 temperature 分别缓存）。

    extra_body 在构造参数层传入（而非 .bind()），保证 .with_structured_output()
    与 .bind_tools() 仍可直接调用；关闭 thinking 实测单次调用提速约 2.5 倍。
    """
    kwargs = {}
    if settings.llm_disable_thinking:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(
        api_key=settings.model_api_key,
        base_url=settings.model_base_url,
        model=settings.model_name,
        temperature=temperature,
        timeout=120,
        max_retries=2,
        max_tokens=settings.model_max_output_tokens,
        **kwargs,
    )


@lru_cache
def _get_extract_llm_cached(
    base_url: str, api_key: str, model_name: str, temperature: float
) -> ChatOpenAI:
    # max_tokens 作为失控生成的保险丝；温度 0 最稳（实测：温度非零易失控烧穿 max_tokens）
    return ChatOpenAI(
        api_key=api_key or "EMPTY",
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        timeout=90,
        max_retries=1,
        max_tokens=2048,
    )


def get_extract_llm(temperature: float = 0.1) -> ChatOpenAI:
    """抽取专用 LLM：EXTRACTION_MODEL_BASE_URL 非空走本地 vLLM 微调模型，否则回退主模型。"""
    if settings.extraction_model_base_url:
        logger.info("Extraction LLM: 抽取使用本地 vLLM 微调模型 (%s)", settings.extraction_model_name)
        return _get_extract_llm_cached(
            settings.extraction_model_base_url,
            settings.extraction_model_api_key,
            settings.extraction_model_name,
            temperature,
        )
    logger.info("Extraction LLM: 抽取使用云端主模型 (%s)", settings.model_name)
    return get_llm(temperature)
