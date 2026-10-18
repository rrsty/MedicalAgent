"""就医指导 Agent：LLM 生成 + 预设清单兜底 + 建议缓存（接入层语义缓存优化）。

相同（症状, 科室）组合的建议直接命中 Redis 缓存（TTL 24h），省一次 LLM 调用；
首查正常走 LLM 并回填缓存。
"""
import hashlib
import json

from src.agents.state import AgentState
from src.common.llm import get_llm
from src.common.utils import parse_llm_json
from src.common.logger import setup_logger
from src.history.store import cache_get, cache_set

logger = setup_logger("agent.previsit", "agents.log")

DEFAULT_ADVICE = [
    "就诊前记录症状出现的时间、频率和加重/缓解因素",
    "携带既往病历、检查报告和正在使用的药物清单",
    "清淡饮食，保证睡眠，避免剧烈运动",
    "提前预约挂号，带好身份证和医保卡",
    "不要自行服用止痛药或抗生素，以免掩盖症状",
]

SYSTEM_PROMPT = """你是一个就医顾问。根据患者症状与推荐科室，提供就诊前的注意事项和建议。

要求：
1. 输出 3-5 条实用建议（JSON 数组）
2. 包括：就诊前准备、症状观察要点、饮食生活注意
3. 不要包含诊断和用药建议
4. 输出 JSON：{"advice": ["建议1", "建议2", ...]}"""


def _advice_cache_key(symptoms: list[str], departments: list[str]) -> str:
    raw = "|".join(sorted(symptoms)) + "##" + "|".join(sorted(departments))
    return "advice:" + hashlib.md5(raw.encode("utf-8")).hexdigest()


def pre_visit_advisor(state: AgentState) -> dict:
    symptoms = state.get("symptoms", [])
    departments = state.get("departments", [])

    cache_key = _advice_cache_key(symptoms, departments)
    cached = cache_get(cache_key)
    if cached:
        try:
            return {"advice": json.loads(cached)}
        except Exception:  # noqa: BLE001
            pass

    try:

        resp = get_llm(0.5).invoke([
            ("system", SYSTEM_PROMPT),
            (
                "human",
                f"症状：{('、'.join(symptoms)) or '未描述'}\n"
                f"推荐科室：{('、'.join(departments)) or '待定'}",
            ),
        ])
        data = parse_llm_json(resp.content) or {}
        advice = [str(a).strip() for a in data.get("advice", []) if str(a).strip()][:5]
        if not advice:
            raise ValueError("LLM 返回空建议")
        cache_set(cache_key, json.dumps(advice, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        logger.warning("就医指导 LLM 失败，使用预设清单兜底: %s", exc)
        advice = list(DEFAULT_ADVICE)

    logger.info("就医指导: %d 条建议", len(advice))
    return {"advice": advice}
