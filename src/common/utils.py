"""通用工具函数：文本清理 / 错误重试装饰器 / 计时装饰器 / JSON 解析兜底。"""
import functools
import json
import logging
import re
import time
from pathlib import Path

from config.paths import PROJECT_ROOT

logger = logging.getLogger("common.utils")


def clean_text(text: str | None) -> str:
    """去除多余空格、换行与特殊不可见字符。"""
    if not text:
        return ""
    text = re.sub(r"[\u200b\u3000\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """错误重试装饰器（指数退避）。"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt < max_retries:
                        wait = delay * (2 ** (attempt - 1))
                        logger.warning(
                            "%s 第 %d/%d 次失败: %s，%.1fs 后重试",
                            func.__name__, attempt, max_retries, exc, wait,
                        )
                        time.sleep(wait)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


def timer(func):
    """函数计时装饰器（毫秒）。"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            logger.info("%s 耗时 %.0f ms", func.__name__, (time.perf_counter() - t0) * 1000)
    return wrapper


def _confine_to_project(file_path: str | Path) -> Path:
    """规范化路径并限制在项目目录内（防路径穿越，拒绝 ../ 越界）。"""
    resolved: Path = Path(file_path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"路径越界（必须在项目目录内）: {resolved}")
    return resolved


def extract_json_block(text: str) -> str | None:
    """从 LLM 输出中提取首个 JSON 对象文本（剥离 ```json 围栏，截取首个 { 到末个 }）。

    供 vLLM 微调模型「裸输出 + 手工解析」链路使用。
    """
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


def parse_llm_json(text: str) -> dict | None:
    """解析 LLM 输出中的 JSON；失败返回 None（绝不抛异常打断主流程）。"""
    block = extract_json_block(text)
    if block is None:
        return None
    try:
        obj = json.loads(block)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError as exc:
        logger.warning("JSON 解析失败: %s", exc)
        return None


def read_json(file_path: str | Path):
    """读取项目目录内的 JSON 文件（路径经规范化与越界校验）。"""
    target = _confine_to_project(file_path)
    return json.loads(target.read_text(encoding="utf-8"))


def write_json(file_path: str | Path, data, ensure_ascii: bool = False, indent: int = 2) -> None:
    """写入 JSON 到项目目录内（路径经规范化与越界校验）。"""
    target = _confine_to_project(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=ensure_ascii, indent=indent), encoding="utf-8")


def coerce_str_list(value) -> list[str]:
    """把 LLM 返回的字符串列表归一化：容忍 [{"name": "..."}] 对象项、单项字符串、None。

    DeepSeek 偶发把 symptoms/side_effects 等返回成对象数组而非纯字符串，
    在 Pydantic field_validator(mode="before") 里调用本函数做统一收敛。
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("content") or "").strip()
            if name:
                out.append(name)
        elif str(item).strip():
            out.append(str(item).strip())
    return out


def truncate(text: str | None, max_len: int, suffix: str = "...") -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + suffix
