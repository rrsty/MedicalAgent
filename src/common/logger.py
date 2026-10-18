"""统一日志配置：同时输出控制台与文件（logs/ 目录），格式：时间 + 级别 + 模块名 + 消息。"""
import logging
from logging.handlers import RotatingFileHandler

from config.paths import LOGS_DIR
from config.settings import settings

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_CONFIGURED = set()


def setup_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """获取统一格式的 logger；重复调用返回同一实例（幂等）。"""
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOGS_DIR / log_file, maxBytes=10 * 1024 * 1024,
            backupCount=5, encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    _CONFIGURED.add(name)
    return logger
