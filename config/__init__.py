"""config 包：全局配置与路径（项目「地基」，所有模块统一从这里 import）。"""

from config.paths import PROJECT_ROOT  # noqa: F401
from config.settings import settings  # noqa: F401

__all__ = ["PROJECT_ROOT", "settings"]
