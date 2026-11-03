"""
Централизованный модуль логирования для отладки голосового ассистента
"""
import os
import sys
import logging
from pathlib import Path

LOG_FILE = Path(__file__).parent / "assistant.log"

# Настройка файла логов (UTF-8, подробный формат)
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

logger = logging.getLogger("Assistant")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Предотвращаем дублирование логов в стандартный вывод консоли
logger.propagate = False

def log_debug(msg: str):
    logger.debug(msg)

def log_info(msg: str):
    logger.info(msg)

def log_warning(msg: str):
    logger.warning(msg)

def log_error(msg: str, exc_info: bool = True):
    logger.error(msg, exc_info=exc_info)
