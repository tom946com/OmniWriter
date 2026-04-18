"""
日志配置模块

本模块提供基于 structlog 的结构化日志配置，支持环境相关的格式化器和处理器。
- 开发环境：使用友好的控制台输出格式，便于调试
- 生产环境：使用结构化的 JSON 格式，便于日志收集和分析

主要功能：
1. 上下文管理：支持请求级别的上下文变量绑定
2. 多输出目标：同时支持控制台输出和文件输出
3. JSONL 文件日志：按日期分割的 JSON Lines 格式日志文件
4. 环境感知：根据运行环境自动选择合适的日志格式
"""

import json
import logging
import sys
import os
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import structlog

# 确保日志目录存在，如果不存在则自动创建（包括父目录）
log_dir = Path(os.getenv("LOG_DIR"))
log_dir.mkdir(parents=True, exist_ok=True)

def get_log_file_path() -> Path:
    """Get the current log file path based on date and environment.

    Returns:
        Path: The path to the log file
    """
    log_dir = Path(os.getenv("LOG_DIR", "./logs"))
    # 2. Path对象支持 / 拼接路径，完美解决报错
    return log_dir / f"log-{datetime.now().strftime('%Y-%m-%d')}.jsonl"

class JsonlFileHandler(logging.Handler):
    """
    自定义日志处理器：将日志以 JSONL 格式写入按日期分割的文件。

    JSONL (JSON Lines) 格式特点：
    - 每行是一个独立的 JSON 对象
    - 便于流式处理和逐行解析
    - 适合日志收集系统（如 ELK、Loki）处理

    继承自 logging.Handler，可以与标准库 logging 无缝集成。
    """

    def __init__(self, file_path: Path):
        """
        初始化 JSONL 文件处理器。

        Args:
            file_path: 日志文件的路径
        """
        super().__init__()
        self.file_path = file_path

    def emit(self, record: logging.LogRecord) -> None:
        """
        将日志记录写入 JSONL 文件。

        此方法由 logging 框架自动调用，处理每条日志记录。

        日志条目格式：
        {
            "timestamp": "2024-01-15T10:30:00.123456",
            "level": "INFO",
            "message": "用户登录成功",
            "module": "auth",
            "function": "login",
            "filename": "/app/src/auth.py",
            "line": 42,
            "environment": "production"
        }

        Args:
            record: 标准库的日志记录对象
        """
        try:
            # 构建日志条目字典
            log_entry = {
                # "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                # "module": record.module,
                "function": record.funcName,
                # "filename": record.pathname,
                "line": record.lineno
            }
            # 如果有额外信息，合并到日志条目中
            if hasattr(record, "extra"):
                log_entry.update(record.extra)

            # 以追加模式写入文件，每行一个 JSON 对象
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            # 发生错误时使用标准错误处理
            self.handleError(record)

    def close(self) -> None:
        """关闭处理器，释放资源。"""
        super().close()


def clean_event_dict(logger, method_name, event_dict):
    """
    清理事件字典，只保留需要的字段：level, event, func_name
    """
    new_dict = {}
    if 'level' in event_dict:
        new_dict['level'] = event_dict['level']
    if 'event' in event_dict:
        new_dict['event'] = event_dict['event']
    if 'func_name' in event_dict:
        new_dict['func_name'] = event_dict['func_name']
    return new_dict


def get_structlog_processors(include_file_info: bool = True) -> List[Any]:
    """
    获取 structlog 处理器链。

    处理器是 structlog 的核心概念，每个处理器按顺序处理日志事件，
    可以添加、修改或过滤日志字段。

    处理器执行顺序（从上到下）：
    1. filter_by_level - 根据日志级别过滤
    2. add_logger_name - 添加日志器名称
    3. add_log_level - 添加日志级别
    4. PositionalArgumentsFormatter - 格式化位置参数
    5. TimeStamper - 添加时间戳
    6. StackInfoRenderer - 渲染堆栈信息
    7. format_exc_info - 格式化异常信息
    8. UnicodeDecoder - 解码 Unicode
    9. add_context_to_event_dict - 添加上下文变量
    10. CallsiteParameterAdder - 添加调用位置信息（可选）
    11. 环境信息添加

    Args:
        include_file_info: 是否在日志中包含文件信息（文件名、行号、函数名等）
                          开发环境建议开启，生产环境可以关闭以减少日志量

    Returns:
        List[Any]: structlog 处理器列表
    """
    # 基础处理器列表：这些处理器对所有环境都是必需的
    processors = [
        # 根据日志级别过滤，低于设定级别的日志不会被处理
        structlog.stdlib.filter_by_level,
        # 添加日志级别字段（如 "INFO", "ERROR"）
        structlog.stdlib.add_log_level,
        # 格式化位置参数，支持 logger.info("Hello %s", name) 这种写法
        structlog.stdlib.PositionalArgumentsFormatter(),
        # 当日志包含堆栈信息时，渲染为可读格式
        structlog.processors.StackInfoRenderer(),
        # 格式化异常信息，将 exc_info 转换为字符串
        structlog.processors.format_exc_info,
        # 确保 Unicode 字符正确解码
        structlog.processors.UnicodeDecoder(),
    ]

    # 添加函数名处理器
    processors.append(
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FUNC_NAME,  # 函数名
            }
        )
    )
    
    # 添加清理处理器，只保留需要的字段
    processors.append(clean_event_dict)
    
    return processors


def setup_logging() -> None:
    """
    配置 structlog 日志系统。

    根据运行环境自动选择合适的日志格式：
    - 开发环境（LOG_FORMAT="console"）：使用 ConsoleRenderer，输出彩色、格式化的控制台日志
    - 生产环境（LOG_FORMAT="json"）：使用 JSONRenderer，输出结构化的 JSON 日志

    日志输出目标：
    1. 控制台（stdout）：实时查看日志
    2. 文件（JSONL）：持久化存储，便于后续分析

    日志级别：
    - DEBUG 模式开启时：DEBUG 级别及以上
    - DEBUG 模式关闭时：INFO 级别及以上
    """
    # 根据 DEBUG 设置确定日志级别
    log_level = logging.DEBUG if os.getenv("LOG_LEVEL") == "DEBUG" else logging.INFO

    # 创建文件处理器：将日志写入 JSONL 文件
    file_handler = JsonlFileHandler(get_log_file_path())
    file_handler.setLevel("ERROR")

    # 创建控制台处理器：将日志输出到标准输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # 获取共享处理器
    # 开发和测试环境包含详细的文件信息，生产环境不包含以减少日志量
    shared_processors = get_structlog_processors(
        include_file_info=True
    )

    # 配置标准库 logging
    logging.basicConfig(
        format="%(message)s",  # 消息格式由 structlog 处理
        level=log_level,
        handlers=[file_handler, console_handler],  # 同时输出到文件和控制台
    )


    # 开发环境：使用 ConsoleRenderer，输出彩色、格式化的控制台日志
    # 便于开发调试，可读性强
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.dev.ConsoleRenderer(),  # 彩色控制台输出
        ],
        wrapper_class=structlog.stdlib.BoundLogger,  # 使用标准库兼容的日志器
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,  # 缓存日志器以提高性能
    )



# 模块加载时初始化日志系统
setup_logging()

# 创建全局日志器实例
# 使用方式：from src.utils.logs import logger; logger.info("message", key="value")
logger = structlog.get_logger()

# 记录日志系统初始化信息
log_level_name = "DEBUG" if os.getenv("LOG_LEVEL") == "DEBUG" else "INFO"
logger.info(
    "logging_initialized",
    log_level=log_level_name,
    log_format="console",
    debug="DEBUG"
)
