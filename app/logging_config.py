import logging
import os
import sys
from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Redirect stdlib logging records into loguru so both ecosystems share sinks.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _resolve_level_name() -> str:
    """
    Determine the desired log level from env variables.
    Prefers APP_LOG_LEVEL, then LOG_LEVEL, then defaults to INFO.
    """
    for env_key in ("APP_LOG_LEVEL", "LOG_LEVEL"):
        value = os.getenv(env_key)
        if value and value.strip():
            return value.strip()
    return "INFO"


def _parse_level(level_name: str) -> int:
    normalized = level_name.upper()
    return getattr(logging, normalized, logging.INFO)


def configure_logging(default_handler: bool = True) -> None:
    """
    Bring uvicorn + application loggers under loguru with unified level/format.
    Invoked during app import so LOG_LEVEL changes take effect immediately.
    """
    level_name = _resolve_level_name()
    level = _parse_level(level_name)

    # Route stdlib logging to loguru and reset existing handlers.
    intercept = InterceptHandler()
    logging.basicConfig(handlers=[intercept], level=level, force=True)

    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "stupidsimplerag",
        "app",
    ):
        logging.getLogger(name).handlers = [intercept]
        logging.getLogger(name).propagate = False
        logging.getLogger(name).setLevel(level)

    if default_handler:
        logger.remove()
        logger.add(
            sys.stdout,
            level=level_name,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
        )

    # Align uvicorn CLI default
    os.environ.setdefault("UVICORN_LOG_LEVEL", level_name.lower())
