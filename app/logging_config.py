import logging
import os


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
    Bring uvicorn + application loggers under the same configurable level.
    Invoked during app import so LOG_LEVEL changes take effect immediately.
    """
    level_name = _resolve_level_name()
    level = _parse_level(level_name)

    if default_handler:
        # If another handler already configured (e.g. uvicorn), basicConfig is a no-op.
        logging.basicConfig(level=level)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "stupidsimplerag",
        "app",
    ):
        logging.getLogger(name).setLevel(level)

    # Align uvicorn CLI default
    os.environ.setdefault("UVICORN_LOG_LEVEL", level_name.lower())
