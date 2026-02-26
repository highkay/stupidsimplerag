import datetime
import logging
import os
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


logger = logging.getLogger(__name__)

DEFAULT_TIMEZONE = "Asia/Shanghai"


def _resolve_timezone_name() -> str:
    for key in ("APP_TIMEZONE", "TZ"):
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return DEFAULT_TIMEZONE


def _resolve_timezone() -> datetime.tzinfo:
    tz_name = _resolve_timezone_name()
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        logger.warning(
            "Unknown timezone %r, fallback to UTC. Set APP_TIMEZONE or TZ to a valid IANA timezone.",
            tz_name,
        )
        return datetime.timezone.utc


APP_TIMEZONE = _resolve_timezone()


def now_in_app_tz() -> datetime.datetime:
    return datetime.datetime.now(APP_TIMEZONE)


def today_in_app_tz() -> datetime.date:
    return now_in_app_tz().date()


def datetime_from_epoch(ts: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(ts, tz=APP_TIMEZONE)
