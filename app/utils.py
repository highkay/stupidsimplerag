import datetime
import os
import re

from dateutil import parser
from llama_index.core.schema import NodeWithScore

from app.time_utils import datetime_from_epoch, today_in_app_tz


def extract_date_from_filename(filename: str) -> str | None:
    """Extract YYYY-MM-DD from complex filenames using timestamp + pattern fallbacks."""
    base_name = os.path.basename(filename)
    today = today_in_app_tz()
    current_year = today.year

    ts_match = re.search(r"(^|[^0-9])(\d{10}|\d{13})([^0-9]|$)", base_name)
    if ts_match:
        try:
            ts = float(ts_match.group(2))
            if ts > 3_000_000_000:
                ts /= 1000
            return datetime_from_epoch(ts).strftime("%Y-%m-%d")
        except Exception:
            pass

    date_match = re.search(r"(\d{4})[-./_]?(\d{2})[-./_]?(\d{2})", base_name)
    if date_match:
        try:
            d_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
            return parser.parse(d_str).strftime("%Y-%m-%d")
        except Exception:
            pass

    # Fallback: MMDD -> fill current year
    mmdd_match = re.search(r"(?<!\d)(\d{2})(\d{2})(?!\d)", base_name)
    if mmdd_match:
        try:
            month = int(mmdd_match.group(1))
            day = int(mmdd_match.group(2))
            dt = datetime.date(current_year, month, day)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Fallback: two-digit year -> assume current century, month/day=01
    yy_match = re.search(r"(?<!\d)(\d{2})(?!\d)", base_name)
    if yy_match:
        try:
            year_suffix = int(yy_match.group(1))
            century = (current_year // 100) * 100
            year = century + year_suffix
            dt = datetime.date(year, 1, 1)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    return None


def get_node_metadata(node: NodeWithScore) -> dict:
    if hasattr(node, "metadata") and isinstance(node.metadata, dict):
        return node.metadata
    inner = getattr(node, "node", None)
    if inner is not None and hasattr(inner, "metadata"):
        return inner.metadata
    return {}
