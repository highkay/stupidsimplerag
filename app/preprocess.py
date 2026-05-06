from __future__ import annotations

from dataclasses import dataclass
import re


HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*$")
QA_PREFIX_RE = re.compile(r"^\s*(?:Q[:：]|A[:：]|问[:：]|答[:：])", re.IGNORECASE)
LIST_BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
APPENDIX_HINTS = (
    "相关标的",
    "受益标的",
    "股票池",
    "建议关注",
    "重点公司",
    "附录",
    "相关公司",
    "受益公司",
)


@dataclass(frozen=True)
class PreprocessedBlock:
    text: str
    section_type: str
    section_order: int
    block_index: int
    heading_path: str | None = None
    is_list_zone: bool = False
    is_qa_zone: bool = False


LIST_SECTION_TYPES = frozenset({"appendix_list", "appendix_table"})
BODY_SECTION_TYPES = frozenset({"title", "body"})


def _normalize_text(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _contains_appendix_hint(text: str) -> bool:
    haystack = text or ""
    return any(keyword in haystack for keyword in APPENDIX_HINTS)


def _looks_like_table_block(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    table_lines = [line for line in lines if TABLE_ROW_RE.match(line)]
    if len(table_lines) >= 2:
        return True
    return any(line.count("|") >= 2 for line in lines[:3]) and len(lines) >= 2


def _looks_like_short_delimited_list(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    sentence_punctuation = ("。", "！", "？")
    if any(any(mark in line for mark in sentence_punctuation) for line in lines):
        return False
    max_len = max((len(line) for line in lines), default=0)
    short_lines = [line for line in lines if len(line) <= 80]
    if len(short_lines) >= 3 and len(short_lines) == len(lines):
        return max_len <= 24
    joined = " ".join(lines)
    delimiters = (
        joined.count("、")
        + joined.count(",")
        + joined.count("，")
        + joined.count("；")
        + joined.count(";")
        + joined.count("/")
    )
    return delimiters >= 2 and max_len <= 36


def _looks_like_list_block(text: str, heading_path: str | None) -> bool:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    bullet_lines = [line for line in lines if LIST_BULLET_RE.match(line)]
    appendix_hint = _contains_appendix_hint(heading_path or "") or _contains_appendix_hint(lines[0])
    if appendix_hint and (bullet_lines or _looks_like_short_delimited_list(text)):
        return True
    return len(bullet_lines) >= 3 or _looks_like_short_delimited_list(text)


def _looks_like_qa_block(text: str, heading_path: str | None) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    if heading_path and any(token in heading_path for token in ("问答", "QA", "Q&A")):
        return True
    qa_lines = [line for line in lines if QA_PREFIX_RE.match(line)]
    return len(qa_lines) >= 1 and len(qa_lines) >= max(1, len(lines) // 2)


def classify_document_block(
    text: str,
    *,
    heading_path: str | None = None,
    force_title: bool = False,
) -> tuple[str, bool, bool]:
    stripped = (text or "").strip()
    if not stripped:
        return "body", False, False
    if force_title:
        return "title", False, False
    if _looks_like_qa_block(stripped, heading_path):
        return "qa", False, True
    if _looks_like_table_block(stripped):
        return "appendix_table", True, False
    if _looks_like_list_block(stripped, heading_path):
        return "appendix_list", True, False
    return "body", False, False


def summarize_chunk_blocks(
    blocks: list[PreprocessedBlock],
) -> tuple[str, str | None, bool, bool]:
    if not blocks:
        return "body", None, False, False

    bucket_lengths = {"body": 0, "qa": 0, "list": 0}
    bucket_blocks = {"body": [], "qa": [], "list": []}
    fallback_heading_path = next((block.heading_path for block in blocks if block.heading_path), None)

    for block in blocks:
        text_len = len((block.text or "").strip())
        if block.section_type in LIST_SECTION_TYPES:
            bucket = "list"
        elif block.section_type == "qa":
            bucket = "qa"
        else:
            bucket = "body"
        bucket_lengths[bucket] += text_len
        bucket_blocks[bucket].append(block)

    if bucket_lengths["body"] >= max(bucket_lengths["qa"], bucket_lengths["list"]):
        bucket = "body"
    elif bucket_lengths["qa"] >= bucket_lengths["list"]:
        bucket = "qa"
    else:
        bucket = "list"

    candidates = bucket_blocks[bucket] or blocks
    dominant_block = max(
        candidates,
        key=lambda block: (
            len((block.text or "").strip()),
            -block.section_order,
            -block.block_index,
        ),
    )
    heading_path = dominant_block.heading_path or fallback_heading_path

    if bucket == "body":
        section_type = dominant_block.section_type if dominant_block.section_type in BODY_SECTION_TYPES else "body"
    elif bucket == "qa":
        section_type = "qa"
    else:
        section_type = dominant_block.section_type if dominant_block.section_type in LIST_SECTION_TYPES else "appendix_list"

    return section_type, heading_path, bucket == "list", bucket == "qa"


def split_document_blocks(content: str, *, allow_title: bool = True) -> list[PreprocessedBlock]:
    text = _normalize_text(content)
    if not text.strip():
        return []

    blocks: list[PreprocessedBlock] = []
    heading_stack: list[str] = []
    pending_heading_line: str | None = None
    current_heading_path: str | None = None
    buffer: list[str] = []
    emitted_title = False

    def emit_block(raw_text: str, *, force_title: bool = False) -> None:
        stripped = raw_text.strip()
        if not stripped:
            return
        section_type, is_list_zone, is_qa_zone = classify_document_block(
            stripped,
            heading_path=current_heading_path,
            force_title=force_title,
        )
        blocks.append(
            PreprocessedBlock(
                text=stripped,
                section_type=section_type,
                section_order=len(blocks),
                block_index=len(blocks),
                heading_path=current_heading_path,
                is_list_zone=is_list_zone,
                is_qa_zone=is_qa_zone,
            )
        )

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        emit_block("\n".join(buffer))
        buffer = []

    for raw_line in text.split("\n"):
        heading_match = HEADING_RE.match(raw_line)
        if heading_match:
            flush_buffer()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            if heading_text:
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading_text)
                current_heading_path = " / ".join(heading_stack)
                if allow_title and not emitted_title and not blocks:
                    emit_block(raw_line.strip(), force_title=True)
                    emitted_title = True
                else:
                    pending_heading_line = raw_line.rstrip()
            continue

        if not raw_line.strip():
            flush_buffer()
            continue

        if pending_heading_line:
            buffer.append(pending_heading_line)
            pending_heading_line = None
        buffer.append(raw_line.rstrip())

    flush_buffer()
    if pending_heading_line:
        emit_block(pending_heading_line)

    if not blocks and text.strip():
        emit_block(text)
    return blocks
