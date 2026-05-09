import asyncio
from collections import deque
import hashlib
import json
import logging
import os
import re
import time
from functools import lru_cache
from typing import Any, List

from chonkie import TokenChunker
from llama_index.core.schema import TextNode

from app.embedding_budget import (
    count_embedding_input_tokens,
    get_embedding_input_token_budget,
)
from app.models import LLMAnalysis
from app.openai_utils import build_llm
from app.preprocess import BODY_SECTION_TYPES, LIST_SECTION_TYPES, PreprocessedBlock, split_document_blocks
from app.utils import extract_date_from_filename

logger = logging.getLogger(__name__)
def compute_doc_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


_llm_context_window = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))
_llm_retry_attempts = int(os.getenv("LLM_MAX_RETRIES", "3"))
_llm_retry_backoff = float(os.getenv("LLM_RETRY_BACKOFF", "1.5"))
_llm_concurrency = int(os.getenv("LLM_CONCURRENCY", "10"))
_llm_semaphore = asyncio.Semaphore(_llm_concurrency)

extractor_llm = build_llm(
    purpose="ingest",
    temperature=0.1,
    context_window=_llm_context_window,
)

DEFAULT_CHUNK_PLAN = (512, 50)
MEDIUM_CHUNK_PLAN = (768, 64)
LARGE_CHUNK_PLAN = (1024, 80)
XL_CHUNK_PLAN = (1536, 96)
XXL_CHUNK_PLAN = (3072, 192)
_CHUNK_PLAN_ORDER = [
    XXL_CHUNK_PLAN,
    XL_CHUNK_PLAN,
    LARGE_CHUNK_PLAN,
    MEDIUM_CHUNK_PLAN,
    DEFAULT_CHUNK_PLAN,
]


@lru_cache(maxsize=8)
def _get_chunker(chunk_size: int, chunk_overlap: int) -> TokenChunker:
    return TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def choose_chunk_plan(*, document_length: int, section_type: str) -> tuple[int, int]:
    """Choose a conservative chunking profile.

    Large body sections benefit most from larger chunks because embedding/write
    costs scale almost linearly with chunk count. Appendix/list/table sections
    keep smaller chunks to preserve grounding precision.
    """
    if section_type in LIST_SECTION_TYPES or section_type == "qa":
        return DEFAULT_CHUNK_PLAN
    if document_length >= 300_000:
        return XXL_CHUNK_PLAN
    if document_length >= 120_000:
        return XL_CHUNK_PLAN
    if document_length >= 40_000:
        return LARGE_CHUNK_PLAN
    if document_length >= 12_000:
        return MEDIUM_CHUNK_PLAN
    return DEFAULT_CHUNK_PLAN


def _next_smaller_chunk_plan(
    chunk_size: int, chunk_overlap: int
) -> tuple[int, int]:
    current = (chunk_size, chunk_overlap)
    if current in _CHUNK_PLAN_ORDER:
        idx = _CHUNK_PLAN_ORDER.index(current)
        if idx + 1 < len(_CHUNK_PLAN_ORDER):
            return _CHUNK_PLAN_ORDER[idx + 1]

    reduced_size = max(128, chunk_size // 2)
    if reduced_size >= chunk_size:
        reduced_size = max(64, chunk_size - 64)
    reduced_overlap = min(chunk_overlap, max(16, reduced_size // 8))
    return reduced_size, reduced_overlap


def _find_midpoint_split(text: str) -> int | None:
    if len(text) < 2:
        return None
    midpoint = len(text) // 2
    for marker in ("\n\n", "\n", "。", "；", "，", " "):
        left = text.rfind(marker, 0, midpoint)
        if left > 0:
            return left + len(marker)
        right = text.find(marker, midpoint)
        if right > 0:
            return right + len(marker)
    return midpoint


def _bounded_chunk_texts(
    *,
    filename: str,
    block_index: int,
    section_type: str,
    header_text: str,
    chunk_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    token_budget = get_embedding_input_token_budget()
    full_text = f"{header_text}{chunk_text}"
    token_count = count_embedding_input_tokens(full_text)
    if token_count is None or token_count <= token_budget:
        return [chunk_text]

    logger.warning(
        "Embedding token budget exceeded file=%s block=%d section=%s tokens=%s budget=%d chunk_size=%d overlap=%d; rechunking",
        filename,
        block_index,
        section_type,
        token_count,
        token_budget,
        chunk_size,
        chunk_overlap,
    )

    bounded: list[str] = []
    pending: deque[tuple[str, int, int]] = deque([(chunk_text, chunk_size, chunk_overlap)])

    while pending:
        current_text, current_size, current_overlap = pending.popleft()
        current_full_text = f"{header_text}{current_text}"
        current_tokens = count_embedding_input_tokens(current_full_text)
        if current_tokens is None or current_tokens <= token_budget:
            bounded.append(current_text)
            continue

        next_size, next_overlap = _next_smaller_chunk_plan(
            current_size, current_overlap
        )
        split_chunks = [
            getattr(chunk, "text", str(chunk)).strip()
            for chunk in _get_chunker(next_size, next_overlap)(current_text)
        ]
        split_chunks = [chunk for chunk in split_chunks if chunk]

        if len(split_chunks) <= 1:
            split_at = _find_midpoint_split(current_text)
            if split_at is None:
                logger.error(
                    "Embedding token budget still exceeded but chunk could not be split file=%s block=%d section=%s tokens=%s budget=%d len=%d",
                    filename,
                    block_index,
                    section_type,
                    current_tokens,
                    token_budget,
                    len(current_text),
                )
                bounded.append(current_text)
                continue
            split_chunks = [
                current_text[:split_at].strip(),
                current_text[split_at:].strip(),
            ]
            split_chunks = [chunk for chunk in split_chunks if chunk]

        for split_chunk in reversed(split_chunks):
            pending.appendleft((split_chunk, next_size, next_overlap))

    return bounded


def _merge_chunkable_blocks(blocks: List[PreprocessedBlock]) -> List[PreprocessedBlock]:
    if not blocks:
        return []

    merged: list[PreprocessedBlock] = []
    current = blocks[0]
    for block in blocks[1:]:
        can_merge = (
            current.section_type in BODY_SECTION_TYPES
            and block.section_type in BODY_SECTION_TYPES
            and current.heading_path == block.heading_path
        )
        if can_merge:
            merged_text = f"{current.text}\n\n{block.text}"
            merged_section_type = "body" if "body" in {current.section_type, block.section_type} else current.section_type
            current = PreprocessedBlock(
                text=merged_text,
                section_type=merged_section_type,
                section_order=current.section_order,
                block_index=current.block_index,
                heading_path=current.heading_path or block.heading_path,
                is_list_zone=False,
                is_qa_zone=False,
            )
            continue
        merged.append(current)
        current = block
    merged.append(current)
    return merged


logger.debug(
    "Initialized adaptive TokenChunker default=%s medium=%s large=%s xl=%s xxl=%s llm_context=%d retries=%d backoff=%.2f concurrency=%d",
    DEFAULT_CHUNK_PLAN,
    MEDIUM_CHUNK_PLAN,
    LARGE_CHUNK_PLAN,
    XL_CHUNK_PLAN,
    XXL_CHUNK_PLAN,
    _llm_context_window,
    _llm_retry_attempts,
    _llm_retry_backoff,
    _llm_concurrency,
)


def _strip_code_fence(content: str) -> str:
    """Remove surrounding markdown fences like ```json ... ``` or single backticks."""
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        inner = text[3:-3].strip()
        if "\n" in inner:
            first, rest = inner.split("\n", 1)
            # Drop language hint lines such as "json" or "markdown"
            if first.strip() and not first.strip().startswith(("{", "[")):
                inner = rest
        text = inner.strip()
    if text.startswith("`") and text.endswith("`"):
        text = text.strip("`").strip()
    return text


def _read_field(value: Any, field: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _normalize_text_candidate(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        text_parts: list[str] = []
        for item in value:
            item_type = _read_field(item, "type")
            if item_type not in (None, "text", "output_text"):
                continue
            item_text = _read_field(item, "text")
            if item_text is None:
                item_text = _read_field(item, "content")
            if item_text:
                text_parts.append(str(item_text))
        text = "".join(text_parts).strip()
        return text or None
    text = str(value).strip()
    return text or None


def _iter_json_object_candidates(text: str):
    starts = [match.start() for match in re.finditer(r"\{", text)]
    for start in starts[:32]:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    break


def _extract_llm_analysis_payload(response: Any) -> tuple[dict[str, Any], str]:
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []

    def add_candidate(source: str, value: Any) -> None:
        text = _normalize_text_candidate(value)
        if not text:
            return
        text = _strip_code_fence(text)
        if not text or text in seen:
            return
        seen.add(text)
        candidates.append((source, text))

    add_candidate("response.text", getattr(response, "text", None))

    raw = getattr(response, "raw", None)
    raw_choices = _read_field(raw, "choices")
    if isinstance(raw_choices, list) and raw_choices:
        first_choice = raw_choices[0]
        message = _read_field(first_choice, "message")
        if message is not None:
            add_candidate("raw.message.content", _read_field(message, "content"))
            add_candidate(
                "raw.message.reasoning_content",
                _read_field(message, "reasoning_content"),
            )
            add_candidate("raw.message.reasoning", _read_field(message, "reasoning"))
        add_candidate("raw.choice.text", _read_field(first_choice, "text"))

    for source, text in candidates:
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload, source
        except json.JSONDecodeError:
            pass
        for fragment in _iter_json_object_candidates(text):
            try:
                payload = json.loads(fragment)
                if isinstance(payload, dict):
                    return payload, f"{source}:substring"
            except json.JSONDecodeError:
                continue

    raise json.JSONDecodeError(
        "No valid JSON object found in LLM response",
        candidates[0][1] if candidates else "",
        0,
    )


async def analyze_document(text: str) -> LLMAnalysis:
    context = text[:4000]
    prompt = f"""
你是一名资深金融分析师，负责将原始 Markdown 片段转化为检索用的富语义数据。请严格按照下列 JSON 结构输出（禁止额外文本或 Markdown 代码块）：
{{
  "summary": "<120-220字的中文文档摘要，概括主旨、核心判断、风险或分歧点；若存在文末名单/附录可简要指出>",
  "table_narrative": "<将表格/数值转写成自然语言，若无表格留空字符串>",
  "keywords": ["<股票代码或公司名>", "<行业关键词>", "...至多8个唯一关键词"]
}}
要求：
- 使用中文输出，保持数值/单位原样。
- summary 优先写成可复用的文档级摘要，不要只写一句标题复述。
- keywords 必须是字符串数组，含 ticker、机构简称、行业术语，避免重复。
- 若无法提取信息，字段填空字符串，keywords 为空数组。

待分析文档：
```markdown
{context}
```
"""
    last_error: Exception | None = None
    async with _llm_semaphore:
        for attempt in range(1, _llm_retry_attempts + 1):
            attempt_start = time.perf_counter()
            try:
                logger.debug(
                    "Analyzing document chunk len=%d attempt=%d/%d",
                    len(context),
                    attempt,
                    _llm_retry_attempts,
                )
                response = await extractor_llm.acomplete(prompt)
                data, source = _extract_llm_analysis_payload(response)
                analysis = LLMAnalysis.model_validate(data)
                duration = time.perf_counter() - attempt_start
                logger.debug(
                    "Analyzer succeeded attempt=%d duration=%.2fs summary_len=%d table_len=%d keywords=%d source=%s",
                    attempt,
                    duration,
                    len(analysis.summary),
                    len(analysis.table_narrative),
                    len(analysis.keywords or []),
                    source,
                )
                return analysis
            except Exception as exc:
                last_error = exc
                duration = time.perf_counter() - attempt_start
                if attempt >= _llm_retry_attempts:
                    logger.exception(
                        "Analyzer failed after %d attempts duration=%.2fs", attempt, duration
                    )
                    break
                delay = _llm_retry_backoff * attempt
                logger.warning(
                    "Analyzer attempt %d/%d failed in %.2fs: %s -- retrying in %.1fs",
                    attempt,
                    _llm_retry_attempts,
                    duration,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
    logger.debug(
        "Analyzer returning empty payload after failures last_error=%s", last_error
    )
    return LLMAnalysis()


async def process_file(
    filename: str,
    content: str,
    ingest_date: str | None = None,
    doc_hash: str | None = None,
    scope: str | None = None,
) -> List[TextNode]:
    content_length = len(content)
    extracted_date = extract_date_from_filename(filename)
    if ingest_date:
        logger.debug(
            "process_file override ingest_date=%s file=%s content_len=%d scope=%s",
            ingest_date,
            filename,
            content_length,
            scope,
        )
    elif extracted_date:
        logger.debug(
            "process_file extracted_date=%s file=%s content_len=%d scope=%s",
            extracted_date,
            filename,
            content_length,
            scope,
        )
    else:
        logger.debug(
            "process_file fallback to default date for file=%s content_len=%d scope=%s",
            filename,
            content_length,
            scope,
        )
    meta_date = ingest_date or extracted_date or "1970-01-01"
    try:
        meta_date_numeric = int(meta_date.replace("-", ""))
    except ValueError:
        meta_date_numeric = 19700101
    logger.info("Processing file=%s extracted_date=%s scope=%s", filename, meta_date, scope)
    analysis_start = time.perf_counter()
    analysis = await analyze_document(content)
    analysis_elapsed = time.perf_counter() - analysis_start
    keywords_preview = ",".join(analysis.keywords or [])
    logger.info(
        "LLM analysis file=%s duration=%.2fs keywords=%d summary_preview=%s",
        filename,
        analysis_elapsed,
        len(analysis.keywords or []),
        keywords_preview or "<empty>",
    )
    logger.info(
        "LLM summary preview file=%s %s",
        filename,
        (analysis.summary or "<empty>")[:80].replace("\n", " "),
    )

    keywords = analysis.keywords
    keywords_str = ",".join(keywords)
    doc_summary = (analysis.summary or "").strip()
    if not doc_summary:
        doc_summary = content.strip().replace("\n", " ")[:180]
    header_text = (
        f"Date: {meta_date}\n"
        f"Summary: {doc_summary}\n"
        f"Key Data: {analysis.table_narrative}\n"
        f"Tags: {keywords_str}\n"
        f"---\n"
    )
    logger.debug(
        "Header prepared file=%s header_len=%d keywords=%s",
        filename,
        len(header_text),
        keywords_str or "<empty>",
    )

    blocks = _merge_chunkable_blocks(split_document_blocks(content))
    logger.debug(
        "Preprocessed file=%s into %d semantic blocks", filename, len(blocks)
    )

    nodes: List[TextNode] = []
    min_chunk_len: int | None = None
    max_chunk_len = 0
    total_chars = 0
    chunk_index = 0
    chunk_plan_counts: dict[tuple[int, int], int] = {}
    source_blocks = blocks
    if not source_blocks and content.strip():
        source_blocks = split_document_blocks(content, allow_title=False)

    for block in source_blocks:
        chunk_size, chunk_overlap = choose_chunk_plan(
            document_length=content_length,
            section_type=block.section_type,
        )
        chunk_plan_counts[(chunk_size, chunk_overlap)] = chunk_plan_counts.get((chunk_size, chunk_overlap), 0) + 1
        block_chunker = _get_chunker(chunk_size, chunk_overlap)
        logger.debug(
            "Chunking block file=%s block_index=%d section=%s block_len=%d chunk_size=%d overlap=%d",
            filename,
            block.block_index,
            block.section_type,
            len(block.text),
            chunk_size,
            chunk_overlap,
        )
        for chunk in block_chunker(block.text):
            chunk_text = getattr(chunk, "text", str(chunk))
            for bounded_chunk_text in _bounded_chunk_texts(
                filename=filename,
                block_index=block.block_index,
                section_type=block.section_type,
                header_text=header_text,
                chunk_text=chunk_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ):
                chunk_len = len(bounded_chunk_text)
                total_chars += chunk_len
                if min_chunk_len is None or chunk_len < min_chunk_len:
                    min_chunk_len = chunk_len
                if chunk_len > max_chunk_len:
                    max_chunk_len = chunk_len
                full_text = f"{header_text}{bounded_chunk_text}"
                node_metadata = {
                    "filename": filename,
                    "date": meta_date,
                    "date_numeric": meta_date_numeric,
                    "doc_hash": doc_hash,
                    "doc_summary": doc_summary,
                    "keywords": keywords_str,
                    "keyword_list": keywords,
                    "original_text": bounded_chunk_text,
                    "section_type": block.section_type,
                    "section_order": block.section_order,
                    "block_index": block.block_index,
                    "chunk_index": chunk_index,
                    "heading_path": block.heading_path,
                    "is_list_zone": block.is_list_zone,
                    "is_qa_zone": block.is_qa_zone,
                }
                if scope:
                    node_metadata["scope"] = scope

                node = TextNode(
                    text=full_text,
                    metadata=node_metadata,
                )
                hash_base = f"{doc_hash or filename}|{scope or ''}"
                node.id_ = hashlib.md5(f"{hash_base}_{chunk_index}".encode()).hexdigest()
                nodes.append(node)
                chunk_index += 1
    if nodes:
        avg_len = total_chars / len(nodes)
        logger.debug(
            "Chunk stats file=%s count=%d avg_len=%.1f min_len=%s max_len=%s plans=%s",
            filename,
            len(nodes),
            avg_len,
            min_chunk_len,
            max_chunk_len,
            {f"{size}/{overlap}": count for (size, overlap), count in sorted(chunk_plan_counts.items())},
        )
    else:
        logger.warning("No chunks produced for file=%s; input may be empty", filename)
    logger.info("File=%s produced %d chunks", filename, len(nodes))
    return nodes
