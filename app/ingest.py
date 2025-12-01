import asyncio
import hashlib
import json
import logging
import os
import time
from typing import List

from chonkie import TokenChunker
from llama_index.core.schema import TextNode

from app.models import LLMAnalysis
from app.openai_utils import OpenAICompatibleLLM, get_openai_kwargs
from app.utils import extract_date_from_filename

logger = logging.getLogger(__name__)


_llm_context_window = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))
_llm_retry_attempts = int(os.getenv("LLM_MAX_RETRIES", "3"))
_llm_retry_backoff = float(os.getenv("LLM_RETRY_BACKOFF", "1.5"))

extractor_llm = OpenAICompatibleLLM(
    model=os.getenv("LLM_MODEL"),
    temperature=0.1,
    context_window=_llm_context_window,
    **get_openai_kwargs("LLM"),
)

chunker = TokenChunker(chunk_size=512, chunk_overlap=50)
logger.debug(
    "Initialized TokenChunker chunk_size=%s chunk_overlap=%s llm_context=%d retries=%d backoff=%.2f",
    getattr(chunker, "chunk_size", "unknown"),
    getattr(chunker, "chunk_overlap", "unknown"),
    _llm_context_window,
    _llm_retry_attempts,
    _llm_retry_backoff,
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


async def analyze_document(text: str) -> LLMAnalysis:
    context = text[:4000]
    prompt = f"""
你是一名资深金融分析师，负责将原始 Markdown 片段转化为检索用的富语义数据。请严格按照下列 JSON 结构输出（禁止额外文本或 Markdown 代码块）：
{{
  "summary": "<<=60字的中文摘要，突出业绩、风险或政策变化>",
  "table_narrative": "<将表格/数值转写成自然语言，若无表格留空字符串>",
  "keywords": ["<股票代码或公司名>", "<行业关键词>", "...至多8个唯一关键词"]
}}
要求：
- 使用中文输出，保持数值/单位原样。
- keywords 必须是字符串数组，含 ticker、机构简称、行业术语，避免重复。
- 若无法提取信息，字段填空字符串，keywords 为空数组。

待分析文档：
```markdown
{context}
```
"""
    last_error: Exception | None = None
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
            content = getattr(response, "text", str(response)).strip()
            content = _strip_code_fence(content)
            data = json.loads(content)
            analysis = LLMAnalysis.model_validate(data)
            duration = time.perf_counter() - attempt_start
            logger.debug(
                "Analyzer succeeded attempt=%d duration=%.2fs summary_len=%d table_len=%d keywords=%d",
                attempt,
                duration,
                len(analysis.summary),
                len(analysis.table_narrative),
                len(analysis.keywords or []),
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


async def process_file(filename: str, content: str, ingest_date: str | None = None) -> List[TextNode]:
    content_length = len(content)
    extracted_date = extract_date_from_filename(filename)
    if ingest_date:
        logger.debug(
            "process_file override ingest_date=%s file=%s content_len=%d",
            ingest_date,
            filename,
            content_length,
        )
    elif extracted_date:
        logger.debug(
            "process_file extracted_date=%s file=%s content_len=%d",
            extracted_date,
            filename,
            content_length,
        )
    else:
        logger.debug(
            "process_file fallback to default date for file=%s content_len=%d",
            filename,
            content_length,
        )
    meta_date = ingest_date or extracted_date or "1970-01-01"
    try:
        meta_date_numeric = int(meta_date.replace("-", ""))
    except ValueError:
        meta_date_numeric = 19700101
    logger.info("Processing file=%s extracted_date=%s", filename, meta_date)
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
    header_text = (
        f"Date: {meta_date}\n"
        f"Summary: {analysis.summary}\n"
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

    nodes: List[TextNode] = []
    min_chunk_len: int | None = None
    max_chunk_len = 0
    total_chars = 0
    chunk_size = getattr(chunker, "chunk_size", "unknown")
    chunk_overlap = getattr(chunker, "chunk_overlap", "unknown")
    logger.debug(
        "Running chunker file=%s total_chars=%d chunk_size=%s overlap=%s",
        filename,
        content_length,
        chunk_size,
        chunk_overlap,
    )
    for idx, chunk in enumerate(chunker(content)):
        chunk_text = getattr(chunk, "text", str(chunk))
        chunk_len = len(chunk_text)
        total_chars += chunk_len
        if min_chunk_len is None or chunk_len < min_chunk_len:
            min_chunk_len = chunk_len
        if chunk_len > max_chunk_len:
            max_chunk_len = chunk_len
        full_text = f"{header_text}{chunk_text}"
        node = TextNode(
            text=full_text,
            metadata={
                "filename": filename,
                "date": meta_date,
                "date_numeric": meta_date_numeric,
                "keywords": keywords_str,
                "keyword_list": keywords,
                "original_text": chunk_text,
            },
        )
        node.id_ = hashlib.md5(f"{filename}_{idx}".encode()).hexdigest()
        nodes.append(node)
    if nodes:
        avg_len = total_chars / len(nodes)
        logger.debug(
            "Chunk stats file=%s count=%d avg_len=%.1f min_len=%s max_len=%s",
            filename,
            len(nodes),
            avg_len,
            min_chunk_len,
            max_chunk_len,
        )
    else:
        logger.warning("No chunks produced for file=%s; input may be empty", filename)
    logger.info("File=%s produced %d chunks", filename, len(nodes))
    return nodes
