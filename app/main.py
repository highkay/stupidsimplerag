import asyncio
import datetime
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from cachetools import TTLCache
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tenacity import AsyncRetrying, RetryCallState, stop_after_attempt, wait_exponential

from llama_index.core import QueryBundle
from llama_index.core.schema import TextNode
from app.core import adoc_exists, get_collection_metrics, get_query_engine, insert_nodes
from app.ingest import compute_doc_hash, process_file
from app.models import (
    ChatRequest,
    ChatResponse,
    IngestResponse,
    SourceItem,
    TextIngestRequest,
)
from app.utils import get_node_metadata

app = FastAPI(title="Finance RAG Engine")
CACHE = TTLCache(maxsize=1000, ttl=3600)
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

logger = logging.getLogger(__name__)
BATCH_INGEST_CONCURRENCY = max(1, int(os.getenv("BATCH_INGEST_CONCURRENCY", "4")))
MAX_BATCH_FILES = max(1, int(os.getenv("BATCH_MAX_FILES", "20")))
INSERT_MAX_RETRIES = max(1, int(os.getenv("INGEST_INSERT_MAX_RETRIES", "3")))
INSERT_RETRY_BACKOFF = float(os.getenv("INGEST_INSERT_RETRY_BACKOFF", "2.0"))
QUERY_MAX_RETRIES = max(1, int(os.getenv("QUERY_MAX_RETRIES", "3")))
QUERY_RETRY_BACKOFF = float(os.getenv("QUERY_RETRY_BACKOFF", "2.0"))


def _log_query_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    sleep = retry_state.next_action.sleep if retry_state.next_action else 0
    logger.warning(
        "Query attempt %d/%d failed: %s -- retrying in %.1fs",
        retry_state.attempt_number,
        QUERY_MAX_RETRIES,
        exc,
        sleep,
    )


def _cache_key(body: ChatRequest) -> str:
    parts = [
        body.query,
        body.start_date or "",
        body.end_date or "",
        body.filename or "",
        body.filename_contains or "",
        ",".join(body.keywords_any or []),
        ",".join(body.keywords_all or []),
        str(body.skip_rerank),
        str(body.skip_generation),
    ]
    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()


def _parse_epoch_date(raw: str) -> Optional[str]:
    try:
        ts = float(raw)
    except (TypeError, ValueError):
        return None
    if ts > 3_000_000_000_000:  # handle ms timestamps
        ts /= 1000.0
    try:
        dt = datetime.datetime.utcfromtimestamp(ts)
    except (OverflowError, OSError, ValueError):
        return None
    return dt.date().isoformat()


def _parse_iso_date(raw: str) -> Optional[str]:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    # Replace trailing Z with +00:00 for fromisoformat compatibility
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt.date().isoformat()


def _extract_date_header(header_value: Optional[str], filename: str) -> Optional[str]:
    if not header_value:
        return None
    date_str = _parse_epoch_date(header_value) or _parse_iso_date(header_value)
    if not date_str:
        logger.warning(
            "Failed to parse X-File-Mtime header=%r for file=%s",
            header_value,
            filename,
        )
    else:
        logger.debug(
            "Resolved ingest date %s from file timestamp header for %s",
            date_str,
            filename,
        )
    return date_str


def _extract_upload_date(upload: UploadFile) -> Optional[str]:
    if not upload:
        return None
    return _extract_date_header(upload.headers.get("x-file-mtime"), upload.filename)


def _decode_upload_bytes(raw_bytes: Optional[bytes], filename: str) -> str:
    data = raw_bytes or b""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        logger.error(
            "Failed to decode upload file=%s byte_len=%d: %s",
            filename,
            len(data),
            exc,
        )
        raise HTTPException(
            status_code=400,
            detail=f"File {filename} must be UTF-8 encoded before uploading.",
        ) from exc


def _parse_keywords(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",")]
    cleaned = [value for value in values if value]
    return cleaned or None


def _parse_bool(raw: Optional[str]) -> bool:
    if raw is None:
        return False
    value = str(raw).strip().lower()
    return value in ("1", "true", "on", "yes")


def _log_retry(retry_state: RetryCallState) -> None:
    """Tenacity hook for logging retry attempts with backoff."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    sleep = retry_state.next_action.sleep if retry_state.next_action else 0
    logger.warning(
        "insert_nodes attempt %d/%d failed: %s -- retrying in %.1fs",
        retry_state.attempt_number,
        INSERT_MAX_RETRIES,
        exc,
        sleep,
    )

def _is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


def _node_to_source(node) -> SourceItem:
    metadata = get_node_metadata(node)
    text = metadata.get("original_text")
    if not text:
        inner = getattr(node, "node", None)
        if inner is not None:
            text = inner.get_content()[:200]
        else:
            text = ""
    return SourceItem(
        filename=metadata.get("filename", "unknown"),
        date=metadata.get("date"),
        score=round(node.score, 4) if node.score is not None else None,
        keywords=metadata.get("keywords"),
        text=text[:200],
    )


def _node_keyword_set(node) -> set[str]:
    metadata = get_node_metadata(node)
    keywords = []
    raw_list = metadata.get("keyword_list")
    if isinstance(raw_list, list):
        keywords.extend(raw_list)
    elif isinstance(raw_list, str):
        keywords.extend(item.strip() for item in raw_list.split(","))
    raw_str = metadata.get("keywords")
    if isinstance(raw_str, str):
        keywords.extend(item.strip() for item in raw_str.split(","))
    return {kw.lower() for kw in keywords if kw and kw.strip()}


def _generate_answer_from_nodes(nodes: List, limit: int = 3) -> str:
    """Fallback answer when generation is skipped: stitch top snippets."""
    snippets = []
    for node in nodes[:limit]:
        metadata = get_node_metadata(node)
        text = metadata.get("original_text") or getattr(node, "text", "") or ""
        if text:
            snippets.append(text[:200])
    if not snippets:
        return "Generation skipped; see sources for details."
    return "\n\n".join(snippets)


async def _process_and_insert_content(
    filename: str, content: str, ingest_date: Optional[str]
) -> tuple[List[TextNode], bool, str]:
    """
    Shared helper to run expensive processing & DB insert without blocking the event loop.
    """
    content_len = len(content)
    doc_hash = compute_doc_hash(content)
    logger.debug(
        "Begin ingest pipeline file=%s ingest_date=%s content_len=%d doc_hash=%s",
        filename,
        ingest_date,
        content_len,
        doc_hash,
    )
    if await adoc_exists(doc_hash):
        logger.info(
            "Skipping ingest for file=%s doc_hash=%s (already exists)", filename, doc_hash
        )
        return [], True, doc_hash
    start = time.perf_counter()
    nodes = await process_file(
        filename, content, ingest_date=ingest_date, doc_hash=doc_hash
    )
    processing_elapsed = time.perf_counter() - start
    logger.debug(
        "process_file finished file=%s nodes=%d duration=%.2fs",
        filename,
        len(nodes),
        processing_elapsed,
    )
    if nodes:
        insert_start = time.perf_counter()
        await _insert_nodes_with_retry(nodes)
        insert_elapsed = time.perf_counter() - insert_start
        logger.debug(
            "Inserted nodes file=%s count=%d duration=%.2fs",
            filename,
            len(nodes),
            insert_elapsed,
        )
    else:
        logger.warning(
            "Skipping insert for file=%s because no nodes were produced",
            filename,
        )
    total_elapsed = time.perf_counter() - start
    logger.debug(
        "Ingest pipeline finished file=%s total_duration=%.2fs",
        filename,
        total_elapsed,
    )
    return nodes, False, doc_hash



async def _insert_nodes_with_retry(nodes: List[TextNode]) -> None:
    if not nodes:
        return
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(INSERT_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=INSERT_RETRY_BACKOFF,
            min=INSERT_RETRY_BACKOFF,
            max=INSERT_RETRY_BACKOFF * 4,
        ),
        reraise=True,
        before_sleep=_log_retry,
    ):
        with attempt:
            await insert_nodes(nodes)
            logger.info(
                "insert_nodes succeeded attempt=%d node_count=%d",
                attempt.retry_state.attempt_number,
                len(nodes),
            )


async def _ingest_single_upload(upload: UploadFile) -> IngestResponse:
    logger.info("Batch ingest processing file=%s", upload.filename)
    try:
        raw_bytes = await upload.read()
        logger.debug(
            "Read batch upload file=%s bytes=%d",
            upload.filename,
            len(raw_bytes or b""),
        )
        content = _decode_upload_bytes(raw_bytes, upload.filename)
        ingest_date = _extract_upload_date(upload)
        nodes, skipped, doc_hash = await _process_and_insert_content(
            upload.filename, content, ingest_date
        )
        status = "skipped" if skipped else "ok"
        result = IngestResponse(
            status=status,
            chunks=len(nodes),
            filename=upload.filename,
            doc_hash=doc_hash,
        )
        logger.info(
            "Batch ingest processed file=%s status=%s chunks=%d",
            upload.filename,
            result.status,
            result.chunks,
        )
        return result
    finally:
        try:
            await upload.close()
        except Exception:
            pass


@app.post("/ingest", response_model=IngestResponse)
async def ingest_api(file: UploadFile = File(...)) -> IngestResponse:
    if not (file.filename.endswith(".md") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only .md or .txt supported")

    logger.info("Ingest single file=%s", file.filename)
    raw_bytes = await file.read()
    byte_len = len(raw_bytes or b"")
    logger.debug("Received upload file=%s size_bytes=%d", file.filename, byte_len)
    content = _decode_upload_bytes(raw_bytes, file.filename)
    ingest_date = _extract_upload_date(file)
    nodes, skipped, doc_hash = await _process_and_insert_content(
        file.filename, content, ingest_date
    )
    status = "skipped" if skipped else "ok"
    logger.info(
        "Ingest complete file=%s status=%s chunks=%d",
        file.filename,
        status,
        len(nodes),
    )
    return IngestResponse(
        status=status, chunks=len(nodes), filename=file.filename, doc_hash=doc_hash
    )


@app.post("/ingest/batch", response_model=List[IngestResponse])
async def ingest_batch(files: List[UploadFile] = File(...)) -> List[IngestResponse]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files in one batch ({len(files)}); max allowed is {MAX_BATCH_FILES}. Please split the upload.",
        )

    for upload in files:
        if not (upload.filename.endswith(".md") or upload.filename.endswith(".txt")):
            raise HTTPException(
                status_code=400,
                detail=f"Only .md or .txt supported: {upload.filename}",
            )

    logger.info(
        "Batch ingest start files=%d concurrency=%d",
        len(files),
        BATCH_INGEST_CONCURRENCY,
    )
    concurrency = BATCH_INGEST_CONCURRENCY
    responses: List[Optional[IngestResponse]] = [None] * len(files)
    in_flight: set[asyncio.Task] = set()

    async def _runner(idx: int, upload: UploadFile) -> tuple[int, IngestResponse]:
        try:
            result = await _ingest_single_upload(upload)
            return idx, result
        except Exception as exc:  # pragma: no cover - defensive logging
            detail = getattr(exc, "detail", str(exc))
            logger.error("Batch ingest failed file=%s error=%s", upload.filename, detail)
            return idx, IngestResponse(
                status="error",
                chunks=0,
                filename=upload.filename,
                error=str(detail),
            )

    # Prime initial workers up to the concurrency limit
    next_index = 0
    while next_index < len(files) and len(in_flight) < concurrency:
        task = asyncio.create_task(_runner(next_index, files[next_index]))
        in_flight.add(task)
        next_index += 1

    while in_flight:
        done, in_flight = await asyncio.wait(
            in_flight, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            idx, result = await task
            responses[idx] = result
            if next_index < len(files):
                task = asyncio.create_task(_runner(next_index, files[next_index]))
                in_flight.add(task)
                next_index += 1

    # Fill any missing (should not happen) with error placeholders
    for i, value in enumerate(responses):
        if value is None:
            responses[i] = IngestResponse(
                status="error",
                chunks=0,
                filename=files[i].filename if i < len(files) else f"unknown_{i}",
                error="unknown failure",
            )

    success_count = sum(
        1 for item in responses if item and item.status and item.status.lower() == "ok"
    )
    logger.info(
        "Batch ingest finished total_files=%d succeeded=%d failed=%d",
        len(responses),
        success_count,
        len(responses) - success_count,
    )
    return [resp for resp in responses if resp is not None]


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text_api(request: Request, body: TextIngestRequest) -> IngestResponse:
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="content must not be empty")

    # 尝试从 X-File-Mtime header 提取日期
    header_value = request.headers.get("x-file-mtime")
    ingest_date = None
    if header_value:
        ingest_date = _parse_epoch_date(header_value) or _parse_iso_date(header_value)
        if not ingest_date:
            logger.warning(
                "Failed to parse X-File-Mtime header=%r for text ingest",
                header_value,
            )
        else:
            logger.debug(
                "Resolved ingest date %s from X-File-Mtime header for text ingest",
                ingest_date,
            )
    
    # 如果 header 中没有日期或解析失败，使用当前时间
    if not ingest_date:
        now = datetime.datetime.utcnow()
        ingest_date = now.strftime("%Y-%m-%d")
    
    filename = body.filename or f"inline_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
    logger.info("Text ingest filename=%s date=%s", filename, ingest_date)
    logger.debug(
        "Text ingest filename=%s raw_len=%d stripped_len=%d",
        filename,
        len(body.content),
        len(content),
    )
    nodes, skipped, doc_hash = await _process_and_insert_content(
        filename, content, ingest_date
    )
    status = "skipped" if skipped else "ok"
    logger.info(
        "Text ingest complete filename=%s status=%s chunks=%d",
        filename,
        status,
        len(nodes),
    )
    return IngestResponse(
        status=status, chunks=len(nodes), filename=filename, doc_hash=doc_hash
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_api(req: ChatRequest) -> ChatResponse:
    logger.info(
        "Chat request query=%r start=%s end=%s filename=%s contains=%s any=%s all=%s",
        req.query,
        req.start_date,
        req.end_date,
        req.filename,
        req.filename_contains,
        req.keywords_any,
        req.keywords_all,
    )
    key = _cache_key(req)
    if key in CACHE:
        cached = CACHE[key]
        logger.info("Cache hit for query hash=%s", key)
        return ChatResponse(**cached)

    engine = get_query_engine(
        start_date=req.start_date,
        end_date=req.end_date,
        filename=req.filename,
        filename_contains=req.filename_contains,
        skip_rerank=req.skip_rerank,
        keywords_any=req.keywords_any,
        keywords_all=req.keywords_all,
    )

    if req.skip_generation:
        retriever = getattr(engine, "_retriever", None)
        if retriever is None:
            raise HTTPException(status_code=500, detail="Retriever unavailable")
        query_bundle = QueryBundle(req.query)
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(QUERY_MAX_RETRIES),
            wait=wait_exponential(
                multiplier=QUERY_RETRY_BACKOFF,
                min=QUERY_RETRY_BACKOFF,
                max=QUERY_RETRY_BACKOFF * 8,
            ),
            before_sleep=_log_query_retry,
        ):
            with attempt:
                source_nodes = await retriever.aretrieve(query_bundle)
        postprocessors = getattr(engine, "_node_postprocessors", []) or []
        for processor in postprocessors:
            try:
                source_nodes = await processor.apostprocess_nodes(source_nodes, query_bundle)
            except Exception as exc:
                logger.warning("Postprocessor failed, skipping: %s", exc)
        response_answer = _generate_answer_from_nodes(source_nodes)
    else:
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(QUERY_MAX_RETRIES),
            wait=wait_exponential(
                multiplier=QUERY_RETRY_BACKOFF,
                min=QUERY_RETRY_BACKOFF,
                max=QUERY_RETRY_BACKOFF * 8,
            ),
            before_sleep=_log_query_retry,
        ):
            with attempt:
                response = await engine.aquery(req.query)
        source_nodes = getattr(response, "source_nodes", []) or []
        response_answer = str(response)

    final_k = int(os.getenv("FINAL_TOP_K", "10"))
    result_nodes = source_nodes[:final_k]

    result_payload: Dict = {
        "answer": response_answer,
        "sources": [_node_to_source(node).dict() for node in result_nodes],
    }
    CACHE[key] = result_payload
    top_filenames = [item["filename"] for item in result_payload["sources"]]
    logger.info(
        "Chat response final_nodes=%d returned=%d top_files=%s answer_len=%d",
        len(source_nodes),
        len(result_nodes),
        top_filenames,
        len(result_payload["answer"]),
    )
    return ChatResponse(**result_payload)


@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> HTMLResponse:
    metrics = await get_collection_metrics()
    cache_stats = {
        "entries": len(CACHE),
        "maxsize": CACHE.maxsize,
        "ttl": getattr(CACHE, "ttl", 0),
    }
    config_snapshot = {
        "llm_model": os.getenv("LLM_MODEL", "unknown"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "unknown"),
        "rerank_model": os.getenv("RERANK_MODEL", "unknown"),
        "collection_name": metrics.get("collection_name"),
        "top_k": int(os.getenv("TOP_K_RETRIEVAL", "100")),
        "top_n_rerank": int(os.getenv("TOP_N_RERANK", "20")),
        "final_top_k": int(os.getenv("FINAL_TOP_K", "10")),
        "batch_concurrency": BATCH_INGEST_CONCURRENCY,
    }
    context = {
        "request": request,
        "metrics": metrics,
        "cache_stats": cache_stats,
        "config": config_snapshot,
    }
    return templates.TemplateResponse("dashboard.html", context)


@app.get("/ui/upload", response_class=HTMLResponse)
async def upload_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("upload_single.html", {"request": request})


@app.get("/ui/upload/batch", response_class=HTMLResponse)
async def upload_batch_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("upload_batch.html", {"request": request})


@app.get("/ui/chat", response_class=HTMLResponse)
async def chat_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/ui/ingest", response_class=HTMLResponse)
async def upload_via_ui(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    is_htmx = _is_htmx(request)
    template_name = "partials/upload_result.html" if is_htmx else "upload_single.html"
    context = {"request": request}
    try:
        result = await ingest_api(file)
        context["result"] = result.dict()
    except HTTPException as exc:
        context["error"] = exc.detail
    except Exception as exc:
        context["error"] = str(exc)
    return templates.TemplateResponse(template_name, context)


@app.post("/ui/ingest/batch", response_class=HTMLResponse)
async def upload_batch_via_ui(
    request: Request, files: List[UploadFile] = File(...)
) -> HTMLResponse:
    is_htmx = _is_htmx(request)
    template_name = "partials/batch_result.html" if is_htmx else "upload_batch.html"
    context = {"request": request}
    try:
        results = await ingest_batch(files)
        context["results"] = [item.dict() for item in results]
    except HTTPException as exc:
        context["error"] = exc.detail
    except Exception as exc:
        context["error"] = str(exc)
    return templates.TemplateResponse(template_name, context)


@app.post("/ui/chat/query", response_class=HTMLResponse)
async def chat_via_ui(
    request: Request,
    query: str = Form(...),
    start_date: Optional[str] = Form(None),
    end_date: Optional[str] = Form(None),
    filename: Optional[str] = Form(None),
    filename_contains: Optional[str] = Form(None),
    keywords_any: Optional[str] = Form(None),
    keywords_all: Optional[str] = Form(None),
    skip_rerank: Optional[str] = Form(None),
    skip_generation: Optional[str] = Form(None),
) -> HTMLResponse:
    req_body = ChatRequest(
        query=query,
        start_date=start_date or None,
        end_date=end_date or None,
        filename=filename or None,
        filename_contains=filename_contains or None,
        keywords_any=_parse_keywords(keywords_any),
        keywords_all=_parse_keywords(keywords_all),
        skip_rerank=_parse_bool(skip_rerank),
        skip_generation=_parse_bool(skip_generation),
    )
    is_htmx = _is_htmx(request)
    template_name = "partials/chat_result.html" if is_htmx else "chat.html"
    context = {"request": request}
    try:
        result = await chat_api(req_body)
        context["result"] = result.dict()
    except HTTPException as exc:
        context["error"] = exc.detail
    except Exception as exc:
        context["error"] = str(exc)
    return templates.TemplateResponse(template_name, context)
