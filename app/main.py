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

from llama_index.core.schema import TextNode
from app.core import doc_exists, get_collection_metrics, get_query_engine, insert_nodes
from app.ingest import compute_doc_hash, process_file
from app.models import (
    ChatRequest,
    ChatResponse,
    IngestResponse,
    SourceItem,
    TextIngestRequest,
)
from app.utils import apply_time_decay, get_node_metadata

app = FastAPI(title="Finance RAG Engine")
CACHE = TTLCache(maxsize=1000, ttl=3600)
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

logger = logging.getLogger("uvicorn.error")
BATCH_INGEST_CONCURRENCY = max(1, int(os.getenv("BATCH_INGEST_CONCURRENCY", "4")))
INSERT_MAX_RETRIES = max(1, int(os.getenv("INGEST_INSERT_MAX_RETRIES", "3")))
INSERT_RETRY_BACKOFF = float(os.getenv("INGEST_INSERT_RETRY_BACKOFF", "2.0"))


def _cache_key(body: ChatRequest) -> str:
    parts = [
        body.query,
        body.start_date or "",
        body.end_date or "",
        body.filename or "",
        body.filename_contains or "",
        ",".join(body.keywords_any or []),
        ",".join(body.keywords_all or []),
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


def _extract_upload_date(upload: UploadFile) -> Optional[str]:
    if not upload:
        return None
    header_value = upload.headers.get("x-file-mtime")
    if not header_value:
        return None
    date_str = _parse_epoch_date(header_value) or _parse_iso_date(header_value)
    if not date_str:
        logger.warning(
            "Failed to parse X-File-Mtime header=%r for file=%s",
            header_value,
            upload.filename,
        )
    else:
        logger.debug(
            "Resolved ingest date %s from file timestamp header for %s",
            date_str,
            upload.filename,
        )
    return date_str


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


def _normalize_keyword_set(values: Optional[List[str]]) -> set[str]:
    if not values:
        return set()
    return {value.strip().lower() for value in values if value and value.strip()}


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


def _filter_nodes_by_keywords(nodes, any_set, all_set):
    if not any_set and not all_set:
        return nodes

    filtered = []
    for node in nodes:
        keyword_set = _node_keyword_set(node)
        if any_set and not (keyword_set & any_set):
            continue
        if all_set and not all_set.issubset(keyword_set):
            continue
        filtered.append(node)
    return filtered


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
    if doc_exists(doc_hash):
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
    last_error: Exception | None = None
    for attempt in range(1, INSERT_MAX_RETRIES + 1):
        try:
            await asyncio.to_thread(insert_nodes, nodes)
            if attempt > 1:
                logger.info(
                    "insert_nodes succeeded after retry attempt=%d node_count=%d",
                    attempt,
                    len(nodes),
                )
            return
        except Exception as exc:
            last_error = exc
            if attempt >= INSERT_MAX_RETRIES:
                logger.exception(
                    "insert_nodes failed after %d attempts", INSERT_MAX_RETRIES
                )
                raise
            delay = INSERT_RETRY_BACKOFF * attempt
            logger.warning(
                "insert_nodes attempt %d/%d failed: %s -- retrying in %.1fs",
                attempt,
                INSERT_MAX_RETRIES,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    if last_error:
        raise last_error


async def _ingest_single_upload(
    upload: UploadFile, semaphore: asyncio.Semaphore
) -> IngestResponse:
    async with semaphore:
        logger.info("Batch ingest processing file=%s", upload.filename)
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
        logger.info(
            "Batch ingest processed file=%s status=%s chunks=%d",
            upload.filename,
            status,
            len(nodes),
        )
        return IngestResponse(
            status=status,
            chunks=len(nodes),
            filename=upload.filename,
            doc_hash=doc_hash,
        )


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
    semaphore = asyncio.Semaphore(BATCH_INGEST_CONCURRENCY)
    tasks = [_ingest_single_upload(upload, semaphore) for upload in files]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    responses: List[IngestResponse] = []
    success_count = 0
    for upload, outcome in zip(files, raw_results):
        if isinstance(outcome, Exception):
            detail = getattr(outcome, "detail", str(outcome))
            logger.error(
                "Batch ingest failed file=%s error=%s",
                upload.filename,
                detail,
            )
            responses.append(
                IngestResponse(
                    status="error",
                    chunks=0,
                    filename=upload.filename,
                    error=str(detail),
                )
            )
        else:
            responses.append(outcome)
            if outcome.status.lower() == "ok":
                success_count += 1
    logger.info(
        "Batch ingest finished total_files=%d succeeded=%d failed=%d",
        len(responses),
        success_count,
        len(responses) - success_count,
    )
    return responses


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text_api(body: TextIngestRequest) -> IngestResponse:
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="content must not be empty")

    now = datetime.datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    filename = body.filename or f"inline_{now.strftime('%Y%m%d_%H%M%S')}.md"
    logger.info("Text ingest filename=%s date=%s", filename, date_str)
    logger.debug(
        "Text ingest filename=%s raw_len=%d stripped_len=%d",
        filename,
        len(body.content),
        len(content),
    )
    nodes, skipped, doc_hash = await _process_and_insert_content(
        filename, content, date_str
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
    )
    response = engine.query(req.query)
    source_nodes = getattr(response, "source_nodes", []) or []
    logger.info("Raw retrieved nodes=%d", len(source_nodes))

    any_set = _normalize_keyword_set(req.keywords_any)
    all_set = _normalize_keyword_set(req.keywords_all)
    filtered_nodes = _filter_nodes_by_keywords(source_nodes, any_set, all_set)
    if len(filtered_nodes) != len(source_nodes):
        logger.info(
            "Keyword filters applied any=%d all=%d dropped=%d nodes",
            len(any_set),
            len(all_set),
            len(source_nodes) - len(filtered_nodes),
        )
    source_nodes = filtered_nodes
    logger.info("Nodes after keyword filtering=%d", len(source_nodes))

    decay_rate = float(os.getenv("TIME_DECAY_RATE", "0.005"))
    nodes = apply_time_decay(source_nodes, decay_rate=decay_rate)
    final_k = int(os.getenv("FINAL_TOP_K", "10"))
    result_nodes = nodes[:final_k]
    logger.info(
        "Time decay rate=%s applied; nodes_post_decay=%d final_top_k=%d",
        decay_rate,
        len(nodes),
        final_k,
    )

    result_payload: Dict = {
        "answer": str(response),
        "sources": [_node_to_source(node).dict() for node in result_nodes],
    }
    CACHE[key] = result_payload
    top_filenames = [item["filename"] for item in result_payload["sources"]]
    logger.info(
        "Chat response final_nodes=%d returned=%d top_files=%s answer_len=%d",
        len(nodes),
        len(result_nodes),
        top_filenames,
        len(result_payload["answer"]),
    )
    return ChatResponse(**result_payload)


@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> HTMLResponse:
    metrics = get_collection_metrics()
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
) -> HTMLResponse:
    req_body = ChatRequest(
        query=query,
        start_date=start_date or None,
        end_date=end_date or None,
        filename=filename or None,
        filename_contains=filename_contains or None,
        keywords_any=_parse_keywords(keywords_any),
        keywords_all=_parse_keywords(keywords_all),
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
