import hashlib
import os
from typing import Dict, List

from cachetools import TTLCache
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.core import get_query_engine, insert_nodes
from app.ingest import process_file
from app.models import ChatRequest, ChatResponse, IngestResponse, SourceItem
from app.utils import apply_time_decay, get_node_metadata

app = FastAPI(title="Finance RAG Engine")
CACHE = TTLCache(maxsize=1000, ttl=3600)


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


@app.post("/ingest", response_model=IngestResponse)
async def ingest_api(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md supported")

    content = (await file.read()).decode("utf-8")
    nodes = await process_file(file.filename, content)
    if nodes:
        insert_nodes(nodes)
    return IngestResponse(status="ok", chunks=len(nodes), filename=file.filename)


@app.post("/ingest/batch", response_model=List[IngestResponse])
async def ingest_batch(files: List[UploadFile] = File(...)) -> List[IngestResponse]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    results: List[IngestResponse] = []
    for upload in files:
        if not upload.filename.endswith(".md"):
            raise HTTPException(status_code=400, detail=f"Only .md supported: {upload.filename}")
        content = (await upload.read()).decode("utf-8")
        nodes = await process_file(upload.filename, content)
        if nodes:
            insert_nodes(nodes)
        results.append(
            IngestResponse(status="ok", chunks=len(nodes), filename=upload.filename)
        )
    return results


@app.post("/chat", response_model=ChatResponse)
async def chat_api(req: ChatRequest) -> ChatResponse:
    key = _cache_key(req)
    if key in CACHE:
        cached = CACHE[key]
        return ChatResponse(**cached)

    engine = get_query_engine(
        start_date=req.start_date,
        end_date=req.end_date,
        filename=req.filename,
        filename_contains=req.filename_contains,
        keywords_any=req.keywords_any,
        keywords_all=req.keywords_all,
    )
    response = engine.query(req.query)
    source_nodes = getattr(response, "source_nodes", []) or []

    decay_rate = float(os.getenv("TIME_DECAY_RATE", "0.005"))
    nodes = apply_time_decay(source_nodes, decay_rate=decay_rate)
    final_k = int(os.getenv("FINAL_TOP_K", "10"))
    result_nodes = nodes[:final_k]

    result_payload: Dict = {
        "answer": str(response),
        "sources": [_node_to_source(node).dict() for node in result_nodes],
    }
    CACHE[key] = result_payload
    return ChatResponse(**result_payload)
