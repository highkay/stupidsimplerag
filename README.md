# stupidsimplerag

Finance-grade single-node RAG blueprint with heavy preprocessing (LLM-assisted summarization + table narration + keyword expansion) and lightweight runtime retrieval (hybrid Qdrant + API rerank + time-decay).

## Quick start

```bash
git clone <repo>
cd stupidsimplerag
cp .env.example .env  # fill API keys & Qdrant config
docker-compose up -d --build
```

FastAPI on `:8000` exposes:

- `POST /ingest`: upload Markdown files (rich context header injected per chunk, dedup via deterministic IDs).
- `POST /ingest/batch`: same as above but accepts multiple Markdown files in a single multipart request.
- `POST /chat`: hybrid dense+sparse retrieval with API rerank + time decay + LRU semantic cache.

## Architecture

- **Ingest**: Chonkie token splitting + LLM tri-pass (summary, table narrative, synonyms) → embeddable TextNodes.
- **Storage**: Qdrant on-disk hybrid collection (`text-dense` + FastEmbed BM42 `text-sparse`).
- **Retrieval**: LlamaIndex QueryEngine (hybrid search → metadata filters → API rerank → apply_time_decay).
- **Runtime APIs**: FastAPI with TTLCache for chat semantics, Markdown-only ingest (pre-converted upstream).

### Preprocessing prompt

LLM prompt enforces strict JSON schema (`summary`, `table_narrative`, `keywords[]`) and outputs are validated via Pydantic (`LLMAnalysis`). Broken or partial responses are auto-normalized (empty strings / keyword dedupe) to keep ingestion deterministic.

## Configuration (.env)

Fill these key variables:

```
LLM_MODEL, OPENAI_API_KEY, OPENAI_API_BASE
EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_API_BASE, EMBEDDING_DIM
RERANK_API_URL, RERANK_API_KEY, RERANK_MODEL
QDRANT_HOST/PORT/COLLECTION_NAME
QDRANT_HTTPS, QDRANT_URL, QDRANT_API_KEY
TOP_K_RETRIEVAL, TOP_N_RERANK, FINAL_TOP_K, TIME_DECAY_RATE, SPARSE_TOP_K
FASTEMBED_CACHE_PATH, FASTEMBED_SPARSE_MODEL
```

`FASTEMBED_*` controls cached downloads of `Qdrant/bm42-all-minilm-l6-v2-attentions`; `preload_models.py` warms the cache during Docker builds.

## Development

- Install deps: `pip install -r requirements.txt`
- Run API: `uvicorn app.main:app --reload`
- Seed data: `curl -X POST http://localhost:8000/ingest -F "file=@docs/sample.md"`
- Query: `curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"query":"英伟达最新财报表现"}'`

## Notes

- Repo assumes upstream crawlers already normalize documents to Markdown.
- Qdrant collection auto-creates; delete it manually if changing `EMBEDDING_DIM`.
- Tune `SPARSE_TOP_K` / `TOP_K_RETRIEVAL` for recall vs throughput.
- Managed Qdrant example (set `QDRANT_URL`/`QDRANT_API_KEY` accordingly):

  ```bash
  curl -X GET 'https://<waiting-for-cluster-host>:6333' \
    --header 'api-key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.JxVRUBtR2jfv4XxYRvFYnBArICgRzhbPv_b3zScsANo'
  ```
