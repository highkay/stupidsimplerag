import json
import logging
import os
import re
import time
from typing import Any, Callable, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import httpx
from pydantic import Field, PrivateAttr
from llama_index.core import QueryBundle, Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.qdrant.base import (
    DEFAULT_SPARSE_VECTOR_NAME,
    DEFAULT_SPARSE_VECTOR_NAME_OLD,
)
from llama_index.vector_stores.qdrant.utils import fastembed_sparse_encoder
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.openai_utils import (
    OpenAICompatibleLLM,
    get_openai_config,
    get_openai_kwargs,
)


logger = logging.getLogger(__name__)


embedding_base, embedding_key = get_openai_config("EMBEDDING")
llm_base, llm_key = get_openai_config("LLM")
embedding_kwargs = get_openai_kwargs("EMBEDDING")
embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
llm_context_window = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
EMBEDDING_RETRY_BACKOFF = float(os.getenv("EMBEDDING_RETRY_BACKOFF", "1.5"))
fastembed_model_name = os.getenv(
    "FASTEMBED_SPARSE_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions"
)
fastembed_cache_dir = os.getenv("FASTEMBED_CACHE_PATH")


class OpenAICompatibleEmbedding(BaseEmbedding):
    """Embedding wrapper that only relies on OpenAI-compatible REST endpoints."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        api_base: Optional[str],
        dimensions: Optional[int] = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model or "openai-compatible", **kwargs)
        self._model = model
        self._api_key = api_key
        self._api_base = (api_base or "https://api.openai.com/v1").rstrip("/")
        self._dimensions = dimensions
        self._timeout = timeout
        self._max_retries = EMBEDDING_MAX_RETRIES
        self._retry_backoff = EMBEDDING_RETRY_BACKOFF

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _send_embedding_request(self, inputs: List[str]) -> List[dict]:
        payload: dict = {"model": self._model, "input": inputs}
        if self._dimensions:
            payload["dimensions"] = self._dimensions
        url = f"{self._api_base}/embeddings"
        last_error: Exception | None = None
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Embedding request model=%s url=%s batch_size=%d dim=%s first_chunk_len=%s",
                self._model,
                url,
                len(inputs),
                self._dimensions or "full",
                len(inputs[0]) if inputs else 0,
            )
        for attempt in range(1, self._max_retries + 1):
            try:
                response = httpx.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self._timeout,
                )
                response.raise_for_status()
                result = response.json()
                data = result.get("data", [])
                if logger.isEnabledFor(logging.INFO):
                    vector_dim = 0
                    preview = []
                    if data:
                        vector = data[0].get("embedding") or []
                        vector_dim = len(vector)
                        preview = vector[: min(len(vector), 4)]
                    logger.info(
                        "Embedding success model=%s batch=%d dim=%s preview=%s",
                        self._model,
                        len(data),
                        vector_dim or "unknown",
                        preview,
                    )
                if logger.isEnabledFor(logging.DEBUG):
                    preview = []
                    if data:
                        vector = data[0].get("embedding") or []
                        preview = vector[: min(len(vector), 8)]
                    logger.debug(
                        "Embedding response success attempt=%d vectors=%d usage=%s preview=%s",
                        attempt,
                        len(data),
                        result.get("usage"),
                        preview,
                    )
                return data
            except Exception as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    logger.error(
                        "Embedding request failed after %d attempts: %s",
                        attempt,
                        exc,
                    )
                    raise
                delay = self._retry_backoff * attempt
                logger.warning(
                    "Embedding request attempt %d/%d failed: %s -- retrying in %.1fs",
                    attempt,
                    self._max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise last_error or RuntimeError("Embedding request failed")

    def _embedding_request(self, inputs: List[str]) -> List[List[float]]:
        data = self._send_embedding_request(inputs)
        if len(data) == len(inputs):
            return [item["embedding"] for item in data]

        # 后端只支持单文本请求时，回退为逐条发送，避免批量 embedding 丢失
        embeddings: List[List[float]] = []
        for text in inputs:
            single_data = self._send_embedding_request([text])
            if not single_data:
                raise ValueError("Embedding API returned empty response for single input")
            embeddings.append(single_data[0]["embedding"])
        return embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embedding_request([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embedding_request([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embedding_request(texts)


class FixedDimensionEmbedding(BaseEmbedding):
    """Ensure embedding outputs never exceed the configured dimension size."""

    def __init__(self, base_model: BaseEmbedding, target_dim: Optional[int]) -> None:
        base_name = getattr(base_model, "model_name", None) or getattr(
            base_model, "_model_name", "embedding"
        )
        super().__init__(model_name=f"{base_name}-fixed-dim")
        self._base_model = base_model
        self._target_dim = target_dim or 0

    def __getattr__(self, name: str) -> Any:
        """Proxy any other attribute access to the wrapped embedding."""
        return getattr(self._base_model, name)

    def _trim_vector(self, embedding: Optional[List[float]]) -> Optional[List[float]]:
        if embedding is None or self._target_dim <= 0:
            return embedding
        if len(embedding) > self._target_dim:
            return embedding[: self._target_dim]
        return embedding

    def _trim_batch(self, embeddings: Optional[List[List[float]]]) -> Optional[List[List[float]]]:
        if embeddings is None or self._target_dim <= 0:
            return embeddings
        return [self._trim_vector(vec) or [] for vec in embeddings]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._trim_vector(self._base_model.get_query_embedding(query)) or []

    async def _aget_query_embedding(self, query: str) -> List[float]:
        base_async = getattr(self._base_model, "aget_query_embedding", None)
        if callable(base_async):
            embedding = await base_async(query)
        else:
            embedding = self._base_model.get_query_embedding(query)
        return self._trim_vector(embedding) or []

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._trim_vector(self._base_model.get_text_embedding(text)) or []

    async def _aget_text_embedding(self, text: str) -> List[float]:
        base_async = getattr(self._base_model, "aget_text_embedding", None)
        if callable(base_async):
            embedding = await base_async(text)
        else:
            embedding = self._base_model.get_text_embedding(text)
        return self._trim_vector(embedding) or []

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        base_batch = getattr(self._base_model, "get_text_embeddings", None)
        if callable(base_batch):
            embeddings = base_batch(texts)
        else:
            embeddings = [self._base_model.get_text_embedding(text) for text in texts]
        trimmed = self._trim_batch(embeddings) or []
        return [vec if vec else [] for vec in trimmed]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        base_async = getattr(self._base_model, "aget_text_embeddings", None)
        if callable(base_async):
            embeddings = await base_async(texts)
        else:
            embeddings = []
            for text in texts:
                embeddings.append(
                    await self._base_model.aget_text_embedding(text)
                    if hasattr(self._base_model, "aget_text_embedding")
                    else self._base_model.get_text_embedding(text)
                )
        trimmed = self._trim_batch(embeddings) or []
        return [vec if vec else [] for vec in trimmed]


def _init_embedding_model() -> BaseEmbedding:
    model_name = os.getenv("EMBEDDING_MODEL")
    base_model: BaseEmbedding
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Initializing embedding model=%s dim=%s", model_name, embedding_dim
        )
    try:
        base_model = OpenAIEmbedding(
            model=model_name,
            dimensions=embedding_dim,
            **embedding_kwargs,
        )
    except ValueError:
        base_model = OpenAICompatibleEmbedding(
            model=model_name,
            api_key=embedding_key,
            api_base=embedding_base,
            dimensions=embedding_dim,
        )
    if embedding_dim > 0:
        return FixedDimensionEmbedding(base_model, embedding_dim)
    return base_model


Settings.embed_model = _init_embedding_model()
if logger.isEnabledFor(logging.INFO):
    logger.info(
        "Embedding model ready: name=%s dim=%s",
        getattr(Settings.embed_model, "model_name", None)
        or getattr(Settings.embed_model, "_model_name", "unknown"),
        embedding_dim,
    )


class APIReranker(BaseNodePostprocessor):
    """HTTP reranker that calls standard OpenAI chat completion endpoints."""

    top_n: int = Field(default=10, description="Number of nodes to return.")
    model: Optional[str] = Field(default=None, description="Reranking model name.")
    api_url: Optional[str] = Field(
        default=None, description="Full rerank endpoint, e.g. https://host/v1/rerank"
    )
    return_documents: bool = Field(
        default=False, description="Whether to ask API to return documents."
    )
    timeout: float = Field(default=180.0, description="Request timeout in seconds.")
    _api_key: Optional[str] = PrivateAttr()
    _client: Any = PrivateAttr()
    _base_url: Optional[str] = PrivateAttr()

    def __init__(
        self,
        top_n: int = 10,
        *,
        timeout: Optional[float] = None,
        return_documents: Optional[bool] = None,
    ):
        model_name = os.getenv("RERANK_MODEL")
        api_key = os.getenv("RERANK_API_KEY") or os.getenv("OPENAI_API_KEY")
        raw_url = os.getenv("RERANK_API_URL") or os.getenv("RERANK_API_BASE")
        resolved_base = self._normalize_base_url(raw_url)
        resolved_url = f"{resolved_base}/chat/completions" if resolved_base else None
        resolved_timeout = timeout or float(os.getenv("RERANK_TIMEOUT", "180"))
        resolved_return_docs = (
            return_documents
            if return_documents is not None
            else os.getenv("RERANK_RETURN_DOCUMENTS", "false").lower() == "true"
        )

        super().__init__(
            top_n=top_n,
            model=model_name,
            api_url=resolved_url,
            return_documents=resolved_return_docs,
            timeout=resolved_timeout,
        )

        self._api_key = api_key
        self._base_url = resolved_base or os.getenv("OPENAI_API_BASE")

        try:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self.timeout,
            )
            logger.info(
                "APIReranker ready openai-style base=%s endpoint=%s top_n=%s model=%s timeout=%ss",
                self._base_url,
                self.api_url,
                top_n,
                model_name,
                self.timeout,
            )
        except Exception as exc:
            logger.warning("Rerank disabled: failed to init OpenAI client (%s)", exc)
            self._client = None

    @staticmethod
    def _normalize_base_url(raw_url: Optional[str]) -> str:
        """Normalize any given URL to an OpenAI-compatible base without the /rerank suffix."""
        fallback = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        if raw_url:
            base = raw_url.strip()
        else:
            base = fallback
        if not base:
            base = fallback
        base = base.rstrip("/")
        lowered = base.lower()
        for suffix in ("/v1/rerank", "/rerank"):
            if lowered.endswith(suffix):
                base = base[: -len(suffix)]
                base = base.rstrip("/")
                lowered = base.lower()
        if not base.lower().endswith("/v1"):
            base = f"{base}/v1"
        return base

    def _call_rerank_api(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if not nodes or not self._client or not self.model:
            return nodes[: self.top_n]

        messages = self._build_messages(query_bundle.query_str, nodes)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("Rerank request failed: %s", exc)
            return nodes[: self.top_n]

        results = self._parse_rerank_response(response, len(nodes))

        ranked: List[NodeWithScore] = []
        for item in results:
            idx = getattr(item, "index", None)
            if idx is None and isinstance(item, dict):
                idx = item.get("index")
            if idx is None or idx >= len(nodes):
                continue
            score = getattr(item, "relevance_score", None)
            if score is None and isinstance(item, dict):
                score = item.get("relevance_score", item.get("score"))
            node = nodes[idx]
            if score is not None:
                node.score = float(score)
            ranked.append(node)

        if not ranked:
            return nodes[: self.top_n]

        ranked.sort(key=lambda x: (x.score or 0.0), reverse=True)
        return ranked[: self.top_n]

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if not nodes or not query_bundle:
            return nodes[: self.top_n]
        return self._call_rerank_api(nodes, query_bundle)

    @staticmethod
    def _build_messages(query: str, nodes: List[NodeWithScore]) -> List[dict]:
        """Construct a prompt that asks the model to rerank documents and return JSON only."""
        doc_blocks = []
        for idx, node in enumerate(nodes):
            text = node.node.get_content() if hasattr(node, "node") else ""
            snippet = text[:800]
            doc_blocks.append(f"[{idx}] {snippet}")
        user_prompt = (
            "You are a reranker. Given a query and candidate documents, return a JSON "
            "array of objects with fields `index` (int) and `relevance_score` (float, higher means more relevant). "
            "Only include the top items. Do not add explanations or extra text.\n\n"
            f"Query: {query}\n\n"
            "Documents:\n" + "\n\n".join(doc_blocks) + "\n\n"
            "Output JSON array like: [{\"index\":0,\"relevance_score\":12.3},{\"index\":4,\"relevance_score\":11.8}]"
        )
        return [
            {"role": "system", "content": "Return only JSON for reranking results."},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_rerank_response(self, response: Any, total_nodes: int) -> List[dict]:
        """Extract rerank results from a chat completion response."""
        content = None
        try:
            choices = getattr(response, "choices", None) or []
            if choices:
                content = getattr(choices[0].message, "content", None)
        except Exception:
            pass

        if content is None and isinstance(response, dict):
            choices = response.get("choices") or []
            if choices and isinstance(choices[0], dict):
                content = choices[0].get("message", {}).get("content")

        if not content:
            return []

        # strip code fences if present
        fenced = re.findall(r"```(?:json)?\\s*(.*?)\\s*```", content, flags=re.S)
        if fenced:
            content = fenced[0]

        try:
            data = json.loads(content)
        except Exception:
            logger.warning("Failed to parse rerank JSON response: %s", content[:200])
            return []

        if isinstance(data, dict):
            if "data" in data:
                data = data["data"]
            elif "results" in data:
                data = data["results"]

        if not isinstance(data, list):
            return []

        cleaned = []
        for item in data:
            idx = None
            score = None
            if isinstance(item, dict):
                idx = item.get("index")
                score = item.get("relevance_score", item.get("score"))
            if idx is None or idx >= total_nodes or idx < 0:
                continue
            if score is None:
                continue
            cleaned.append({"index": int(idx), "relevance_score": float(score)})
        return cleaned


class SafeQdrantVectorStore(QdrantVectorStore):
    """Qdrant adapter that normalizes outputs and enforces FastEmbed sparse encoders."""

    def _resolve_sparse_model(self, override: Optional[str] = None) -> str:
        return override or fastembed_model_name

    def _build_fastembed_encoder(
        self, fastembed_sparse_model: Optional[str]
    ) -> Callable[[List[str]], Tuple[List[List[int]], List[List[float]]]]:
        return fastembed_sparse_encoder(
            model_name=self._resolve_sparse_model(fastembed_sparse_model),
            cache_dir=fastembed_cache_dir,
        )

    def get_default_sparse_doc_encoder(
        self,
        collection_name: str,
        fastembed_sparse_model: Optional[str] = None,
    ):
        if self._client is not None and self.use_old_sparse_encoder(collection_name):
            self.sparse_vector_name = DEFAULT_SPARSE_VECTOR_NAME_OLD
        return self._build_fastembed_encoder(fastembed_sparse_model)

    def get_default_sparse_query_encoder(
        self,
        collection_name: str,
        fastembed_sparse_model: Optional[str] = None,
    ):
        if self._client is not None and self.use_old_sparse_encoder(collection_name):
            self.sparse_vector_name = DEFAULT_SPARSE_VECTOR_NAME_OLD
        return self._build_fastembed_encoder(fastembed_sparse_model)

    def query(self, *args: Any, **kwargs: Any):
        result = super().query(*args, **kwargs)
        if result.ids is None:
            result.ids = []
        if result.similarities is None:
            result.similarities = []
        if result.nodes is None:
            result.nodes = []
        return result

    def _reinitialize_sparse_encoders(self) -> None:
        """Override default behavior to keep FastEmbed encoders even for legacy collections."""
        if not self.enable_hybrid:
            return
        if not self._user_provided_sparse_doc_fn:
            self._sparse_doc_fn = self._build_fastembed_encoder(
                self.fastembed_sparse_model
            )
        if not self._user_provided_sparse_query_fn:
            self._sparse_query_fn = self._build_fastembed_encoder(
                self.fastembed_sparse_model
            )


def _ensure_hybrid_collection(
    client: QdrantClient, collection_name: str, vector_size: int
) -> None:
    if client.collection_exists(collection_name):
        pass
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "text-dense": qdrant_models.VectorParams(
                    size=vector_size, distance=qdrant_models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                DEFAULT_SPARSE_VECTOR_NAME: qdrant_models.SparseVectorParams(
                    index=qdrant_models.SparseIndexParams()
                )
            },
        )
    _ensure_payload_indexes(client, collection_name)


def _ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Create payload indexes used by filters (idempotent if they already exist)."""
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="date_numeric",
            field_schema=qdrant_models.PayloadSchemaType.INTEGER,
        )
    except Exception:
        pass
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="doc_hash",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass


def _append_port_if_missing(raw_url: str, default_port: int) -> str:
    parsed = urlparse(raw_url)
    if not parsed.netloc:
        return raw_url
    host_port = parsed.netloc.split("@")[-1]
    if ":" in host_port:
        return raw_url
    new_netloc = parsed.netloc + f":{default_port}"
    return urlunparse(parsed._replace(netloc=new_netloc))


def _build_qdrant_store() -> QdrantVectorStore:
    host = os.getenv("QDRANT_HOST", "qdrant")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    collection = os.getenv("COLLECTION_NAME", "financial_reports")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
    use_https = os.getenv("QDRANT_HTTPS", "false").lower() == "true"
    client_timeout = float(os.getenv("QDRANT_CLIENT_TIMEOUT", "10"))

    def _create_store(q_client: QdrantClient) -> QdrantVectorStore:
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Ensuring Qdrant hybrid collection=%s dim=%s",
                collection,
                embedding_dim,
            )
        _ensure_hybrid_collection(q_client, collection, embedding_dim)
        return SafeQdrantVectorStore(
            client=q_client,
            collection_name=collection,
            enable_hybrid=True,
            dense_vector_name="text-dense",
            sparse_vector_name=DEFAULT_SPARSE_VECTOR_NAME,
            fastembed_sparse_model=fastembed_model_name,
        )

    if qdrant_url:
        qdrant_url = _append_port_if_missing(qdrant_url, port)
        try:
            if logger.isEnabledFor(logging.INFO):
                logger.info("Connecting to managed Qdrant url=%s", qdrant_url)
            remote_client = QdrantClient(
                url=qdrant_url, api_key=qdrant_api_key, timeout=client_timeout
            )
            return _create_store(remote_client)
        except Exception as exc:
            logger.warning(
                "Failed to reach managed Qdrant cluster at %s (%s), falling back to host %s:%s",
                qdrant_url,
                exc,
                host,
                port,
            )

    resolved_host = host
    running_in_container = os.path.exists("/.dockerenv")
    if resolved_host == "qdrant" and not running_in_container:
        resolved_host = "127.0.0.1"

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Connecting to Qdrant host=%s port=%s https=%s",
            resolved_host,
            port,
            use_https,
        )
    local_client = QdrantClient(
        host=resolved_host,
        port=port,
        https=use_https,
        api_key=qdrant_api_key if use_https else None,
        timeout=client_timeout,
    )
    return _create_store(local_client)


vector_store = _build_qdrant_store()
storage_context = StorageContext.from_defaults(vector_store=vector_store)
_INDEX_CACHE: VectorStoreIndex | None = None


def _get_index() -> VectorStoreIndex:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        if logger.isEnabledFor(logging.INFO):
            logger.info("Building VectorStoreIndex with hybrid store (cached)")
        _INDEX_CACHE = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
    return _INDEX_CACHE


def insert_nodes(nodes: List[TextNode]) -> None:
    if not nodes:
        logger.debug("insert_nodes called with empty payload; skipping")
        return
    logger.info("Inserting %d nodes into Qdrant (will trigger embedding)", len(nodes))
    start = time.perf_counter()
    index = _get_index()
    index.insert_nodes(nodes)
    elapsed = time.perf_counter() - start
    logger.debug(
        "Qdrant insert complete collection=%s count=%d duration=%.2fs",
        getattr(vector_store, "collection_name", "unknown"),
        len(nodes),
        elapsed,
    )


def _date_str_to_int(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) < 8:
        return None
    return int(digits[:8])


def get_query_engine(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filename: Optional[str] = None,
    filename_contains: Optional[str] = None,
):
    filters: Optional[MetadataFilters] = None
    filter_items: List[MetadataFilter] = []
    start_num = _date_str_to_int(start_date)
    end_num = _date_str_to_int(end_date)
    if start_num is not None:
        filter_items.append(
            MetadataFilter(
                key="date_numeric", value=start_num, operator=FilterOperator.GTE
            )
        )
    if end_num is not None:
        filter_items.append(
            MetadataFilter(
                key="date_numeric", value=end_num, operator=FilterOperator.LTE
            )
        )
    if filename:
        filter_items.append(
            MetadataFilter(key="filename", value=filename, operator=FilterOperator.EQ)
        )
    if filename_contains:
        filter_items.append(
            MetadataFilter(
                key="filename",
                value=filename_contains,
                operator=FilterOperator.TEXT_MATCH_INSENSITIVE,
            )
        )
    if filter_items:
        filters = MetadataFilters(filters=filter_items)

    index = _get_index()
    top_k = int(os.getenv("TOP_K_RETRIEVAL", "100"))
    raw_rerank_n = os.getenv("TOP_N_RERANK", "20")
    logger.info("get_query_engine: raw TOP_N_RERANK env=%s", raw_rerank_n)
    rerank_top_n = int(raw_rerank_n)
    logger.info("get_query_engine: parsed rerank_top_n=%s", rerank_top_n)
    sparse_top_k = int(os.getenv("SPARSE_TOP_K", "12"))
    logger.info("Creating query_engine with APIReranker(top_n=%s)", rerank_top_n)
    return index.as_query_engine(
        similarity_top_k=top_k,
        vector_store_query_mode="hybrid",
        sparse_top_k=sparse_top_k,
        filters=filters,
        node_postprocessors=[APIReranker(top_n=rerank_top_n)],
    )


def get_collection_metrics() -> dict:
    """Expose lightweight collection metrics for the web dashboard."""
    metrics = {
        "collection_name": getattr(vector_store, "collection_name", None),
        "vectors_count": None,
        "segments_count": None,
        "status": "unknown",
        "points_count": None,
    }
    collection = metrics["collection_name"]
    client = getattr(vector_store, "client", None)
    if not client or not collection:
        return metrics

    try:
        info = client.collection_info(collection_name=collection)
        metrics["status"] = getattr(info, "status", metrics["status"])
        metrics["vectors_count"] = getattr(info, "vectors_count", None)
        metrics["segments_count"] = getattr(info, "segments_count", None)
    except Exception:
        pass

    try:
        count_resp = client.count(collection_name=collection, exact=False)
        metrics["points_count"] = getattr(count_resp, "count", None)
    except Exception:
        pass

    return metrics


Settings.llm = OpenAICompatibleLLM(
    model=os.getenv("LLM_MODEL"),
    api_key=llm_key,
    api_base=llm_base,
    temperature=0.1,
    context_window=llm_context_window,
)


def doc_exists(doc_hash: str) -> bool:
    """Return True if collection already has a payload with the given doc_hash."""
    if not doc_hash:
        return False
    client = getattr(vector_store, "client", None)
    collection = getattr(vector_store, "collection_name", None)
    if not client or not collection:
        return False
    _ensure_payload_indexes(client, collection)
    try:
        flt = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="doc_hash", match=qdrant_models.MatchValue(value=doc_hash)
                )
            ]
        )
        result = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            with_payload=False,
            limit=1,
        )
        points = result[0] if isinstance(result, (list, tuple)) else result
        return bool(points)
    except Exception as exc:
        if "Index required" in str(exc):
            try:
                _ensure_payload_indexes(client, collection)
                result = client.scroll(
                    collection_name=collection,
                    scroll_filter=flt,
                    with_payload=False,
                    limit=1,
                )
                points = result[0] if isinstance(result, (list, tuple)) else result
                return bool(points)
            except Exception as inner_exc:
                logger.warning("doc_exists retry failed: %s", inner_exc)
                return False
        logger.warning("doc_exists check failed: %s", exc)
        return False
