import json
import logging
import os
import re
import time
import datetime
import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, List, Optional, Tuple
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
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import RetryCallState, retry, stop_after_attempt, wait_exponential

from app.models import (
    GroundingCandidate,
    GroundingDocumentSelector,
    GroundingRequest,
    GroundingResponse,
    GroundingResult,
    GroundingDocumentInfo,
    GroundingExcerpt,
)
from app.openai_utils import (
    build_llm,
    get_openai_config,
    get_openai_kwargs,
)
from app.preprocess import classify_document_block, split_document_blocks
from app.time_utils import today_in_app_tz


logger = logging.getLogger(__name__)

RELATION_KEYWORDS = (
    "竞争",
    "替代",
    "受益",
    "订单",
    "客户",
    "供应链",
    "产能",
    "风险",
    "份额",
    "进入",
    "导入",
)

GENERATED_STOCK_ALIAS_CONTEXT_KEYWORDS = (
    "推荐",
    "看好",
    "关注",
    "布局",
    "增持",
    "受益",
    "弹性",
    "催化",
)

CHINESE_STOCK_ALIAS_SUFFIXES = (
    "控股股份有限公司",
    "集团股份有限公司",
    "股份有限公司",
    "控股有限公司",
    "集团有限公司",
    "有限公司",
    "控股股份",
    "集团股份",
    "控股",
    "集团",
    "股份",
    "银行",
    "证券",
    "保险",
)


@dataclass(frozen=True)
class _GroundingChunk:
    filename: str
    scope: Optional[str]
    doc_hash: Optional[str]
    date: Optional[str]
    metadata: dict
    full_text: str
    original_text: str
    order_key: tuple


@dataclass(frozen=True)
class _GroundingBlock:
    filename: str
    scope: Optional[str]
    doc_hash: Optional[str]
    date: Optional[str]
    section_type: str
    heading_path: Optional[str]
    text: str
    block_index: int
    chunk_index: int


embedding_base, embedding_key = get_openai_config("EMBEDDING")
llm_base, llm_key = get_openai_config("LLM")
embedding_kwargs = get_openai_kwargs("EMBEDDING")
embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
embedding_query_prefix = os.getenv("EMBEDDING_QUERY_PREFIX", "")
embedding_document_prefix = os.getenv("EMBEDDING_DOCUMENT_PREFIX", "")
embedding_timeout = float(os.getenv("EMBEDDING_TIMEOUT", "60"))
llm_context_window = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
EMBEDDING_RETRY_BACKOFF = float(os.getenv("EMBEDDING_RETRY_BACKOFF", "1.5"))
EMBEDDING_REQUEST_BATCH_SIZE = max(
    1, int(os.getenv("EMBEDDING_REQUEST_BATCH_SIZE", "4"))
)
EMBEDDING_CONCURRENCY = max(1, int(os.getenv("EMBEDDING_CONCURRENCY", "2")))
RERANK_MAX_RETRIES = int(os.getenv("RERANK_MAX_RETRIES", "3"))
RERANK_RETRY_BACKOFF = float(os.getenv("RERANK_RETRY_BACKOFF", "1.5"))
fastembed_model_name = os.getenv(
    "FASTEMBED_SPARSE_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions"
)
_default_fastembed_cache_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "model_cache",
)
fastembed_cache_dir = os.getenv("FASTEMBED_CACHE_PATH") or (
    _default_fastembed_cache_dir if os.path.isdir(_default_fastembed_cache_dir) else None
)


def _normalize_embedding_prefix(value: Optional[str]) -> str:
    if value is None:
        return ""
    normalized = value.strip()
    if not normalized:
        return ""
    return normalized if normalized.endswith(" ") else f"{normalized} "


def _log_rerank_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    sleep = retry_state.next_action.sleep if retry_state.next_action else 0
    logger.warning(
        "Rerank request attempt %d/%d failed: %s -- retrying in %.1fs",
        retry_state.attempt_number,
        RERANK_MAX_RETRIES,
        exc,
        sleep,
    )


class OpenAICompatibleEmbedding(BaseEmbedding):
    """Embedding wrapper that only relies on OpenAI-compatible REST endpoints."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        api_base: Optional[str],
        dimensions: Optional[int] = None,
        timeout: float = 30.0,
        query_prefix: str = "",
        text_prefix: str = "",
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
        self._request_batch_size = EMBEDDING_REQUEST_BATCH_SIZE
        self._max_concurrency = EMBEDDING_CONCURRENCY
        self._query_prefix = _normalize_embedding_prefix(query_prefix)
        self._text_prefix = _normalize_embedding_prefix(text_prefix)
        self._sync_semaphore = threading.BoundedSemaphore(self._max_concurrency)
        self._async_semaphore = asyncio.Semaphore(self._max_concurrency)
        
        # Persistent clients for connection pooling
        self._client = httpx.Client(timeout=self._timeout)
        self._aclient = httpx.AsyncClient(timeout=self._timeout)

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _prepare_query_input(self, query: str) -> str:
        return f"{self._query_prefix}{query}" if self._query_prefix else query

    def _prepare_text_input(self, text: str) -> str:
        return f"{self._text_prefix}{text}" if self._text_prefix else text

    def _prepare_text_inputs(self, texts: List[str]) -> List[str]:
        if not self._text_prefix:
            return texts
        return [self._prepare_text_input(text) for text in texts]

    def _iter_input_batches(self, inputs: List[str]) -> List[List[str]]:
        return [
            inputs[index : index + self._request_batch_size]
            for index in range(0, len(inputs), self._request_batch_size)
        ]

    def _send_embedding_request(self, inputs: List[str]) -> List[dict]:
        payload: dict = {"model": self._model, "input": inputs}
        if self._dimensions:
            payload["dimensions"] = self._dimensions
        url = f"{self._api_base}/embeddings"
        last_error: Exception | None = None
        
        for attempt in range(1, self._max_retries + 1):
            try:
                with self._sync_semaphore:
                    response = self._client.post(
                        url,
                        json=payload,
                        headers=self._headers(),
                    )
                response.raise_for_status()
                result = response.json()
                data = result.get("data", [])
                return data
            except Exception as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    logger.error("Embedding request failed after %d attempts: %s", attempt, exc)
                    raise
                delay = self._retry_backoff * attempt
                time.sleep(delay)
        raise last_error or RuntimeError("Embedding request failed")

    async def _asend_embedding_request(self, inputs: List[str]) -> List[dict]:
        payload: dict = {"model": self._model, "input": inputs}
        if self._dimensions:
            payload["dimensions"] = self._dimensions
        url = f"{self._api_base}/embeddings"
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                async with self._async_semaphore:
                    response = await self._aclient.post(
                        url,
                        json=payload,
                        headers=self._headers(),
                    )
                response.raise_for_status()
                result = response.json()
                data = result.get("data", [])
                return data
            except Exception as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    logger.error("Async embedding request failed after %d attempts: %s", attempt, exc)
                    raise
                delay = self._retry_backoff * attempt
                await asyncio.sleep(delay)
        raise last_error or RuntimeError("Async embedding request failed")

    def _embedding_request(self, inputs: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for batch in self._iter_input_batches(inputs):
            data = self._send_embedding_request(batch)
            if len(data) == len(batch):
                embeddings.extend(item["embedding"] for item in data)
                continue
            for text in batch:
                single_data = self._send_embedding_request([text])
                if not single_data:
                    raise ValueError("Embedding API returned empty response for single input")
                embeddings.append(single_data[0]["embedding"])
        return embeddings

    async def _aembedding_request(self, inputs: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for batch in self._iter_input_batches(inputs):
            data = await self._asend_embedding_request(batch)
            if len(data) == len(batch):
                embeddings.extend(item["embedding"] for item in data)
                continue
            for text in batch:
                single_data = await self._asend_embedding_request([text])
                if not single_data:
                    raise ValueError("Embedding API returned empty response for single input")
                embeddings.append(single_data[0]["embedding"])
        return embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embedding_request([self._prepare_query_input(query)])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await self._aembedding_request([self._prepare_query_input(query)]))[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embedding_request([self._prepare_text_input(text)])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await self._aembedding_request([self._prepare_text_input(text)]))[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embedding_request(self._prepare_text_inputs(texts))
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self._aembedding_request(self._prepare_text_inputs(texts))


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
    query_prefix = embedding_query_prefix
    text_prefix = embedding_document_prefix
    base_model: BaseEmbedding
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Initializing embedding model=%s dim=%s query_prefix=%s text_prefix=%s",
            model_name,
            embedding_dim,
            bool(query_prefix),
            bool(text_prefix),
        )
    if query_prefix or text_prefix:
        base_model = OpenAICompatibleEmbedding(
            model=model_name,
            api_key=embedding_key,
            api_base=embedding_base,
            dimensions=embedding_dim,
            timeout=embedding_timeout,
            query_prefix=query_prefix,
            text_prefix=text_prefix,
        )
    else:
        try:
            base_model = OpenAIEmbedding(
                model=model_name,
                dimensions=embedding_dim,
                timeout=embedding_timeout,
                **embedding_kwargs,
            )
        except ValueError:
            base_model = OpenAICompatibleEmbedding(
                model=model_name,
                api_key=embedding_key,
                api_base=embedding_base,
                dimensions=embedding_dim,
                timeout=embedding_timeout,
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


_RERANK_HTTP_CLIENT: Optional[httpx.Client] = None
_RERANK_HTTP_ACLIENT: Optional[httpx.AsyncClient] = None
_RERANK_OPENAI_CLIENT: Optional[Any] = None
_RERANK_OPENAI_ACLIENT: Optional[Any] = None


def _get_rerank_clients(timeout: float) -> Tuple[httpx.Client, httpx.AsyncClient, Any, Any]:
    global _RERANK_HTTP_CLIENT, _RERANK_HTTP_ACLIENT, _RERANK_OPENAI_CLIENT, _RERANK_OPENAI_ACLIENT
    
    # Initialize HTTP clients if needed
    if _RERANK_HTTP_CLIENT is None:
        _RERANK_HTTP_CLIENT = httpx.Client(timeout=timeout)
    if _RERANK_HTTP_ACLIENT is None:
        _RERANK_HTTP_ACLIENT = httpx.AsyncClient(timeout=timeout)
        
    # Initialize OpenAI clients if needed
    api_key = os.getenv("RERANK_API_KEY") or os.getenv("OPENAI_API_KEY")
    raw_url = os.getenv("RERANK_API_URL") or os.getenv("RERANK_API_BASE")
    
    # Determine base URL logic similar to APIReranker init, but simplified for client creation
    # We just need the base_url for OpenAI client
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
    
    if _RERANK_OPENAI_CLIENT is None:
        try:
            _RERANK_OPENAI_CLIENT = OpenAI(
                api_key=api_key,
                base_url=base,
                timeout=timeout,
            )
        except Exception as exc:
            logger.warning("Failed to init cached OpenAI client for rerank: %s", exc)
            
    if _RERANK_OPENAI_ACLIENT is None:
        try:
            from openai import AsyncOpenAI
            _RERANK_OPENAI_ACLIENT = AsyncOpenAI(
                api_key=api_key,
                base_url=base,
                timeout=timeout,
            )
        except Exception as exc:
            logger.warning("Failed to init cached AsyncOpenAI client for rerank: %s", exc)
            
    return _RERANK_HTTP_CLIENT, _RERANK_HTTP_ACLIENT, _RERANK_OPENAI_CLIENT, _RERANK_OPENAI_ACLIENT


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
    _aclient: Any = PrivateAttr()
    _http_client: Any = PrivateAttr()
    _http_aclient: Any = PrivateAttr()
    _base_url: Optional[str] = PrivateAttr()
    _use_rerank_endpoint: bool = PrivateAttr()

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
        use_rerank_endpoint = self._looks_like_rerank_endpoint(raw_url)
        resolved_base = None if use_rerank_endpoint else self._normalize_base_url(raw_url)
        resolved_url = (
            self._normalize_rerank_url(raw_url or "")
            if use_rerank_endpoint
            else f"{resolved_base}/chat/completions" if resolved_base else None
        )
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
        self._use_rerank_endpoint = use_rerank_endpoint
        self._base_url = resolved_base or os.getenv("OPENAI_API_BASE")
        
        # Use cached clients to avoid per-request connection overhead
        (
            self._http_client, 
            self._http_aclient, 
            self._client, 
            self._aclient
        ) = _get_rerank_clients(self.timeout)

        if self._use_rerank_endpoint:
            # If using raw endpoint, we might not need OpenAI clients, but they are cached anyway
            # We enforce they are None here if we strictly don't want to use them, 
            # but the logic in _call_rerank_api checks _use_rerank_endpoint first.
            pass
        else:
             if not self._client:
                 logger.warning("Rerank disabled: OpenAI client not initialized")

        if logger.isEnabledFor(logging.INFO):
             type_str = "rerank-endpoint" if self._use_rerank_endpoint else "openai-style"
             logger.info(
                "APIReranker initialized type=%s url=%s top_n=%s model=%s timeout=%ss",
                type_str,
                self.api_url,
                top_n,
                model_name,
                self.timeout,
            )

    @staticmethod
    def _looks_like_rerank_endpoint(raw_url: Optional[str]) -> bool:
        if not raw_url:
            return False
        lowered = raw_url.strip().lower().rstrip("/")
        return "/rerank" in lowered

    @staticmethod
    def _normalize_rerank_url(raw_url: str) -> str:
        url = raw_url.strip()
        if not url:
            return ""
        url = url.rstrip("/")
        lowered = url.lower()
        if lowered.endswith("/rerank"):
            return url
        if "/rerank" in lowered:
            return url
        if lowered.endswith("/v1"):
            return f"{url}/rerank"
        return f"{url}/v1/rerank"

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
        if not nodes or not self.model:
            return nodes[: self.top_n]
        if self._use_rerank_endpoint:
            return self._call_rerank_endpoint(nodes, query_bundle)
        if not self._client:
            return nodes[: self.top_n]
        return self._call_chat_rerank(nodes, query_bundle)

    def _call_rerank_endpoint(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self.model,
            "query": query_bundle.query_str,
            "documents": [n.node.get_content() for n in nodes],
            "top_n": self.top_n,
            "return_documents": self.return_documents,
        }
        try:
            data = self._perform_rerank_request(payload, headers)
            results = self._extract_rerank_items(data)
        except Exception as exc:
            logger.warning(
                "Rerank request failed after retries (endpoint=%s): %s",
                self.api_url,
                exc,
            )
            return nodes[: self.top_n]
        return self._apply_results(nodes, results)

    def _call_chat_rerank(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
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
        return self._apply_results(nodes, results)

    def _apply_results(
        self, nodes: List[NodeWithScore], results: List[dict]
    ) -> List[NodeWithScore]:
        if not results:
            return nodes[: self.top_n]
        ranked: List[NodeWithScore] = []
        for item in results:
            idx = getattr(item, "index", None)
            if idx is None and isinstance(item, dict):
                idx = item.get("index")
            if idx is None or idx >= len(nodes) or idx < 0:
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

    async def _apostprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if not nodes or not query_bundle:
            return nodes[: self.top_n]
        return await self._acall_rerank_api(nodes, query_bundle)

    async def _acall_rerank_api(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if not nodes or not self.model:
            return nodes[: self.top_n]
        if self._use_rerank_endpoint:
            return await self._acall_rerank_endpoint(nodes, query_bundle)
        if not self._aclient:
            return nodes[: self.top_n]
        return await self._acall_chat_rerank(nodes, query_bundle)

    async def _acall_rerank_endpoint(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self.model,
            "query": query_bundle.query_str,
            "documents": [n.node.get_content() for n in nodes],
            "top_n": self.top_n,
            "return_documents": self.return_documents,
        }
        try:
            data = await self._aperform_rerank_request(payload, headers)
            results = self._extract_rerank_items(data)
        except Exception as exc:
            logger.warning(
                "Async rerank request failed after retries (endpoint=%s): %s",
                self.api_url,
                exc,
            )
            return nodes[: self.top_n]
        return self._apply_results(nodes, results)

    async def _acall_chat_rerank(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        messages = self._build_messages(query_bundle.query_str, nodes)
        try:
            response = await self._aclient.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("Async rerank request failed: %s", exc)
            return nodes[: self.top_n]

        results = self._parse_rerank_response(response, len(nodes))
        return self._apply_results(nodes, results)

    @retry(
        reraise=True,
        stop=stop_after_attempt(RERANK_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=RERANK_RETRY_BACKOFF,
            min=RERANK_RETRY_BACKOFF,
            max=RERANK_RETRY_BACKOFF * 4,
        ),
        before_sleep=_log_rerank_retry,
    )
    async def _aperform_rerank_request(self, payload: dict, headers: dict) -> dict:
        response = await self._http_aclient.post(self.api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    @retry(
        reraise=True,
        stop=stop_after_attempt(RERANK_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=RERANK_RETRY_BACKOFF,
            min=RERANK_RETRY_BACKOFF,
            max=RERANK_RETRY_BACKOFF * 4,
        ),
        before_sleep=_log_rerank_retry,
    )
    def _perform_rerank_request(self, payload: dict, headers: dict) -> dict:
        response = self._http_client.post(self.api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

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

        return self._extract_rerank_items(data)

    @staticmethod
    def _extract_rerank_items(data: Any) -> List[dict]:
        items: List[Any] = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in ("results", "data"):
                value = data.get(key)
                if isinstance(value, list):
                    items = value
                    break
        cleaned: List[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if idx is None or score is None:
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
        try:
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
        except UnexpectedResponse as exc:
            if exc.status_code != 409 or not client.collection_exists(collection_name):
                raise
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
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="scope",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="filename",
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
    global _QDRANT_CLIENT_CACHE, _QDRANT_ASYNC_CLIENT_CACHE
    host = os.getenv("QDRANT_HOST", "qdrant")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    collection = os.getenv("COLLECTION_NAME", "financial_reports")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
    use_https = os.getenv("QDRANT_HTTPS", "false").lower() == "true"
    client_timeout = float(os.getenv("QDRANT_CLIENT_TIMEOUT", "60"))

    def _create_store(q_client: QdrantClient, q_aclient: AsyncQdrantClient) -> QdrantVectorStore:
        _QDRANT_CLIENT_CACHE = q_client
        _QDRANT_ASYNC_CLIENT_CACHE = q_aclient
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Ensuring Qdrant hybrid collection=%s dim=%s",
                collection,
                embedding_dim,
            )
        _ensure_hybrid_collection(q_client, collection, embedding_dim)
        return SafeQdrantVectorStore(
            client=q_client,
            aclient=q_aclient,
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
            remote_aclient = AsyncQdrantClient(
                url=qdrant_url, api_key=qdrant_api_key, timeout=client_timeout
            )
            return _create_store(remote_client, remote_aclient)
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
    local_aclient = AsyncQdrantClient(
        host=resolved_host,
        port=port,
        https=use_https,
        api_key=qdrant_api_key if use_https else None,
        timeout=client_timeout,
    )
    return _create_store(local_client, local_aclient)


vector_store = _build_qdrant_store()
storage_context = StorageContext.from_defaults(vector_store=vector_store)
_QDRANT_CLIENT_CACHE: QdrantClient | None = getattr(vector_store, "_client", None)
_QDRANT_ASYNC_CLIENT_CACHE: AsyncQdrantClient | None = getattr(
    vector_store, "_aclient", None
)
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


def _get_async_client() -> AsyncQdrantClient:
    if _QDRANT_ASYNC_CLIENT_CACHE is None:
        raise RuntimeError("Qdrant async client is not initialized")
    return _QDRANT_ASYNC_CLIENT_CACHE


def _get_collection_name() -> Optional[str]:
    return getattr(vector_store, "collection_name", None)


def _unscoped_payload_conditions() -> List[Any]:
    scope_field = qdrant_models.PayloadField(key="scope")
    return [
        qdrant_models.IsEmptyCondition(is_empty=scope_field),
        qdrant_models.IsNullCondition(is_null=scope_field),
    ]


def _build_scoped_filter(
    *,
    filename: Optional[str] = None,
    doc_hash: Optional[str] = None,
    exclude_doc_hash: Optional[str] = None,
    scope: Optional[str],
) -> qdrant_models.Filter:
    must: List[Any] = []
    must_not: List[Any] = []
    if filename:
        must.append(
            qdrant_models.FieldCondition(
                key="filename", match=qdrant_models.MatchValue(value=filename)
            )
        )
    if doc_hash:
        must.append(
            qdrant_models.FieldCondition(
                key="doc_hash", match=qdrant_models.MatchValue(value=doc_hash)
            )
        )
    if exclude_doc_hash:
        must_not.append(
            qdrant_models.FieldCondition(
                key="doc_hash", match=qdrant_models.MatchValue(value=exclude_doc_hash)
            )
        )
    if scope:
        must.append(
            qdrant_models.FieldCondition(
                key="scope", match=qdrant_models.MatchValue(value=scope)
            )
        )
        return qdrant_models.Filter(must=must, must_not=must_not)
    return qdrant_models.Filter(
        must=must,
        must_not=must_not,
        min_should=qdrant_models.MinShould(
            conditions=_unscoped_payload_conditions(),
            min_count=1,
        ),
    )


async def insert_nodes(nodes: List[TextNode]) -> None:
    if not nodes:
        logger.debug("insert_nodes called with empty payload; skipping")
        return
    
    # Batch size for insertion to avoid overloading embedding API or Vector DB
    BATCH_SIZE = int(os.getenv("INSERT_BATCH_SIZE", "8"))
    total_nodes = len(nodes)
    logger.info("Inserting %d nodes into Qdrant (will trigger embedding) in batches of %d", total_nodes, BATCH_SIZE)
    
    start = time.perf_counter()
    index = _get_index()
    
    for i in range(0, total_nodes, BATCH_SIZE):
        batch = nodes[i : i + BATCH_SIZE]
        batch_start = time.perf_counter()
        try:
            await index.ainsert_nodes(batch)
            logger.debug("Inserted batch %d-%d (%.2fs)", i, i + len(batch), time.perf_counter() - batch_start)
        except Exception as exc:
            logger.error("Failed to insert batch %d-%d: %s", i, i + len(batch), exc)
            raise exc

    elapsed = time.perf_counter() - start
    logger.debug(
        "Qdrant insert complete collection=%s count=%d duration=%.2fs",
        getattr(vector_store, "collection_name", "unknown"),
        total_nodes,
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
    skip_rerank: bool = False,
    keywords_any: Optional[List[str]] = None,
    keywords_all: Optional[List[str]] = None,
    time_decay_rate: Optional[float] = None,
    scope: Optional[str] = None,
    filenames_in: Optional[List[str]] = None,
    scopes_in: Optional[List[str]] = None,
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
    if scope:
        # Match scope exactly or as a path prefix?
        # User suggestion: "simulate file system path... use prefix match"
        # Qdrant KEYWORD index supports exact match. TEXT match supports token match.
        # For path hierarchy like "reports/2025", we might want "reports" to match "reports/2025".
        # But Qdrant TEXT match is token based.
        # Let's use TEXT_MATCH_INSENSITIVE for flexibility, or EQ if strict.
        # Given "optional logical scope", let's use EQ for simplicity first, or prefix if possible?
        # LlamaIndex FilterOperator doesn't have "PREFIX".
        # Let's stick to EQ for strict scope or TEXT_MATCH for partial.
        # OpenViking uses "viking://scope/path".
        # Let's use EQ for now as it's safer for strict isolation.
        filter_items.append(
            MetadataFilter(key="scope", value=scope, operator=FilterOperator.EQ)
        )
    elif scopes_in:
        filter_items.append(
            MetadataFilter(key="scope", value=scopes_in, operator=FilterOperator.IN)
        )
    if filenames_in:
        filter_items.append(
            MetadataFilter(key="filename", value=filenames_in, operator=FilterOperator.IN)
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
    any_set = {kw.strip().lower() for kw in (keywords_any or []) if kw and kw.strip()}
    all_set = {kw.strip().lower() for kw in (keywords_all or []) if kw and kw.strip()}

    postprocessors: List[BaseNodePostprocessor] = [
        ContentFilterPostprocessor(),
        KeywordFilterPostprocessor(any_set, all_set),
    ]
    if not skip_rerank:
        postprocessors.append(APIReranker(top_n=rerank_top_n))
    decay = time_decay_rate if time_decay_rate is not None else float(os.getenv("TIME_DECAY_RATE", "0.005"))
    postprocessors.append(TimeDecayPostprocessor(decay_rate=decay))

    return index.as_query_engine(
        similarity_top_k=top_k,
        vector_store_query_mode="hybrid",
        sparse_top_k=sparse_top_k,
        filters=filters,
        node_postprocessors=postprocessors,
        response_mode="simple_summarize",
    )


async def perform_hierarchical_search(
    query: str,
    scope: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filename: Optional[str] = None,
    filename_contains: Optional[str] = None,
    keywords_any: Optional[List[str]] = None,
    keywords_all: Optional[List[str]] = None,
    top_docs: int = 3,
):
    """
    Two-stage retrieval (LOD):
    1. Retrieve potential chunks (L1)
    2. Aggregate by filename to find Top N relevant documents
    3. Re-retrieve/Re-rank focused strictly on those documents (L2)
    """
    logger.info(
        "Starting hierarchical search query=%r scope=%s filename=%s contains=%s",
        query,
        scope,
        filename,
        filename_contains,
    )
    
    # L1: Fast retrieval (skip rerank for speed, or keep it for better L1 precision? 
    # Let's keep rerank to ensure the "Summary" (if matched) actually bubbles up.
    # But to save cost, we might skip it or use a cheaper one. 
    # For now, reuse standard logic but maybe just retrieval.)
    
    # We use get_query_engine to get a retriever-like object.
    # We'll use the query engine's retrieve method.
    engine_l1 = get_query_engine(
        start_date=start_date,
        end_date=end_date,
        filename=filename,
        filename_contains=filename_contains,
        scope=scope,
        skip_rerank=False, # Use rerank to get quality candidates for document selection
        keywords_any=keywords_any,
        keywords_all=keywords_all,
    )
    
    nodes_l1 = await engine_l1.aretrieve(QueryBundle(query))
    if not nodes_l1:
        logger.info("L1 search returned 0 nodes")
        return None # Let caller handle empty

    # Aggregate by document identity
    doc_scores = {}
    for node in nodes_l1:
        fname = node.metadata.get("filename")
        if not fname:
            continue
        node_scope = node.metadata.get("scope")
        doc_key = (fname, node_scope)
        # Strategy: Max score per doc
        score = node.score or 0.0
        if doc_key not in doc_scores or score > doc_scores[doc_key]:
            doc_scores[doc_key] = score
            
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    top_doc_keys = [x[0] for x in sorted_docs[:top_docs]]
    top_files = list(dict.fromkeys(fname for fname, _node_scope in top_doc_keys))
    selected_scopes = [node_scope for _fname, node_scope in top_doc_keys]
    top_scopes = [
        node_scope
        for node_scope in dict.fromkeys(selected_scopes)
        if node_scope
    ]
    l2_scopes_in = None if scope or not selected_scopes or not all(selected_scopes) else top_scopes
    logger.info("L1 identified top %d docs: %s", len(top_doc_keys), top_doc_keys)
    
    if not top_files:
        return None

    # L2: Focused Search
    # We restrict search to these files. 
    # We can perform generation here.
    engine_l2 = get_query_engine(
        start_date=start_date,
        end_date=end_date,
        filename=None, # We use filenames_in
        filename_contains=filename_contains,
        filenames_in=top_files,
        scope=scope,
        scopes_in=l2_scopes_in,
        keywords_any=keywords_any,
        keywords_all=keywords_all,
        skip_rerank=False, # Rerank again within the focused set? Yes, to order chunks for context.
        # We assume L1 rerank was coarse or global. L2 is focused.
        # Actually, if we already reranked in L1, maybe we just use those nodes?
        # But L1 might have missed some chunks from the "good doc" because of global Top K.
        # L2 brings ALL relevant chunks from the "good doc" into scope (if we used a higher top_k for L2?)
        # Let's trust standard Top K but constrained to these docs.
    )
    
    response = await engine_l2.aquery(query)
    return response


async def get_collection_metrics() -> dict:
    """Expose lightweight collection metrics for the web dashboard (async client)."""
    metrics = {
        "collection_name": _get_collection_name(),
        "vectors_count": None,
        "segments_count": None,
        "status": "uninitialized",
        "points_count": None,
        "error": None,
    }
    collection = metrics["collection_name"]
    if not collection:
        return metrics

    client = _get_async_client()

    try:
        await client.get_collections()
        metrics["status"] = "reachable"
    except Exception as exc:
        metrics["status"] = "unreachable"
        metrics["error"] = str(exc)
        logger.warning("Qdrant reachability check failed: %s", exc)
        return metrics

    try:
        info = await client.get_collection(collection_name=collection)
        status = getattr(info, "status", None)
        if hasattr(status, "value"):
            status = status.value
        elif hasattr(status, "name"):
            status = status.name
        metrics["status"] = status or "ready"
        metrics["vectors_count"] = getattr(info, "vectors_count", None)
        metrics["segments_count"] = getattr(info, "segments_count", None)
    except Exception as exc:
        metrics["error"] = str(exc)
        logger.warning("get_collection failed for %s: %s", collection, exc)

    try:
        count_resp = await client.count(collection_name=collection, exact=False)
        metrics["points_count"] = getattr(count_resp, "count", None)
    except Exception as exc:
        metrics["error"] = metrics["error"] or str(exc)
        logger.warning("count query failed for %s: %s", collection, exc)

    return metrics


Settings.llm = build_llm(
    purpose="chat",
    temperature=0.1,
    context_window=llm_context_window,
)


class ContentFilterPostprocessor(BaseNodePostprocessor):
    """Drop nodes without usable content."""

    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        cleaned: List[NodeWithScore] = []
        for node in nodes:
            try:
                inner = getattr(node, "node", None)
                text = inner.get_content() if inner is not None else getattr(node, "text", "")
                if text:
                    cleaned.append(node)
            except Exception:
                continue
        return cleaned


class KeywordFilterPostprocessor(BaseNodePostprocessor):
    """Filter nodes by keyword sets."""

    _any_set: set[str] = PrivateAttr()
    _all_set: set[str] = PrivateAttr()

    _ALIASES: ClassVar[dict[str, set[str]]] = {
        "high": {"strong"},
        "strong": {"high"},
        "medium": {"moderate"},
        "moderate": {"medium"},
        "low": {"weak"},
        "weak": {"low"},
    }

    def __init__(self, any_set: set[str], all_set: set[str]):
        super().__init__()
        self._any_set = any_set
        self._all_set = all_set

    def _expand_term(self, value: str) -> set[str]:
        term = value.strip().lower()
        if not term:
            return set()
        return {term, *self._ALIASES.get(term, set())}

    def _add_keyword_value(self, keywords: set[str], value: Any) -> None:
        if value is None:
            return
        text = str(value).strip()
        if not text:
            return
        for term in self._expand_term(text):
            keywords.add(term)
        for part in re.split(r"[|=,;\s/_-]+", text):
            for term in self._expand_term(part):
                keywords.add(term)

    def _normalize_keywords(self, node) -> set[str]:
        metadata = getattr(node, "metadata", {}) or {}
        keywords: set[str] = set()
        raw_list = metadata.get("keyword_list")
        if isinstance(raw_list, list):
            for item in raw_list:
                self._add_keyword_value(keywords, item)
        elif isinstance(raw_list, str):
            for item in raw_list.split(","):
                self._add_keyword_value(keywords, item)
        raw_str = metadata.get("keywords")
        if isinstance(raw_str, str):
            for item in raw_str.split(","):
                self._add_keyword_value(keywords, item)
        self._add_keyword_value(keywords, metadata.get("scope"))
        self._add_keyword_value(keywords, metadata.get("filename"))
        return keywords

    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        if not self._any_set and not self._all_set:
            return nodes
        filtered: List[NodeWithScore] = []
        for node in nodes:
            keyword_set = self._normalize_keywords(node)
            if self._any_set and not (keyword_set & self._any_set):
                continue
            if self._all_set and not self._all_set.issubset(keyword_set):
                continue
            filtered.append(node)
        return filtered


class TimeDecayPostprocessor(BaseNodePostprocessor):
    """Apply linear time decay to node scores."""

    _decay_rate: float = PrivateAttr()

    def __init__(self, decay_rate: float = 0.005):
        super().__init__()
        self._decay_rate = decay_rate

    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        today = today_in_app_tz()
        for node in nodes:
            metadata = getattr(node, "metadata", {}) or {}
            date_str = metadata.get("date")
            if not date_str:
                continue
            try:
                doc_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                delta = (today - doc_date).days
                if delta > 0 and node.score is not None:
                    node.score *= 1.0 / (1.0 + self._decay_rate * delta)
            except Exception:
                continue
        nodes.sort(key=lambda x: (x.score or 0.0), reverse=True)
        return nodes


async def adoc_exists(doc_hash: str, scope: Optional[str] = None) -> bool:
    """Return True if collection already has a payload with the given doc_hash (async)."""
    if not doc_hash:
        return False

    collection = _get_collection_name()
    if not collection:
        return False

    client = _get_async_client()
    try:
        flt = _build_scoped_filter(doc_hash=doc_hash, scope=scope)
        # Using scroll with limit=1 to check existence
        result = await client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            with_payload=False,
            limit=1,
        )
        points = result[0] if isinstance(result, (list, tuple)) else result
        return bool(points)
    except Exception as exc:
        logger.warning("adoc_exists check failed: %s", exc)
        return False


async def delete_nodes_by_filename(
    filename: str,
    scope: Optional[str] = None,
    *,
    exclude_doc_hash: Optional[str] = None,
) -> bool:
    """Delete all nodes associated with a specific scoped document."""
    if not filename:
        return False
    client = _get_async_client()
    collection = _get_collection_name()
    if not collection:
        return False
    try:
        flt = _build_scoped_filter(
            filename=filename,
            scope=scope,
            exclude_doc_hash=exclude_doc_hash,
        )
        points, _ = await client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            with_payload=False,
            limit=1,
        )
        if not points:
            return False
        await client.delete(
            collection_name=collection,
            points_selector=flt,
        )
        logger.info(
            "Deleted document filename=%s scope=%s exclude_doc_hash=%s",
            filename,
            scope,
            exclude_doc_hash,
        )
        return True
    except Exception as exc:
        logger.error(
            "Failed to delete nodes for filename=%s scope=%s: %s",
            filename,
            scope,
            exc,
        )
        return False


async def _count_document_chunks(
    client: AsyncQdrantClient,
    collection: str,
    *,
    filename: str,
    scope: Optional[str],
) -> int:
    result = await client.count(
        collection_name=collection,
        count_filter=_build_scoped_filter(filename=filename, scope=scope),
        exact=True,
    )
    return int(getattr(result, "count", 0) or 0)


async def _fill_document_chunk_counts(
    client: AsyncQdrantClient,
    collection: str,
    docs: List[dict],
) -> None:
    if not docs:
        return
    concurrency = max(1, int(os.getenv("DOCUMENT_LIST_COUNT_CONCURRENCY", "16")))
    semaphore = asyncio.Semaphore(concurrency)

    async def _fill_one(doc: dict) -> None:
        async with semaphore:
            try:
                doc["chunks"] = await _count_document_chunks(
                    client,
                    collection,
                    filename=doc["filename"],
                    scope=doc.get("scope"),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to count chunks for document filename=%s scope=%s: %s",
                    doc.get("filename"),
                    doc.get("scope"),
                    exc,
                )

    await asyncio.gather(*(_fill_one(doc) for doc in docs))


async def list_all_documents(limit: int = 1000, search: Optional[str] = None) -> List[dict]:
    """
    List unique documents in the collection by aggregating filename info.
    Returns a list of dicts: [{'filename': str, 'date': str, 'chunks': int}]
    """
    client = _get_async_client()
    collection = _get_collection_name()
    if not collection:
        return []

    limit = max(1, min(int(limit or 1000), 1000))
    docs: dict[tuple[str, Optional[str]], dict] = {}
    offset = None
    points_scanned = 0
    MAX_POINTS_TO_SCAN = int(os.getenv("DOCUMENT_LIST_MAX_POINTS_TO_SCAN", "50000"))
    search_text = (search or "").strip().lower()

    while points_scanned < MAX_POINTS_TO_SCAN:
        try:
            result = await client.scroll(
                collection_name=collection,
                limit=1000,
                offset=offset,
                with_payload=["filename", "date", "scope"],
                with_vectors=False,
            )
            points, next_offset = result
            if not points:
                break
                
            for point in points:
                payload = point.payload
                fname = payload.get("filename")
                if not fname:
                    continue

                scope = payload.get("scope") or None
                date = payload.get("date")
                if search_text:
                    haystack = " ".join(
                        part for part in [fname, scope or "", date or ""] if part
                    ).lower()
                    if search_text not in haystack:
                        continue
                doc_key = (fname, scope)
                if doc_key not in docs:
                    if len(docs) >= limit:
                        continue
                    docs[doc_key] = {
                        "filename": fname,
                        "date": date,
                        "chunks": 0,
                        "scope": scope,
                    }
                docs[doc_key]["chunks"] += 1
            
            points_scanned += len(points)
            offset = next_offset
            if len(docs) >= limit:
                break
            if offset is None:
                break
        except Exception as exc:
            logger.error("Error scrolling documents for listing: %s", exc)
            break

    result = sorted(
        list(docs.values()),
        key=lambda item: (item["filename"], item.get("scope") or ""),
    )
    await _fill_document_chunk_counts(client, collection, result)
    logger.debug(
        "Listed documents returned=%d limit=%d search=%s points_scanned=%d",
        len(result),
        limit,
        search_text or None,
        points_scanned,
    )
    return result


def _parse_optional_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_summary_from_full_text(full_text: str) -> str:
    if not full_text:
        return ""
    match = re.search(r"^Summary:\s*(.+)$", full_text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def _extract_original_text_from_full_text(full_text: str) -> str:
    if not full_text:
        return ""
    marker = "\n---\n"
    if marker in full_text:
        return full_text.split(marker, 1)[1].strip()
    return full_text.strip()


def _normalize_grounding_payload(payload: dict) -> tuple[dict, str, str]:
    metadata = dict(payload or {})
    full_text = ""
    raw_node = metadata.get("_node_content")
    if isinstance(raw_node, str):
        try:
            node_data = json.loads(raw_node)
        except json.JSONDecodeError:
            node_data = {}
        node_metadata = node_data.get("metadata")
        if isinstance(node_metadata, dict):
            metadata.update(node_metadata)
        full_text = str(node_data.get("text") or "")
    original_text = str(metadata.get("original_text") or "")
    if not original_text:
        original_text = _extract_original_text_from_full_text(full_text)
        metadata["original_text"] = original_text
    if "doc_summary" not in metadata or not metadata.get("doc_summary"):
        metadata["doc_summary"] = _extract_summary_from_full_text(full_text)
    return metadata, full_text, original_text


def _build_unrestricted_doc_hash_filter(doc_hash: str) -> qdrant_models.Filter:
    return qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="doc_hash",
                match=qdrant_models.MatchValue(value=doc_hash),
            )
        ]
    )


async def _scroll_points_by_filter(flt: qdrant_models.Filter) -> List[Any]:
    client = _get_async_client()
    collection = _get_collection_name()
    if not collection:
        return []

    all_points: List[Any] = []
    offset = None
    while True:
        points, next_offset = await client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        if not points:
            break
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset
    return all_points


def _selector_scope(selector: GroundingDocumentSelector, request_scope: Optional[str]) -> Optional[str]:
    selector_scope = selector.scope.strip() if selector.scope else None
    root_scope = request_scope.strip() if request_scope else None
    if selector_scope and root_scope and selector_scope != root_scope:
        raise ValueError("document.scope and request scope do not match")
    return selector_scope or root_scope


async def _resolve_grounding_chunks(
    selector: GroundingDocumentSelector,
    request_scope: Optional[str],
) -> List[_GroundingChunk]:
    scope = _selector_scope(selector, request_scope)
    if selector.doc_hash:
        flt = (
            _build_scoped_filter(doc_hash=selector.doc_hash, scope=scope)
            if scope
            else _build_unrestricted_doc_hash_filter(selector.doc_hash)
        )
    else:
        flt = _build_scoped_filter(filename=selector.filename, scope=scope, doc_hash=None)

    points = await _scroll_points_by_filter(flt)
    if not points:
        return []

    grouped: dict[tuple[str, Optional[str]], List[_GroundingChunk]] = {}
    for order, point in enumerate(points):
        payload = getattr(point, "payload", None) or {}
        metadata, full_text, original_text = _normalize_grounding_payload(payload)
        filename = str(metadata.get("filename") or selector.filename or "")
        point_scope = metadata.get("scope") or None
        group_key = (filename, point_scope)
        grouped.setdefault(group_key, []).append(
            _GroundingChunk(
                filename=filename,
                scope=point_scope,
                doc_hash=metadata.get("doc_hash"),
                date=metadata.get("date"),
                metadata=metadata,
                full_text=full_text,
                original_text=original_text,
                order_key=(
                    _parse_optional_int(metadata.get("section_order"), order),
                    _parse_optional_int(metadata.get("block_index"), order),
                    _parse_optional_int(metadata.get("chunk_index"), order),
                    order,
                ),
            )
        )

    if len(grouped) > 1:
        if selector.doc_hash and not scope:
            raise ValueError(
                "doc_hash matched multiple scoped documents; provide scope to disambiguate"
            )
        if selector.filename and not scope:
            raise ValueError(
                "filename matched multiple documents; provide scope or doc_hash to disambiguate"
            )

    selected = next(iter(grouped.values()))
    return sorted(selected, key=lambda chunk: chunk.order_key)


def _build_document_info(chunks: List[_GroundingChunk]) -> GroundingDocumentInfo:
    first = chunks[0]
    summary = ""
    for chunk in chunks:
        summary = str(chunk.metadata.get("doc_summary") or "").strip()
        if summary:
            break
        summary = _extract_summary_from_full_text(chunk.full_text)
        if summary:
            break
    return GroundingDocumentInfo(
        doc_hash=first.doc_hash,
        filename=first.filename,
        scope=first.scope,
        date=first.date,
        summary=summary,
        chunk_count=len(chunks),
    )


def _chunk_to_blocks(chunks: List[_GroundingChunk]) -> List[_GroundingBlock]:
    blocks: List[_GroundingBlock] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        local_blocks = split_document_blocks(chunk.original_text, allow_title=False)
        if not local_blocks:
            section_type, is_list_zone, is_qa_zone = classify_document_block(
                chunk.original_text,
                heading_path=chunk.metadata.get("heading_path"),
            )
            local_blocks = [
                type("FallbackBlock", (), {
                    "text": chunk.original_text,
                    "section_type": section_type,
                    "heading_path": chunk.metadata.get("heading_path"),
                    "is_list_zone": is_list_zone,
                    "is_qa_zone": is_qa_zone,
                })()
            ]

        metadata_heading_path = chunk.metadata.get("heading_path")

        for local_index, block in enumerate(local_blocks):
            block_text = str(block.text or "").strip()
            if not block_text:
                continue
            section_type = block.section_type
            heading_path = block.heading_path or metadata_heading_path

            signature = (section_type, re.sub(r"\s+", "", block_text))
            if signature in seen:
                continue
            seen.add(signature)
            blocks.append(
                _GroundingBlock(
                    filename=chunk.filename,
                    scope=chunk.scope,
                    doc_hash=chunk.doc_hash,
                    date=chunk.date,
                    section_type=section_type,
                    heading_path=heading_path,
                    text=block_text,
                    block_index=len(blocks),
                    chunk_index=_parse_optional_int(chunk.metadata.get("chunk_index"), local_index),
                )
            )
    return blocks


def _normalized_match_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "")).lower()


def _candidate_name(candidate: GroundingCandidate) -> str:
    return candidate.name or candidate.identifier or "unknown"


def _normalize_candidate_variant(value: Any) -> str:
    return str(value or "").strip()


def _generate_stock_alias_variants(candidate: GroundingCandidate) -> List[str]:
    if (candidate.candidate_type or "").lower() != "stock":
        return []
    name = _normalize_candidate_variant(candidate.name)
    if not name:
        return []

    generated: List[str] = []
    for suffix in CHINESE_STOCK_ALIAS_SUFFIXES:
        if not name.endswith(suffix):
            continue
        alias = name[: -len(suffix)].strip()
        if len(alias) < 2:
            continue
        if alias not in generated:
            generated.append(alias)
    return generated


def _candidate_variants(candidate: GroundingCandidate) -> List[str]:
    values: List[str] = []
    for raw in [candidate.name, candidate.identifier, *(candidate.aliases or [])]:
        text = _normalize_candidate_variant(raw)
        if text and text not in values:
            values.append(text)
        if "." in text:
            code_prefix = text.split(".", 1)[0]
            if code_prefix and code_prefix not in values:
                values.append(code_prefix)
    return values


def _candidate_generated_variants(candidate: GroundingCandidate) -> List[str]:
    values: List[str] = []
    explicit = set(_candidate_variants(candidate))
    for text in _generate_stock_alias_variants(candidate):
        if text and text not in explicit and text not in values:
            values.append(text)
    return values


def _variant_hit(text: str, normalized_text: str, variant: str) -> bool:
    if not variant:
        return False
    token = variant.strip()
    if not token:
        return False
    if re.fullmatch(r"\d{6}(?:\.[A-Za-z]{2,4})?", token):
        pattern = re.compile(rf"(?<!\d){re.escape(token)}(?!\d)", re.IGNORECASE)
        return bool(pattern.search(text))
    if token.lower() == token.upper():
        return token in text or _normalized_match_text(token) in normalized_text
    return token.lower() in text.lower() or _normalized_match_text(token) in normalized_text


def _relation_keyword_hits(text: str) -> int:
    return sum(1 for keyword in RELATION_KEYWORDS if keyword in text)


def _supports_generated_stock_alias_match(
    candidate: GroundingCandidate,
    block: _GroundingBlock,
    text: str,
) -> bool:
    if (candidate.candidate_type or "").lower() != "stock":
        return False
    if block.section_type in {"appendix_list", "appendix_table"}:
        return True
    return any(keyword in text for keyword in GENERATED_STOCK_ALIAS_CONTEXT_KEYWORDS)


def _section_weight(section_type: str) -> int:
    weights = {
        "title": 5,
        "body": 4,
        "qa": 3,
        "appendix_table": 2,
        "appendix_list": 1,
    }
    return weights.get(section_type, 0)


async def _rank_grounding_blocks(
    candidate: GroundingCandidate,
    candidate_blocks: List[dict],
    *,
    skip_rerank: bool,
) -> List[dict]:
    ranked = sorted(
        candidate_blocks,
        key=lambda item: (
            item["base_score"],
            -item["block"].block_index,
            -item["block"].chunk_index,
        ),
        reverse=True,
    )
    if skip_rerank or len(ranked) <= 1:
        return ranked

    top_n = min(8, len(ranked))
    nodes = [
        NodeWithScore(
            node=TextNode(
                text=item["block"].text,
                metadata={
                    "section_type": item["block"].section_type,
                    "original_text": item["block"].text,
                },
            ),
            score=float(item["base_score"]),
        )
        for item in ranked[:top_n]
    ]
    query = " ".join(
        part for part in [_candidate_name(candidate), *candidate.aliases, "与本文主题的直接相关证据"] if part
    )
    reranker = APIReranker(top_n=top_n)
    try:
        reranked_nodes = await reranker.apostprocess_nodes(nodes, QueryBundle(query))
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Grounding rerank failed for candidate=%s: %s", _candidate_name(candidate), exc)
        return ranked

    score_map = {node.node.get_content(): float(node.score or 0.0) for node in reranked_nodes}
    for item in ranked:
        item["final_score"] = score_map.get(item["block"].text, float(item["base_score"]))
    ranked.sort(
        key=lambda item: (
            item.get("final_score", float(item["base_score"])),
            item["base_score"],
        ),
        reverse=True,
    )
    return ranked


def _compact_text(text: str, limit: int = 220) -> str:
    collapsed = re.sub(r"\s+", " ", (text or "")).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _build_candidate_brief(
    candidate_name: str,
    tier: str,
    doc_summary: str,
    source_reason: str,
) -> str:
    doc_summary_text = _compact_text(doc_summary or "该文档")
    if tier == "body_grounded":
        return f"本文主旨为{doc_summary_text}。正文明确提到{candidate_name}，{source_reason}"
    if tier == "relation_grounded":
        return f"本文主旨为{doc_summary_text}。正文提到{candidate_name}，但主要体现为关系型关联：{source_reason}"
    if tier == "list_only":
        return f"本文主旨为{doc_summary_text}。{candidate_name}仅在文末名单或附录中出现，正文无直接展开，属于弱证据候选。"
    return f"本文主旨为{doc_summary_text}。未在正文或附录中找到{candidate_name}的明确提及。"


async def _ground_single_candidate(
    candidate: GroundingCandidate,
    blocks: List[_GroundingBlock],
    doc_summary: str,
    *,
    max_excerpts: int,
    skip_rerank: bool,
) -> GroundingResult:
    candidate_name = _candidate_name(candidate)
    variants = _candidate_variants(candidate)
    generated_variants = _candidate_generated_variants(candidate)
    candidate_blocks: List[dict] = []
    body_hit_count = 0
    qa_hit_count = 0
    list_hit_count = 0
    title_hit_count = 0
    relation_hits = 0
    used_generated_variant = False

    for block in blocks:
        text = block.text
        normalized = _normalized_match_text(text)
        matched_variants = [variant for variant in variants if _variant_hit(text, normalized, variant)]
        matched_generated_variants = [
            variant
            for variant in generated_variants
            if _variant_hit(text, normalized, variant)
        ]
        if not matched_variants and matched_generated_variants:
            if not _supports_generated_stock_alias_match(candidate, block, text):
                matched_generated_variants = []
        if not matched_variants and not matched_generated_variants:
            continue

        keyword_hits = _relation_keyword_hits(text)
        if block.section_type in {"title", "body"}:
            if block.section_type == "title":
                title_hit_count += 1
            else:
                body_hit_count += 1
        elif block.section_type == "qa":
            qa_hit_count += 1
        elif block.section_type in {"appendix_list", "appendix_table"}:
            list_hit_count += 1
        relation_hits += keyword_hits
        used_generated_variant = used_generated_variant or bool(
            matched_generated_variants and not matched_variants
        )

        candidate_blocks.append(
            {
                "block": block,
                "matched_variants": matched_variants or matched_generated_variants,
                "is_alias_hit": bool(matched_variants or matched_generated_variants),
                "used_generated_variant": bool(
                    matched_generated_variants and not matched_variants
                ),
                "base_score": (
                    _section_weight(block.section_type) * 100
                    + min(keyword_hits, 5) * 10
                    + min(len(matched_variants or matched_generated_variants), 3) * 5
                ),
            }
        )

    if not candidate_blocks:
        return GroundingResult(
            identifier=candidate.identifier,
            name=candidate_name,
            relevance_tier="not_found",
            source_zone="not_found",
            source_reason="文档中未找到该资产的显式提及。",
            candidate_brief=_build_candidate_brief(
                candidate_name,
                "not_found",
                doc_summary,
                "文档中未找到该资产的显式提及。",
            ),
        )

    has_non_list = any(item["block"].section_type in {"title", "body", "qa"} for item in candidate_blocks)
    if has_non_list:
        tier = "body_grounded" if (title_hit_count or body_hit_count >= 2 or relation_hits > 0) else "relation_grounded"
        if used_generated_variant and tier == "body_grounded":
            tier = "relation_grounded"
        source_candidates = [
            item for item in candidate_blocks if item["block"].section_type in {"title", "body", "qa"}
        ]
        if used_generated_variant:
            source_reason = "正文未写全称，但以股票简称/短称出现，且上下文支持该候选的关系型提及。"
        else:
            source_reason = (
                "正文存在显式提及，并伴随竞争、供应链、订单或风险等关系论述。"
                if tier == "body_grounded"
                else "正文存在显式提及，但论述更偏关系映射或轻度展开。"
            )
    else:
        tier = "list_only"
        source_candidates = [
            item for item in candidate_blocks if item["block"].section_type in {"appendix_list", "appendix_table"}
        ]
        source_reason = "仅在文末名单或附录中出现，正文未找到直接论据。"

    ranked = await _rank_grounding_blocks(candidate, source_candidates, skip_rerank=skip_rerank)
    top_excerpts = ranked[:max_excerpts]
    source_zone = top_excerpts[0]["block"].section_type if top_excerpts else "not_found"
    excerpts = [
        GroundingExcerpt(
            section_type=item["block"].section_type,
            score=round(float(item.get("final_score", item["base_score"])), 4),
            text=_compact_text(item["block"].text, limit=320),
            is_alias_hit=bool(item["is_alias_hit"]),
        )
        for item in top_excerpts
    ]
    return GroundingResult(
        identifier=candidate.identifier,
        name=candidate_name,
        relevance_tier=tier,
        source_zone=source_zone,
        source_reason=source_reason,
        body_hit_count=body_hit_count + title_hit_count,
        qa_hit_count=qa_hit_count,
        list_hit_count=list_hit_count,
        candidate_brief=_build_candidate_brief(candidate_name, tier, doc_summary, source_reason),
        excerpts=excerpts,
    )


async def run_grounding_query(
    request: GroundingRequest,
) -> GroundingResponse:
    chunks = await _resolve_grounding_chunks(request.document, request.scope)
    if not chunks:
        raise LookupError("document not found")

    document = _build_document_info(chunks)
    blocks = _chunk_to_blocks(chunks)
    candidate_results: List[GroundingResult] = []
    for candidate in request.candidates:
        candidate_results.append(
            await _ground_single_candidate(
                candidate,
                blocks,
                document.summary,
                max_excerpts=request.max_excerpts,
                skip_rerank=request.skip_rerank,
            )
        )

    return GroundingResponse(document=document, candidate_results=candidate_results)


async def shutdown_resources() -> None:
    """Close recreatable transports without tearing down module-level core state."""
    global _RERANK_HTTP_CLIENT, _RERANK_HTTP_ACLIENT
    global _RERANK_OPENAI_CLIENT, _RERANK_OPENAI_ACLIENT

    for client_name in ("_RERANK_HTTP_CLIENT",):
        client = globals().get(client_name)
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    for client_name in ("_RERANK_HTTP_ACLIENT",):
        client = globals().get(client_name)
        aclose = getattr(client, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:
                pass

    for client_name in ("_RERANK_OPENAI_CLIENT",):
        client = globals().get(client_name)
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    for client_name in ("_RERANK_OPENAI_ACLIENT",):
        client = globals().get(client_name)
        aclose = getattr(client, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:
                pass

    _RERANK_HTTP_CLIENT = None
    _RERANK_HTTP_ACLIENT = None
    _RERANK_OPENAI_CLIENT = None
    _RERANK_OPENAI_ACLIENT = None
