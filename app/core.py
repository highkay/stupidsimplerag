import json
import logging
import os
import re
import time
import datetime
import asyncio
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
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from tenacity import RetryCallState, retry, stop_after_attempt, wait_exponential

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
RERANK_MAX_RETRIES = int(os.getenv("RERANK_MAX_RETRIES", "3"))
RERANK_RETRY_BACKOFF = float(os.getenv("RERANK_RETRY_BACKOFF", "1.5"))
fastembed_model_name = os.getenv(
    "FASTEMBED_SPARSE_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions"
)
fastembed_cache_dir = os.getenv("FASTEMBED_CACHE_PATH")


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
        
        # Persistent clients for connection pooling
        self._client = httpx.Client(timeout=self._timeout)
        self._aclient = httpx.AsyncClient(timeout=self._timeout)

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
        
        for attempt in range(1, self._max_retries + 1):
            try:
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
        data = self._send_embedding_request(inputs)
        if len(data) == len(inputs):
            return [item["embedding"] for item in data]
        
        embeddings: List[List[float]] = []
        for text in inputs:
            single_data = self._send_embedding_request([text])
            if not single_data:
                raise ValueError("Embedding API returned empty response for single input")
            embeddings.append(single_data[0]["embedding"])
        return embeddings

    async def _aembedding_request(self, inputs: List[str]) -> List[List[float]]:
        data = await self._asend_embedding_request(inputs)
        if len(data) == len(inputs):
            return [item["embedding"] for item in data]
        
        embeddings: List[List[float]] = []
        for text in inputs:
            single_data = await self._asend_embedding_request([text])
            if not single_data:
                raise ValueError("Embedding API returned empty response for single input")
            embeddings.append(single_data[0]["embedding"])
        return embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embedding_request([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await self._aembedding_request([query]))[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embedding_request([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await self._aembedding_request([text]))[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embedding_request(texts)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self._aembedding_request(texts)


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

    def _create_store(q_client: QdrantClient, q_aclient: AsyncQdrantClient) -> QdrantVectorStore:
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
_ASYNC_CLIENT_CACHE: AsyncQdrantClient | None = None
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


async def insert_nodes(nodes: List[TextNode]) -> None:
    if not nodes:
        logger.debug("insert_nodes called with empty payload; skipping")
        return
    
    # Batch size for insertion to avoid overloading embedding API or Vector DB
    BATCH_SIZE = int(os.getenv("INSERT_BATCH_SIZE", "50"))
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
    any_set = {kw.strip().lower() for kw in (keywords_any or []) if kw and kw.strip()}
    all_set = {kw.strip().lower() for kw in (keywords_all or []) if kw and kw.strip()}

    postprocessors: List[BaseNodePostprocessor] = []
    if not skip_rerank:
        postprocessors.append(APIReranker(top_n=rerank_top_n))
    postprocessors.append(ContentFilterPostprocessor())
    postprocessors.append(KeywordFilterPostprocessor(any_set, all_set))
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


def _build_async_client() -> AsyncQdrantClient:
    global _ASYNC_CLIENT_CACHE
    if _ASYNC_CLIENT_CACHE is not None:
        return _ASYNC_CLIENT_CACHE

    host = os.getenv("QDRANT_HOST", "qdrant")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
    use_https = os.getenv("QDRANT_HTTPS", "false").lower() == "true"
    client_timeout = float(os.getenv("QDRANT_CLIENT_TIMEOUT", "10"))

    if qdrant_url:
        qdrant_url = _append_port_if_missing(qdrant_url, port)
        _ASYNC_CLIENT_CACHE = AsyncQdrantClient(
            url=qdrant_url, api_key=qdrant_api_key, timeout=client_timeout
        )
        return _ASYNC_CLIENT_CACHE

    resolved_host = host
    running_in_container = os.path.exists("/.dockerenv")
    if resolved_host == "qdrant" and not running_in_container:
        resolved_host = "127.0.0.1"

    _ASYNC_CLIENT_CACHE = AsyncQdrantClient(
        host=resolved_host,
        port=port,
        https=use_https,
        api_key=qdrant_api_key if use_https else None,
        timeout=client_timeout,
    )
    return _ASYNC_CLIENT_CACHE


async def get_collection_metrics() -> dict:
    """Expose lightweight collection metrics for the web dashboard (async client)."""
    metrics = {
        "collection_name": getattr(vector_store, "collection_name", None),
        "vectors_count": None,
        "segments_count": None,
        "status": "uninitialized",
        "points_count": None,
        "error": None,
    }
    collection = metrics["collection_name"]
    if not collection:
        return metrics

    client = _build_async_client()

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


Settings.llm = OpenAICompatibleLLM(
    model=os.getenv("LLM_MODEL"),
    api_key=llm_key,
    api_base=llm_base,
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

    def __init__(self, any_set: set[str], all_set: set[str]):
        super().__init__()
        self._any_set = any_set
        self._all_set = all_set

    def _normalize_keywords(self, node) -> set[str]:
        metadata = getattr(node, "metadata", {}) or {}
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
        today = datetime.date.today()
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


async def adoc_exists(doc_hash: str) -> bool:
    """Return True if collection already has a payload with the given doc_hash (async)."""
    if not doc_hash:
        return False
    
    collection = getattr(vector_store, "collection_name", None)
    if not collection:
        return False
        
    client = _build_async_client()
    try:
        flt = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="doc_hash", match=qdrant_models.MatchValue(value=doc_hash)
                )
            ]
        )
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
