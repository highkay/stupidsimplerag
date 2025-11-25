import logging
import os
from typing import Any, List, Optional
from urllib.parse import urlparse, urlunparse

import httpx
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
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


logger = logging.getLogger(__name__)


Settings.llm = OpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_API_BASE"),
    temperature=0.1,
)
embedding_api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY"))
embedding_api_base = os.getenv("EMBEDDING_API_BASE", os.getenv("OPENAI_API_BASE"))
embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))


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

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _embedding_request(self, inputs: List[str]) -> List[List[float]]:
        payload: dict = {"model": self._model, "input": inputs}
        if self._dimensions:
            payload["dimensions"] = self._dimensions
        url = f"{self._api_base}/embeddings"
        response = httpx.post(url, json=payload, headers=self._headers(), timeout=self._timeout)
        response.raise_for_status()
        data = response.json().get("data", [])
        return [item["embedding"] for item in data]

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


def _init_embedding_model() -> BaseEmbedding:
    model_name = os.getenv("EMBEDDING_MODEL")
    try:
        return OpenAIEmbedding(
            model=model_name,
            api_key=embedding_api_key,
            api_base=embedding_api_base,
        )
    except ValueError:
        return OpenAICompatibleEmbedding(
            model=model_name,
            api_key=embedding_api_key,
            api_base=embedding_api_base,
            dimensions=embedding_dim,
        )


Settings.embed_model = _init_embedding_model()


class APIReranker(BaseNodePostprocessor):
    def __init__(self, top_n: int = 10):
        super().__init__()
        self.top_n = top_n
        self.api_url = os.getenv("RERANK_API_URL")
        self.api_key = os.getenv("RERANK_API_KEY")
        self.model = os.getenv("RERANK_MODEL")

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if not nodes or not query_bundle:
            return nodes[: self.top_n]

        try:
            payload = {
                "model": self.model,
                "query": query_bundle.query_str,
                "documents": [n.node.get_content() for n in nodes],
                "top_n": self.top_n,
                "return_documents": False,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(self.api_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return nodes[: self.top_n]

        results = data.get("results", data.get("data", []))
        ranked: List[NodeWithScore] = []
        for item in results:
            idx = item.get("index")
            if idx is None or idx >= len(nodes):
                continue
            score = item.get("relevance_score", item.get("score"))
            node = nodes[idx]
            if score is not None:
                node.score = float(score)
            ranked.append(node)

        if not ranked:
            return nodes[: self.top_n]

        ranked.sort(key=lambda x: (x.score or 0.0), reverse=True)
        return ranked


def _ensure_hybrid_collection(
    client: QdrantClient, collection_name: str, vector_size: int
) -> None:
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text-dense": qdrant_models.VectorParams(
                size=vector_size, distance=qdrant_models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "text-sparse": qdrant_models.SparseVectorParams(
                index=qdrant_models.SparseIndexParams()
            )
        },
    )


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
        _ensure_hybrid_collection(q_client, collection, embedding_dim)
        return QdrantVectorStore(
            client=q_client,
            collection_name=collection,
            enable_hybrid=True,
            dense_vector_name="text-dense",
            sparse_vector_name="text-sparse",
            fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
        )

    if qdrant_url:
        qdrant_url = _append_port_if_missing(qdrant_url, port)
        try:
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
    if resolved_host == "qdrant":
        resolved_host = "127.0.0.1"

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


def _build_index() -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )


def insert_nodes(nodes: List[TextNode]) -> None:
    index = _build_index()
    index.insert_nodes(nodes)


def get_query_engine(start_date: Optional[str] = None, end_date: Optional[str] = None):
    filters: Optional[MetadataFilters] = None
    filter_items: List[MetadataFilter] = []
    if start_date:
        filter_items.append(
            MetadataFilter(key="date", value=start_date, operator=FilterOperator.GTE)
        )
    if end_date:
        filter_items.append(
            MetadataFilter(key="date", value=end_date, operator=FilterOperator.LTE)
        )
    if filter_items:
        filters = MetadataFilters(filters=filter_items)

    index = _build_index()
    top_k = int(os.getenv("TOP_K_RETRIEVAL", "100"))
    rerank_top_n = int(os.getenv("TOP_N_RERANK", "20"))
    sparse_top_k = int(os.getenv("SPARSE_TOP_K", "12"))
    return index.as_query_engine(
        similarity_top_k=top_k,
        vector_store_query_mode="hybrid",
        sparse_top_k=sparse_top_k,
        filters=filters,
        node_postprocessors=[APIReranker(top_n=rerank_top_n)],
    )
