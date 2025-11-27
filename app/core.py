import logging
import os
from typing import Any, List, Optional
from urllib.parse import urlparse, urlunparse

import httpx
from llama_index.core import QueryBundle, Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLMMetadata, MessageRole
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


embedding_api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY"))
embedding_api_base = os.getenv("EMBEDDING_API_BASE", os.getenv("OPENAI_API_BASE"))
embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
llm_context_window = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))


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

    def _send_embedding_request(self, inputs: List[str]) -> List[dict]:
        payload: dict = {"model": self._model, "input": inputs}
        if self._dimensions:
            payload["dimensions"] = self._dimensions
        url = f"{self._api_base}/embeddings"
        response = httpx.post(
            url,
            json=payload,
            headers=self._headers(),
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json().get("data", [])

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
    top_n: int = 10
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None

    def __init__(self, top_n: int = 10):
        super().__init__(
            top_n=top_n,
            api_url=os.getenv("RERANK_API_URL"),
            api_key=os.getenv("RERANK_API_KEY"),
            model=os.getenv("RERANK_MODEL"),
        )

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if not nodes or not query_bundle or not self.api_url or not self.model:
            return nodes[: self.top_n]

        try:
            payload = {
                "model": self.model,
                "query": query_bundle.query_str,
                "documents": [n.node.get_content() for n in nodes],
                "top_n": self.top_n,
                "return_documents": False,
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

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


class SafeQdrantVectorStore(QdrantVectorStore):
    """Qdrant 适配层，兜底缺失的返回字段。"""

    def query(self, *args: Any, **kwargs: Any):
        result = super().query(*args, **kwargs)
        if result.ids is None:
            result.ids = []
        if result.similarities is None:
            result.similarities = []
        if result.nodes is None:
            result.nodes = []
        return result


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
                "text-sparse": qdrant_models.SparseVectorParams(
                    index=qdrant_models.SparseIndexParams()
                )
            },
        )
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="date_numeric",
            field_schema=qdrant_models.PayloadSchemaType.INTEGER,
        )
    except Exception:
        # 索引已存在或集群不支持重复创建时忽略
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
        _ensure_hybrid_collection(q_client, collection, embedding_dim)
        return SafeQdrantVectorStore(
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
    keywords_any: Optional[List[str]] = None,
    keywords_all: Optional[List[str]] = None,
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
    if keywords_any:
        filter_items.append(
            MetadataFilter(
                key="keyword_list",
                value=keywords_any,
                operator=FilterOperator.ANY,
            )
        )
    if keywords_all:
        filter_items.append(
            MetadataFilter(
                key="keyword_list",
                value=keywords_all,
                operator=FilterOperator.ALL,
            )
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
class OpenAICompatibleLLM(OpenAI):
    """LLM wrapper that tolerates非标准 OpenAI 模型名."""

    def __init__(self, *args: Any, context_window: int = 8192, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._context_window_override = context_window

    def _tokenizer(self):
        try:
            return super()._tokenizer()
        except Exception:
            return None

    @property
    def metadata(self) -> LLMMetadata:
        try:
            return super().metadata
        except Exception:
            return LLMMetadata(
                context_window=self._context_window_override,
                num_output=self.max_tokens or -1,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=self.model,
                system_role=MessageRole.SYSTEM,
            )


Settings.llm = OpenAICompatibleLLM(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_API_BASE"),
    temperature=0.1,
    context_window=llm_context_window,
)
