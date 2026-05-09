import asyncio

import pytest

from app import core


def _fake_embedding_response(inputs):
    return [{"embedding": [float(index), float(len(text))]} for index, text in enumerate(inputs, 1)]


def test_openai_compatible_embedding_applies_query_and_document_prefixes(monkeypatch):
    model = core.OpenAICompatibleEmbedding(
        model="jina-embeddings-v5-text-small-retrieval",
        api_key=None,
        api_base="http://embedding:8080/v1",
        dimensions=1024,
        query_prefix="Query:",
        text_prefix="Document:",
    )
    captured: list[list[str]] = []

    monkeypatch.setattr(
        model,
        "_send_embedding_request",
        lambda inputs: captured.append(list(inputs)) or _fake_embedding_response(inputs),
    )

    assert model._get_query_embedding("立讯精密") == [1.0, float(len("Query: 立讯精密"))]
    assert model._get_text_embedding("正文段落") == [1.0, float(len("Document: 正文段落"))]
    assert model._get_text_embeddings(["段落A", "段落B"]) == [
        [1.0, float(len("Document: 段落A"))],
        [2.0, float(len("Document: 段落B"))],
    ]
    assert captured == [
        ["Query: 立讯精密"],
        ["Document: 正文段落"],
        ["Document: 段落A", "Document: 段落B"],
    ]


def test_openai_compatible_embedding_leaves_inputs_unchanged_without_prefixes(monkeypatch):
    model = core.OpenAICompatibleEmbedding(
        model="text-embedding-3-small",
        api_key="sk-test",
        api_base="https://api.example.com/v1",
        dimensions=1536,
    )
    captured: list[list[str]] = []

    monkeypatch.setattr(
        model,
        "_send_embedding_request",
        lambda inputs: captured.append(list(inputs)) or _fake_embedding_response(inputs),
    )

    model._get_query_embedding("hello")
    model._get_text_embeddings(["alpha", "beta"])
    assert captured == [["hello"], ["alpha", "beta"]]


def test_openai_compatible_embedding_splits_large_batches(monkeypatch):
    monkeypatch.setattr(core, "EMBEDDING_REQUEST_BATCH_SIZE", 2)
    model = core.OpenAICompatibleEmbedding(
        model="jina-embeddings-v5-text-small-retrieval",
        api_key=None,
        api_base="http://embedding:8080/v1",
        dimensions=1024,
    )
    captured: list[list[str]] = []

    monkeypatch.setattr(
        model,
        "_send_embedding_request",
        lambda inputs: captured.append(list(inputs)) or _fake_embedding_response(inputs),
    )

    result = model._get_text_embeddings(["A", "B", "C", "D", "E"])

    assert result == [
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 1.0],
    ]
    assert captured == [["A", "B"], ["C", "D"], ["E"]]


@pytest.mark.asyncio
async def test_openai_compatible_embedding_limits_async_request_concurrency(monkeypatch):
    monkeypatch.setattr(core, "EMBEDDING_CONCURRENCY", 1)
    model = core.OpenAICompatibleEmbedding(
        model="jina-embeddings-v5-text-small-retrieval",
        api_key=None,
        api_base="http://embedding:8080/v1",
        dimensions=1024,
    )

    in_flight = 0
    max_in_flight = 0

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"embedding": [1.0, 2.0]}]}

    async def _fake_post(*args, **kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _Response()

    monkeypatch.setattr(model._aclient, "post", _fake_post)

    await asyncio.gather(
        model._aget_text_embedding("first"),
        model._aget_text_embedding("second"),
        model._aget_text_embedding("third"),
    )

    assert max_in_flight == 1


def test_init_embedding_model_uses_compat_client_when_prefixes_enabled(monkeypatch):
    monkeypatch.setattr(core, "embedding_query_prefix", "Query:")
    monkeypatch.setattr(core, "embedding_document_prefix", "Document:")
    monkeypatch.setattr(core, "embedding_dim", 1024)
    monkeypatch.setattr(core, "embedding_base", "http://embedding:8080/v1")
    monkeypatch.setattr(core, "embedding_key", None)
    monkeypatch.setattr(core, "embedding_timeout", 75.0)

    def _unexpected_openai_embedding(**kwargs):
        raise AssertionError("OpenAIEmbedding should not be used when prefixes are configured")

    monkeypatch.setattr(core, "OpenAIEmbedding", _unexpected_openai_embedding)
    monkeypatch.setenv("EMBEDDING_MODEL", "jina-embeddings-v5-text-small-retrieval")

    model = core._init_embedding_model()

    assert isinstance(model, core.FixedDimensionEmbedding)
    assert isinstance(model._base_model, core.OpenAICompatibleEmbedding)
    assert model._base_model._query_prefix == "Query: "
    assert model._base_model._text_prefix == "Document: "
    assert model._base_model._timeout == 75.0
