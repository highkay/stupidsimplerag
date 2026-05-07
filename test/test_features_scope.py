import datetime
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from llama_index.core.schema import TextNode, NodeWithScore
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core import KeywordFilterPostprocessor, _ensure_hybrid_collection, perform_hierarchical_search
from app.ingest import compute_doc_hash, process_file
from app.main import _parse_epoch_date
from app.models import LLMAnalysis

@pytest.fixture
def mock_deps():
    # Mock ingest analysis
    with patch("app.ingest.analyze_document", new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = LLMAnalysis(
            summary="Summary", table_narrative="", keywords=["k1"]
        )
        # Mock Qdrant checks
        with patch("app.main.adoc_exists", new_callable=AsyncMock) as mock_exists:
            mock_exists.return_value = False
            # Mock DB insert
            with patch("app.main.insert_nodes", new_callable=AsyncMock) as mock_insert:
                mock_insert.return_value = None
                # Mock Query Engine
                with patch("app.main.get_query_engine") as mock_engine_builder:
                    mock_engine = MagicMock()
                    mock_engine.aquery = AsyncMock(return_value=MagicMock(
                        source_nodes=[], __str__=lambda x: "Answer"
                    ))
                    mock_engine_builder.return_value = mock_engine
                    yield {
                        "analyze": mock_analyze,
                        "insert": mock_insert,
                        "engine_builder": mock_engine_builder
                    }

def test_ingest_with_scope(mock_deps, client):
    response = client.post(
        "/ingest/text",
        json={
            "content": "Test content",
            "filename": "test.md",
            "scope": "myscope"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["scope"] == "myscope"
    assert data["filename"] == "test.md"

def test_chat_with_scope(mock_deps, client):
    response = client.post(
        "/chat",
        json={
            "query": "Test",
            "scope": "myscope"
        }
    )
    assert response.status_code == 200
    # Verify get_query_engine was called with scope
    mock_deps["engine_builder"].assert_called()
    call_kwargs = mock_deps["engine_builder"].call_args.kwargs
    assert call_kwargs.get("scope") == "myscope"

def test_chat_lod_endpoint(client):
    # Mock specifically for LOD endpoint which uses perform_hierarchical_search
    with patch("app.main.perform_hierarchical_search", new_callable=AsyncMock) as mock_hier:
        mock_hier.return_value = MagicMock(
            source_nodes=[], __str__=lambda x: "LOD Answer"
        )
        response = client.post(
            "/chat/lod",
            json={
                "query": "Test",
                "scope": "myscope",
                "filename": "target.md",
                "filename_contains": "target",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "LOD Answer"
        mock_hier.assert_called_once()
        kwargs = mock_hier.call_args.kwargs
        assert kwargs["scope"] == "myscope"
        assert kwargs["filename"] == "target.md"
        assert kwargs["filename_contains"] == "target"


def test_parse_epoch_date_supports_milliseconds():
    sec = _parse_epoch_date("1700000000")
    ms = _parse_epoch_date("1700000000000")
    assert sec is not None
    assert sec == ms


def test_ingest_text_default_date_uses_app_timezone(client):
    fixed_now = datetime.datetime(
        2026, 2, 1, 0, 30, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=8))
    )
    captured = {}

    async def _fake_process(
        filename: str,
        content: str,
        ingest_date: str | None,
        force_update: bool = False,
        scope: str | None = None,
        doc_hash: str | None = None,
    ):
        captured["ingest_date"] = ingest_date
        return [], False, "fake_hash"

    with patch("app.main.now_in_app_tz", return_value=fixed_now), patch(
        "app.main._process_and_insert_content", new=AsyncMock(side_effect=_fake_process)
    ):
        response = client.post("/ingest/text", json={"content": "Test content"})

    assert response.status_code == 200
    assert captured.get("ingest_date") == "2026-02-01"


def test_chat_source_includes_scope(client):
    class FakeResponse:
        def __init__(self):
            self.source_nodes = [
                NodeWithScore(
                    node=TextNode(
                        text="example text",
                        metadata={
                            "filename": "scope_test.md",
                            "date": "2026-01-01",
                            "keywords": "k1",
                            "original_text": "example text",
                            "scope": "reports/2026",
                        },
                    ),
                    score=0.88,
                )
            ]

        def __str__(self):
            return "answer"

    class FakeEngine:
        async def aquery(self, query):
            return FakeResponse()

    with patch("app.main.get_query_engine", return_value=FakeEngine()):
        response = client.post("/chat", json={"query": "scope?"})
    assert response.status_code == 200
    source = response.json()["sources"][0]
    assert source["scope"] == "reports/2026"


def test_chat_lod_source_includes_scope(client):
    class FakeResponse:
        def __init__(self):
            self.source_nodes = [
                NodeWithScore(
                    node=TextNode(
                        text="lod example text",
                        metadata={
                            "filename": "lod_scope_test.md",
                            "date": "2026-01-02",
                            "keywords": "k2",
                            "original_text": "lod example text",
                            "scope": "reports/lod/2026",
                        },
                    ),
                    score=0.77,
                )
            ]

        def __str__(self):
            return "lod answer"

    with patch(
        "app.main.perform_hierarchical_search",
        new=AsyncMock(return_value=FakeResponse()),
    ):
        response = client.post("/chat/lod", json={"query": "lod scope?"})
    assert response.status_code == 200
    source = response.json()["sources"][0]
    assert source["scope"] == "reports/lod/2026"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_keyword_filter_matches_scope_tokens_and_strength_aliases():
    medium_node = NodeWithScore(
        node=TextNode(
            text="agent report chunk",
            metadata={
                "filename": "agent_analysis_300037_SZ_20260329_WATCH.md",
                "keywords": "300037.SZ,新宙邦,电解液",
                "scope": "agent_report|stock=300037.SZ|signal=WATCH|confidence=2|keywords=WATCH-MEDIUM-300037.SZ-300037",
            },
        ),
        score=0.9,
    )
    high_node = NodeWithScore(
        node=TextNode(
            text="high conviction agent report chunk",
            metadata={
                "filename": "agent_analysis_000000_GROUP_20260401_LONG.md",
                "scope": "agent_report|stock=000000.GROUP|signal=LONG|confidence=3|keywords=LONG-HIGH-000000.GROUP",
            },
        ),
        score=0.9,
    )

    assert KeywordFilterPostprocessor({"moderate"}, set())._postprocess_nodes([medium_node])
    assert KeywordFilterPostprocessor({"strong"}, set())._postprocess_nodes([high_node])
    assert KeywordFilterPostprocessor({"long"}, set())._postprocess_nodes([high_node])
    assert KeywordFilterPostprocessor({"weak"}, set())._postprocess_nodes([]) == []
    assert not KeywordFilterPostprocessor({"weak"}, set())._postprocess_nodes([medium_node])


def test_ensure_hybrid_collection_tolerates_concurrent_create_conflict():
    class RaceClient:
        def __init__(self):
            self.exists_calls = 0
            self.indexes = []

        def collection_exists(self, _collection_name):
            self.exists_calls += 1
            return self.exists_calls > 1

        def create_collection(self, **_kwargs):
            raise UnexpectedResponse(409, "Conflict", b"already exists", {})

        def create_payload_index(self, **kwargs):
            self.indexes.append(kwargs["field_name"])

    client = RaceClient()

    _ensure_hybrid_collection(client, "race", 1536)

    assert client.indexes == ["date_numeric", "doc_hash", "scope", "filename"]


@pytest.mark.asyncio
async def test_lod_l2_keeps_original_filters():
    l1_nodes = [
        NodeWithScore(
            node=TextNode(
                text="l1",
                metadata={
                    "filename": "same.md",
                    "scope": "scope-a",
                    "keywords": "k1",
                },
            ),
            score=0.8,
        )
    ]

    class FakeL1Engine:
        async def aretrieve(self, _query_bundle):
            return l1_nodes

    class FakeL2Engine:
        async def aquery(self, _query):
            return MagicMock(source_nodes=[], __str__=lambda _self: "answer")

    with patch("app.core.get_query_engine", side_effect=[FakeL1Engine(), FakeL2Engine()]) as mock_get:
        await perform_hierarchical_search(
            query="test",
            scope="scope-a",
            start_date="2026-01-01",
            end_date="2026-12-31",
            filename_contains="same",
            keywords_any=["moderate"],
            keywords_all=["k1"],
        )

    l2_kwargs = mock_get.call_args_list[1].kwargs
    assert l2_kwargs["start_date"] == "2026-01-01"
    assert l2_kwargs["end_date"] == "2026-12-31"
    assert l2_kwargs["filename_contains"] == "same"
    assert l2_kwargs["filenames_in"] == ["same.md"]
    assert l2_kwargs["scope"] == "scope-a"
    assert l2_kwargs["keywords_any"] == ["moderate"]
    assert l2_kwargs["keywords_all"] == ["k1"]


@pytest.mark.asyncio
async def test_lod_l2_uses_scoped_doc_identity_without_scope_filter():
    l1_nodes = [
        NodeWithScore(
            node=TextNode(text="a", metadata={"filename": "same.md", "scope": "scope-a"}),
            score=0.9,
        ),
        NodeWithScore(
            node=TextNode(text="b", metadata={"filename": "same.md", "scope": "scope-b"}),
            score=0.8,
        ),
    ]

    class FakeL1Engine:
        async def aretrieve(self, _query_bundle):
            return l1_nodes

    class FakeL2Engine:
        async def aquery(self, _query):
            return MagicMock(source_nodes=[], __str__=lambda _self: "answer")

    with patch("app.core.get_query_engine", side_effect=[FakeL1Engine(), FakeL2Engine()]) as mock_get:
        await perform_hierarchical_search(query="test", top_docs=1)

    l2_kwargs = mock_get.call_args_list[1].kwargs
    assert l2_kwargs["filenames_in"] == ["same.md"]
    assert l2_kwargs["scopes_in"] == ["scope-a"]


def test_ingest_scope_dedup_is_scoped(client):
    content = "Scoped content"
    expected_hash = compute_doc_hash(content)

    with patch("app.main.adoc_exists", new=AsyncMock(return_value=True)) as mock_exists:
        response = client.post(
            "/ingest/text",
            json={"content": content, "filename": "scoped.md", "scope": "reports/2026"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    mock_exists.assert_awaited_once_with(expected_hash, scope="reports/2026")


def test_documents_api_includes_scope(client):
    docs = [
        {
            "filename": "scoped.md",
            "date": "2026-01-01",
            "chunks": 3,
            "scope": "reports/2026",
        }
    ]
    with patch("app.main.list_all_documents", new=AsyncMock(return_value=docs)):
        response = client.get("/documents")

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["scope"] == "reports/2026"


def test_delete_document_passes_scope(client):
    with patch(
        "app.main.delete_nodes_by_filename",
        new=AsyncMock(return_value=True),
    ) as mock_delete:
        response = client.delete("/documents/demo.md", params={"scope": "reports/2026"})

    assert response.status_code == 200
    mock_delete.assert_awaited_once_with("demo.md", scope="reports/2026")


def test_documents_ui_delete_uses_scoped_query(client):
    docs = [
        {
            "filename": "scoped.md",
            "date": "2026-01-01",
            "chunks": 1,
            "scope": "reports/2026",
        }
    ]
    with patch("app.main.list_all_documents", new=AsyncMock(return_value=docs)):
        response = client.get("/ui/documents")

    assert response.status_code == 200
    assert "?scope=reports/2026" in response.text


@pytest.mark.asyncio
async def test_process_file_node_ids_include_scope():
    with patch("app.ingest.analyze_document", new=AsyncMock(return_value=LLMAnalysis())):
        nodes_a = await process_file(
            "same.md",
            "same content",
            ingest_date="2026-01-01",
            doc_hash="same-hash",
            scope="a",
        )
        nodes_b = await process_file(
            "same.md",
            "same content",
            ingest_date="2026-01-01",
            doc_hash="same-hash",
            scope="b",
        )

    assert nodes_a
    assert nodes_b
    assert nodes_a[0].id_ != nodes_b[0].id_


def test_chat_cache_cleared_after_successful_ingest(client):
    call_count = {"n": 0}

    class FakeResponse:
        def __init__(self, idx: int):
            self.source_nodes = [
                NodeWithScore(
                    node=TextNode(
                        text=f"text-{idx}",
                        metadata={
                            "filename": f"file-{idx}.md",
                            "date": "2026-01-01",
                            "original_text": f"text-{idx}",
                        },
                    ),
                    score=0.9,
                )
            ]
            self._answer = f"answer-{idx}"

        def __str__(self):
            return self._answer

    class FakeEngine:
        async def aquery(self, query):
            call_count["n"] += 1
            return FakeResponse(call_count["n"])

    async def fake_process_file(
        filename: str,
        content: str,
        ingest_date: str | None = None,
        doc_hash: str | None = None,
        scope: str | None = None,
    ):
        return [
            TextNode(
                text=content,
                metadata={
                    "filename": filename,
                    "date": ingest_date or "2026-01-01",
                    "date_numeric": 20260101,
                    "doc_hash": doc_hash,
                    "keywords": "",
                    "keyword_list": [],
                    "original_text": content,
                },
            )
        ]

    with patch("app.main.get_query_engine", return_value=FakeEngine()), patch(
        "app.main.adoc_exists", new=AsyncMock(return_value=False)
    ), patch("app.main.process_file", new=AsyncMock(side_effect=fake_process_file)), patch(
        "app.main.insert_nodes", new=AsyncMock(return_value=None)
    ):
        from app.main import CACHE

        CACHE.clear()
        first = client.post("/chat", json={"query": "same question"})
        ingest = client.post(
            "/ingest/text", json={"content": "new content", "filename": "new.md"}
        )
        second = client.post("/chat", json={"query": "same question"})

    assert first.status_code == 200
    assert ingest.status_code == 200
    assert second.status_code == 200
    assert first.json()["answer"] == "answer-1"
    assert second.json()["answer"] == "answer-2"
    assert call_count["n"] == 2
