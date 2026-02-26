import datetime
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from llama_index.core.schema import TextNode, NodeWithScore
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
