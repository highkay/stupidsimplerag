import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from app.main import app
from app.models import LLMAnalysis

client = TestClient(app)

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

def test_ingest_with_scope(mock_deps):
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

def test_chat_with_scope(mock_deps):
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

def test_chat_lod_endpoint():
    # Mock specifically for LOD endpoint which uses perform_hierarchical_search
    with patch("app.main.perform_hierarchical_search", new_callable=AsyncMock) as mock_hier:
        mock_hier.return_value = MagicMock(
            source_nodes=[], __str__=lambda x: "LOD Answer"
        )
        response = client.post(
            "/chat/lod",
            json={
                "query": "Test",
                "scope": "myscope"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "LOD Answer"
        mock_hier.assert_called_once()
        kwargs = mock_hier.call_args.kwargs
        assert kwargs["scope"] == "myscope"
