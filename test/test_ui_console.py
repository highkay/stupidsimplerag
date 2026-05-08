from unittest.mock import AsyncMock, patch

from app.models import (
    ChatResponse,
    GroundingDocumentInfo,
    GroundingExcerpt,
    GroundingResponse,
    GroundingResult,
)


def test_dashboard_mentions_grounding_and_lod(client):
    with patch(
        "app.main.get_collection_metrics",
        new=AsyncMock(
            return_value={
                "status": "ready",
                "collection_name": "financial_reports",
                "points_count": 123,
                "segments_count": 4,
                "vectors_count": 123,
                "error": None,
            }
        ),
    ):
        response = client.get("/")

    assert response.status_code == 200
    body = response.text
    assert "/grounding/query" in body
    assert "/chat/lod" in body
    assert "/ui/grounding" in body


def test_upload_ui_accepts_markdown_and_txt(client):
    response = client.get("/ui/upload")

    assert response.status_code == 200
    body = response.text
    assert 'accept=".md,.txt"' in body
    assert "doc_hash" in body


def test_batch_upload_ui_accepts_markdown_and_txt(client):
    response = client.get("/ui/upload/batch")

    assert response.status_code == 200
    assert 'accept=".md,.txt"' in response.text


def test_chat_ui_exposes_lod_mode(client):
    response = client.get("/ui/chat")

    assert response.status_code == 200
    assert 'value="lod"' in response.text


def test_chat_ui_lod_mode_calls_lod_endpoint(client):
    with patch(
        "app.main.chat_api",
        new=AsyncMock(return_value=ChatResponse(answer="chat", sources=[])),
    ) as mock_chat, patch(
        "app.main.chat_lod_api",
        new=AsyncMock(return_value=ChatResponse(answer="lod", sources=[])),
    ) as mock_lod:
        response = client.post(
            "/ui/chat/query",
            data={"query": "test", "query_mode": "lod"},
            headers={"hx-request": "true"},
        )

    assert response.status_code == 200
    assert "lod" in response.text
    mock_lod.assert_awaited_once()
    mock_chat.assert_not_awaited()


def test_grounding_page_exists(client):
    response = client.get("/ui/grounding")

    assert response.status_code == 200
    body = response.text
    assert 'name="doc_hash"' in body
    assert 'name="candidates_text"' in body


def test_grounding_ui_parses_candidates_and_renders_response(client):
    fake_response = GroundingResponse(
        document=GroundingDocumentInfo(
            doc_hash="doc-1",
            filename="sample.md",
            scope="reports/2026",
            date="2026-05-08",
            summary="这是一篇测试文档。",
            chunk_count=4,
        ),
        candidate_results=[
            GroundingResult(
                identifier="002475.SZ",
                name="立讯精密",
                relevance_tier="body_grounded",
                source_zone="body",
                source_reason="正文明确讨论。",
                body_hit_count=2,
                qa_hit_count=0,
                list_hit_count=0,
                candidate_brief="正文主叙事直接提及立讯精密。",
                excerpts=[
                    GroundingExcerpt(
                        section_type="body",
                        score=0.97,
                        text="立讯精密进入光模块市场。",
                        is_alias_hit=False,
                    )
                ],
            )
        ],
    )

    with patch(
        "app.main.grounding_api",
        new=AsyncMock(return_value=fake_response),
    ) as mock_grounding:
        response = client.post(
            "/ui/grounding/query",
            data={
                "doc_hash": "doc-1",
                "scope": "reports/2026",
                "candidates_text": "立讯精密\n002475.SZ|立讯精密|立讯|stock",
                "max_excerpts": "4",
            },
            headers={"hx-request": "true"},
        )

    assert response.status_code == 200
    body = response.text
    assert "sample.md" in body
    assert "body_grounded" in body
    assert "立讯精密进入光模块市场" in body

    request_body = mock_grounding.await_args.args[0]
    assert request_body.document.doc_hash == "doc-1"
    assert request_body.document.scope == "reports/2026"
    assert request_body.max_excerpts == 4
    assert len(request_body.candidates) == 2
    assert request_body.candidates[0].name == "立讯精密"
    assert request_body.candidates[1].identifier == "002475.SZ"
    assert request_body.candidates[1].aliases == ["立讯"]
    assert request_body.candidates[1].candidate_type == "stock"


def test_documents_page_renders_filter_and_rows(client):
    docs = [
        {
            "filename": "sample.md",
            "scope": "reports/2026",
            "date": "2026-05-08",
            "chunks": 3,
        }
    ]
    with patch("app.main.list_all_documents", new=AsyncMock(return_value=docs)):
        response = client.get("/ui/documents/manage")

    assert response.status_code == 200
    body = response.text
    assert "搜索 filename / scope / date" in body
    assert "sample.md" in body
