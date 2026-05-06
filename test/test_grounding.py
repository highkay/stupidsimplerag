import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core import run_grounding_query
from app.ingest import process_file
from app.models import GroundingRequest, LLMAnalysis


def _make_point(
    *,
    filename: str,
    content: str,
    doc_hash: str = "doc-hash",
    scope: str | None = "reports/grounding/2026",
    date: str = "2026-05-06",
    summary: str = "本文聚焦AI光模块产业链变化，并在文末附带相关标的名单。",
):
    metadata = {
        "filename": filename,
        "date": date,
        "date_numeric": int(date.replace("-", "")),
        "doc_hash": doc_hash,
        "doc_summary": summary,
        "keywords": "立讯精密,仕佳光子,天孚通信,光模块",
        "keyword_list": ["立讯精密", "仕佳光子", "天孚通信", "光模块"],
        "original_text": content,
    }
    if scope:
        metadata["scope"] = scope
    node_payload = {
        "class_name": "TextNode",
        "metadata": metadata,
        "text": f"Date: {date}\nSummary: {summary}\nKey Data: \nTags: 立讯精密,仕佳光子\n---\n{content}",
    }
    return SimpleNamespace(
        payload={
            **metadata,
            "_node_type": "TextNode",
            "_node_content": json.dumps(node_payload, ensure_ascii=False),
        }
    )


@pytest.mark.asyncio
async def test_grounding_query_returns_body_and_list_only():
    content = """# AI光模块行业跟踪

立讯精密正在进入光模块市场，并讨论其对现有龙头格局的竞争影响。

## 相关标的
- 仕佳光子
- 天孚通信
"""
    point = _make_point(filename="ai_optical.md", content=content)
    request = GroundingRequest.model_validate(
        {
            "document": {"doc_hash": "doc-hash"},
            "candidates": [
                {"identifier": "002475.SZ", "name": "立讯精密", "aliases": ["立讯"], "candidate_type": "stock"},
                {"identifier": "688313.SH", "name": "仕佳光子", "candidate_type": "stock"},
            ],
            "skip_rerank": True,
        }
    )

    with patch("app.core._scroll_points_by_filter", new=AsyncMock(return_value=[point])):
        response = await run_grounding_query(request)

    assert response.document.filename == "ai_optical.md"
    assert response.document.doc_hash == "doc-hash"
    assert response.document.summary.startswith("本文聚焦AI光模块产业链")
    assert response.candidate_results[0].relevance_tier == "body_grounded"
    assert response.candidate_results[0].source_zone == "body"
    assert response.candidate_results[0].body_hit_count >= 1
    assert response.candidate_results[1].relevance_tier == "list_only"
    assert response.candidate_results[1].source_zone == "appendix_list"
    assert response.candidate_results[1].list_hit_count >= 1


@pytest.mark.asyncio
async def test_grounding_doc_hash_and_filename_scope_match():
    content = """产业链梳理

中际旭创在正文中被提及，但更多体现为供应链受益关系。
"""
    point = _make_point(
        filename="supply_chain.md",
        content=content,
        doc_hash="same-doc",
        scope="reports/a",
        summary="本文主要梳理光模块供应链中的受益与风险映射。",
    )

    with patch("app.core._scroll_points_by_filter", new=AsyncMock(return_value=[point])):
        by_hash = await run_grounding_query(
            GroundingRequest.model_validate(
                {
                    "document": {"doc_hash": "same-doc"},
                    "scope": "reports/a",
                    "candidates": [{"name": "中际旭创", "aliases": ["旭创"], "candidate_type": "stock"}],
                    "skip_rerank": True,
                }
            )
        )
        by_name = await run_grounding_query(
            GroundingRequest.model_validate(
                {
                    "document": {"filename": "supply_chain.md", "scope": "reports/a"},
                    "candidates": [{"name": "中际旭创", "aliases": ["旭创"], "candidate_type": "stock"}],
                    "skip_rerank": True,
                }
            )
        )

    assert by_hash.model_dump() == by_name.model_dump()


@pytest.mark.asyncio
async def test_grounding_returns_not_found_for_missing_candidate():
    point = _make_point(
        filename="policy.md",
        content="这篇文档讨论行业政策与需求，不包含具体公司名称。",
        doc_hash="missing-doc",
        scope=None,
        summary="本文讨论行业政策与需求变化。",
    )

    with patch("app.core._scroll_points_by_filter", new=AsyncMock(return_value=[point])):
        response = await run_grounding_query(
            GroundingRequest.model_validate(
                {
                    "document": {"doc_hash": "missing-doc"},
                    "candidates": [{"name": "立讯精密", "candidate_type": "stock"}],
                    "skip_rerank": True,
                }
            )
        )

    result = response.candidate_results[0]
    assert result.relevance_tier == "not_found"
    assert result.source_zone == "not_found"
    assert result.excerpts == []


def test_grounding_route_returns_404_for_missing_document(client):
    with patch("app.main.run_grounding_query", new=AsyncMock(side_effect=LookupError("document not found"))):
        response = client.post(
            "/grounding/query",
            json={
                "document": {"doc_hash": "missing"},
                "candidates": [{"name": "立讯精密", "candidate_type": "stock"}],
            },
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "document not found"


@pytest.mark.asyncio
async def test_process_file_adds_grounding_metadata():
    with patch(
        "app.ingest.analyze_document",
        new=AsyncMock(
            return_value=LLMAnalysis(
                summary="本文分析AI光模块行业，并给出相关标的名单。",
                table_narrative="",
                keywords=["立讯精密", "仕佳光子"],
            )
        ),
    ):
        nodes = await process_file(
            "grounding.md",
            "# 标题\n\n立讯精密正在进入光模块市场。\n\n## 相关标的\n- 仕佳光子\n",
            ingest_date="2026-05-06",
            doc_hash="grounding-hash",
            scope="reports/grounding",
        )

    assert nodes
    metadata = nodes[0].metadata
    assert metadata["doc_summary"].startswith("本文分析AI光模块行业")
    assert "section_type" in metadata
    assert "chunk_index" in metadata
    assert "is_list_zone" in metadata
