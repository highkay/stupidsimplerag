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
    metadata_overrides: dict | None = None,
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
    if metadata_overrides:
        metadata.update(metadata_overrides)
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


@pytest.mark.asyncio
async def test_grounding_ignores_stale_list_metadata_for_pdf_body_chunk():
    content = """<<<< FILE_START: 276654 >>>>
---
## 附件内容（自动转换）
证券代码：600636
证券简称：*ST 国化
国新文化控股股份有限公司
关于以集中竞价交易方式回购股份的回购报告书
特别风险提示：
1、因回购比例有限，可能短期内即达到回购上限：国新文化控股股份有限公司本次回购比例不低于公司股本5%（含）且不超过10%（含）。
2、存在因股票价格持续超出回购价格上限而导致本次回购计划无法顺利实施的风险。
"""
    point = _make_point(
        filename="issuer_report.md",
        content=content,
        doc_hash="issuer-doc",
        summary="国新文化回购股份报告，正文围绕回购比例、金额和实施风险展开。",
        metadata_overrides={
            "section_type": "appendix_list",
            "heading_path": "附件内容（自动转换）",
            "is_list_zone": True,
            "is_qa_zone": False,
            "section_order": 0,
            "block_index": 0,
            "chunk_index": 0,
        },
    )

    with patch("app.core._scroll_points_by_filter", new=AsyncMock(return_value=[point])):
        response = await run_grounding_query(
            GroundingRequest.model_validate(
                {
                    "document": {"doc_hash": "issuer-doc"},
                    "candidates": [{"name": "国新文化", "candidate_type": "stock"}],
                    "skip_rerank": True,
                }
            )
        )

    result = response.candidate_results[0]
    assert result.relevance_tier == "body_grounded"
    assert result.source_zone == "body"
    assert result.excerpts[0].section_type == "body"


@pytest.mark.asyncio
async def test_grounding_keeps_same_tier_across_pdf_line_break_variants():
    compact = """## 附件内容（自动转换）
国新文化控股股份有限公司关于第十一届董事会第十二次会议决议公告
国新文化控股股份有限公司董事会审议通过回购股份议案，并提示股价交易风险与实施风险。"""
    broken = """## 附件内容（自动转换）
国新文化控股股份有限公司
关于第十一届董事会第十二次会议决议公告
国新文化控股股份有限公司
董事会审议通过回购股份议案，
并提示股价交易风险与实施风险。"""

    with patch(
        "app.core._scroll_points_by_filter",
        new=AsyncMock(
            side_effect=[
                [_make_point(filename="compact.md", content=compact, doc_hash="same-family")],
                [_make_point(filename="broken.md", content=broken, doc_hash="same-family-2")],
            ]
        ),
    ):
        compact_resp = await run_grounding_query(
            GroundingRequest.model_validate(
                {
                    "document": {"filename": "compact.md"},
                    "candidates": [{"name": "国新文化", "candidate_type": "stock"}],
                    "skip_rerank": True,
                }
            )
        )
        broken_resp = await run_grounding_query(
            GroundingRequest.model_validate(
                {
                    "document": {"filename": "broken.md"},
                    "candidates": [{"name": "国新文化", "candidate_type": "stock"}],
                    "skip_rerank": True,
                }
            )
        )

    assert compact_resp.candidate_results[0].relevance_tier == "body_grounded"
    assert broken_resp.candidate_results[0].relevance_tier == "body_grounded"


@pytest.mark.asyncio
async def test_grounding_preserves_semicolon_appendix_lists():
    content = """## 相关标的
立讯精密；仕佳光子；天孚通信"""
    point = _make_point(
        filename="appendix_semicolon.md",
        content=content,
        doc_hash="semicolon-doc",
        summary="文末附带相关标的名单。",
    )

    with patch("app.core._scroll_points_by_filter", new=AsyncMock(return_value=[point])):
        response = await run_grounding_query(
            GroundingRequest.model_validate(
                {
                    "document": {"doc_hash": "semicolon-doc"},
                    "candidates": [{"name": "仕佳光子", "candidate_type": "stock"}],
                    "skip_rerank": True,
                }
            )
        )

    result = response.candidate_results[0]
    assert result.relevance_tier == "list_only"
    assert result.source_zone == "appendix_list"


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
    assert metadata["section_type"] == "body"
    assert "section_type" in metadata
    assert "chunk_index" in metadata
    assert "is_list_zone" in metadata
    assert metadata["is_list_zone"] is False
