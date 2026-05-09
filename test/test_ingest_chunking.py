from unittest.mock import AsyncMock, patch

import pytest
from chonkie import TokenChunker

from app.ingest import (
    DEFAULT_CHUNK_PLAN,
    MEDIUM_CHUNK_PLAN,
    LARGE_CHUNK_PLAN,
    XL_CHUNK_PLAN,
    XXL_CHUNK_PLAN,
    choose_chunk_plan,
    process_file,
)
from app.models import LLMAnalysis


def test_choose_chunk_plan_keeps_list_sections_small():
    assert choose_chunk_plan(document_length=200_000, section_type="appendix_list") == DEFAULT_CHUNK_PLAN
    assert choose_chunk_plan(document_length=200_000, section_type="appendix_table") == DEFAULT_CHUNK_PLAN
    assert choose_chunk_plan(document_length=200_000, section_type="qa") == DEFAULT_CHUNK_PLAN


def test_choose_chunk_plan_uses_xxl_for_ultra_large_body_documents():
    assert choose_chunk_plan(document_length=350_000, section_type="body") == XXL_CHUNK_PLAN


@pytest.mark.asyncio
async def test_process_file_uses_larger_body_chunks_for_large_documents():
    paragraph = (
        "徐工机械发布年度报告，核心看点包括工程机械景气修复、海外收入增长、"
        "费用率改善以及资产负债表优化。"
    )
    repeated_body = "\n\n".join([paragraph] * 600)
    content = "# 标题\n\n" + repeated_body

    with patch(
        "app.ingest.analyze_document",
        new=AsyncMock(
            return_value=LLMAnalysis(
                summary="年度报告摘要",
                table_narrative="",
                keywords=["徐工机械"],
            )
        ),
    ):
        nodes = await process_file(
            "large-body.md",
            content,
            ingest_date="2026-05-07",
            doc_hash="large-body-hash",
        )

    baseline_count = len(
        TokenChunker(chunk_size=DEFAULT_CHUNK_PLAN[0], chunk_overlap=DEFAULT_CHUNK_PLAN[1])(content)
    )
    assert nodes
    assert len(nodes) < baseline_count
    assert choose_chunk_plan(document_length=len(content), section_type="body") in {
        MEDIUM_CHUNK_PLAN,
        LARGE_CHUNK_PLAN,
        XL_CHUNK_PLAN,
        XXL_CHUNK_PLAN,
    }
    assert all(node.metadata["section_type"] in {"title", "body"} for node in nodes)


@pytest.mark.asyncio
async def test_process_file_preserves_appendix_list_metadata_for_large_documents():
    body = "\n\n".join(["公司主营业务稳定增长。"] * 400)
    appendix = "## 相关标的\n立讯精密、仕佳光子、天孚通信、剑桥科技、工业富联、源杰科技"
    content = "# 标题\n\n" + body + "\n\n" + appendix

    with patch(
        "app.ingest.analyze_document",
        new=AsyncMock(
            return_value=LLMAnalysis(
                summary="包含正文和附录名单的文档",
                table_narrative="",
                keywords=["立讯精密", "仕佳光子"],
            )
        ),
    ):
        nodes = await process_file(
            "large-appendix.md",
            content,
            ingest_date="2026-05-07",
            doc_hash="large-appendix-hash",
        )

    list_nodes = [node for node in nodes if node.metadata["section_type"] == "appendix_list"]
    assert list_nodes
    assert all(node.metadata["is_list_zone"] is True for node in list_nodes)
    assert all("相关标的" in (node.metadata.get("heading_path") or "") for node in list_nodes)


@pytest.mark.asyncio
async def test_process_file_rechunks_when_embedding_budget_is_exceeded(monkeypatch):
    paragraph = (
        "公司在海外市场、毛利率、现金流、研发进展和订单储备方面均有细节披露，"
        "同时包含多段英文、数字和表格转写内容以抬高 embedding token 密度。"
    )
    content = "# 标题\n\n" + "\n\n".join([paragraph] * 900)

    with patch(
        "app.ingest.analyze_document",
        new=AsyncMock(
            return_value=LLMAnalysis(
                summary="预算校验测试",
                table_narrative="",
                keywords=["预算", "切块"],
            )
        ),
    ):
        monkeypatch.setattr("app.ingest.get_embedding_input_token_budget", lambda: 1800)
        monkeypatch.setattr(
            "app.ingest.count_embedding_input_tokens",
            lambda text: max(1, len(text) // 2),
        )
        nodes = await process_file(
            "budget-rechunk.md",
            content,
            ingest_date="2026-05-09",
            doc_hash="budget-rechunk-hash",
        )

    assert nodes
    assert all(len(node.text) // 2 <= 1800 for node in nodes)
    assert max(len(node.metadata["original_text"]) for node in nodes) < XXL_CHUNK_PLAN[0]
