import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from llama_index.core.schema import TextNode

import app.main as main_module


@pytest.mark.asyncio
async def test_identical_ingest_requests_join_same_inflight_work():
    main_module._INGEST_DOC_LOCKS.clear()
    main_module._INGEST_REQUEST_FUTURES.clear()

    async def fake_inner(
        filename: str,
        content: str,
        ingest_date: str | None,
        force_update: bool = False,
        scope: str | None = None,
        doc_hash: str | None = None,
    ):
        await asyncio.sleep(0.05)
        return (
            [TextNode(text=content, metadata={"filename": filename})],
            False,
            doc_hash or "missing-hash",
        )

    with patch("app.main._process_and_insert_content", new=AsyncMock(side_effect=fake_inner)) as mock_inner:
        results = await asyncio.gather(
            main_module._process_and_insert_content_deduped(
                "same.md",
                "same content",
                "2026-05-07",
                force_update=True,
                scope=None,
            ),
            main_module._process_and_insert_content_deduped(
                "same.md",
                "same content",
                "2026-05-07",
                force_update=True,
                scope=None,
            ),
        )

    assert mock_inner.await_count == 1
    assert results[0][2] == results[1][2]


@pytest.mark.asyncio
async def test_same_document_different_content_is_serialized():
    main_module._INGEST_DOC_LOCKS.clear()
    main_module._INGEST_REQUEST_FUTURES.clear()
    first_started = asyncio.Event()
    first_released = asyncio.Event()
    second_started = asyncio.Event()
    call_order: list[str] = []

    async def fake_inner(
        filename: str,
        content: str,
        ingest_date: str | None,
        force_update: bool = False,
        scope: str | None = None,
        doc_hash: str | None = None,
    ):
        if content == "first":
            call_order.append("first-start")
            first_started.set()
            await first_released.wait()
            call_order.append("first-end")
        else:
            call_order.append("second-start")
            second_started.set()
            call_order.append("second-end")
        return (
            [TextNode(text=content, metadata={"filename": filename})],
            False,
            doc_hash or "missing-hash",
        )

    with patch("app.main._process_and_insert_content", new=AsyncMock(side_effect=fake_inner)):
        task1 = asyncio.create_task(
            main_module._process_and_insert_content_deduped(
                "same.md",
                "first",
                "2026-05-07",
                force_update=True,
                scope=None,
            )
        )
        await first_started.wait()

        task2 = asyncio.create_task(
            main_module._process_and_insert_content_deduped(
                "same.md",
                "second",
                "2026-05-07",
                force_update=True,
                scope=None,
            )
        )

        await asyncio.sleep(0.05)
        assert not second_started.is_set()
        first_released.set()
        await asyncio.gather(task1, task2)

    assert call_order == ["first-start", "first-end", "second-start", "second-end"]
