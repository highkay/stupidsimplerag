import json
from unittest.mock import patch

from offline_ingest import ingest_batch, maybe_delay_between_requests


class _Response:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text


class _Client:
    def __init__(self, response: _Response):
        self._response = response

    def post(self, url, files=None, timeout=None):
        return self._response


def test_ingest_batch_handles_partial_success(tmp_path):
    p1 = tmp_path / "a.md"
    p2 = tmp_path / "b.md"
    p1.write_text("a", encoding="utf-8")
    p2.write_text("b", encoding="utf-8")

    payload = [
        {"status": "ok", "filename": "a.md", "chunks": 1},
        {"status": "error", "filename": "b.md", "error": "failed"},
    ]
    client = _Client(_Response(200, json.dumps(payload, ensure_ascii=False)))

    ok, _message, success_paths, item_errors = ingest_batch(
        client, "http://localhost:8000/ingest/batch", [p1, p2], timeout=5.0
    )

    assert ok is True
    assert success_paths == [p1]
    assert len(item_errors) == 1
    assert "b.md" in item_errors[0]


def test_ingest_batch_all_failed_returns_false(tmp_path):
    p1 = tmp_path / "a.md"
    p1.write_text("a", encoding="utf-8")

    payload = [{"status": "error", "filename": "a.md", "error": "failed"}]
    client = _Client(_Response(200, json.dumps(payload, ensure_ascii=False)))

    ok, _message, success_paths, item_errors = ingest_batch(
        client, "http://localhost:8000/ingest/batch", [p1], timeout=5.0
    )

    assert ok is False
    assert success_paths == []
    assert len(item_errors) == 1


def test_maybe_delay_between_requests_skips_non_positive():
    with patch("offline_ingest.time.sleep") as sleep:
        maybe_delay_between_requests(0)
        maybe_delay_between_requests(-1)

    sleep.assert_not_called()


def test_maybe_delay_between_requests_sleeps_when_positive():
    with patch("offline_ingest.time.sleep") as sleep:
        maybe_delay_between_requests(1.25)

    sleep.assert_called_once_with(1.25)
