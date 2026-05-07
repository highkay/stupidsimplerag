from types import SimpleNamespace

from app.ingest import _extract_llm_analysis_payload


def test_extract_llm_analysis_payload_prefers_response_text_json():
    response = SimpleNamespace(
        text='{"summary":"ok","table_narrative":"","keywords":["a"]}',
        raw=None,
    )

    payload, source = _extract_llm_analysis_payload(response)

    assert payload["summary"] == "ok"
    assert payload["keywords"] == ["a"]
    assert source == "response.text"


def test_extract_llm_analysis_payload_recovers_json_substring():
    response = SimpleNamespace(
        text='下面是结果：\n```json\n{"summary":"ok","table_narrative":"","keywords":[]}\n```\n请查收',
        raw=None,
    )

    payload, source = _extract_llm_analysis_payload(response)

    assert payload["summary"] == "ok"
    assert payload["keywords"] == []
    assert source == "response.text:substring"


def test_extract_llm_analysis_payload_falls_back_to_raw_message_content():
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"summary":"raw","table_narrative":"","keywords":["x"]}',
                    reasoning_content="ignored",
                    reasoning="ignored",
                ),
                text=None,
            )
        ]
    )
    response = SimpleNamespace(text="", raw=raw)

    payload, source = _extract_llm_analysis_payload(response)

    assert payload["summary"] == "raw"
    assert payload["keywords"] == ["x"]
    assert source == "raw.message.content"


def test_extract_llm_analysis_payload_falls_back_to_reasoning_json():
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    reasoning_content='推理如下：{"summary":"reasoned","table_narrative":"","keywords":["r"]}',
                    reasoning=None,
                ),
                text=None,
            )
        ]
    )
    response = SimpleNamespace(text="", raw=raw)

    payload, source = _extract_llm_analysis_payload(response)

    assert payload["summary"] == "reasoned"
    assert payload["keywords"] == ["r"]
    assert source == "raw.message.reasoning_content:substring"
