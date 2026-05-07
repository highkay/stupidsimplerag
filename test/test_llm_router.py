import json
from pathlib import Path

from app import openai_utils


def test_build_llm_uses_purpose_specific_legacy_env(monkeypatch):
    monkeypatch.delenv("LLM_ROUTER_CONFIG", raising=False)
    monkeypatch.delenv("LLM_ROUTER_CONFIG_FILE", raising=False)
    monkeypatch.setattr(openai_utils, "_ROUTER_RUNTIME", None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "chat-a,chat-b")
    monkeypatch.setenv("INGEST_LLM_MODEL", "ingest-a,ingest-b")
    monkeypatch.setenv("LLM_API_BASE", "http://chat.example.com/v1")
    monkeypatch.setenv("INGEST_LLM_API_BASE", "http://ingest.example.com/v1")

    llm = openai_utils.build_llm(purpose="ingest", temperature=0.1, context_window=4096)

    assert isinstance(llm, openai_utils.OpenAICompatibleLLM)
    assert [delegate.model for delegate in llm._delegates] == ["ingest-a", "ingest-b"]
    assert all(delegate.api_base == "http://ingest.example.com/v1" for delegate in llm._delegates)


def test_router_runtime_opens_circuit_and_skips_failed_deployment():
    config = openai_utils.LLMRouterConfig.model_validate(
        {
            "deployments": [
                {
                    "name": "chat-primary",
                    "model": "grok-4.20-fast",
                    "weight": 1,
                    "max_inflight": 1,
                    "failure_threshold": 1,
                    "cooldown_s": 30,
                    "purposes": ["chat"],
                },
                {
                    "name": "chat-secondary",
                    "model": "qwen-3-235b-a22b-instruct-2507",
                    "weight": 1,
                    "max_inflight": 1,
                    "failure_threshold": 1,
                    "cooldown_s": 30,
                    "purposes": ["chat"],
                },
            ]
        }
    )
    runtime = openai_utils.LLMRouterRuntime(config)

    first = runtime.acquire_sync("chat")
    runtime.release_failure_sync(first, 0.1, RuntimeError("429 Too Many Requests"))

    second = runtime.acquire_sync("chat")
    assert second.deployment_name != first.deployment_name


def test_build_llm_uses_router_config_file_for_purpose_pools(monkeypatch, tmp_path):
    config_path = tmp_path / "llm_router.json"
    config_path.write_text(
        json.dumps(
            {
                "deployments": [
                    {
                        "name": "chat-primary",
                        "model": "grok-4.20-fast",
                        "api_base": "http://gateway.example.com/v1",
                        "weight": 4,
                        "max_inflight": 2,
                    },
                    {
                        "name": "ingest-primary",
                        "model": "LongCat-Flash-Chat",
                        "api_base": "http://gateway.example.com/v1",
                        "weight": 2,
                        "max_inflight": 1,
                    },
                ],
                "pools": {
                    "chat": {"deployments": ["chat-primary"], "max_inflight": 4, "retry_budget": 1},
                    "ingest": {"deployments": ["ingest-primary"], "max_inflight": 2, "retry_budget": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_ROUTER_CONFIG_FILE", str(config_path))
    monkeypatch.setattr(openai_utils, "_ROUTER_RUNTIME", None)

    chat_llm = openai_utils.build_llm(purpose="chat", temperature=0.1, context_window=4096)
    ingest_llm = openai_utils.build_llm(purpose="ingest", temperature=0.1, context_window=4096)

    assert isinstance(chat_llm, openai_utils.GatewayRoutedLLM)
    assert isinstance(ingest_llm, openai_utils.GatewayRoutedLLM)
    assert sorted(chat_llm._delegates_by_name) == ["chat-primary"]
    assert sorted(ingest_llm._delegates_by_name) == ["ingest-primary"]


def test_example_router_config_inherits_shared_gateway_env(monkeypatch):
    example_path = Path(__file__).resolve().parents[1] / "llm_router.example.json"
    payload = json.loads(example_path.read_text(encoding="utf-8"))

    monkeypatch.setenv("LLM_API_BASE", "http://gateway.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-shared")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = openai_utils.LLMRouterConfig.model_validate(payload)
    runtime = openai_utils.LLMRouterRuntime(config)
    chat_llm = openai_utils.GatewayRoutedLLM(
        router=runtime,
        purpose="chat",
        temperature=0.1,
        context_window=4096,
    )
    ingest_llm = openai_utils.GatewayRoutedLLM(
        router=runtime,
        purpose="ingest",
        temperature=0.1,
        context_window=4096,
    )

    assert sorted(chat_llm._delegates_by_name) == [
        "chat-gemma4-31b",
        "chat-gpt-oss-120b",
        "chat-grok-fast",
        "chat-longcat-flash",
        "chat-qwen-235b",
        "chat-step-flash",
    ]
    assert sorted(ingest_llm._delegates_by_name) == [
        "ingest-deepseek-v4-flash",
        "ingest-gpt-oss-120b",
        "ingest-longcat-flash",
        "ingest-nemotron-3-super",
        "ingest-qwen3-5-122b-a10b",
        "ingest-qwen3-next-80b",
        "ingest-step-flash",
    ]
    assert all(
        delegate.api_base == "http://gateway.example.com/v1"
        for delegate in chat_llm._delegates_by_name.values()
    )
    assert all(
        delegate.api_base == "http://gateway.example.com/v1"
        for delegate in ingest_llm._delegates_by_name.values()
    )
    assert all(
        delegate.api_key == "sk-shared"
        for delegate in chat_llm._delegates_by_name.values()
    )
    assert all(
        delegate.api_key == "sk-shared"
        for delegate in ingest_llm._delegates_by_name.values()
    )
