import json
from pathlib import Path

import pytest

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


def test_router_runtime_logs_periodic_stats(monkeypatch):
    monkeypatch.setenv("LLM_ROUTER_STATS_INTERVAL", "2")
    logged_messages = []

    def fake_info(message, *args, **kwargs):
        if args:
            message = message % args
        logged_messages.append(str(message))

    monkeypatch.setattr(openai_utils.logger, "info", fake_info)
    config = openai_utils.LLMRouterConfig.model_validate(
        {
            "deployments": [
                {
                    "name": "chat-primary",
                    "model": "grok-4.20-fast",
                    "weight": 1,
                    "max_inflight": 1,
                    "failure_threshold": 2,
                    "cooldown_s": 30,
                    "purposes": ["chat"],
                }
            ]
        }
    )
    runtime = openai_utils.LLMRouterRuntime(config)

    first = runtime.acquire_sync("chat")
    runtime.release_success_sync(first, 0.25)
    second = runtime.acquire_sync("chat")
    runtime.release_success_sync(second, 0.50)

    output = "\n".join(logged_messages)
    assert "LLM router stats pool=chat" in output
    assert "chat-primary{req=2 ok=2 fail=0" in output


def test_router_runtime_reserves_chat_capacity_from_ingest():
    config = openai_utils.LLMRouterConfig.model_validate(
        {
            "deployments": [
                {
                    "name": "chat-primary",
                    "model": "grok-4.20-fast",
                    "weight": 1,
                    "max_inflight": 2,
                    "purposes": ["chat"],
                },
                {
                    "name": "ingest-primary",
                    "model": "LongCat-Flash-Chat",
                    "weight": 1,
                    "max_inflight": 3,
                    "purposes": ["ingest"],
                },
            ],
            "pools": {
                "chat": {"deployments": ["chat-primary"], "max_inflight": 2, "retry_budget": 1, "acquire_timeout_s": 0.05},
                "ingest": {"deployments": ["ingest-primary"], "max_inflight": 3, "retry_budget": 1, "acquire_timeout_s": 0.05},
            },
            "scheduler": {
                "global_max_inflight": 3,
                "reserved_by_purpose": {"chat": 2},
            },
        }
    )
    runtime = openai_utils.LLMRouterRuntime(config)

    first_ingest = runtime.acquire_sync("ingest")
    assert first_ingest.deployment_name == "ingest-primary"

    with pytest.raises(RuntimeError, match="no healthy capacity"):
        runtime.acquire_sync("ingest")

    first_chat = runtime.acquire_sync("chat")
    second_chat = runtime.acquire_sync("chat")
    assert first_chat.deployment_name == "chat-primary"
    assert second_chat.deployment_name == "chat-primary"


def test_router_runtime_adaptive_ingest_limit_grows_after_clean_successes():
    config = openai_utils.LLMRouterConfig.model_validate(
        {
            "deployments": [
                {
                    "name": "ingest-primary",
                    "model": "LongCat-Flash-Chat",
                    "weight": 1,
                    "max_inflight": 4,
                    "purposes": ["ingest"],
                },
            ],
            "pools": {
                "ingest": {"deployments": ["ingest-primary"], "max_inflight": 4, "retry_budget": 1, "acquire_timeout_s": 0.05},
            },
            "scheduler": {
                "adaptive_by_purpose": {
                    "ingest": {
                        "enabled": True,
                        "min_inflight": 1,
                        "max_inflight": 3,
                        "increase_every": 2,
                        "latency_threshold_ms": 5000,
                    }
                }
            },
        }
    )
    runtime = openai_utils.LLMRouterRuntime(config)

    first = runtime.acquire_sync("ingest")
    runtime.release_success_sync(first, 0.2)
    second = runtime.acquire_sync("ingest")
    runtime.release_success_sync(second, 0.2)

    assert runtime._pool_states["ingest"].adaptive_limit == 2

    lease_a = runtime.acquire_sync("ingest")
    lease_b = runtime.acquire_sync("ingest")
    assert lease_a.deployment_name == "ingest-primary"
    assert lease_b.deployment_name == "ingest-primary"


def test_router_runtime_adaptive_ingest_limit_shrinks_after_retryable_failure():
    config = openai_utils.LLMRouterConfig.model_validate(
        {
            "deployments": [
                {
                    "name": "ingest-primary",
                    "model": "LongCat-Flash-Chat",
                    "weight": 1,
                    "max_inflight": 4,
                    "failure_threshold": 1,
                    "cooldown_s": 10,
                    "purposes": ["ingest"],
                },
            ],
            "pools": {
                "ingest": {"deployments": ["ingest-primary"], "max_inflight": 4, "retry_budget": 1, "acquire_timeout_s": 0.05},
            },
            "scheduler": {
                "adaptive_by_purpose": {
                    "ingest": {
                        "enabled": True,
                        "min_inflight": 1,
                        "max_inflight": 3,
                        "increase_every": 1,
                        "latency_threshold_ms": 5000,
                    }
                }
            },
        }
    )
    runtime = openai_utils.LLMRouterRuntime(config)

    first = runtime.acquire_sync("ingest")
    runtime.release_success_sync(first, 0.1)
    assert runtime._pool_states["ingest"].adaptive_limit == 2

    second = runtime.acquire_sync("ingest")
    runtime.release_failure_sync(second, 0.1, RuntimeError("429 Too Many Requests"))

    assert runtime._pool_states["ingest"].adaptive_limit == 1


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
        "chat-gpt-oss-120b",
        "chat-gpt-oss-20b",
        "chat-grok-fast",
        "chat-longcat-flash",
    ]
    assert sorted(ingest_llm._delegates_by_name) == [
        "ingest-deepseek-v4-flash",
        "ingest-gpt-oss-120b",
        "ingest-longcat-flash",
        "ingest-qwen3-next-80b",
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
