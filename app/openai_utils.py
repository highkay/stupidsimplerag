import os
from functools import lru_cache
from typing import Optional

from llama_index.core.llms import LLMMetadata, MessageRole
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from openai import OpenAI as OpenAIClient


def _env(key: str) -> Optional[str]:
    value = os.getenv(key)
    if value and value.strip():
        return value.strip()
    return None


def _resolve_api_key(kind: Optional[str]) -> Optional[str]:
    if kind:
        value = _env(f"{kind}_API_KEY")
        if value:
            return value
    return _env("OPENAI_API_KEY")


def _resolve_api_base(kind: Optional[str], base_alias_env: Optional[str]) -> str:
    candidates = []
    if kind:
        candidates.append(f"{kind}_API_BASE")
    if base_alias_env:
        candidates.append(base_alias_env)
    candidates.append("OPENAI_API_BASE")
    for env_key in candidates:
        value = _env(env_key)
        if value:
            return value.rstrip("/")
    return "https://api.openai.com/v1"


def get_openai_config(
    kind: Optional[str] = None,
    *,
    base_alias_env: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Return (api_base, api_key) with minimal logic."""
    api_base = _resolve_api_base(kind, base_alias_env)
    api_key = _resolve_api_key(kind)
    return api_base, api_key


def get_openai_kwargs(kind: Optional[str] = None) -> dict:
    """Return kwargs usable for llama-index OpenAI integrations."""
    api_base, api_key = get_openai_config(kind)
    kwargs = {"api_base": api_base}
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


@lru_cache(maxsize=4)
def get_openai_client(
    kind: Optional[str] = None,
    *,
    base_alias_env: Optional[str] = None,
) -> OpenAIClient:
    """Shared OpenAI client builder with caching."""
    api_base, api_key = get_openai_config(kind, base_alias_env=base_alias_env)
    if not api_key:
        raise ValueError(f"Missing API key for {kind or 'OPENAI'} client")
    return OpenAIClient(base_url=api_base, api_key=api_key)


class OpenAICompatibleLLM(LlamaOpenAI):
    """LLM wrapper that tolerates非标准 OpenAI 模型名."""

    def __init__(self, *args, context_window: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._context_window_override = (
            context_window or int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))
        )

    def _tokenizer(self):
        try:
            return super()._tokenizer()
        except Exception:
            return None

    @property
    def metadata(self) -> LLMMetadata:
        try:
            return super().metadata
        except Exception:
            return LLMMetadata(
                context_window=self._context_window_override,
                num_output=self.max_tokens or -1,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=self.model,
                system_role=MessageRole.SYSTEM,
            )
