import os
from functools import lru_cache
from typing import Optional

from openai import OpenAI


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
) -> OpenAI:
    """Shared OpenAI client builder with caching."""
    api_base, api_key = get_openai_config(kind, base_alias_env=base_alias_env)
    if not api_key:
        raise ValueError(f"Missing API key for {kind or 'OPENAI'} client")
    return OpenAI(base_url=api_base, api_key=api_key)
