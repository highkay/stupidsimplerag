import os
from functools import lru_cache
from typing import Optional, List

from pydantic import PrivateAttr
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
    """LLM wrapper that tolerates非标准 OpenAI 模型名 and supports round-robin for multiple models."""
    
    _delegates: List["OpenAICompatibleLLM"] = PrivateAttr(default_factory=list)
    _delegate_index: int = PrivateAttr(default=0)

    def __init__(self, *args, context_window: Optional[int] = None, **kwargs):
        # Extract model from kwargs to check for multiple models
        model = kwargs.get("model")
        delegates_to_create = []
        
        if model and isinstance(model, str) and "," in model:
            models = [m.strip() for m in model.split(",") if m.strip()]
            if len(models) > 1:
                # Initialize this instance with the first model to satisfy parent requirements
                kwargs["model"] = models[0]
                # Prepare delegates creation AFTER super init to avoid issues? 
                # Or just prepare data.
                # We can't create delegates before super init if delegates depend on this class 
                # but we are just calling the class constructor recursively.
                
                # Create delegates for each model
                for m in models:
                    d_kwargs = kwargs.copy()
                    d_kwargs["model"] = m
                    delegates_to_create.append((args, d_kwargs))
        
        super().__init__(*args, **kwargs)
        
        # Now create delegates if any
        if delegates_to_create:
            for d_args, d_kwargs in delegates_to_create:
                delegate = OpenAICompatibleLLM(*d_args, context_window=context_window, **d_kwargs)
                self._delegates.append(delegate)

        self._context_window_override = (
            context_window or int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))
        )

    def _get_next_delegate(self) -> "OpenAICompatibleLLM":
        if not self._delegates:
            return self
        idx = self._delegate_index
        self._delegate_index = (idx + 1) % len(self._delegates)
        return self._delegates[idx]

    def chat(self, *args, **kwargs):
        if self._delegates:
            return self._get_next_delegate().chat(*args, **kwargs)
        return super().chat(*args, **kwargs)

    async def achat(self, *args, **kwargs):
        if self._delegates:
            return await self._get_next_delegate().achat(*args, **kwargs)
        return await super().achat(*args, **kwargs)

    def stream_chat(self, *args, **kwargs):
        if self._delegates:
            return self._get_next_delegate().stream_chat(*args, **kwargs)
        return super().stream_chat(*args, **kwargs)

    async def astream_chat(self, *args, **kwargs):
        if self._delegates:
            return await self._get_next_delegate().astream_chat(*args, **kwargs)
        return await super().astream_chat(*args, **kwargs)

    def complete(self, *args, **kwargs):
        if self._delegates:
            return self._get_next_delegate().complete(*args, **kwargs)
        return super().complete(*args, **kwargs)

    async def acomplete(self, *args, **kwargs):
        if self._delegates:
            return await self._get_next_delegate().acomplete(*args, **kwargs)
        return await super().acomplete(*args, **kwargs)

    def stream_complete(self, *args, **kwargs):
        if self._delegates:
            return self._get_next_delegate().stream_complete(*args, **kwargs)
        return super().stream_complete(*args, **kwargs)

    async def astream_complete(self, *args, **kwargs):
        if self._delegates:
            return await self._get_next_delegate().astream_complete(*args, **kwargs)
        return await super().astream_complete(*args, **kwargs)

    def _tokenizer(self):
        try:
            return super()._tokenizer()
        except Exception:
            return None

    @property
    def metadata(self) -> LLMMetadata:
        # If router, maybe return metadata of the next delegate or the first one?
        # Usually metadata is static. Returning self (initialized with first model) is fine.
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
