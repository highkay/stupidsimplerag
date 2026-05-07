import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, InternalServerError, RateLimitError
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from llama_index.core.llms import LLMMetadata, MessageRole
from llama_index.llms.openai import OpenAI as LlamaOpenAI


logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_PURPOSE_DEFAULTS = {
    "chat": {"max_inflight": 6, "acquire_timeout_s": 3.0, "retry_budget": 1},
    "ingest": {"max_inflight": 2, "acquire_timeout_s": 15.0, "retry_budget": 2},
}


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


def _resolve_api_base(kind: Optional[str]) -> str:
    candidates = []
    if kind:
        candidates.append(f"{kind}_API_BASE")
    candidates.append("OPENAI_API_BASE")
    for env_key in candidates:
        value = _env(env_key)
        if value:
            return value.rstrip("/")
    return "https://api.openai.com/v1"


def get_openai_config(
    kind: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Return (api_base, api_key) with minimal logic."""
    api_base = _resolve_api_base(kind)
    api_key = _resolve_api_key(kind)
    return api_base, api_key


def get_openai_kwargs(kind: Optional[str] = None) -> dict:
    """Return kwargs usable for llama-index OpenAI integrations."""
    api_base, api_key = get_openai_config(kind)
    kwargs = {"api_base": api_base}
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def _resolve_purpose_llm_model(purpose: Optional[str]) -> Optional[str]:
    candidates: List[str] = []
    if purpose:
        candidates.append(f"{purpose.upper()}_LLM_MODEL")
    candidates.append("LLM_MODEL")
    for env_key in candidates:
        value = _env(env_key)
        if value:
            return value
    return None


def _resolve_purpose_llm_api_base(purpose: Optional[str]) -> str:
    candidates: List[str] = []
    if purpose:
        candidates.append(f"{purpose.upper()}_LLM_API_BASE")
    candidates.extend(["LLM_API_BASE", "OPENAI_API_BASE"])
    for env_key in candidates:
        value = _env(env_key)
        if value:
            return value.rstrip("/")
    return "https://api.openai.com/v1"


def _resolve_purpose_llm_api_key(purpose: Optional[str]) -> Optional[str]:
    candidates: List[str] = []
    if purpose:
        candidates.append(f"{purpose.upper()}_LLM_API_KEY")
    candidates.extend(["LLM_API_KEY", "OPENAI_API_KEY"])
    for env_key in candidates:
        value = _env(env_key)
        if value:
            return value
    return None


class LLMDeploymentConfig(BaseModel):
    name: str
    model: str
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None
    weight: int = Field(default=1, ge=1)
    max_inflight: int = Field(default=2, ge=1)
    timeout_s: float = Field(default=60.0, gt=0)
    failure_threshold: int = Field(default=3, ge=1)
    cooldown_s: float = Field(default=30.0, gt=0)
    success_threshold: int = Field(default=1, ge=1)
    enabled: bool = True
    purposes: List[str] = Field(default_factory=list)

    def resolved_api_base(self) -> str:
        return (self.api_base or _resolve_api_base("LLM")).rstrip("/")

    def resolved_api_key(self) -> Optional[str]:
        if self.api_key_env:
            return _env(self.api_key_env)
        return _resolve_api_key("LLM")


class LLMPoolConfig(BaseModel):
    deployments: List[str] = Field(default_factory=list)
    max_inflight: int = Field(default=4, ge=1)
    retry_budget: int = Field(default=1, ge=0)
    acquire_timeout_s: float = Field(default=5.0, gt=0)


class LLMAdaptivePoolConfig(BaseModel):
    enabled: bool = False
    min_inflight: int = Field(default=1, ge=1)
    max_inflight: Optional[int] = Field(default=None, ge=1)
    increase_every: int = Field(default=8, ge=1)
    latency_threshold_ms: float = Field(default=30000.0, gt=0)
    decrease_step: int = Field(default=1, ge=1)


class LLMRouterSchedulerConfig(BaseModel):
    global_max_inflight: Optional[int] = Field(default=None, ge=1)
    reserved_by_purpose: Dict[str, int] = Field(default_factory=dict)
    adaptive_by_purpose: Dict[str, LLMAdaptivePoolConfig] = Field(default_factory=dict)


class LLMRouterConfig(BaseModel):
    deployments: List[LLMDeploymentConfig] = Field(default_factory=list)
    pools: Dict[str, LLMPoolConfig] = Field(default_factory=dict)
    scheduler: LLMRouterSchedulerConfig = Field(default_factory=LLMRouterSchedulerConfig)

    @model_validator(mode="after")
    def validate_config(self):
        deployment_names = {deployment.name for deployment in self.deployments}
        if not deployment_names:
            raise ValueError("llm router config requires at least one deployment")

        if not self.pools:
            inferred: Dict[str, List[str]] = {}
            for deployment in self.deployments:
                purposes = deployment.purposes or ["default"]
                for purpose in purposes:
                    inferred.setdefault(purpose, []).append(deployment.name)
            self.pools = {
                purpose: LLMPoolConfig(
                    deployments=names,
                    max_inflight=_PURPOSE_DEFAULTS.get(purpose, {}).get("max_inflight", 4),
                    retry_budget=_PURPOSE_DEFAULTS.get(purpose, {}).get("retry_budget", 1),
                    acquire_timeout_s=_PURPOSE_DEFAULTS.get(purpose, {}).get("acquire_timeout_s", 5.0),
                )
                for purpose, names in inferred.items()
            }

        for pool_name, pool in self.pools.items():
            if not pool.deployments:
                raise ValueError(f"llm router pool {pool_name!r} has no deployments")
            unknown = [name for name in pool.deployments if name not in deployment_names]
            if unknown:
                raise ValueError(
                    f"llm router pool {pool_name!r} references unknown deployments: {', '.join(sorted(unknown))}"
                )

        for purpose, reserved in self.scheduler.reserved_by_purpose.items():
            if purpose not in self.pools:
                raise ValueError(f"llm router scheduler reserved_by_purpose references unknown pool {purpose!r}")
            if reserved < 0:
                raise ValueError(f"llm router scheduler reserved_by_purpose[{purpose!r}] must be >= 0")

        for purpose, adaptive in self.scheduler.adaptive_by_purpose.items():
            if purpose not in self.pools:
                raise ValueError(f"llm router scheduler adaptive_by_purpose references unknown pool {purpose!r}")
            pool = self.pools[purpose]
            ceiling = adaptive.max_inflight or pool.max_inflight
            if adaptive.min_inflight > ceiling:
                raise ValueError(
                    f"llm router adaptive pool {purpose!r} min_inflight={adaptive.min_inflight} exceeds ceiling={ceiling}"
                )
        return self


@dataclass
class _DeploymentRuntimeState:
    inflight: int = 0
    consecutive_failures: int = 0
    circuit_state: str = "closed"
    opened_until: float = 0.0
    half_open_successes: int = 0
    ewma_latency_ms: float = 0.0
    last_error: str = ""
    total_attempts: int = 0
    total_successes: int = 0
    total_failures: int = 0
    retryable_failures: int = 0


@dataclass
class _PoolRuntimeState:
    inflight: int = 0
    adaptive_limit: Optional[int] = None
    adaptive_success_streak: int = 0


@dataclass
class _RouterLease:
    pool_name: str
    deployment_name: str
    acquired_at: float


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)):
        return True
    if isinstance(exc, APIStatusError):
        return int(getattr(exc, "status_code", 0) or 0) in _RETRYABLE_STATUS_CODES
    if isinstance(exc, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ConnectError, httpx.ReadError, asyncio.TimeoutError, TimeoutError)):
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "429",
            "503",
            "rate limit",
            "timed out",
            "timeout",
            "no active api keys available",
            "connection reset",
        )
    )


class LLMRouterRuntime:
    def __init__(self, config: LLMRouterConfig):
        self.config = config
        self._deployments = {deployment.name: deployment for deployment in config.deployments}
        self._deployment_states = {
            deployment.name: _DeploymentRuntimeState() for deployment in config.deployments
        }
        self._pool_states = {pool_name: _PoolRuntimeState() for pool_name in config.pools}
        self._lock = threading.Lock()
        self._stats_interval = max(0, int(os.getenv("LLM_ROUTER_STATS_INTERVAL", "20")))
        self._stats_events = 0
        for pool_name, adaptive in config.scheduler.adaptive_by_purpose.items():
            if adaptive.enabled:
                self._pool_states[pool_name].adaptive_limit = adaptive.min_inflight

    @property
    def deployments(self) -> Dict[str, LLMDeploymentConfig]:
        return self._deployments

    def acquire_sync(
        self,
        purpose: str,
        exclude: Optional[set[str]] = None,
    ) -> _RouterLease:
        pool_name = purpose if purpose in self.config.pools else "default"
        if pool_name not in self.config.pools:
            raise RuntimeError(f"no llm router pool configured for purpose={purpose!r}")
        pool = self.config.pools[pool_name]
        deadline = time.monotonic() + pool.acquire_timeout_s
        excluded = exclude or set()

        while True:
            with self._lock:
                lease = self._try_acquire_locked(pool_name, excluded)
                if lease is not None:
                    return lease
            if time.monotonic() >= deadline:
                raise RuntimeError(f"llm router pool {pool_name!r} has no healthy capacity")
            time.sleep(0.05)

    async def acquire(self, purpose: str, exclude: Optional[set[str]] = None) -> _RouterLease:
        pool_name = purpose if purpose in self.config.pools else "default"
        if pool_name not in self.config.pools:
            raise RuntimeError(f"no llm router pool configured for purpose={purpose!r}")
        pool = self.config.pools[pool_name]
        deadline = time.monotonic() + pool.acquire_timeout_s
        excluded = exclude or set()

        while True:
            with self._lock:
                lease = self._try_acquire_locked(pool_name, excluded)
                if lease is not None:
                    return lease
            if time.monotonic() >= deadline:
                raise RuntimeError(f"llm router pool {pool_name!r} has no healthy capacity")
            await asyncio.sleep(0.05)

    async def release_success(self, lease: _RouterLease, latency_s: float) -> None:
        self.release_success_sync(lease, latency_s)

    def release_success_sync(self, lease: _RouterLease, latency_s: float) -> None:
        with self._lock:
            self._release_locked(lease)
            state = self._deployment_states[lease.deployment_name]
            deployment = self._deployments[lease.deployment_name]
            state.total_attempts += 1
            state.total_successes += 1
            state.last_error = ""
            state.consecutive_failures = 0
            latency_ms = max(latency_s * 1000.0, 0.0)
            if state.ewma_latency_ms <= 0:
                state.ewma_latency_ms = latency_ms
            else:
                state.ewma_latency_ms = (state.ewma_latency_ms * 0.7) + (latency_ms * 0.3)

            if state.circuit_state == "half_open":
                state.half_open_successes += 1
                if state.half_open_successes >= deployment.success_threshold:
                    state.circuit_state = "closed"
                    state.half_open_successes = 0
                    state.opened_until = 0.0
            self._adjust_pool_after_success_locked(lease.pool_name, latency_ms)
            self._stats_events += 1
            self._maybe_log_pool_stats_locked(lease.pool_name)

    async def release_failure(self, lease: _RouterLease, latency_s: float, exc: Exception) -> None:
        self.release_failure_sync(lease, latency_s, exc)

    def release_failure_sync(self, lease: _RouterLease, latency_s: float, exc: Exception) -> None:
        retryable = _is_retryable_exception(exc)
        with self._lock:
            self._release_locked(lease)
            state = self._deployment_states[lease.deployment_name]
            deployment = self._deployments[lease.deployment_name]
            state.total_attempts += 1
            state.total_failures += 1
            state.last_error = str(exc)
            latency_ms = max(latency_s * 1000.0, 0.0)
            if state.ewma_latency_ms <= 0:
                state.ewma_latency_ms = latency_ms
            else:
                state.ewma_latency_ms = (state.ewma_latency_ms * 0.7) + (latency_ms * 0.3)

            if not retryable:
                self._stats_events += 1
                self._maybe_log_pool_stats_locked(lease.pool_name)
                return

            state.retryable_failures += 1
            state.consecutive_failures += 1
            state.half_open_successes = 0
            if state.circuit_state == "half_open" or state.consecutive_failures >= deployment.failure_threshold:
                state.circuit_state = "open"
                state.opened_until = time.monotonic() + deployment.cooldown_s
                logger.warning(
                    "LLM router opened circuit deployment=%s purpose=%s cooldown=%.1fs reason=%s",
                    lease.deployment_name,
                    lease.pool_name,
                    deployment.cooldown_s,
                    exc,
                )
            self._adjust_pool_after_failure_locked(lease.pool_name, retryable=retryable, exc=exc)
            self._stats_events += 1
            self._maybe_log_pool_stats_locked(lease.pool_name)

    def _release_locked(self, lease: _RouterLease) -> None:
        deployment_state = self._deployment_states[lease.deployment_name]
        pool_state = self._pool_states[lease.pool_name]
        deployment_state.inflight = max(0, deployment_state.inflight - 1)
        pool_state.inflight = max(0, pool_state.inflight - 1)

    def _try_acquire_locked(self, pool_name: str, exclude: set[str]) -> Optional[_RouterLease]:
        pool = self.config.pools[pool_name]
        pool_state = self._pool_states[pool_name]
        effective_pool_limit = self._effective_pool_limit_locked(pool_name)
        if pool_state.inflight >= effective_pool_limit:
            return None
        if not self._can_acquire_under_global_budget_locked(pool_name):
            return None

        now = time.monotonic()
        candidates: list[tuple[tuple[float, float, str], str]] = []

        for deployment_name in pool.deployments:
            if deployment_name in exclude:
                continue
            deployment = self._deployments[deployment_name]
            if not deployment.enabled:
                continue

            state = self._deployment_states[deployment_name]
            if state.circuit_state == "open":
                if now < state.opened_until:
                    continue
                state.circuit_state = "half_open"
                state.half_open_successes = 0

            if state.circuit_state == "half_open" and state.inflight >= 1:
                continue
            if state.inflight >= deployment.max_inflight:
                continue

            half_open_penalty = 1.0 if state.circuit_state == "half_open" else 0.0
            inflight_score = state.inflight / max(deployment.weight, 1)
            latency_score = state.ewma_latency_ms / 10000.0 if state.ewma_latency_ms > 0 else 0.0
            candidates.append(((half_open_penalty, inflight_score + latency_score, deployment_name), deployment_name))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        selected = candidates[0][1]
        self._deployment_states[selected].inflight += 1
        pool_state.inflight += 1
        return _RouterLease(pool_name=pool_name, deployment_name=selected, acquired_at=now)

    def _effective_pool_limit_locked(self, pool_name: str) -> int:
        pool = self.config.pools[pool_name]
        adaptive = self.config.scheduler.adaptive_by_purpose.get(pool_name)
        state = self._pool_states[pool_name]
        if adaptive and adaptive.enabled:
            ceiling = adaptive.max_inflight or pool.max_inflight
            current = state.adaptive_limit if state.adaptive_limit is not None else adaptive.min_inflight
            return max(adaptive.min_inflight, min(current, ceiling))
        return pool.max_inflight

    def _can_acquire_under_global_budget_locked(self, pool_name: str) -> bool:
        scheduler = self.config.scheduler
        if not scheduler.global_max_inflight:
            return True
        total_inflight = sum(state.inflight for state in self._pool_states.values())
        blocked_reserved = 0
        for other_pool_name, reserved in scheduler.reserved_by_purpose.items():
            if other_pool_name == pool_name:
                continue
            other_pool_inflight = self._pool_states.get(other_pool_name, _PoolRuntimeState()).inflight
            blocked_reserved += max(reserved - other_pool_inflight, 0)
        allowed_total = max(scheduler.global_max_inflight - blocked_reserved, 0)
        return total_inflight < allowed_total

    def _adjust_pool_after_success_locked(self, pool_name: str, latency_ms: float) -> None:
        adaptive = self.config.scheduler.adaptive_by_purpose.get(pool_name)
        if not adaptive or not adaptive.enabled:
            return
        pool = self.config.pools[pool_name]
        state = self._pool_states[pool_name]
        ceiling = adaptive.max_inflight or pool.max_inflight
        if state.adaptive_limit is None:
            state.adaptive_limit = adaptive.min_inflight
        if latency_ms > adaptive.latency_threshold_ms:
            self._decrease_pool_limit_locked(pool_name, reason=f"latency={latency_ms:.0f}ms")
            return
        state.adaptive_success_streak += 1
        if state.adaptive_success_streak < adaptive.increase_every:
            return
        if state.adaptive_limit >= ceiling:
            state.adaptive_success_streak = 0
            return
        state.adaptive_limit += 1
        state.adaptive_success_streak = 0
        logger.info(
            "LLM router increased adaptive pool limit pool=%s new_limit=%d ceiling=%d",
            pool_name,
            state.adaptive_limit,
            ceiling,
        )

    def _adjust_pool_after_failure_locked(self, pool_name: str, retryable: bool, exc: Exception) -> None:
        adaptive = self.config.scheduler.adaptive_by_purpose.get(pool_name)
        if not adaptive or not adaptive.enabled:
            return
        state = self._pool_states[pool_name]
        state.adaptive_success_streak = 0
        if retryable:
            self._decrease_pool_limit_locked(pool_name, reason=str(exc))

    def _decrease_pool_limit_locked(self, pool_name: str, reason: str) -> None:
        adaptive = self.config.scheduler.adaptive_by_purpose.get(pool_name)
        if not adaptive or not adaptive.enabled:
            return
        state = self._pool_states[pool_name]
        current = state.adaptive_limit if state.adaptive_limit is not None else adaptive.min_inflight
        new_limit = max(adaptive.min_inflight, current - adaptive.decrease_step)
        state.adaptive_success_streak = 0
        if new_limit == current:
            return
        state.adaptive_limit = new_limit
        logger.warning(
            "LLM router decreased adaptive pool limit pool=%s new_limit=%d reason=%s",
            pool_name,
            new_limit,
            reason,
        )

    def _maybe_log_pool_stats_locked(self, pool_name: str) -> None:
        if self._stats_interval <= 0 or self._stats_events % self._stats_interval != 0:
            return
        pool = self.config.pools[pool_name]
        deployment_summaries: list[str] = []
        for deployment_name in pool.deployments:
            state = self._deployment_states[deployment_name]
            if state.total_attempts <= 0 and state.inflight <= 0 and not state.last_error:
                continue
            parts = [
                f"req={state.total_attempts}",
                f"ok={state.total_successes}",
                f"fail={state.total_failures}",
                f"retry={state.retryable_failures}",
                f"ewma_ms={state.ewma_latency_ms:.0f}",
                f"st={state.circuit_state}",
                f"inflight={state.inflight}",
            ]
            if state.last_error:
                parts.append(f"last={state.last_error[:80]}")
            deployment_summaries.append(f"{deployment_name}{{{' '.join(parts)}}}")
        logger.info(
            "LLM router stats pool=%s total_events=%d pool_inflight=%d pool_limit=%d global_inflight=%d deployments=%s",
            pool_name,
            self._stats_events,
            self._pool_states[pool_name].inflight,
            self._effective_pool_limit_locked(pool_name),
            sum(state.inflight for state in self._pool_states.values()),
            "; ".join(deployment_summaries) if deployment_summaries else "none",
        )


class OpenAICompatibleLLM(LlamaOpenAI):
    """LLM wrapper that tolerates non-standard OpenAI model names and supports round-robin for multiple models."""

    _delegates: List["OpenAICompatibleLLM"] = PrivateAttr(default_factory=list)
    _delegate_index: int = PrivateAttr(default=0)

    def __init__(self, *args, context_window: Optional[int] = None, **kwargs):
        model = kwargs.get("model")
        delegates_to_create = []

        if model and isinstance(model, str) and "," in model:
            models = [m.strip() for m in model.split(",") if m.strip()]
            if len(models) > 1:
                kwargs["model"] = models[0]
                for m in models:
                    d_kwargs = kwargs.copy()
                    d_kwargs["model"] = m
                    delegates_to_create.append((args, d_kwargs))

        super().__init__(*args, **kwargs)

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


class GatewayRoutedLLM(LlamaOpenAI):
    _router: LLMRouterRuntime = PrivateAttr()
    _purpose: str = PrivateAttr()
    _delegates_by_name: Dict[str, OpenAICompatibleLLM] = PrivateAttr(default_factory=dict)
    _pool_name: str = PrivateAttr()
    _retry_budget: int = PrivateAttr(default=1)
    _context_window_override: int = PrivateAttr(default=8192)

    def __init__(
        self,
        *,
        router: LLMRouterRuntime,
        purpose: str,
        context_window: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        pool_name = purpose if purpose in router.config.pools else "default"
        if pool_name not in router.config.pools:
            raise ValueError(f"no llm router pool configured for purpose={purpose!r}")

        pool = router.config.pools[pool_name]
        first_deployment = router.deployments[pool.deployments[0]]
        super().__init__(
            model=first_deployment.model,
            api_key=first_deployment.resolved_api_key(),
            api_base=first_deployment.resolved_api_base(),
            timeout=first_deployment.timeout_s,
            max_retries=0,
            **kwargs,
        )
        self._router = router
        self._purpose = purpose
        self._pool_name = pool_name
        self._retry_budget = pool.retry_budget
        self._context_window_override = (
            context_window or int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))
        )

        for deployment_name in pool.deployments:
            deployment = router.deployments[deployment_name]
            self._delegates_by_name[deployment_name] = OpenAICompatibleLLM(
                model=deployment.model,
                api_key=deployment.resolved_api_key(),
                api_base=deployment.resolved_api_base(),
                timeout=deployment.timeout_s,
                max_retries=0,
                context_window=context_window,
                **kwargs,
            )

    @property
    def metadata(self) -> LLMMetadata:
        try:
            first_delegate = next(iter(self._delegates_by_name.values()))
            return first_delegate.metadata
        except Exception:
            return LLMMetadata(
                context_window=self._context_window_override,
                num_output=self.max_tokens or -1,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=self.model,
                system_role=MessageRole.SYSTEM,
            )

    async def _route_async(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        errors: list[Exception] = []
        tried: set[str] = set()
        attempts = self._retry_budget + 1
        for _ in range(attempts):
            lease = await self._router.acquire(self._purpose, exclude=tried)
            tried.add(lease.deployment_name)
            delegate = self._delegates_by_name[lease.deployment_name]
            started = time.perf_counter()
            try:
                result = await getattr(delegate, method_name)(*args, **kwargs)
                await self._router.release_success(lease, time.perf_counter() - started)
                return result
            except Exception as exc:
                await self._router.release_failure(lease, time.perf_counter() - started, exc)
                errors.append(exc)
                if not _is_retryable_exception(exc):
                    raise
        raise errors[-1] if errors else RuntimeError("llm router request failed")

    def _route_sync(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        errors: list[Exception] = []
        tried: set[str] = set()
        attempts = self._retry_budget + 1
        for _ in range(attempts):
            lease = self._router.acquire_sync(self._purpose, exclude=tried)
            tried.add(lease.deployment_name)
            delegate = self._delegates_by_name[lease.deployment_name]
            started = time.perf_counter()
            try:
                result = getattr(delegate, method_name)(*args, **kwargs)
                self._router.release_success_sync(lease, time.perf_counter() - started)
                return result
            except Exception as exc:
                self._router.release_failure_sync(lease, time.perf_counter() - started, exc)
                errors.append(exc)
                if not _is_retryable_exception(exc):
                    raise
        raise errors[-1] if errors else RuntimeError("llm router request failed")

    async def achat(self, *args, **kwargs):
        return await self._route_async("achat", *args, **kwargs)

    async def astream_chat(self, *args, **kwargs):
        return await self._route_async("astream_chat", *args, **kwargs)

    async def acomplete(self, *args, **kwargs):
        return await self._route_async("acomplete", *args, **kwargs)

    async def astream_complete(self, *args, **kwargs):
        return await self._route_async("astream_complete", *args, **kwargs)

    def chat(self, *args, **kwargs):
        return self._route_sync("chat", *args, **kwargs)

    def stream_chat(self, *args, **kwargs):
        return self._route_sync("stream_chat", *args, **kwargs)

    def complete(self, *args, **kwargs):
        return self._route_sync("complete", *args, **kwargs)

    def stream_complete(self, *args, **kwargs):
        return self._route_sync("stream_complete", *args, **kwargs)


def _load_router_payload() -> Optional[dict]:
    inline = _env("LLM_ROUTER_CONFIG")
    if inline:
        return json.loads(inline)

    file_path = _env("LLM_ROUTER_CONFIG_FILE")
    if not file_path:
        return None
    payload = Path(file_path).read_text(encoding="utf-8")
    return json.loads(payload)


_ROUTER_RUNTIME: Optional[LLMRouterRuntime] = None


def get_llm_router_runtime() -> Optional[LLMRouterRuntime]:
    global _ROUTER_RUNTIME
    if _ROUTER_RUNTIME is not None:
        return _ROUTER_RUNTIME
    payload = _load_router_payload()
    if payload is None:
        return None
    config = LLMRouterConfig.model_validate(payload)
    _ROUTER_RUNTIME = LLMRouterRuntime(config)
    logger.info(
        "Initialized LLM router pools=%s deployments=%s",
        ",".join(sorted(config.pools)),
        ",".join(deployment.name for deployment in config.deployments),
    )
    return _ROUTER_RUNTIME


def build_llm(
    *,
    purpose: str,
    context_window: Optional[int] = None,
    **kwargs: Any,
) -> LlamaOpenAI:
    router = get_llm_router_runtime()
    if router is not None and (purpose in router.config.pools or "default" in router.config.pools):
        logger.info("Building routed LLM purpose=%s pool=%s", purpose, purpose if purpose in router.config.pools else "default")
        return GatewayRoutedLLM(
            router=router,
            purpose=purpose,
            context_window=context_window,
            **kwargs,
        )

    model = _resolve_purpose_llm_model(purpose)
    api_base = _resolve_purpose_llm_api_base(purpose)
    api_key = _resolve_purpose_llm_api_key(purpose)
    logger.info(
        "Building legacy LLM purpose=%s model=%s api_base=%s",
        purpose,
        model,
        api_base,
    )
    return OpenAICompatibleLLM(
        model=model,
        api_key=api_key,
        api_base=api_base,
        context_window=context_window,
        **kwargs,
    )
