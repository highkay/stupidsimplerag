import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


def normalize_embedding_prefix(prefix: str) -> str:
    stripped = (prefix or "").strip()
    if not stripped:
        return ""
    return stripped if stripped.endswith(" ") else f"{stripped} "


def _tokenize_endpoint(api_base: Optional[str]) -> Optional[str]:
    base = (api_base or "").rstrip("/")
    if not base:
        return None
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}/tokenize"


class EmbeddingTokenCounter:
    def __init__(
        self,
        *,
        api_base: Optional[str],
        api_key: Optional[str],
        timeout: float,
        text_prefix: str,
    ) -> None:
        self._tokenize_url = _tokenize_endpoint(api_base)
        self._api_key = api_key
        self._text_prefix = normalize_embedding_prefix(text_prefix)
        self._client = httpx.Client(timeout=timeout)
        self._supported: Optional[bool] = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def count(self, text: str) -> Optional[int]:
        if not self._tokenize_url or self._supported is False:
            return None
        content = f"{self._text_prefix}{text}" if self._text_prefix else text
        try:
            response = self._client.post(
                self._tokenize_url,
                json={"content": content},
                headers=self._headers(),
            )
            response.raise_for_status()
            payload = response.json()
            tokens = payload.get("tokens")
            if isinstance(tokens, list):
                self._supported = True
                return len(tokens)
            ids = payload.get("ids")
            if isinstance(ids, list):
                self._supported = True
                return len(ids)
            raise ValueError("tokenize response missing tokens/ids list")
        except ValueError as exc:
            self._supported = False
            logger.debug(
                "Embedding tokenize probe unavailable url=%s error=%s",
                self._tokenize_url,
                exc,
            )
            return None
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            # Treat capability errors as durable; allow transient server/client
            # issues to retry on the next chunk instead of disabling budget checks.
            self._supported = False if status_code in {404, 405, 501} else None
            logger.debug(
                "Embedding tokenize probe unavailable url=%s status=%s error=%s",
                self._tokenize_url,
                status_code,
                exc,
            )
            return None
        except httpx.HTTPError as exc:
            self._supported = None
            logger.debug(
                "Embedding tokenize probe unavailable url=%s error=%s",
                self._tokenize_url,
                exc,
            )
            return None


def _build_counter() -> EmbeddingTokenCounter:
    return EmbeddingTokenCounter(
        api_base=os.getenv("EMBEDDING_API_BASE"),
        api_key=os.getenv("EMBEDDING_API_KEY"),
        timeout=float(
            os.getenv(
                "EMBEDDING_TOKEN_COUNT_TIMEOUT",
                os.getenv("EMBEDDING_TIMEOUT", "60"),
            )
        ),
        text_prefix=os.getenv("EMBEDDING_DOCUMENT_PREFIX", ""),
    )


_TOKEN_COUNTER = _build_counter()


def count_embedding_input_tokens(text: str) -> Optional[int]:
    return _TOKEN_COUNTER.count(text)


def get_embedding_input_token_budget() -> int:
    explicit = os.getenv("EMBEDDING_MAX_INPUT_TOKENS")
    if explicit:
        return max(256, int(explicit))

    ctx = max(1024, int(os.getenv("EMBEDDING_CONTEXT_WINDOW", "8192")))
    parallel = max(
        1,
        int(
            os.getenv(
                "EMBEDDING_SERVER_PARALLEL",
                os.getenv("EMBEDDING_CONCURRENCY", "2"),
            )
        ),
    )
    budget = int((ctx / parallel) * 0.85) if parallel > 1 else int(ctx * 0.5)
    return max(1024, min(ctx - 256, budget))
