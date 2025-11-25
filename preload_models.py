"""
Utility script to pre-download FastEmbed models into a deterministic cache.

This is used during Docker builds so the final image can run completely
offline. You can also invoke it locally (`python preload_models.py`) to
warm the cache before starting the API.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastembed import SparseTextEmbedding


FASTEMBED_CACHE_PATH = os.getenv("FASTEMBED_CACHE_PATH", "./model_cache")
FASTEMBED_SPARSE_MODEL = os.getenv(
    "FASTEMBED_SPARSE_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions"
)


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _download_sparse() -> None:
    if not FASTEMBED_SPARSE_MODEL:
        print("FASTEMBED_SPARSE_MODEL empty, skip sparse download")
        return
    print(f"Downloading sparse model: {FASTEMBED_SPARSE_MODEL}")
    SparseTextEmbedding(
        model_name=FASTEMBED_SPARSE_MODEL,
        cache_dir=FASTEMBED_CACHE_PATH,
    )


def main() -> None:
    print(f"Using cache dir: {FASTEMBED_CACHE_PATH}")
    _ensure_dir(FASTEMBED_CACHE_PATH)
    _download_sparse()
    print("FastEmbed models cached successfully.")


if __name__ == "__main__":
    main()
