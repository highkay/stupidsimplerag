import argparse
import logging
import os
from urllib.parse import urlparse, urlunparse

from dotenv import find_dotenv, load_dotenv
from llama_index.vector_stores.qdrant.base import DEFAULT_SPARSE_VECTOR_NAME
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("reset_qdrant")


def _append_port_if_missing(raw_url: str, default_port: int) -> str:
    parsed = urlparse(raw_url)
    if not parsed.netloc:
        return raw_url
    host_port = parsed.netloc.split("@")[-1]
    if ":" in host_port:
        return raw_url
    new_netloc = parsed.netloc + f":{default_port}"
    return urlunparse(parsed._replace(netloc=new_netloc))


def _build_client() -> QdrantClient:
    host = os.getenv("QDRANT_HOST", "qdrant")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    api_key = os.getenv("QDRANT_API_KEY") or None
    use_https = os.getenv("QDRANT_HTTPS", "false").lower() == "true"
    timeout = float(os.getenv("QDRANT_CLIENT_TIMEOUT", "10"))
    managed_url = os.getenv("QDRANT_URL")

    if managed_url:
        managed_url = _append_port_if_missing(managed_url, port)
        logger.info("Connecting to managed Qdrant url=%s", managed_url)
        return QdrantClient(url=managed_url, api_key=api_key, timeout=timeout)

    resolved_host = host
    if host == "qdrant" and not os.path.exists("/.dockerenv"):
        resolved_host = "127.0.0.1"
    logger.info(
        "Connecting to Qdrant host=%s port=%s https=%s", resolved_host, port, use_https
    )
    return QdrantClient(
        host=resolved_host,
        port=port,
        https=use_https,
        api_key=api_key if use_https else None,
        timeout=timeout,
    )


def _reset_collection(client: QdrantClient, collection: str, dim: int) -> None:
    if client.collection_exists(collection):
        logger.info("Deleting existing collection=%s", collection)
        client.delete_collection(collection_name=collection)
    logger.info("Creating collection=%s dim=%d", collection, dim)
    client.create_collection(
        collection_name=collection,
        vectors_config={
            "text-dense": qmodels.VectorParams(
                size=dim, distance=qmodels.Distance.COSINE
            )
        },
        sparse_vectors_config={
            DEFAULT_SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
                index=qmodels.SparseIndexParams()
            )
        },
    )
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name="date_numeric",
            field_schema=qmodels.PayloadSchemaType.INTEGER,
        )
    except Exception:
        pass
    logger.info("Collection reset complete: %s", collection)


def main() -> None:
    load_dotenv(find_dotenv(), override=False)

    parser = argparse.ArgumentParser(
        description="Dangerous: drops and recreates the Qdrant collection defined in .env"
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--collection",
        help="Override collection name (defaults to env COLLECTION_NAME or financial_reports)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        help="Override embedding dimension (defaults to env EMBEDDING_DIM or 1536)",
    )
    args = parser.parse_args()

    collection = args.collection or os.getenv("COLLECTION_NAME", "financial_reports")
    dim = args.dim or int(os.getenv("EMBEDDING_DIM", "1536"))

    if not args.yes:
        prompt = (
            f"This will DROP and recreate collection '{collection}' "
            f"on the Qdrant instance from .env. Proceed? [y/N]: "
        )
        answer = input(prompt).strip().lower()
        if answer not in ("y", "yes"):
            logger.info("Aborted by user.")
            return

    client = _build_client()
    _reset_collection(client, collection, dim)


if __name__ == "__main__":
    main()
