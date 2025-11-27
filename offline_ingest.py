import argparse
import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import httpx


logger = logging.getLogger("offline_ingest")


def find_documents(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        candidates.append(path)
    return sorted(candidates)


def chunked(seq: Sequence[Path], size: int) -> Iterable[List[Path]]:
    if size <= 0:
        size = 1
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def _mime_type(path: Path) -> str:
    return "text/markdown" if path.suffix.lower() == ".md" else "text/plain"


def _file_headers(path: Path) -> dict:
    try:
        mtime = path.stat().st_mtime
        return {"X-File-Mtime": str(mtime)}
    except OSError:
        return {}


def ingest_single(client: httpx.Client, url: str, path: Path) -> Tuple[bool, str]:
    files = {
        "file": (
            path.name,
            path.read_bytes(),
            _mime_type(path),
            _file_headers(path),
        )
    }
    resp = client.post(url, files=files, timeout=30.0)
    if resp.status_code == 200:
        return True, resp.text
    return False, resp.text


def ingest_batch(
    client: httpx.Client,
    url: str,
    paths: List[Path],
) -> Tuple[bool, str]:
    files: List[Tuple[str, Tuple[Any, ...]]] = []
    for path in paths:
        files.append(
            (
                "files",
                (
                    path.name,
                    path.read_bytes(),
                    _mime_type(path),
                    _file_headers(path),
                ),
            )
        )
    resp = client.post(url, files=files, timeout=60.0)
    if resp.status_code == 200:
        return True, resp.text
    return False, resp.text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline ingest utility for stupidsimplerag."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("./data"),
        help="Directory containing .md/.txt files (recursive).",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("RAG_API_BASE", "http://localhost:8000"),
        help="Base URL of the RAG server, e.g. http://localhost:8000",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Send multiple files per request using /ingest/batch when >1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files that would be ingested.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    files = find_documents(args.dir)
    if not files:
        logger.warning("No .md or .txt files found under %s", args.dir)
        return

    logger.info("Discovered %d files under %s", len(files), args.dir)
    if args.dry_run:
        for path in files:
            print(path)
        return

    api_base = args.api_base.rstrip("/")
    ingest_url = f"{api_base}/ingest"
    ingest_batch_url = f"{api_base}/ingest/batch"
    total = len(files)
    success = 0

    with httpx.Client() as client:
        if args.batch_size <= 1:
            for path in files:
                ok, message = ingest_single(client, ingest_url, path)
                if ok:
                    success += 1
                    logger.info("Ingested %s", path)
                else:
                    logger.error("Failed ingest %s: %s", path, message)
        else:
            for chunk in chunked(files, args.batch_size):
                ok, message = ingest_batch(client, ingest_batch_url, chunk)
                if ok:
                    success += len(chunk)
                    logger.info("Ingested batch (%d files)", len(chunk))
                else:
                    names = ", ".join(p.name for p in chunk)
                    logger.error("Failed batch [%s]: %s", names, message)

    logger.info("Finished ingest: %d/%d files succeeded", success, total)


if __name__ == "__main__":
    main()
