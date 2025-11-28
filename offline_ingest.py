import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple

import httpx


logger = logging.getLogger("offline_ingest")


def load_progress(progress_file: Path) -> Set[str]:
    """从进度文件加载已处理的文件列表"""
    if not progress_file.exists():
        return set()
    try:
        with progress_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("completed", []))
    except Exception as e:
        logger.warning("无法加载进度文件 %s: %s", progress_file, e)
        return set()


def save_progress(progress_file: Path, completed: Set[str]) -> None:
    """保存已处理文件列表到进度文件"""
    try:
        with progress_file.open("w", encoding="utf-8") as f:
            json.dump({"completed": sorted(completed)}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("无法保存进度文件 %s: %s", progress_file, e)


def find_documents(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    
    # 如果是文件，直接返回该文件
    if root.is_file():
        if root.suffix.lower() not in {".md", ".txt"}:
            raise ValueError(f"File must be .md or .txt, got: {root.suffix}")
        return [root]
    
    # 如果是目录，递归查找
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


def ingest_single(
    client: httpx.Client,
    url: str,
    path: Path,
    timeout: Optional[float],
) -> Tuple[bool, str]:
    files = {
        "file": (
            path.name,
            path.read_bytes(),
            _mime_type(path),
            _file_headers(path),
        )
    }
    resp = client.post(url, files=files, timeout=timeout)
    if resp.status_code == 200:
        return True, resp.text
    return False, resp.text


def ingest_batch(
    client: httpx.Client,
    url: str,
    paths: List[Path],
    timeout: Optional[float],
    max_retries: int = 3,
) -> Tuple[bool, str]:
    """批量上传文件，支持自动重试"""
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
    
    # 重试逻辑
    last_error = ""
    for attempt in range(max_retries):
        try:
            resp = client.post(url, files=files, timeout=timeout)
            if resp.status_code == 200:
                return True, resp.text
            last_error = resp.text
            # 非超时错误不重试
            if resp.status_code != 408:  # 408 Request Timeout
                return False, last_error
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_error = f"超时错误: {e}"
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避: 1s, 2s, 4s
                logger.warning("批次上传超时 (尝试 %d/%d)，等待 %ds 后重试...",
                             attempt + 1, max_retries, wait_time)
                time.sleep(wait_time)
                continue
        except Exception as e:
            last_error = f"未知错误: {e}"
            logger.error("批次上传出错: %s", e)
            break
    
    return False, last_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline ingest utility for stupidsimplerag."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("./data"),
        help="Directory containing .md/.txt files (recursive), or a single .md/.txt file.",
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
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds (set <=0 for no timeout).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files that would be ingested.",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="进度文件路径，用于记录已处理文件（默认: <dir>/.ingest_progress.json）",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="跳过已经成功处理的文件（需配合进度文件使用，会自动启用进度记录）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="单个批次的最大重试次数（默认: 3）",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    files = find_documents(args.dir)
    if not files:
        logger.warning("No .md or .txt files found under %s", args.dir)
        return

    logger.info("Discovered %d files under %s", len(files), args.dir)
    
    # 确定进度文件路径
    if args.progress_file:
        progress_file = args.progress_file
    else:
        # 默认在目标目录下创建进度文件
        if args.dir.is_file():
            progress_file = args.dir.parent / ".ingest_progress.json"
        else:
            progress_file = args.dir / ".ingest_progress.json"
    
    # 加载已完成的文件
    completed_files: Set[str] = set()
    if args.skip_completed:
        completed_files = load_progress(progress_file)
        if completed_files:
            logger.info("加载进度文件: %d 个文件已处理", len(completed_files))
    
    # 过滤已完成的文件
    if completed_files:
        original_count = len(files)
        files = [f for f in files if str(f.absolute()) not in completed_files]
        skipped = original_count - len(files)
        if skipped > 0:
            logger.info("跳过 %d 个已处理文件，剩余 %d 个文件待处理", skipped, len(files))
        if not files:
            logger.info("所有文件已处理完成")
            return
    
    if args.dry_run:
        for path in files:
            print(path)
        return

    api_base = args.api_base.rstrip("/")
    ingest_url = f"{api_base}/ingest"
    ingest_batch_url = f"{api_base}/ingest/batch"
    total = len(files)
    success = 0
    timeout: Optional[float] = None if args.timeout <= 0 else args.timeout

    with httpx.Client() as client:
        if args.batch_size <= 1:
            # 单文件模式
            for path in files:
                ok, message = ingest_single(client, ingest_url, path, timeout)
                if ok:
                    success += 1
                    logger.info("Ingested %s", path)
                    # 记录成功
                    completed_files.add(str(path.absolute()))
                    if args.skip_completed:
                        save_progress(progress_file, completed_files)
                else:
                    logger.error("Failed ingest %s: %s", path, message)
        else:
            # 批量模式
            for chunk in chunked(files, args.batch_size):
                ok, message = ingest_batch(
                    client, ingest_batch_url, chunk, timeout, args.max_retries
                )
                if ok:
                    success += len(chunk)
                    logger.info("Ingested batch (%d files)", len(chunk))
                    # 记录成功的批次
                    for path in chunk:
                        completed_files.add(str(path.absolute()))
                    if args.skip_completed:
                        save_progress(progress_file, completed_files)
                else:
                    names = ", ".join(p.name for p in chunk)
                    logger.error("Failed batch [%s]: %s", names, message)

    logger.info("Finished ingest: %d/%d files succeeded", success, total)
    if args.skip_completed:
        logger.info("进度已保存至: %s", progress_file)


if __name__ == "__main__":
    main()
