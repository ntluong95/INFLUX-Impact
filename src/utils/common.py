from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_whitespace(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def sha256_text(value: str) -> str:
    return stable_hash(value or "")


def slugify_text(value: str, max_length: int = 72) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower())
    cleaned = cleaned.strip("_") or "item"
    if len(cleaned) <= max_length:
        return cleaned
    digest = stable_hash(value)[:10]
    return f"{cleaned[: max_length - 11]}_{digest}"


def extract_domain(value: str | None) -> str:
    text = normalize_whitespace(value)
    if not text:
        return ""
    if "://" not in text and "." in text and "/" not in text:
        host = text
    else:
        host = urlsplit(text).hostname or ""
    host = host.lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def domain_matches_blocklist(domain: str, blocked_domains: set[str]) -> bool:
    if not domain:
        return False
    normalized = extract_domain(domain)
    if not normalized:
        return False
    if normalized in blocked_domains:
        return True
    return any(normalized.endswith(f".{blocked}") for blocked in blocked_domains)


def text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def chunk_iterable(values: Iterable[Any], size: int) -> Iterable[list[Any]]:
    size = max(1, int(size))
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def estimate_prompt_tokens(text: str) -> int:
    compact = text_or_empty(text)
    if not compact:
        return 0
    return max(1, int(len(compact) / 4))


def atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{path.stem}_",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def write_dataframe_atomic(
    df: pd.DataFrame,
    csv_path: Path,
    parquet_path: Path | None = None,
) -> None:
    ensure_dir(csv_path.parent)
    fd, csv_temp = tempfile.mkstemp(
        prefix=f"{csv_path.stem}_",
        suffix=".csv.tmp",
        dir=csv_path.parent,
    )
    os.close(fd)
    try:
        df.to_csv(csv_temp, index=False)
        os.replace(csv_temp, csv_path)
    finally:
        if os.path.exists(csv_temp):
            os.unlink(csv_temp)

    if parquet_path is None:
        return

    ensure_dir(parquet_path.parent)
    fd, parquet_temp = tempfile.mkstemp(
        prefix=f"{parquet_path.stem}_",
        suffix=".parquet.tmp",
        dir=parquet_path.parent,
    )
    os.close(fd)
    try:
        df.to_parquet(parquet_temp, index=False)
        os.replace(parquet_temp, parquet_path)
    finally:
        if os.path.exists(parquet_temp):
            os.unlink(parquet_temp)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def setup_logger(name: str, log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
