"""Shared common helpers used across the pipeline."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if missing."""
    path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    """Current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(value: str) -> str:
    """Stable SHA-256 hash of a string."""
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def normalize_whitespace(value: str) -> str:
    """Collapse repeated whitespace and trim."""
    return re.sub(r"\s+", " ", value).strip()


def save_dataframe_with_parquet_fallback(df: pd.DataFrame, target_parquet: Path) -> Path:
    """Try saving parquet; fallback to CSV when parquet dependencies are missing."""
    try:
        df.to_parquet(target_parquet, index=False)
        return target_parquet
    except Exception:
        csv_path = target_parquet.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def setup_logger(log_file: Path) -> logging.Logger:
    """Set up pipeline logger with file + console handlers."""
    ensure_dir(log_file.parent)
    logger = logging.getLogger("classification_pipeline")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def text_or_empty(value: Any) -> str:
    """Return empty string for null-like values, otherwise string form."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)
