"""Schema detection and CSV loading helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

URL_COLUMN_PRIORITY = [
    "canonical_url",
    "url",
    "item_link",
    "item_url",
    "link",
    "article_url",
    "permalink",
    "href",
]

TITLE_COLUMN_PRIORITY = [
    "rss_title",
    "item_title",
    "title",
    "headline",
    "news_title",
]

DESCRIPTION_COLUMN_PRIORITY = [
    "rss_description",
    "item_description",
    "description",
    "summary",
    "snippet",
    "abstract",
]

DATE_COLUMN_PRIORITY = [
    "pub_date",
    "item_pub_date",
    "published_at",
    "publish_date",
    "date",
    "datetime",
    "item_date",
]


def is_probable_url(value: Any) -> bool:
    """Fast URL heuristic for schema inference."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip()
    return bool(re.match(r"^https?://", text, flags=re.IGNORECASE))


def _best_by_priority(columns: Sequence[str], priority_names: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for name in priority_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _best_column_by_url_ratio(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    scored: List[Tuple[float, str]] = []
    for col in candidates:
        if col not in df.columns:
            continue
        sample = df[col].dropna().astype(str).head(300)
        if sample.empty:
            continue
        ratio = float(sample.apply(is_probable_url).mean())
        name_bonus = 0.0
        name_lower = col.lower()
        if "url" in name_lower or "link" in name_lower or "href" in name_lower:
            name_bonus = 0.15
        if "guid" in name_lower:
            name_bonus = 0.05
        scored.append((ratio + name_bonus, col))

    if not scored:
        return None
    scored.sort(reverse=True, key=lambda x: x[0])
    winner_score, winner_col = scored[0]
    if winner_score < 0.2:
        return None
    return winner_col


def _best_textual_column(
    df: pd.DataFrame,
    priority_names: Sequence[str],
    fallback_patterns: Sequence[str],
) -> Optional[str]:
    direct = _best_by_priority(df.columns, priority_names)
    if direct:
        return direct

    pattern = re.compile("|".join(re.escape(p) for p in fallback_patterns), flags=re.IGNORECASE)
    for col in df.columns:
        if pattern.search(col):
            return col
    return None


def _best_date_column(df: pd.DataFrame) -> Optional[str]:
    direct = _best_by_priority(df.columns, DATE_COLUMN_PRIORITY)
    if direct:
        return direct

    date_like = [c for c in df.columns if re.search(r"date|time|published", c, flags=re.IGNORECASE)]
    if not date_like:
        return None

    best_col = None
    best_ratio = -1.0
    for col in date_like:
        sample = df[col].dropna().astype(str).head(300)
        if sample.empty:
            continue
        parsed_any = pd.to_datetime(sample, errors="coerce", utc=False)
        ratio_any = float(parsed_any.notna().mean())
        parsed_epoch = pd.to_datetime(sample, errors="coerce", utc=False, unit="s")
        ratio_epoch = float(parsed_epoch.notna().mean())
        ratio = max(ratio_any, ratio_epoch)
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = col
    return best_col


def detect_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Infer key RSS fields from unknown schemas."""
    columns = list(df.columns)

    direct_url = _best_by_priority(columns, URL_COLUMN_PRIORITY)
    url_candidates = list(columns)
    inferred_url = _best_column_by_url_ratio(df, [direct_url] + url_candidates if direct_url else url_candidates)

    title_col = _best_textual_column(
        df,
        TITLE_COLUMN_PRIORITY,
        fallback_patterns=["title", "headline"],
    )
    description_col = _best_textual_column(
        df,
        DESCRIPTION_COLUMN_PRIORITY,
        fallback_patterns=["description", "summary", "snippet", "abstract"],
    )
    date_col = _best_date_column(df)

    return {
        "url_col": inferred_url,
        "title_col": title_col,
        "description_col": description_col,
        "date_col": date_col,
    }


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV with fallback encodings."""
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception as exc:  # pragma: no cover - fallback behavior
            last_error = exc
    raise RuntimeError(f"Failed to read CSV at {path}: {last_error}")
