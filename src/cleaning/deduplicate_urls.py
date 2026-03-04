"""URL canonicalization, deduplication, and persistent SQLite registry."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
from pathlib import Path
import sys
from typing import Dict, Optional

import pandas as pd

from src.utils.common import (
    ensure_dir,
    text_or_empty,
    utc_now_iso,
)
from src.utils.deduplication import (
    canonicalize_url,
    decode_google_news_url,
    is_google_news_wrapper_url,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def init_registry(db_path: Path) -> None:
    """Create URL registry schema if missing."""
    ensure_dir(db_path.parent)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            --sql
            CREATE TABLE IF NOT EXISTS url_registry (
                canonical_url TEXT PRIMARY KEY,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                original_url TEXT,
                source TEXT,
                content_hash TEXT
            );
            """
        )
        conn.execute(
            """
            --sql
            CREATE INDEX IF NOT EXISTS idx_url_registry_last_seen ON url_registry(last_seen);
            """
        )
        conn.commit()


def _upsert_registry_rows(df: pd.DataFrame, db_path: Path, source: str = "rss") -> None:
    if df.empty:
        return

    now = utc_now_iso()
    rows = []
    for _, row in df.iterrows():
        canonical = row.get("canonical_url")
        if not canonical:
            continue
        rows.append(
            (
                canonical,
                now,
                now,
                text_or_empty(row.get("original_url")) or None,
                source,
                text_or_empty(row.get("content_hash")) or None,
            )
        )

    if not rows:
        return

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            --sql
            INSERT INTO url_registry (
                canonical_url, first_seen, last_seen, original_url, source, content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical_url) DO UPDATE SET
                last_seen = excluded.last_seen,
                source = excluded.source,
                original_url = COALESCE(url_registry.original_url, excluded.original_url),
                content_hash = COALESCE(excluded.content_hash, url_registry.content_hash);
            """,
            rows,
        )
        conn.commit()


def _resolve_google_wrapper_urls(
    urls: pd.Series,
    timeout_seconds: int = 20,
    max_workers: int = 6,
    logger=None,
) -> tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Resolve Google News wrapper URLs to publisher URLs for a unique URL set."""
    unique_urls = [
        str(u).strip() for u in urls.dropna().astype(str).unique() if str(u).strip()
    ]
    if not unique_urls:
        return {}, {}, {}

    resolved_map: Dict[str, str] = {}
    method_map: Dict[str, str] = {}
    error_map: Dict[str, str] = {}

    def _resolve_one(raw_url: str) -> tuple[str, str, str]:
        if not is_google_news_wrapper_url(raw_url):
            return raw_url, "not_google_wrapper", ""
        resolved, method, err = decode_google_news_url(
            raw_url,
            timeout_seconds=timeout_seconds,
        )
        return (resolved or raw_url), method, err

    if max_workers <= 1:
        for idx, url in enumerate(unique_urls, start=1):
            resolved, method, err = _resolve_one(url)
            resolved_map[url] = resolved
            method_map[url] = method
            error_map[url] = err
            if logger is not None and (idx % 100 == 0 or idx == len(unique_urls)):
                logger.info(
                    "Google wrapper resolution progress: %s/%s", idx, len(unique_urls)
                )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_resolve_one, url): url for url in unique_urls}
            for idx, future in enumerate(as_completed(futures), start=1):
                source_url = futures[future]
                try:
                    resolved, method, err = future.result()
                except Exception as exc:
                    resolved, method, err = source_url, "resolver_exception", str(exc)

                resolved_map[source_url] = resolved
                method_map[source_url] = method
                error_map[source_url] = err

                if logger is not None and (idx % 100 == 0 or idx == len(unique_urls)):
                    logger.info(
                        "Google wrapper resolution progress: %s/%s",
                        idx,
                        len(unique_urls),
                    )

    return resolved_map, method_map, error_map


def deduplicate_feed_urls(
    df: pd.DataFrame,
    schema: Dict[str, Optional[str]],
    db_path: Path,
    output_path: Path,
    source: str = "rss",
    tracking_params=None,
    resolve_google_news: bool = True,
    resolver_timeout_seconds: int = 20,
    resolver_workers: int = 6,
    logger=None,
) -> pd.DataFrame:
    """Canonicalize URLs, drop duplicate URLs, update registry, and save CSV."""
    url_col = schema.get("url_col")
    if not url_col or url_col not in df.columns:
        raise ValueError("URL column not found in dataframe.")

    work = df.copy()

    work["original_url"] = work[url_col].astype(str)

    work["resolved_url"] = work["original_url"]
    work["url_resolution_method"] = "disabled"
    work["url_resolution_error"] = ""

    if resolve_google_news:
        resolved_map, method_map, error_map = _resolve_google_wrapper_urls(
            work["original_url"],
            timeout_seconds=int(resolver_timeout_seconds),
            max_workers=int(resolver_workers),
            logger=logger,
        )
        if resolved_map:
            work["resolved_url"] = work["original_url"].map(
                lambda u: resolved_map.get(str(u), str(u))
            )
            work["url_resolution_method"] = work["original_url"].map(
                lambda u: method_map.get(str(u), "not_resolved")
            )
            work["url_resolution_error"] = work["original_url"].map(
                lambda u: error_map.get(str(u), "")
            )

    work["canonical_url"] = work["resolved_url"].apply(
        lambda x: canonicalize_url(x, tracking_params=tracking_params)
    )

    title_col = schema.get("title_col")
    description_col = schema.get("description_col")
    date_col = schema.get("date_col")

    work["rss_title"] = (
        work[title_col].astype(str) if title_col and title_col in work.columns else ""
    )
    work["rss_description"] = (
        work[description_col].astype(str)
        if description_col and description_col in work.columns
        else ""
    )
    work["pub_date"] = (
        work[date_col].astype(str) if date_col and date_col in work.columns else ""
    )

    before = len(work)
    work = work[work["canonical_url"].notna() & (work["canonical_url"] != "")].copy()
    valid_count = len(work)

    init_registry(db_path)
    _upsert_registry_rows(work, db_path=db_path, source=source)

    deduped = work.drop_duplicates(subset=["canonical_url"], keep="first").copy()
    after = len(deduped)

    ensure_dir(output_path.parent)
    deduped.to_csv(output_path, index=False)

    if logger is not None:
        wrapper_rows = int(work["original_url"].apply(is_google_news_wrapper_url).sum())
        resolved_rows = int(
            work["url_resolution_method"]
            .isin({"google_decode_offline", "google_decode_batchexecute"})
            .sum()
        )
        resolver_failures = int(
            work["url_resolution_method"].eq("google_decode_failed").sum()
            + work["url_resolution_method"].eq("resolver_exception").sum()
        )
        logger.info(
            "Dedup complete: input_rows=%s valid_urls=%s unique_urls=%s dropped=%s wrappers=%s resolved=%s resolver_failures=%s output=%s",
            before,
            valid_count,
            after,
            valid_count - after,
            wrapper_rows,
            resolved_rows,
            resolver_failures,
            output_path,
        )

    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate RSS URLs and update URL registry."
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output deduped CSV path")
    parser.add_argument(
        "--db", default="data/url_registry.sqlite", help="SQLite URL registry path"
    )
    parser.add_argument("--url-col", required=True, help="URL column name")
    parser.add_argument("--title-col", default=None, help="RSS title column name")
    parser.add_argument(
        "--description-col", default=None, help="RSS description column name"
    )
    parser.add_argument("--date-col", default=None, help="RSS date column name")
    parser.add_argument(
        "--resolve-google-news",
        action="store_true",
        help="Resolve Google News wrapper URLs before canonicalization",
    )
    parser.add_argument(
        "--resolver-timeout-seconds",
        type=int,
        default=20,
        help="Timeout for Google wrapper resolution calls",
    )
    parser.add_argument(
        "--resolver-workers",
        type=int,
        default=6,
        help="Thread workers for URL resolution",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    df = pd.read_csv(in_path, low_memory=False)

    schema = {
        "url_col": args.url_col,
        "title_col": args.title_col,
        "description_col": args.description_col,
        "date_col": args.date_col,
    }

    deduplicate_feed_urls(
        df=df,
        schema=schema,
        db_path=Path(args.db),
        output_path=Path(args.output),
        resolve_google_news=bool(args.resolve_google_news),
        resolver_timeout_seconds=int(args.resolver_timeout_seconds),
        resolver_workers=int(args.resolver_workers),
    )


if __name__ == "__main__":
    main()
