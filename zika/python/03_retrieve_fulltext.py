from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from scrape_utils import (
    canonicalize_url,
    extract_full_text,
    extract_metadata,
    load_config,
    load_existing_table,
    polite_fetch,
    resolve_google_news_url,
    setup_logger,
    sha256_text,
    utc_now_iso,
    write_dataframe,
)


SUCCESS_COLUMNS = [
    "record_id",
    "google_news_redirect_url",
    "rss_title",
    "source_hint",
    "final_action",
    "final_url",
    "canonical_url",
    "domain",
    "page_title",
    "published_at",
    "authors",
    "full_text",
    "word_count",
    "language",
    "extraction_method",
    "fetch_status",
    "error_reason",
    "resolution_method",
    "resolution_error",
    "content_hash",
    "fetched_at",
]

FAILED_COLUMNS = [
    "record_id",
    "google_news_redirect_url",
    "rss_title",
    "final_url",
    "fetch_status",
    "error_reason",
    "resolution_method",
    "resolution_error",
    "fetched_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve full text for kept Zika URLs.")
    parser.add_argument("--config", default="zika/config/zika.yaml")
    return parser.parse_args()


def build_success_record(row: pd.Series, metadata: dict[str, Any], final_url: str, full_text: str, extraction_method: str, resolution_method: str, resolution_error: str) -> dict[str, Any]:
    canonical_url = canonicalize_url(metadata.get("canonical_url")) or canonicalize_url(final_url) or final_url
    domain = metadata.get("domain") or (urlsplit(final_url).hostname or "").lower()
    authors = metadata.get("authors") or []
    word_count = len(full_text.split())
    return {
        "record_id": row["record_id"],
        "google_news_redirect_url": row["google_news_redirect_url"],
        "rss_title": row.get("rss_title", ""),
        "source_hint": row.get("source_hint", ""),
        "final_action": row.get("final_action", ""),
        "final_url": final_url,
        "canonical_url": canonical_url,
        "domain": domain,
        "page_title": metadata.get("page_title", ""),
        "published_at": metadata.get("published_at", ""),
        "authors": " | ".join(authors),
        "full_text": full_text,
        "word_count": word_count,
        "language": metadata.get("language", ""),
        "extraction_method": extraction_method,
        "fetch_status": "success",
        "error_reason": "",
        "resolution_method": resolution_method,
        "resolution_error": resolution_error,
        "content_hash": sha256_text(full_text),
        "fetched_at": utc_now_iso(),
    }


def build_failure_record(row: pd.Series, final_url: str, fetch_status: str, error_reason: str, resolution_method: str, resolution_error: str) -> dict[str, Any]:
    return {
        "record_id": row["record_id"],
        "google_news_redirect_url": row["google_news_redirect_url"],
        "rss_title": row.get("rss_title", ""),
        "final_url": final_url,
        "fetch_status": fetch_status,
        "error_reason": error_reason,
        "resolution_method": resolution_method,
        "resolution_error": resolution_error,
        "fetched_at": utc_now_iso(),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()
    zika_root = config_path.parent.parent
    cfg = load_config(config_path)
    fulltext_cfg = cfg.get("fulltext", {})

    log_path = zika_root / "logs" / "03_retrieve_fulltext.log"
    logger = setup_logger(log_path)

    input_csv = zika_root / "data" / "intermediate" / "zika_headlines_scored.csv"
    output_csv = zika_root / "data" / "final" / "zika_fulltext.csv"
    output_parquet = zika_root / "data" / "final" / "zika_fulltext.parquet"
    failed_csv = zika_root / "data" / "final" / "zika_fulltext_failed.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing stage-2 output: {input_csv}")

    scored_df = pd.read_csv(input_csv)
    keep_df = scored_df[scored_df["final_action"] == fulltext_cfg.get("input_action", "keep")].copy()
    logger.info("Stage 3 starting with %s kept URLs", len(keep_df))

    existing_success = load_existing_table(output_csv)
    existing_failed = load_existing_table(failed_csv)
    done_ids = set(existing_success.get("record_id", pd.Series(dtype=str)).astype(str)) | set(existing_failed.get("record_id", pd.Series(dtype=str)).astype(str))

    success_rows = existing_success.to_dict(orient="records") if not existing_success.empty else []
    failed_rows = existing_failed.to_dict(orient="records") if not existing_failed.empty else []

    seen_url_keys = set()
    seen_content_hashes = set()
    if success_rows:
        for record in success_rows:
            if record.get("canonical_url"):
                seen_url_keys.add(str(record["canonical_url"]))
            if record.get("final_url"):
                seen_url_keys.add(str(record["final_url"]))
            if record.get("content_hash"):
                seen_content_hashes.add(str(record["content_hash"]))

    if keep_df.empty:
        write_dataframe(pd.DataFrame(success_rows, columns=SUCCESS_COLUMNS), output_csv, output_parquet)
        write_dataframe(pd.DataFrame(failed_rows, columns=FAILED_COLUMNS), failed_csv, None)
        logger.info("No kept URLs found. Wrote empty outputs.")
        return

    session = requests.Session()
    checkpoint_every = max(1, int(fulltext_cfg.get("checkpoint_every", 10)))
    processed = 0

    for _, row in keep_df.iterrows():
        record_id = str(row["record_id"])
        if record_id in done_ids:
            continue

        wrapper_url = str(row["google_news_redirect_url"])
        resolved_url, resolution_method, resolution_error = resolve_google_news_url(
            wrapper_url=wrapper_url,
            timeout_seconds=int(fulltext_cfg.get("timeout_seconds", 30)),
            session=session,
        )

        try:
            response = polite_fetch(session, resolved_url, fulltext_cfg, logger)
        except Exception as exc:
            failed_rows.append(
                build_failure_record(
                    row=row,
                    final_url=resolved_url,
                    fetch_status="request_failed",
                    error_reason=str(exc),
                    resolution_method=resolution_method,
                    resolution_error=resolution_error,
                )
            )
            done_ids.add(record_id)
            processed += 1
            logger.warning("Fetch failed for %s: %s", wrapper_url, exc)
            continue

        final_url = str(response.url)
        content_type = response.headers.get("Content-Type", "")
        if response.status_code >= 400:
            failed_rows.append(
                build_failure_record(row, final_url, f"http_{response.status_code}", f"HTTP {response.status_code}", resolution_method, resolution_error)
            )
            done_ids.add(record_id)
            processed += 1
            continue
        if "html" not in content_type.lower() and "<html" not in (response.text or "")[:2000].lower():
            failed_rows.append(
                build_failure_record(row, final_url, "non_html", content_type, resolution_method, resolution_error)
            )
            done_ids.add(record_id)
            processed += 1
            continue

        metadata = extract_metadata(response.text or "", final_url, response=response)
        full_text, extraction_method = extract_full_text(response.text or "", final_url)
        if not full_text:
            failed_rows.append(
                build_failure_record(row, final_url, "empty_text", "no_extraction_output", resolution_method, resolution_error)
            )
            done_ids.add(record_id)
            processed += 1
            continue

        word_count = len(full_text.split())
        if word_count < int(fulltext_cfg.get("min_text_words", 120)):
            failed_rows.append(
                build_failure_record(row, final_url, "too_short", f"word_count={word_count}", resolution_method, resolution_error)
            )
            done_ids.add(record_id)
            processed += 1
            continue

        success_record = build_success_record(row, metadata, final_url, full_text, extraction_method, resolution_method, resolution_error)
        url_keys = {key for key in [success_record["canonical_url"], success_record["final_url"]] if key}
        if seen_url_keys.intersection(url_keys):
            failed_rows.append(
                build_failure_record(row, final_url, "duplicate_skipped", "duplicate_url", resolution_method, resolution_error)
            )
            done_ids.add(record_id)
            processed += 1
            continue
        if success_record["content_hash"] in seen_content_hashes:
            failed_rows.append(
                build_failure_record(row, final_url, "duplicate_skipped", "duplicate_content_hash", resolution_method, resolution_error)
            )
            done_ids.add(record_id)
            processed += 1
            continue

        seen_url_keys.update(url_keys)
        seen_content_hashes.add(success_record["content_hash"])
        success_rows.append(success_record)
        done_ids.add(record_id)
        processed += 1
        logger.info("Fetched %s -> %s (%s words)", wrapper_url, final_url, success_record["word_count"])

        if processed % checkpoint_every == 0:
            write_dataframe(pd.DataFrame(success_rows, columns=SUCCESS_COLUMNS), output_csv, output_parquet)
            write_dataframe(pd.DataFrame(failed_rows, columns=FAILED_COLUMNS), failed_csv, None)
            logger.info("Checkpoint written after %s processed rows", processed)

    write_dataframe(pd.DataFrame(success_rows, columns=SUCCESS_COLUMNS), output_csv, output_parquet)
    write_dataframe(pd.DataFrame(failed_rows, columns=FAILED_COLUMNS), failed_csv, None)
    logger.info("Full-text retrieval complete: success=%s failed=%s", len(success_rows), len(failed_rows))


if __name__ == "__main__":
    main()
