from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.common import read_csv_if_exists, setup_logger, text_or_empty, write_dataframe_atomic
from src.utils.config import load_project_config, project_paths
from src.utils.project import DatasetKey, resolve_datasets, resolve_languages, resolve_pathogen_domains
from src.utils.scrape import (
    canonicalize_url,
    extract_full_text,
    extract_metadata,
    polite_fetch,
    resolve_google_news_url,
    sha256_text,
)


SUCCESS_COLUMNS = [
    "record_id",
    "dataset_key",
    "pathogen_domain",
    "language_code",
    "search_string",
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
    "article_language",
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
    "dataset_key",
    "pathogen_domain",
    "language_code",
    "search_string",
    "google_news_redirect_url",
    "rss_title",
    "final_url",
    "fetch_status",
    "error_reason",
    "resolution_method",
    "resolution_error",
    "fetched_at",
]


def existing_success_is_valid(record: dict[str, Any], min_text_words: int) -> bool:
    if str(record.get("fetch_status", "")).strip().lower() != "success":
        return False
    full_text = str(record.get("full_text", "") or "").strip()
    if not full_text:
        return False
    content_hash = str(record.get("content_hash", "") or "").strip()
    if not content_hash:
        return False
    try:
        word_count = int(float(record.get("word_count", 0) or 0))
    except Exception:
        word_count = 0
    return word_count >= min_text_words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve full text for pathogen news articles classified as keep."
    )
    parser.add_argument("--config", default="src/config/pipeline.yaml")
    parser.add_argument("--pathogen-domains", default="all")
    parser.add_argument("--languages", default="all")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_success_record(
    row: pd.Series,
    metadata: dict[str, Any],
    final_url: str,
    full_text: str,
    extraction_method: str,
    resolution_method: str,
    resolution_error: str,
) -> dict[str, Any]:
    canonical_url = (
        canonicalize_url(metadata.get("canonical_url"))
        or canonicalize_url(final_url)
        or final_url
    )
    domain = metadata.get("domain") or ""
    authors = metadata.get("authors") or []
    word_count = len(full_text.split())
    return {
        "record_id": row["record_id"],
        "dataset_key": row["dataset_key"],
        "pathogen_domain": row["pathogen_domain"],
        "language_code": row["language_code"],
        "search_string": row.get("search_string", ""),
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
        "article_language": metadata.get("language", ""),
        "extraction_method": extraction_method,
        "fetch_status": "success",
        "error_reason": "",
        "resolution_method": resolution_method,
        "resolution_error": resolution_error,
        "content_hash": sha256_text(full_text),
        "fetched_at": text_or_empty(pd.Timestamp.now(tz="UTC").isoformat()),
    }


def build_failure_record(
    row: pd.Series,
    final_url: str,
    fetch_status: str,
    error_reason: str,
    resolution_method: str,
    resolution_error: str,
) -> dict[str, Any]:
    return {
        "record_id": row["record_id"],
        "dataset_key": row["dataset_key"],
        "pathogen_domain": row["pathogen_domain"],
        "language_code": row["language_code"],
        "search_string": row.get("search_string", ""),
        "google_news_redirect_url": row["google_news_redirect_url"],
        "rss_title": row.get("rss_title", ""),
        "final_url": final_url,
        "fetch_status": fetch_status,
        "error_reason": error_reason,
        "resolution_method": resolution_method,
        "resolution_error": resolution_error,
        "fetched_at": text_or_empty(pd.Timestamp.now(tz="UTC").isoformat()),
    }


def retrieve_dataset(
    dataset: DatasetKey,
    config: dict[str, Any],
    force: bool,
    logger: Any,
) -> None:
    paths = project_paths(config)
    fulltext_cfg = config["fulltext"]
    input_csv = paths.classified_headlines_csv(dataset)
    output_csv = paths.fulltext_csv(dataset)
    output_parquet = paths.fulltext_parquet(dataset)
    failed_csv = paths.fulltext_failed_csv(dataset)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Missing classified headlines for {dataset.stem}: {input_csv}"
        )

    scored_df = pd.read_csv(input_csv, low_memory=False)
    keep_df = scored_df.loc[
        scored_df["final_action"].fillna("").astype(str)
        == str(fulltext_cfg.get("input_action", "keep"))
    ].copy()

    if force:
        existing_success = pd.DataFrame()
        existing_failed = pd.DataFrame()
    else:
        existing_success = read_csv_if_exists(output_csv)
        existing_failed = read_csv_if_exists(failed_csv)

    min_text_words = int(fulltext_cfg.get("min_text_words", 120))
    valid_success_rows: list[dict[str, Any]] = []
    stale_success_ids: set[str] = set()
    if not existing_success.empty:
        for record in existing_success.to_dict(orient="records"):
            record_id = str(record.get("record_id", "") or "")
            if existing_success_is_valid(record, min_text_words):
                valid_success_rows.append(record)
            elif record_id:
                stale_success_ids.add(record_id)

    existing_failed_rows = (
        existing_failed.to_dict(orient="records") if not existing_failed.empty else []
    )
    existing_failed_ids = {
        str(record.get("record_id", "") or "")
        for record in existing_failed_rows
        if str(record.get("record_id", "") or "")
    }
    valid_success_ids = {
        str(record.get("record_id", "") or "")
        for record in valid_success_rows
        if str(record.get("record_id", "") or "")
    }
    done_ids = valid_success_ids | existing_failed_ids

    logger.info(
        "Full-text retrieval starting for %s with kept_rows=%s pending=%s cached_success=%s cached_failed=%s",
        dataset.stem,
        len(keep_df),
        max(0, len(keep_df) - len(done_ids)),
        len(valid_success_ids),
        len(existing_failed_ids),
    )

    success_rows = list(valid_success_rows)
    failed_rows = list(existing_failed_rows)
    seen_url_keys = set()
    seen_content_hashes = set()
    for record in success_rows:
        if record.get("canonical_url"):
            seen_url_keys.add(str(record["canonical_url"]))
        if record.get("final_url"):
            seen_url_keys.add(str(record["final_url"]))
        if record.get("content_hash"):
            seen_content_hashes.add(str(record["content_hash"]))

    if keep_df.empty:
        write_dataframe_atomic(pd.DataFrame(success_rows, columns=SUCCESS_COLUMNS), output_csv, output_parquet)
        write_dataframe_atomic(pd.DataFrame(failed_rows, columns=FAILED_COLUMNS), failed_csv)
        logger.info("No kept URLs found for %s. Wrote empty outputs.", dataset.stem)
        return

    session = requests.Session()
    checkpoint_every = max(1, int(fulltext_cfg.get("checkpoint_every", 10)))
    processed = 0
    new_successes = 0
    new_failures = 0

    for _, row in keep_df.iterrows():
        record_id = str(row["record_id"])
        if record_id in done_ids and record_id not in stale_success_ids:
            continue

        wrapper_url = str(row["google_news_redirect_url"])
        resolved_url, resolution_method, resolution_error = resolve_google_news_url(
            wrapper_url=wrapper_url,
            timeout_seconds=int(fulltext_cfg.get("timeout_seconds", 30)),
            session=session,
        )
        try:
            response = polite_fetch(session, resolved_url, fulltext_cfg)
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
            new_failures += 1
            continue

        final_url = str(response.url)
        content_type = response.headers.get("Content-Type", "")
        if response.status_code >= 400:
            failed_rows.append(
                build_failure_record(
                    row,
                    final_url,
                    f"http_{response.status_code}",
                    f"HTTP {response.status_code}",
                    resolution_method,
                    resolution_error,
                )
            )
            done_ids.add(record_id)
            processed += 1
            new_failures += 1
            continue
        if "html" not in content_type.lower() and "<html" not in (response.text or "")[:2000].lower():
            failed_rows.append(
                build_failure_record(
                    row,
                    final_url,
                    "non_html",
                    content_type,
                    resolution_method,
                    resolution_error,
                )
            )
            done_ids.add(record_id)
            processed += 1
            new_failures += 1
            continue

        metadata = extract_metadata(response.text or "", final_url, response=response)
        full_text, extraction_method = extract_full_text(response.text or "", final_url)
        if not full_text:
            failed_rows.append(
                build_failure_record(
                    row,
                    final_url,
                    "empty_text",
                    "no_extraction_output",
                    resolution_method,
                    resolution_error,
                )
            )
            done_ids.add(record_id)
            processed += 1
            new_failures += 1
            continue

        word_count = len(full_text.split())
        if word_count < min_text_words:
            failed_rows.append(
                build_failure_record(
                    row,
                    final_url,
                    "too_short",
                    f"word_count={word_count}",
                    resolution_method,
                    resolution_error,
                )
            )
            done_ids.add(record_id)
            processed += 1
            new_failures += 1
            continue

        success_record = build_success_record(
            row,
            metadata,
            final_url,
            full_text,
            extraction_method,
            resolution_method,
            resolution_error,
        )
        url_keys = {
            key
            for key in [success_record["canonical_url"], success_record["final_url"]]
            if key
        }
        if seen_url_keys.intersection(url_keys):
            failed_rows.append(
                build_failure_record(
                    row,
                    final_url,
                    "duplicate_skipped",
                    "duplicate_url",
                    resolution_method,
                    resolution_error,
                )
            )
            done_ids.add(record_id)
            processed += 1
            new_failures += 1
            continue
        if success_record["content_hash"] in seen_content_hashes:
            failed_rows.append(
                build_failure_record(
                    row,
                    final_url,
                    "duplicate_skipped",
                    "duplicate_content_hash",
                    resolution_method,
                    resolution_error,
                )
            )
            done_ids.add(record_id)
            processed += 1
            new_failures += 1
            continue

        seen_url_keys.update(url_keys)
        seen_content_hashes.add(success_record["content_hash"])
        success_rows.append(success_record)
        done_ids.add(record_id)
        processed += 1
        new_successes += 1

        if processed % checkpoint_every == 0:
            write_dataframe_atomic(pd.DataFrame(success_rows, columns=SUCCESS_COLUMNS), output_csv, output_parquet)
            write_dataframe_atomic(pd.DataFrame(failed_rows, columns=FAILED_COLUMNS), failed_csv)
            logger.info("Full-text checkpoint written for %s after %s processed rows", dataset.stem, processed)

    write_dataframe_atomic(pd.DataFrame(success_rows, columns=SUCCESS_COLUMNS), output_csv, output_parquet)
    write_dataframe_atomic(pd.DataFrame(failed_rows, columns=FAILED_COLUMNS), failed_csv)
    logger.info(
        "Full-text retrieval complete for %s: new_success=%s new_failed=%s total_success=%s total_failed=%s",
        dataset.stem,
        new_successes,
        new_failures,
        len(success_rows),
        len(failed_rows),
    )


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    paths = project_paths(config)
    logger = setup_logger(
        "pathogen_fulltext",
        paths.logs_dir() / "fulltext_retrieval.log",
    )
    datasets = resolve_datasets(
        resolve_pathogen_domains(args.pathogen_domains),
        resolve_languages(args.languages),
    )
    for dataset in datasets:
        retrieve_dataset(dataset, config, args.force, logger)


if __name__ == "__main__":
    main()
