from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.common import text_or_empty
from src.utils.config import load_project_config, project_paths
from src.utils.project import (
    DatasetKey,
    language_spec_for,
    load_search_strings,
    resolve_datasets,
    resolve_languages,
    resolve_pathogen_domains,
)
from src.utils.rss import (
    RSS_OUTPUT_COLUMNS,
    build_date_windows,
    build_proxy_settings,
    compute_backoff_seconds,
    dedupe_rss_records,
    detect_throttling,
    is_completed_manifest_row,
    load_or_initialize_manifest,
    parse_cached_payload,
    should_pause_until_next_attempt,
    should_retry,
    write_manifest,
    InstrumentedGoogleNews,
)
from src.utils.common import (
    ensure_dir,
    setup_logger,
    utc_now,
    utc_now_iso,
    write_dataframe_atomic,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve Google News RSS headlines for pathogen-domain datasets."
    )
    parser.add_argument("--config", default="src/config/pipeline.yaml")
    parser.add_argument("--pathogen-domains", default="all")
    parser.add_argument("--languages", default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Skip API retrieval and only parse existing cached RSS payloads.",
    )
    return parser.parse_args()


def frame_value(df: pd.DataFrame, idx: Any, column: str) -> Any:
    return df.loc[idx, column]


def set_frame_value(df: pd.DataFrame, idx: Any, column: str, value: Any) -> None:
    df.loc[idx, column] = value


def update_manifest_row(manifest_df: pd.DataFrame, idx: Any, **values: Any) -> None:
    for key, value in values.items():
        set_frame_value(manifest_df, idx, key, "" if value is None else str(value))


def parse_manifest_outputs(
    dataset: DatasetKey,
    manifest_csv: Path,
    paths: Any,
    logger: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing RSS manifest: {manifest_csv}")

    manifest_df = pd.read_csv(manifest_csv, dtype=str).fillna("")
    completed = manifest_df.loc[manifest_df["status"].isin(["success", "empty"])].copy()
    frames: list[pd.DataFrame] = []
    missing_cache = 0

    for _, row in completed.iterrows():
        cached_rel = text_or_empty(row["cached_path"])
        cached_path = paths.repo_root / cached_rel
        if not cached_rel or not cached_path.exists():
            missing_cache += 1
            logger.warning(
                "Skipping manifest row %s/%s %s..%s because cache is missing at %s",
                dataset.stem,
                row["search_string"],
                row["window_start"],
                row["window_end"],
                cached_rel,
            )
            continue

        with cached_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        frames.append(parse_cached_payload(payload, cached_rel))

    if frames:
        parsed_df = pd.concat(frames, ignore_index=True)
    else:
        parsed_df = pd.DataFrame(columns=RSS_OUTPUT_COLUMNS)

    deduped_df, counts_df = dedupe_rss_records(parsed_df)
    counts_df = counts_df.copy()
    counts_df["dataset_key"] = dataset.stem
    counts_df.loc[len(counts_df)] = {
        "metric": "missing_cache",
        "value": missing_cache,
        "dataset_key": dataset.stem,
    }

    write_dataframe_atomic(
        deduped_df,
        paths.rss_output_csv(dataset),
        paths.rss_output_parquet(dataset),
    )
    write_dataframe_atomic(counts_df, paths.rss_metrics_csv(dataset))
    logger.info(
        "Parsed RSS output for %s: parsed_rows=%s deduped_rows=%s missing_cache=%s",
        dataset.stem,
        len(parsed_df),
        len(deduped_df),
        missing_cache,
    )
    return deduped_df, counts_df


def retrieve_dataset(
    dataset: DatasetKey,
    config: dict[str, Any],
    force: bool,
    parse_only: bool,
    logger: Any,
) -> None:
    paths = project_paths(config)
    paths.ensure_parent_dirs(dataset)

    input_csv = REPO_ROOT / str(config["inputs"]["query_files"][dataset.pathogen_domain])
    search_strings = load_search_strings(input_csv)
    search_cfg = config["search"]
    rss_cfg = config["rss"]
    windows = build_date_windows(
        start_date=str(search_cfg.get("start_date", "2005-01-01")),
        end_date=str(search_cfg.get("end_date", "2025-12-31")),
        chunk_size_days=int(rss_cfg.get("chunk_size_days", 1)),
    )

    manifest_csv = paths.rss_manifest_path(dataset)
    manifest_df = load_or_initialize_manifest(
        manifest_csv=manifest_csv,
        dataset=dataset,
        search_strings=search_strings,
        locale_specs=list(language_spec_for(config, dataset.language_code)["locales"]),
        windows=windows,
        cache_format=str(rss_cfg.get("cache_format", "json")),
    )
    write_manifest(manifest_df, manifest_csv)
    logger.info(
        "RSS retrieval configured for %s with %s search strings and %s windows",
        dataset.stem,
        len(search_strings),
        len(windows),
    )

    if parse_only:
        parse_manifest_outputs(dataset, manifest_csv, paths, logger)
        return

    proxy_settings = build_proxy_settings(rss_cfg)
    client_cache: dict[str, InstrumentedGoogleNews] = {}

    def get_client(locale_id: str, request_lang: str, request_country: str, accept_language: str) -> InstrumentedGoogleNews:
        if locale_id in client_cache:
            return client_cache[locale_id]
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": str(
                    rss_cfg.get(
                        "user_agent",
                        "Mozilla/5.0 (compatible; INFLUX-PathogenNews/1.0; +https://news.google.com/)",
                    )
                ),
                "Accept-Language": accept_language,
                "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
            }
        )
        client_cache[locale_id] = InstrumentedGoogleNews(
            lang=request_lang,
            country=request_country,
            session=session,
            timeout_seconds=int(rss_cfg.get("timeout_seconds", 30)),
        )
        return client_cache[locale_id]

    max_retries = max(1, int(rss_cfg.get("max_retries", 3)))
    base_backoff = float(rss_cfg.get("retry_backoff_seconds", 2.0))
    max_backoff = float(rss_cfg.get("max_backoff_seconds", 90.0))
    request_sleep_seconds = float(rss_cfg.get("request_sleep_seconds", 1.5))

    skipped_completed = 0
    fetched_success = 0
    fetched_empty = 0
    failed_windows = 0
    paused_for_throttle = False
    force_refresh = bool(force or rss_cfg.get("force_refresh", False))

    for idx, row in manifest_df.iterrows():
        cached_rel = text_or_empty(row["cached_path"])
        cached_path = REPO_ROOT / cached_rel
        status = text_or_empty(row["status"] or "pending")
        locale_id = text_or_empty(row["locale_id"])
        request_lang = text_or_empty(row["request_lang"])
        request_country = text_or_empty(row["request_country"])
        request_accept_language = text_or_empty(row["request_accept_language"])
        window_start = text_or_empty(row["window_start"])
        window_end = text_or_empty(row["window_end"])
        search_string = text_or_empty(row["search_string"])
        query_to_exclusive = (pd.Timestamp(window_end) + timedelta(days=1)).date().isoformat()
        window_id = f"{dataset.stem}:{locale_id}:{search_string}:{window_start}_{window_end}"

        if is_completed_manifest_row(status, cached_path, force_refresh):
            skipped_completed += 1
            continue

        now = utc_now()
        if should_pause_until_next_attempt(
            status,
            text_or_empty(row["next_eligible_attempt_at"]),
            now,
        ):
            logger.warning(
                "Stopping before %s because next eligible retry is %s",
                window_id,
                row["next_eligible_attempt_at"],
            )
            paused_for_throttle = True
            break

        window_completed = False
        for attempt_number in range(1, max_retries + 1):
            attempt_started_at = utc_now_iso()
            attempts_total = int(text_or_empty(frame_value(manifest_df, idx, "attempts")) or "0") + 1
            update_manifest_row(
                manifest_df,
                idx,
                attempts=attempts_total,
                last_attempt_at=attempt_started_at,
            )

            try:
                client = get_client(
                    locale_id=locale_id,
                    request_lang=request_lang,
                    request_country=request_country,
                    accept_language=request_accept_language,
                )
                result = client.search_window(
                    query=search_string,
                    from_=window_start,
                    to_=query_to_exclusive,
                    proxy_backend=str(proxy_settings["proxy_backend"]),
                    proxies=proxy_settings["proxies"],
                    scraping_bee_api_key=str(proxy_settings["scraping_bee_api_key"]),
                )
                http_status = result["http_status"]
                response_text = result["response_text"]
                throttled = detect_throttling(
                    http_status=http_status,
                    response_text=response_text,
                    response_url=result["response_url"],
                )
                if throttled:
                    raise RuntimeError("throttled")

                payload = {
                    "dataset_key": dataset.stem,
                    "pathogen_domain": dataset.pathogen_domain,
                    "language_code": dataset.language_code,
                    "request_locale_id": locale_id,
                    "request_lang": request_lang,
                    "request_country": request_country,
                    "request_accept_language": request_accept_language,
                    "search_string": search_string,
                    "window_start": window_start,
                    "window_end": window_end,
                    "rss_language": request_lang,
                    "lang": request_lang,
                    "country": request_country,
                    "feed_url": result["feed_url"],
                    "response_url": result["response_url"],
                    "http_status": http_status,
                    "retrieved_at": attempt_started_at,
                    "feed": result["payload"]["feed"],
                    "entries": result["payload"]["entries"],
                }
                write_json_atomic(payload, cached_path)

                final_status = "empty" if not payload["entries"] else "success"
                update_manifest_row(
                    manifest_df,
                    idx,
                    status=final_status,
                    http_status=http_status,
                    next_eligible_attempt_at="",
                    error_summary="",
                )
                write_manifest(manifest_df, manifest_csv)
                if final_status == "success":
                    fetched_success += 1
                else:
                    fetched_empty += 1
                logger.info(
                    "Retrieved %s status=%s entries=%s",
                    window_id,
                    final_status,
                    len(payload["entries"]),
                )
                window_completed = True
                break
            except Exception as exc:
                throttled = detect_throttling(None, str(exc), "")
                retry_allowed = should_retry(None, throttled, exc)
                next_eligible = ""
                if throttled:
                    next_eligible = (
                        utc_now()
                        + timedelta(
                            seconds=compute_backoff_seconds(
                                attempt_number=attempt_number,
                                base_seconds=base_backoff,
                                max_backoff_seconds=max_backoff,
                            )
                        )
                    ).replace(microsecond=0).isoformat()

                update_manifest_row(
                    manifest_df,
                    idx,
                    status="throttled" if throttled else "failed",
                    http_status="",
                    next_eligible_attempt_at=next_eligible,
                    error_summary=str(exc),
                )
                write_manifest(manifest_df, manifest_csv)

                if attempt_number < max_retries and retry_allowed and not throttled:
                    sleep_seconds = compute_backoff_seconds(
                        attempt_number=attempt_number,
                        base_seconds=base_backoff,
                        max_backoff_seconds=max_backoff,
                    )
                    logger.warning(
                        "Retrying %s after error=%s sleep=%.1fs",
                        window_id,
                        exc,
                        sleep_seconds,
                    )
                    time.sleep(sleep_seconds)
                    continue

                if throttled:
                    logger.warning(
                        "Throttled on %s. Will resume after %s",
                        window_id,
                        next_eligible,
                    )
                    paused_for_throttle = True
                    break
                logger.warning("Failed %s: %s", window_id, exc)
                failed_windows += 1
                break

        if paused_for_throttle:
            break
        if not window_completed:
            continue

        time.sleep(request_sleep_seconds)

    write_manifest(manifest_df, manifest_csv)
    logger.info(
        "RSS retrieval summary for %s: skipped_completed=%s success=%s empty=%s failed=%s paused_for_throttle=%s",
        dataset.stem,
        skipped_completed,
        fetched_success,
        fetched_empty,
        failed_windows,
        paused_for_throttle,
    )
    parse_manifest_outputs(dataset, manifest_csv, paths, logger)


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    paths = project_paths(config)
    logger = setup_logger(
        "pathogen_rss_retrieval",
        paths.logs_dir() / "rss_retrieval.log",
    )
    datasets = resolve_datasets(
        resolve_pathogen_domains(args.pathogen_domains),
        resolve_languages(args.languages),
    )
    for dataset in datasets:
        retrieve_dataset(
            dataset=dataset,
            config=config,
            force=args.force,
            parse_only=args.parse_only,
            logger=logger,
        )


if __name__ == "__main__":
    main()
