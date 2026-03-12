from __future__ import annotations

import argparse
import time
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast
import requests
import pandas as pd

from zika.python.rss_utils import (
    InstrumentedGoogleNews,
    build_date_windows,
    build_proxy_settings,
    cache_relative_path,
    compute_backoff_seconds,
    detect_throttling,
    ensure_dir,
    is_completed_manifest_row,
    load_config,
    load_or_initialize_manifest,
    manifest_path,
    setup_logger,
    should_pause_until_next_attempt,
    should_retry,
    truncate_for_log,
    utc_now,
    utc_now_iso,
    write_json_atomic,
    write_manifest,
)

warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve Google News RSS windows for Zika using pygooglenews."
    )
    parser.add_argument("--config", default="zika/config/zika.yaml")
    return parser.parse_args()


def frame_value(df: pd.DataFrame, idx: Any, column: str) -> Any:
    return cast(Any, df).loc[idx, column]


def set_frame_value(df: pd.DataFrame, idx: Any, column: str, value: Any) -> None:
    cast(Any, df).loc[idx, column] = value


def update_manifest_row(manifest_df: pd.DataFrame, idx: Any, **values: Any) -> None:
    for key, value in values.items():
        set_frame_value(manifest_df, idx, key, "" if value is None else str(value))


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()
    zika_root = config_path.parent.parent
    cfg = load_config(config_path)

    log_path = zika_root / "logs" / "01_retrieve_google_rss.log"
    logger = setup_logger("zika_rss_fetch", log_path)

    query_cfg = cfg.get("query", {})
    rss_cfg = cfg.get("rss", {})
    query = str(query_cfg.get("term", "zika"))
    windows = build_date_windows(
        start_date=str(query_cfg.get("start_date")),
        end_date=str(query_cfg.get("end_date")),
        chunk_size_days=int(rss_cfg.get("chunk_size_days", 1)),
    )

    raw_dir = zika_root / "data" / "raw" / "rss"
    ensure_dir(raw_dir)
    manifest_csv = manifest_path(
        zika_root,
        str(rss_cfg.get("manifest_filename", f"{query}_rss_manifest.csv")),
    )
    manifest_df = load_or_initialize_manifest(
        manifest_csv=manifest_csv,
        windows=windows,
        query=query,
        cache_format=str(rss_cfg.get("cache_format", "json")),
    )
    write_manifest(manifest_df, manifest_csv)

    logger.info(
        "Stage 1 retrieval configured for %s windows from %s to %s using backend=%s",
        len(windows),
        query_cfg.get("start_date"),
        query_cfg.get("end_date"),
        rss_cfg.get("backend", "pygooglenews"),
    )

    proxy_settings = build_proxy_settings(rss_cfg)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": str(
                rss_cfg.get(
                    "user_agent",
                    "Mozilla/5.0 (compatible; INFLUX-Zika/1.0; +https://news.google.com/)",
                )
            ),
            "Accept-Language": str(rss_cfg.get("accept_language", "en-US,en;q=0.9")),
            "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
        }
    )

    client = InstrumentedGoogleNews(
        lang=str(rss_cfg.get("lang", "en")),
        country=str(rss_cfg.get("country", "US")),
        session=session,
        timeout_seconds=int(rss_cfg.get("timeout_seconds", 30)),
    )

    max_retries = max(1, int(rss_cfg.get("max_retries", 3)))
    base_backoff = float(rss_cfg.get("retry_backoff_seconds", 2.0))
    max_backoff = float(rss_cfg.get("max_backoff_seconds", 90.0))
    request_sleep_seconds = float(rss_cfg.get("request_sleep_seconds", 1.5))
    force_refresh = str(rss_cfg.get("force_refresh", "false")).lower() == "true"

    skipped_completed = 0
    fetched_success = 0
    fetched_empty = 0
    failed_windows = 0
    paused_for_throttle = False

    for idx, row in manifest_df.iterrows():
        cached_rel = str(row["cached_path"] or "")
        cached_path = zika_root / cached_rel
        status = str(row["status"] or "pending")
        window_start = str(row["window_start"])
        window_end = str(row["window_end"])
        query_to_exclusive = (
            date.fromisoformat(window_end) + timedelta(days=1)
        ).isoformat()
        window_id = f"{window_start}_{window_end}"

        if is_completed_manifest_row(status, cached_path, force_refresh):
            skipped_completed += 1
            continue

        now = utc_now()
        if should_pause_until_next_attempt(
            status, str(row["next_eligible_attempt_at"]), now
        ):
            logger.warning(
                "Stopping before %s because next eligible retry is %s",
                window_id,
                row["next_eligible_attempt_at"],
            )
            paused_for_throttle = True
            break

        set_frame_value(
            manifest_df,
            idx,
            "cached_path",
            cached_rel
            or cache_relative_path(
                query=query,
                window_start=window_start,
                window_end=window_end,
                cache_format=str(rss_cfg.get("cache_format", "json")),
            ),
        )
        cached_path = zika_root / str(frame_value(manifest_df, idx, "cached_path"))

        window_completed = False

        for attempt_number in range(1, max_retries + 1):
            attempt_started_at = utc_now_iso()
            attempts_total = (
                int(str(frame_value(manifest_df, idx, "attempts") or "0")) + 1
            )
            update_manifest_row(
                manifest_df,
                idx,
                attempts=attempts_total,
                last_attempt_at=attempt_started_at,
            )

            try:
                result = client.search_window(
                    query=query,
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
                    raise RuntimeError(
                        f"throttled:{http_status}:{truncate_for_log(response_text)}"
                    )

                payload = {
                    "backend": "pygooglenews",
                    "query": query,
                    "window_start": window_start,
                    "window_end": window_end,
                    "language": query_cfg.get("language", "en"),
                    "lang": rss_cfg.get("lang", "en"),
                    "country": rss_cfg.get("country", "US"),
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
                    last_attempt_at=attempt_started_at,
                    next_eligible_attempt_at="",
                    error_summary="",
                )
                write_manifest(manifest_df, manifest_csv)

                if final_status == "success":
                    fetched_success += 1
                else:
                    fetched_empty += 1
                logger.info(
                    "Window %s fetched status=%s entries=%s",
                    window_id,
                    final_status,
                    len(payload["entries"]),
                )
                window_completed = True
                if request_sleep_seconds > 0:
                    time.sleep(request_sleep_seconds)
                break

            except Exception as exc:
                error_text = str(exc)
                http_status = ""
                throttled = False
                if error_text.startswith("throttled:"):
                    throttled = True
                    parts = error_text.split(":", 2)
                    if len(parts) >= 2:
                        http_status = parts[1]
                    error_summary = parts[2] if len(parts) == 3 else error_text
                else:
                    error_summary = truncate_for_log(error_text)

                retryable = should_retry(
                    http_status=int(http_status)
                    if str(http_status).isdigit()
                    else None,
                    throttled=throttled,
                    exc=exc if isinstance(exc, requests.RequestException) else None,
                )

                if retryable and attempt_number < max_retries:
                    backoff_seconds = compute_backoff_seconds(
                        attempt_number=attempt_number,
                        base_seconds=base_backoff,
                        max_backoff_seconds=max_backoff,
                    )
                    next_eligible = utc_now() + pd.Timedelta(seconds=backoff_seconds)
                    update_manifest_row(
                        manifest_df,
                        idx,
                        status="throttled" if throttled else "failed",
                        http_status=http_status,
                        error_summary=error_summary,
                        next_eligible_attempt_at=next_eligible.isoformat().replace(
                            "+00:00", "Z"
                        ),
                    )
                    write_manifest(manifest_df, manifest_csv)
                    logger.warning(
                        "Window %s attempt %s/%s failed status=%s throttled=%s; retrying in %.1fs",
                        window_id,
                        attempt_number,
                        max_retries,
                        http_status or "NA",
                        throttled,
                        backoff_seconds,
                    )
                    time.sleep(backoff_seconds)
                    continue

                update_manifest_row(
                    manifest_df,
                    idx,
                    status="throttled" if throttled else "failed",
                    http_status=http_status,
                    error_summary=error_summary,
                    next_eligible_attempt_at=(
                        (
                            utc_now()
                            + pd.Timedelta(
                                seconds=compute_backoff_seconds(
                                    attempt_number=max_retries,
                                    base_seconds=base_backoff,
                                    max_backoff_seconds=max_backoff,
                                )
                            )
                        )
                        .isoformat()
                        .replace("+00:00", "Z")
                        if throttled
                        else ""
                    ),
                )
                write_manifest(manifest_df, manifest_csv)

                if throttled:
                    logger.warning(
                        "Stopping retrieval after repeated throttling on %s status=%s error=%s",
                        window_id,
                        http_status or "NA",
                        error_summary,
                    )
                    paused_for_throttle = True
                else:
                    failed_windows += 1
                    logger.warning(
                        "Window %s failed after %s attempts status=%s error=%s",
                        window_id,
                        attempt_number,
                        http_status or "NA",
                        error_summary,
                    )
                break

        if paused_for_throttle:
            break
        if not window_completed and not paused_for_throttle:
            continue

    write_manifest(manifest_df, manifest_csv)
    logger.info(
        "Retrieval summary completed=%s success=%s empty=%s failed=%s manifest=%s",
        skipped_completed + fetched_success + fetched_empty,
        fetched_success,
        fetched_empty,
        failed_windows,
        manifest_csv,
    )
    if paused_for_throttle:
        logger.warning(
            "Retrieval paused. First unfinished window remains resumable via manifest %s",
            manifest_csv,
        )


if __name__ == "__main__":
    main()
