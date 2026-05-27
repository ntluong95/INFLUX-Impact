"""Polite HTML fetching and raw HTML persistence."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests

from src.utils.common import ensure_dir, sha256_text, text_or_empty, utc_now_iso
from src.utils.deduplication import decode_google_news_url, is_google_consent_interstitial


def _fetch_with_retries(
    session: requests.Session,
    url: str,
    user_agent: str,
    timeout_seconds: int,
    max_retries: int,
    backoff_base_seconds: float,
    min_delay_seconds: float,
    max_delay_seconds: float,
    retry_http_statuses,
):
    retry_http_statuses = set(retry_http_statuses or [])
    last_error = ""

    for attempt in range(1, max_retries + 1):
        time.sleep(random.uniform(min_delay_seconds, max_delay_seconds))

        try:
            response = session.get(
                url,
                headers={"User-Agent": user_agent},
                timeout=timeout_seconds,
                allow_redirects=True,
            )

            if response.status_code in retry_http_statuses and attempt < max_retries:
                sleep_s = backoff_base_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_s)
                continue

            return response, ""
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < max_retries:
                sleep_s = backoff_base_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_s)

    return None, last_error


def fetch_html_for_urls(
    urls_df: pd.DataFrame,
    raw_html_dir: Path,
    fetch_cfg: Dict[str, Any],
    force: bool = False,
    previous_results_df: pd.DataFrame | None = None,
    logger=None,
) -> pd.DataFrame:
    """Fetch and cache raw HTML for each canonical URL."""
    ensure_dir(raw_html_dir)

    user_agent = fetch_cfg.get(
        "user_agent",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    )
    timeout_seconds = int(fetch_cfg.get("timeout_seconds", 20))
    max_retries = int(fetch_cfg.get("max_retries", 3))
    backoff_base_seconds = float(fetch_cfg.get("backoff_base_seconds", 1.5))
    min_delay_seconds = float(fetch_cfg.get("min_delay_seconds", 1.0))
    max_delay_seconds = float(fetch_cfg.get("max_delay_seconds", 3.0))
    retry_http_statuses = fetch_cfg.get("retry_http_statuses", [429, 500, 502, 503, 504])
    resolver_timeout_seconds = int(fetch_cfg.get("resolver_timeout_seconds", timeout_seconds))
    reuse_previous_failures = bool(fetch_cfg.get("reuse_previous_failures", True))

    results = []
    session = requests.Session()
    previous_lookup = (
        previous_results_df.set_index("canonical_url", drop=False)
        if previous_results_df is not None and not previous_results_df.empty
        else None
    )

    for i, (_, row) in enumerate(urls_df.iterrows(), start=1):
        canonical_url = str(row.get("canonical_url", "")).strip()
        original_url = str(row.get("original_url", "")).strip() or canonical_url
        resolved_url = str(row.get("resolved_url", "")).strip()
        fetch_target = resolved_url or canonical_url or original_url

        html_filename = f"{sha256_text(canonical_url)}.html"
        html_path = raw_html_dir / html_filename

        out = {
            "canonical_url": canonical_url,
            "original_url": original_url,
            "resolved_url": resolved_url,
            "fetch_url": fetch_target,
            "html_path": str(html_path),
            "fetch_status": "",
            "http_status": None,
            "fetched_at": utc_now_iso(),
            "resolution_method": row.get("url_resolution_method", ""),
            "error": "",
        }

        if html_path.exists() and not force:
            out["fetch_status"] = "cached"
            results.append(out)
            if logger is not None and (i % 50 == 0 or i == len(urls_df)):
                logger.info("Fetch progress: %s/%s (cached skip)", i, len(urls_df))
            continue

        if not force and reuse_previous_failures and previous_lookup is not None:
            if canonical_url in previous_lookup.index:
                prev = previous_lookup.loc[canonical_url]
                prev_status = text_or_empty(prev.get("fetch_status"))
                if prev_status in {"http_error", "failed_request", "non_html", "blocked_consent"}:
                    out["fetch_status"] = prev_status
                    out["http_status"] = prev.get("http_status")
                    out["fetched_at"] = text_or_empty(prev.get("fetched_at")) or out["fetched_at"]
                    out["error"] = text_or_empty(prev.get("error"))
                    out["fetch_url"] = text_or_empty(prev.get("fetch_url")) or out["fetch_url"]
                    out["resolved_url"] = text_or_empty(prev.get("resolved_url")) or out["resolved_url"]
                    out["resolution_method"] = (
                        text_or_empty(prev.get("resolution_method")) or out["resolution_method"]
                    )
                    results.append(out)
                    if logger is not None and (i % 50 == 0 or i == len(urls_df)):
                        logger.info("Fetch progress: %s/%s (reused previous failure)", i, len(urls_df))
                    continue

        response, error = _fetch_with_retries(
            session=session,
            url=fetch_target,
            user_agent=user_agent,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_base_seconds=backoff_base_seconds,
            min_delay_seconds=min_delay_seconds,
            max_delay_seconds=max_delay_seconds,
            retry_http_statuses=retry_http_statuses,
        )

        if response is None:
            out["fetch_status"] = "failed_request"
            out["error"] = error
            results.append(out)
            if logger is not None:
                logger.warning("Fetch failed: %s | %s", fetch_target, error)
            continue

        out["http_status"] = int(response.status_code)
        out["fetch_url"] = str(response.url)

        if response.status_code >= 400:
            out["fetch_status"] = "http_error"
            out["error"] = f"HTTP {response.status_code}"
            results.append(out)
            if logger is not None:
                logger.warning("HTTP error for %s: %s", original_url, response.status_code)
            continue

        content_type = response.headers.get("Content-Type", "")
        body = response.text or ""
        looks_html = "text/html" in content_type.lower() or "<html" in body[:2000].lower()

        if not looks_html:
            out["fetch_status"] = "non_html"
            out["error"] = f"Content-Type={content_type}"
            results.append(out)
            continue

        # Fallback: if consent/interstitial page slipped through, decode Google wrapper
        # and retry against resolved publisher URL.
        if is_google_consent_interstitial(body, fetched_url=str(response.url)):
            decoded_url, method, resolver_error = decode_google_news_url(
                original_url,
                timeout_seconds=resolver_timeout_seconds,
                session=session,
            )
            if decoded_url and decoded_url != fetch_target:
                response2, error2 = _fetch_with_retries(
                    session=session,
                    url=decoded_url,
                    user_agent=user_agent,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    backoff_base_seconds=backoff_base_seconds,
                    min_delay_seconds=min_delay_seconds,
                    max_delay_seconds=max_delay_seconds,
                    retry_http_statuses=retry_http_statuses,
                )
                if response2 is not None and response2.status_code < 400:
                    body2 = response2.text or ""
                    if not is_google_consent_interstitial(body2, fetched_url=str(response2.url)):
                        response = response2
                        body = body2
                        out["http_status"] = int(response2.status_code)
                        out["fetch_url"] = str(response2.url)
                        out["resolved_url"] = decoded_url
                        out["resolution_method"] = method
                    else:
                        out["fetch_status"] = "blocked_consent"
                        out["error"] = "google_consent_interstitial"
                        results.append(out)
                        continue
                else:
                    out["fetch_status"] = "blocked_consent"
                    out["error"] = f"resolver_refetch_failed: {error2 or resolver_error}"
                    results.append(out)
                    continue
            else:
                out["fetch_status"] = "blocked_consent"
                out["error"] = f"google_consent_interstitial: {resolver_error}"
                results.append(out)
                continue

        html_path.write_text(body, encoding="utf-8", errors="ignore")
        out["fetch_status"] = "fetched"
        results.append(out)

        if logger is not None and (i % 25 == 0 or i == len(urls_df)):
            logger.info("Fetch progress: %s/%s", i, len(urls_df))

    session.close()
    return pd.DataFrame(results)
