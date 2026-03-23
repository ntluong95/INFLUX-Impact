from __future__ import annotations

import json
import os
import random
import re
import tempfile
import warnings
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, cast

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pygooglenews import GoogleNews

from src.utils.common import (
    ensure_dir,
    extract_domain,
    normalize_whitespace,
    setup_logger,
    slugify_text,
    stable_hash,
    utc_now,
    utc_now_iso,
    write_dataframe_atomic,
    write_json_atomic,
)
from src.utils.project import DatasetKey


warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")
REQUESTS_WARNING = getattr(requests.exceptions, "RequestsDependencyWarning", Warning)
warnings.filterwarnings("ignore", category=REQUESTS_WARNING)

DEFAULT_WINDOW_HIERARCHY_DAYS = [30, 14, 7, 3, 1]

THROTTLE_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [
        r"<title>\s*sorry",
        r"unusual traffic from your computer network",
        r"our systems have detected unusual traffic",
        r"/sorry/",
        r"to continue, please type the characters below",
        r"detected unusual traffic",
        r"captcha",
    ]
]

RSS_MANIFEST_COLUMNS = [
    "pathogen_domain",
    "language_code",
    "locale_id",
    "request_lang",
    "request_country",
    "request_accept_language",
    "search_string",
    "query_slug",
    "window_start",
    "window_end",
    "window_days",
    "window_role",
    "parent_window_start",
    "parent_window_end",
    "split_trigger_entries",
    "status",
    "attempts",
    "http_status",
    "cached_path",
    "last_attempt_at",
    "next_eligible_attempt_at",
    "error_summary",
]

RSS_MANIFEST_KEY_COLUMNS = [
    "pathogen_domain",
    "language_code",
    "locale_id",
    "request_lang",
    "request_country",
    "request_accept_language",
    "search_string",
    "query_slug",
    "window_start",
    "window_end",
]

RSS_OUTPUT_COLUMNS = [
    "record_id",
    "dataset_key",
    "pathogen_domain",
    "language_code",
    "request_locale_id",
    "request_lang",
    "request_country",
    "search_string",
    "date_window_start",
    "date_window_end",
    "rss_language",
    "rss_title",
    "rss_pubdate",
    "rss_pubdate_utc",
    "rss_description",
    "rss_description_text",
    "google_rss_url",
    "google_news_redirect_url",
    "guid",
    "source_hint",
    "source_url",
    "source_domain",
    "feed_channel_title",
    "feed_last_build_date",
    "retrieved_at",
    "cached_path",
]


def build_date_windows(
    start_date: str,
    end_date: str,
    chunk_size_days: int,
) -> list[dict[str, str]]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    size = max(1, int(chunk_size_days))
    windows: list[dict[str, str]] = []

    current = start
    while current <= end:
        window_end = min(current + timedelta(days=size - 1), end)
        windows.append(
            {
                "window_start": current.isoformat(),
                "window_end": window_end.isoformat(),
                "window_id": f"{current.isoformat()}_{window_end.isoformat()}",
            }
        )
        current = window_end + timedelta(days=1)

    return windows


def compute_window_days(window_start: str, window_end: str) -> int:
    start = date.fromisoformat(window_start)
    end = date.fromisoformat(window_end)
    return max(1, (end - start).days + 1)


def normalize_window_hierarchy(raw_value: Any) -> list[int]:
    if raw_value is None or raw_value == "":
        values = list(DEFAULT_WINDOW_HIERARCHY_DAYS)
    elif isinstance(raw_value, str):
        values = [int(chunk.strip()) for chunk in raw_value.split(",") if chunk.strip()]
    elif isinstance(raw_value, (list, tuple)):
        values = [int(item) for item in raw_value]
    else:
        values = [int(raw_value)]

    cleaned = sorted({max(1, int(value)) for value in values}, reverse=True)
    if not cleaned:
        raise ValueError("RSS window hierarchy must contain at least one positive integer.")
    if cleaned[-1] != 1:
        cleaned.append(1)
    return cleaned


def next_split_window_days(
    current_window_days: int,
    window_hierarchy_days: list[int],
) -> int | None:
    current = max(1, int(current_window_days))
    hierarchy = normalize_window_hierarchy(window_hierarchy_days)
    smaller_levels = [level for level in hierarchy if level < current]
    if not smaller_levels:
        return None
    return max(smaller_levels)


def cache_relative_path(
    dataset: DatasetKey,
    locale_id: str,
    search_string: str,
    window_start: str,
    window_end: str,
    cache_format: str = "json",
) -> str:
    extension = cache_format.lower().strip(".") or "json"
    query_slug = slugify_text(search_string)
    return (
        f"data/raw/rss/{dataset.stem}/{locale_id}/{query_slug}/"
        f"{dataset.stem}_{window_start}_{window_end}.{extension}"
    )


def manifest_row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "") or "") for column in RSS_MANIFEST_KEY_COLUMNS)


def build_manifest_row(
    dataset: DatasetKey,
    locale_id: str,
    request_lang: str,
    request_country: str,
    request_accept_language: str,
    search_string: str,
    window_start: str,
    window_end: str,
    cache_format: str,
    *,
    window_role: str = "base",
    parent_window_start: str = "",
    parent_window_end: str = "",
    split_trigger_entries: str = "",
    status: str = "pending",
    attempts: str = "0",
    http_status: str = "",
    last_attempt_at: str = "",
    next_eligible_attempt_at: str = "",
    error_summary: str = "",
) -> dict[str, str]:
    query_slug = slugify_text(search_string)
    return {
        "pathogen_domain": dataset.pathogen_domain,
        "language_code": dataset.language_code,
        "locale_id": locale_id,
        "request_lang": request_lang,
        "request_country": request_country,
        "request_accept_language": request_accept_language,
        "search_string": search_string,
        "query_slug": query_slug,
        "window_start": window_start,
        "window_end": window_end,
        "window_days": str(compute_window_days(window_start, window_end)),
        "window_role": window_role,
        "parent_window_start": parent_window_start,
        "parent_window_end": parent_window_end,
        "split_trigger_entries": split_trigger_entries,
        "status": status,
        "attempts": attempts,
        "http_status": http_status,
        "cached_path": cache_relative_path(
            dataset=dataset,
            locale_id=locale_id,
            search_string=search_string,
            window_start=window_start,
            window_end=window_end,
            cache_format=cache_format,
        ),
        "last_attempt_at": last_attempt_at,
        "next_eligible_attempt_at": next_eligible_attempt_at,
        "error_summary": error_summary,
    }


def load_or_initialize_manifest(
    manifest_csv: Path,
    dataset: DatasetKey,
    search_strings: list[str],
    locale_specs: list[dict[str, str]],
    windows: list[dict[str, str]],
    cache_format: str,
) -> pd.DataFrame:
    default_rows = []
    for locale in locale_specs:
        locale_id = str(locale["id"])
        for search_string in search_strings:
            for window in windows:
                default_rows.append(
                    build_manifest_row(
                        dataset=dataset,
                        locale_id=locale_id,
                        request_lang=str(locale["lang"]),
                        request_country=str(locale["country"]),
                        request_accept_language=str(locale["accept_language"]),
                        search_string=search_string,
                        window_start=window["window_start"],
                        window_end=window["window_end"],
                        cache_format=cache_format,
                    )
                )

    manifest_df = pd.DataFrame(default_rows, columns=RSS_MANIFEST_COLUMNS)
    if not manifest_csv.exists():
        return manifest_df

    existing = pd.read_csv(manifest_csv, dtype=str).fillna("")
    for column in RSS_MANIFEST_COLUMNS:
        if column not in existing.columns:
            existing[column] = ""
    existing = existing[RSS_MANIFEST_COLUMNS]

    default_by_key = {manifest_row_key(row): row for row in default_rows}
    normalized_existing: list[dict[str, str]] = []
    existing_keys: set[tuple[str, ...]] = set()

    for record in existing.to_dict(orient="records"):
        key = manifest_row_key(record)
        default_row = default_by_key.get(key, {})
        normalized: dict[str, str] = {}
        for column in RSS_MANIFEST_COLUMNS:
            existing_value = str(record.get(column, "") or "")
            default_value = str(default_row.get(column, "") or "")
            normalized[column] = existing_value or default_value

        if not normalized["query_slug"]:
            normalized["query_slug"] = slugify_text(normalized["search_string"])
        if (
            not normalized["cached_path"]
            and normalized["locale_id"]
            and normalized["search_string"]
            and normalized["window_start"]
            and normalized["window_end"]
        ):
            normalized["cached_path"] = cache_relative_path(
                dataset=dataset,
                locale_id=normalized["locale_id"],
                search_string=normalized["search_string"],
                window_start=normalized["window_start"],
                window_end=normalized["window_end"],
                cache_format=cache_format,
            )
        if (
            not normalized["window_days"]
            and normalized["window_start"]
            and normalized["window_end"]
        ):
            normalized["window_days"] = str(
                compute_window_days(
                    normalized["window_start"],
                    normalized["window_end"],
                )
            )
        if not normalized["window_role"]:
            normalized["window_role"] = "base" if default_row else "adaptive_child"

        normalized_existing.append(normalized)
        existing_keys.add(manifest_row_key(normalized))

    missing_default_rows = [
        row for row in default_rows if manifest_row_key(row) not in existing_keys
    ]
    combined_rows = normalized_existing + missing_default_rows
    return pd.DataFrame(combined_rows, columns=RSS_MANIFEST_COLUMNS).fillna("")


def write_manifest(manifest_df: pd.DataFrame, manifest_csv: Path) -> None:
    write_dataframe_atomic(manifest_df[RSS_MANIFEST_COLUMNS].fillna(""), manifest_csv)


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_rfc822_to_utc(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = parsedate_to_datetime(value)
    except TypeError:
        return ""
    except ValueError:
        return ""
    except IndexError:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (
        parsed.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def clean_html_text(html: str | None) -> str:
    if not html:
        return ""
    return normalize_whitespace(
        BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    )


def serialize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [serialize_for_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_for_json(item) for key, item in value.items()}
    if hasattr(value, "items"):
        return {str(key): serialize_for_json(item) for key, item in value.items()}
    return str(value)


def normalize_title_for_dedupe(value: str | None) -> str:
    return normalize_whitespace(re.sub(r"[^0-9A-Za-z ]+", " ", (value or "").lower()))


def dedupe_rss_records(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        counts = pd.DataFrame(
            [
                {"metric": "input_rows", "value": 0},
                {"metric": "duplicate_redirect_url", "value": 0},
                {"metric": "duplicate_guid", "value": 0},
                {"metric": "duplicate_title_pubdate", "value": 0},
                {"metric": "kept_rows", "value": 0},
            ]
        )
        return df.reindex(columns=RSS_OUTPUT_COLUMNS), counts

    work = df.copy()
    work["dedupe_redirect_key"] = work["google_news_redirect_url"].replace("", pd.NA)
    work["dedupe_guid_key"] = work["guid"].replace("", pd.NA)
    work["dedupe_title_pubdate_key"] = work.apply(
        lambda row: (
            f"{normalize_title_for_dedupe(row['rss_title'])}||{row['rss_pubdate_utc']}"
            if row.get("rss_pubdate_utc")
            else pd.NA
        ),
        axis=1,
    )
    work = work.sort_values(
        by=[
            "date_window_start",
            "date_window_end",
            "retrieved_at",
            "google_news_redirect_url",
            "guid",
            "rss_title",
        ],
        kind="stable",
    ).reset_index(drop=True)

    duplicate_redirect = (
        work["dedupe_redirect_key"].notna() & work["dedupe_redirect_key"].duplicated()
    )
    remaining_after_redirect = work.loc[~duplicate_redirect].copy()
    duplicate_guid_remaining = (
        remaining_after_redirect["dedupe_guid_key"].notna()
        & remaining_after_redirect["dedupe_guid_key"].duplicated()
    )
    duplicate_guid = pd.Series(False, index=work.index)
    duplicate_guid.loc[remaining_after_redirect.index] = duplicate_guid_remaining

    remaining_after_guid = work.loc[~(duplicate_redirect | duplicate_guid)].copy()
    duplicate_title_remaining = (
        remaining_after_guid["dedupe_title_pubdate_key"].notna()
        & remaining_after_guid["dedupe_title_pubdate_key"].duplicated()
    )
    duplicate_title = pd.Series(False, index=work.index)
    duplicate_title.loc[remaining_after_guid.index] = duplicate_title_remaining

    kept = work.loc[~(duplicate_redirect | duplicate_guid | duplicate_title)].copy()
    kept = kept.drop(
        columns=["dedupe_redirect_key", "dedupe_guid_key", "dedupe_title_pubdate_key"],
        errors="ignore",
    )
    kept = kept.reindex(columns=RSS_OUTPUT_COLUMNS)

    counts = pd.DataFrame(
        [
            {"metric": "input_rows", "value": int(len(work))},
            {"metric": "duplicate_redirect_url", "value": int(duplicate_redirect.sum())},
            {"metric": "duplicate_guid", "value": int(duplicate_guid.sum())},
            {"metric": "duplicate_title_pubdate", "value": int(duplicate_title.sum())},
            {"metric": "kept_rows", "value": int(len(kept))},
        ]
    )
    return kept, counts


def is_completed_manifest_row(status: str, cached_path: Path, force_refresh: bool) -> bool:
    if force_refresh:
        return False
    if status == "split":
        return True
    return status in {"success", "empty"} and cached_path.exists()


def should_pause_until_next_attempt(
    status: str,
    next_eligible_attempt_at: str,
    now: datetime,
) -> bool:
    if status not in {"throttled", "failed"}:
        return False
    next_dt = parse_iso_datetime(next_eligible_attempt_at)
    if next_dt is None:
        return False
    return now < next_dt


def compute_backoff_seconds(
    attempt_number: int,
    base_seconds: float,
    max_backoff_seconds: float,
) -> float:
    raw = min(max_backoff_seconds, base_seconds * (2 ** max(0, attempt_number - 1)))
    return raw + random.uniform(0, 1)


def detect_throttling(
    http_status: int | None,
    response_text: str | None,
    response_url: str | None,
) -> bool:
    if http_status in {429, 503}:
        return True
    haystack = " ".join(
        part
        for part in [str(response_url or ""), str(response_text or "")[:4000]]
        if part
    )
    return any(pattern.search(haystack) for pattern in THROTTLE_PATTERNS)


def should_retry(
    http_status: int | None,
    throttled: bool,
    exc: Exception | None = None,
) -> bool:
    if throttled:
        return True
    if http_status in {408, 425, 429, 500, 502, 503, 504}:
        return True
    return isinstance(exc, requests.RequestException)


def build_proxy_settings(rss_cfg: dict[str, Any]) -> dict[str, Any]:
    mode = str(rss_cfg.get("proxy_backend", "direct")).strip().lower() or "direct"
    http_proxy = str(rss_cfg.get("http_proxy", "") or "").strip()
    https_proxy = str(rss_cfg.get("https_proxy", "") or "").strip()
    scraping_bee_api_key = str(rss_cfg.get("scraping_bee_api_key", "") or "").strip()

    if mode == "scraping_bee" and not scraping_bee_api_key:
        raise ValueError(
            "SCRAPING_BEE_API_KEY is required when rss.proxy_backend=scraping_bee"
        )

    proxies: dict[str, str] | None = None
    if mode == "requests":
        proxies = {}
        if http_proxy:
            proxies["http"] = http_proxy
        if https_proxy:
            proxies["https"] = https_proxy
        if not proxies:
            mode = "direct"
            proxies = None

    return {
        "proxy_backend": mode,
        "proxies": proxies,
        "scraping_bee_api_key": scraping_bee_api_key,
    }


class InstrumentedGoogleNews(GoogleNews):
    def __init__(
        self,
        *,
        lang: str,
        country: str,
        session: requests.Session,
        timeout_seconds: int,
    ) -> None:
        super().__init__(lang=lang, country=country)
        self.session = session
        self.timeout_seconds = timeout_seconds

    def _from_to_helper(self, value: str) -> str:
        return cast(Any, self)._GoogleNews__from_to_helper(validate=value)

    def _search_helper(self, query_text: str) -> str:
        return cast(Any, self)._GoogleNews__search_helper(query_text)

    def _ceid(self) -> str:
        return cast(Any, self)._GoogleNews__ceid()

    def _add_sub_articles(self, entries: Any) -> Any:
        return cast(Any, self)._GoogleNews__add_sub_articles(entries)

    def build_search_url(
        self,
        query: str,
        *,
        helper: bool = True,
        when: str | None = None,
        from_: str | None = None,
        to_: str | None = None,
    ) -> str:
        query_text = query
        if when:
            query_text += f" when:{when}"
        if from_ and not when:
            query_text += f" after:{self._from_to_helper(from_)}"
        if to_ and not when:
            query_text += f" before:{self._from_to_helper(to_)}"
        if helper:
            query_text = self._search_helper(query_text)
        search_ceid = self._ceid().replace("?", "&")
        return f"{self.BASE_URL}/search?q={query_text}{search_ceid}"

    def _perform_request(
        self,
        feed_url: str,
        *,
        proxy_backend: str,
        proxies: dict[str, str] | None,
        scraping_bee_api_key: str,
    ) -> requests.Response:
        if proxy_backend == "scraping_bee":
            response = self.session.get(
                url="https://app.scrapingbee.com/api/v1/",
                params={
                    "api_key": scraping_bee_api_key,
                    "url": feed_url,
                    "render_js": "false",
                },
                timeout=self.timeout_seconds,
            )
        else:
            response = self.session.get(
                feed_url,
                timeout=self.timeout_seconds,
                proxies=proxies,
            )
        return response

    def search_window(
        self,
        *,
        query: str,
        from_: str,
        to_: str,
        proxy_backend: str,
        proxies: dict[str, str] | None,
        scraping_bee_api_key: str,
    ) -> dict[str, Any]:
        feed_url = self.build_search_url(query=query, from_=from_, to_=to_)
        response = self._perform_request(
            feed_url,
            proxy_backend=proxy_backend,
            proxies=proxies,
            scraping_bee_api_key=scraping_bee_api_key,
        )
        response_text = response.text or ""
        if "https://news.google.com/rss/unsupported" in str(response.url):
            raise RuntimeError("This feed is not available")

        parsed = feedparser.parse(response_text)
        entries = self._add_sub_articles(parsed.get("entries", []))

        return {
            "feed_url": feed_url,
            "response_url": str(response.url),
            "http_status": int(response.status_code),
            "response_text": response_text,
            "payload": {
                "feed": serialize_for_json(parsed.get("feed", {})),
                "entries": serialize_for_json(entries),
            },
        }


def parse_cached_payload(payload: dict[str, Any], cached_path: str) -> pd.DataFrame:
    feed = payload.get("feed", {}) or {}
    entries = payload.get("entries", []) or []
    rows: list[dict[str, str]] = []
    dataset_key = str(payload.get("dataset_key", "") or "")
    pathogen_domain = str(payload.get("pathogen_domain", "") or "")
    language_code = str(payload.get("language_code", "") or "")
    request_locale_id = str(payload.get("request_locale_id", "") or "")
    request_lang = str(payload.get("request_lang", "") or "")
    request_country = str(payload.get("request_country", "") or "")
    search_string = str(payload.get("search_string", "") or "")
    window_start = str(payload.get("window_start", "") or "")
    window_end = str(payload.get("window_end", "") or "")
    retrieved_at = str(payload.get("retrieved_at", "") or "")
    language = normalize_whitespace(
        str(
            feed.get("language")
            or payload.get("rss_language")
            or payload.get("lang")
            or ""
        )
    )
    google_rss_url = str(payload.get("feed_url", "") or "")
    feed_channel_title = normalize_whitespace(str(feed.get("title", "") or ""))
    feed_last_build_date = normalize_whitespace(str(feed.get("updated", "") or ""))

    for entry in entries:
        title = normalize_whitespace(str(entry.get("title", "") or ""))
        pubdate = normalize_whitespace(str(entry.get("published", "") or ""))
        description = str(
            entry.get("summary")
            or (entry.get("summary_detail") or {}).get("value")
            or ""
        )
        redirect_url = normalize_whitespace(str(entry.get("link", "") or ""))
        guid = normalize_whitespace(str(entry.get("id", "") or ""))
        source = entry.get("source") or {}
        source_url = normalize_whitespace(str(source.get("href", "") or ""))
        row = {
            "dataset_key": dataset_key,
            "pathogen_domain": pathogen_domain,
            "language_code": language_code,
            "request_locale_id": request_locale_id,
            "request_lang": request_lang,
            "request_country": request_country,
            "search_string": search_string,
            "date_window_start": window_start,
            "date_window_end": window_end,
            "rss_language": language,
            "rss_title": title,
            "rss_pubdate": pubdate,
            "rss_pubdate_utc": parse_rfc822_to_utc(pubdate),
            "rss_description": description,
            "rss_description_text": clean_html_text(description),
            "google_rss_url": google_rss_url,
            "google_news_redirect_url": redirect_url,
            "guid": guid,
            "source_hint": normalize_whitespace(str(source.get("title", "") or "")),
            "source_url": source_url,
            "source_domain": extract_domain(source_url),
            "feed_channel_title": feed_channel_title,
            "feed_last_build_date": feed_last_build_date,
            "retrieved_at": retrieved_at,
            "cached_path": cached_path,
        }
        row["record_id"] = stable_hash(
            "||".join(
                [
                    row["dataset_key"],
                    row["search_string"],
                    row["date_window_start"],
                    row["date_window_end"],
                    row["google_news_redirect_url"],
                    row["guid"],
                    row["rss_title"],
                    row["rss_pubdate_utc"],
                ]
            )
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=RSS_OUTPUT_COLUMNS)
    return pd.DataFrame(rows, columns=RSS_OUTPUT_COLUMNS)
