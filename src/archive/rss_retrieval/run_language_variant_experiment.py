from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.common import ensure_dir, normalize_whitespace, setup_logger, stable_hash, utc_now_iso, write_dataframe_atomic, write_json_atomic
from src.utils.config import load_project_config, project_paths
from src.utils.project import language_spec_for
from src.utils.rss import InstrumentedGoogleNews, compute_backoff_seconds, detect_throttling, parse_cached_payload


DEFAULT_BASE_COUNTRY = {
    "en": "US",
    "fr": "FR",
    "es": "ES",
    "pt": "BR",
}

SPECIAL_VARIANT_COUNTRY = {
    "es-419": "MX",
}

LANGUAGE_LABELS = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
}


@dataclass(frozen=True)
class VariantSpec:
    language_code: str
    requested_variant: str
    variant_rank: int
    request_lang: str
    request_country: str
    request_accept_language: str

    @property
    def variant_id(self) -> str:
        return (
            self.requested_variant.lower()
            .replace("-", "_")
            .replace(" ", "_")
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a yearly Google News RSS overlap experiment across language variants."
    )
    parser.add_argument("--config", default="src/config/pipeline.yaml")
    parser.add_argument(
        "--topics",
        default="data/inputs/language_variant_experiment_topics.csv",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--languages", default="en,fr,es,pt")
    parser.add_argument("--topics-filter", default="all")
    parser.add_argument("--sleep-seconds", type=float, default=0.4)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=90.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    normalized = normalize_whitespace(value)
    if not normalized or normalized.lower() == "all":
        return []
    return [chunk.strip() for chunk in normalized.split(",") if chunk.strip()]


def load_topics(topics_path: Path, selected_topics: list[str]) -> pd.DataFrame:
    topics_df = pd.read_csv(topics_path, low_memory=False).fillna("")
    required_columns = {
        "topic_id",
        "pathogen_domain",
        "topic_label",
        "search_string_en",
        "search_string_fr",
        "search_string_es",
        "search_string_pt",
    }
    missing = sorted(required_columns.difference(topics_df.columns))
    if missing:
        raise ValueError(
            f"{topics_path} is missing required columns: {', '.join(missing)}"
        )

    topics_df["topic_id"] = topics_df["topic_id"].map(normalize_whitespace)
    topics_df["pathogen_domain"] = topics_df["pathogen_domain"].map(normalize_whitespace)
    topics_df["topic_label"] = topics_df["topic_label"].map(normalize_whitespace)

    if selected_topics:
        topics_df = topics_df[topics_df["topic_id"].isin(selected_topics)].copy()
    if topics_df.empty:
        raise ValueError("No experiment topics matched the requested filter.")
    return topics_df.reset_index(drop=True)


def country_for_variant(language_code: str, requested_variant: str) -> str:
    normalized = normalize_whitespace(requested_variant)
    if normalized in SPECIAL_VARIANT_COUNTRY:
        return SPECIAL_VARIANT_COUNTRY[normalized]
    if "-" in normalized:
        suffix = normalized.rsplit("-", 1)[-1]
        if len(suffix) == 2 and suffix.isalpha():
            return suffix.upper()
    return DEFAULT_BASE_COUNTRY[language_code]


def build_accept_language(
    language_code: str,
    requested_variant: str,
    all_variants: list[str],
) -> str:
    ordered: list[str] = []
    for value in [requested_variant, *all_variants]:
        normalized = normalize_whitespace(value)
        if normalized and normalized not in ordered:
            ordered.append(normalized)

    parts: list[str] = []
    for idx, value in enumerate(ordered):
        if idx == 0:
            parts.append(value)
            continue
        quality = max(0.4, 1.0 - (0.1 * idx))
        parts.append(f"{value};q={quality:.1f}")

    if language_code != "en":
        parts.append("en;q=0.3")
    return ",".join(parts)


def build_variant_specs(config: dict[str, Any], language_codes: list[str]) -> list[VariantSpec]:
    specs: list[VariantSpec] = []
    for language_code in language_codes:
        spec = language_spec_for(config, language_code)
        all_variants = [str(value) for value in spec.get("language_variants", []) if str(value).strip()]
        for rank, requested_variant in enumerate(all_variants, start=1):
            specs.append(
                VariantSpec(
                    language_code=language_code,
                    requested_variant=requested_variant,
                    variant_rank=rank,
                    request_lang=requested_variant,
                    request_country=country_for_variant(language_code, requested_variant),
                    request_accept_language=build_accept_language(
                        language_code=language_code,
                        requested_variant=requested_variant,
                        all_variants=all_variants,
                    ),
                )
            )
    return specs


def parse_response_locale(response_url: str) -> dict[str, str]:
    text = normalize_whitespace(response_url)
    if not text:
        return {
            "response_hl": "",
            "response_gl": "",
            "response_ceid": "",
        }
    query = parse_qs(urlparse(text).query)
    return {
        "response_hl": normalize_whitespace("".join(query.get("hl", []))),
        "response_gl": normalize_whitespace("".join(query.get("gl", []))),
        "response_ceid": normalize_whitespace("".join(query.get("ceid", []))),
    }


def article_key(row: pd.Series) -> str:
    for field_name in ["guid", "google_news_redirect_url"]:
        value = normalize_whitespace(str(row.get(field_name, "") or ""))
        if value:
            return stable_hash(f"{field_name}::{value}")
    fallback = "||".join(
        [
            normalize_whitespace(str(row.get("rss_title", "") or "")),
            normalize_whitespace(
                str(row.get("rss_pubdate_utc", "") or row.get("rss_pubdate", "") or "")
            ),
            normalize_whitespace(str(row.get("source_hint", "") or "")),
        ]
    )
    return stable_hash(f"fallback::{fallback}")


def canonical_variant_label(response_hl: str, response_gl: str) -> str:
    if response_hl and response_gl:
        return f"{response_hl}|{response_gl}"
    return response_hl or response_gl or ""


def run_payload_cache_path(
    output_dir: Path,
    topic_id: str,
    language_code: str,
    requested_variant: str,
    year: int,
) -> Path:
    return (
        output_dir
        / "raw"
        / topic_id
        / language_code
        / requested_variant.lower().replace("-", "_")
        / f"{year}.json"
    )


def build_client_cache_key(variant: VariantSpec) -> tuple[str, str, str]:
    return (
        variant.request_lang,
        variant.request_country,
        variant.request_accept_language,
    )


def get_client(
    *,
    client_cache: dict[tuple[str, str, str], InstrumentedGoogleNews],
    variant: VariantSpec,
    timeout_seconds: int,
    user_agent: str,
) -> InstrumentedGoogleNews:
    cache_key = build_client_cache_key(variant)
    if cache_key in client_cache:
        return client_cache[cache_key]

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Language": variant.request_accept_language,
        }
    )
    client = InstrumentedGoogleNews(
        lang=variant.request_lang,
        country=variant.request_country,
        session=session,
        timeout_seconds=timeout_seconds,
    )
    client_cache[cache_key] = client
    return client


def fetch_payload(
    *,
    client: InstrumentedGoogleNews,
    topic_row: pd.Series,
    variant: VariantSpec,
    year: int,
    max_retries: int,
    retry_backoff_seconds: float,
    max_backoff_seconds: float,
    logger: Any,
) -> dict[str, Any]:
    window_start = f"{year}-01-01"
    window_end = f"{year}-12-31"
    search_string = normalize_whitespace(str(topic_row[f"search_string_{variant.language_code}"]))
    last_error = ""
    last_result: dict[str, Any] = {}

    for attempt in range(1, max_retries + 1):
        started_at = time.monotonic()
        try:
            result = client.search_window(
                query=search_string,
                from_=window_start,
                to_=window_end,
                proxy_backend="direct",
                proxies=None,
                scraping_bee_api_key="",
            )
            elapsed = round(time.monotonic() - started_at, 3)
            throttled = detect_throttling(
                http_status=result.get("http_status"),
                response_text=result.get("response_text"),
                response_url=result.get("response_url"),
            )
            entry_count = len(((result.get("payload") or {}).get("entries") or []))
            status = "success" if entry_count else "empty"
            if throttled:
                status = "throttled"
            last_result = {
                "status": status,
                "error_summary": "",
                "attempts": attempt,
                "elapsed_request_seconds": elapsed,
                "http_status": int(result.get("http_status") or 0),
                "feed_url": str(result.get("feed_url") or ""),
                "response_url": str(result.get("response_url") or ""),
                "feed": ((result.get("payload") or {}).get("feed") or {}),
                "entries": ((result.get("payload") or {}).get("entries") or []),
            }
            if not throttled:
                return last_result
        except Exception as exc:
            elapsed = round(time.monotonic() - started_at, 3)
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Request failed for %s | %s | %s | %s | attempt=%s | %s",
                topic_row["topic_id"],
                variant.language_code,
                variant.requested_variant,
                year,
                attempt,
                last_error,
            )
            last_result = {
                "status": "failed",
                "error_summary": last_error,
                "attempts": attempt,
                "elapsed_request_seconds": elapsed,
                "http_status": 0,
                "feed_url": "",
                "response_url": "",
                "feed": {},
                "entries": [],
            }

        if attempt < max_retries:
            backoff = compute_backoff_seconds(
                attempt_number=attempt,
                base_seconds=retry_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
            )
            logger.info(
                "Backing off %.1fs for %s | %s | %s | %s",
                backoff,
                topic_row["topic_id"],
                variant.language_code,
                variant.requested_variant,
                year,
            )
            time.sleep(backoff)

    if not last_result:
        last_result = {
            "status": "failed",
            "error_summary": last_error,
            "attempts": max_retries,
            "elapsed_request_seconds": 0.0,
            "http_status": 0,
            "feed_url": "",
            "response_url": "",
            "feed": {},
            "entries": [],
        }
    return last_result


def build_run_record(
    *,
    topic_row: pd.Series,
    variant: VariantSpec,
    year: int,
    payload: dict[str, Any],
    cached_path: Path,
) -> dict[str, Any]:
    locale = parse_response_locale(str(payload.get("response_url", "") or ""))
    record = {
        "topic_id": topic_row["topic_id"],
        "topic_label": topic_row["topic_label"],
        "pathogen_domain": topic_row["pathogen_domain"],
        "language_code": variant.language_code,
        "language_label": LANGUAGE_LABELS[variant.language_code],
        "year": int(year),
        "window_start": f"{year}-01-01",
        "window_end": f"{year}-12-31",
        "requested_variant": variant.requested_variant,
        "variant_rank": int(variant.variant_rank),
        "request_lang": variant.request_lang,
        "request_country": variant.request_country,
        "request_accept_language": variant.request_accept_language,
        "search_string": normalize_whitespace(
            str(topic_row[f"search_string_{variant.language_code}"])
        ),
        "status": str(payload.get("status") or ""),
        "attempts": int(payload.get("attempts") or 0),
        "elapsed_request_seconds": float(payload.get("elapsed_request_seconds") or 0.0),
        "http_status": int(payload.get("http_status") or 0),
        "feed_url": str(payload.get("feed_url") or ""),
        "response_url": str(payload.get("response_url") or ""),
        "entry_count": int(len(payload.get("entries") or [])),
        "error_summary": str(payload.get("error_summary") or ""),
        "cached_path": str(cached_path.relative_to(REPO_ROOT)),
        **locale,
    }
    record["resolved_variant"] = canonical_variant_label(
        response_hl=record["response_hl"],
        response_gl=record["response_gl"],
    )
    return record


def parse_article_rows(
    *,
    topic_row: pd.Series,
    variant: VariantSpec,
    year: int,
    payload: dict[str, Any],
    cached_path: Path,
) -> pd.DataFrame:
    parser_payload = {
        "dataset_key": f"{topic_row['topic_id']}_{variant.language_code}",
        "pathogen_domain": topic_row["pathogen_domain"],
        "language_code": variant.language_code,
        "request_locale_id": variant.variant_id,
        "request_lang": variant.request_lang,
        "request_country": variant.request_country,
        "search_string": normalize_whitespace(
            str(topic_row[f"search_string_{variant.language_code}"])
        ),
        "window_start": f"{year}-01-01",
        "window_end": f"{year}-12-31",
        "retrieved_at": str(payload.get("retrieved_at") or utc_now_iso()),
        "feed_url": str(payload.get("feed_url") or ""),
        "feed": payload.get("feed") or {},
        "entries": payload.get("entries") or [],
    }
    df = parse_cached_payload(
        payload=parser_payload,
        cached_path=str(cached_path.relative_to(REPO_ROOT)),
    )
    if df.empty:
        return df

    locale = parse_response_locale(str(payload.get("response_url") or ""))
    df = df.copy()
    df["topic_id"] = topic_row["topic_id"]
    df["topic_label"] = topic_row["topic_label"]
    df["pathogen_domain"] = topic_row["pathogen_domain"]
    df["year"] = int(year)
    df["language_label"] = LANGUAGE_LABELS[variant.language_code]
    df["requested_variant"] = variant.requested_variant
    df["variant_rank"] = int(variant.variant_rank)
    df["request_accept_language"] = variant.request_accept_language
    df["response_url"] = str(payload.get("response_url") or "")
    df["response_hl"] = locale["response_hl"]
    df["response_gl"] = locale["response_gl"]
    df["response_ceid"] = locale["response_ceid"]
    df["resolved_variant"] = canonical_variant_label(
        response_hl=locale["response_hl"],
        response_gl=locale["response_gl"],
    )
    df["article_key"] = df.apply(article_key, axis=1)
    return df


def summarize_group(
    *,
    group_key: dict[str, Any],
    run_slice: pd.DataFrame,
    article_slice: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    variant_order = (
        run_slice[["requested_variant", "variant_rank"]]
        .drop_duplicates()
        .sort_values(["variant_rank", "requested_variant"], kind="stable")
    )
    variants = variant_order["requested_variant"].tolist()
    article_sets: dict[str, set[str]] = {}
    for requested_variant in variants:
        subset = article_slice[article_slice["requested_variant"] == requested_variant]
        article_sets[requested_variant] = set(subset["article_key"].tolist())

    sum_variant_articles = int(sum(len(values) for values in article_sets.values()))
    all_articles = set().union(*article_sets.values()) if article_sets else set()
    intersection = set(all_articles)
    if article_sets:
        for values in article_sets.values():
            intersection &= values

    memberships = (
        article_slice.groupby("article_key")["requested_variant"]
        .agg(lambda values: tuple(sorted(set(values), key=variants.index)))
        .reset_index(name="membership")
        if not article_slice.empty
        else pd.DataFrame(columns=["article_key", "membership"])
    )
    membership_sizes = memberships["membership"].map(len) if not memberships.empty else pd.Series(dtype=int)
    exclusive_articles = int((membership_sizes == 1).sum())
    shared_articles_any = int((membership_sizes >= 2).sum())

    pairwise_rows: list[dict[str, Any]] = []
    jaccard_values: list[float] = []
    for variant_a, variant_b in combinations(variants, 2):
        articles_a = article_sets[variant_a]
        articles_b = article_sets[variant_b]
        union_articles = articles_a | articles_b
        shared_articles = articles_a & articles_b
        union_count = len(union_articles)
        shared_count = len(shared_articles)
        overlap_coefficient = (
            shared_count / min(len(articles_a), len(articles_b))
            if min(len(articles_a), len(articles_b)) > 0
            else 0.0
        )
        jaccard = (shared_count / union_count) if union_count > 0 else 0.0
        pairwise_rows.append(
            {
                **group_key,
                "variant_a": variant_a,
                "variant_b": variant_b,
                "articles_a": len(articles_a),
                "articles_b": len(articles_b),
                "shared_articles": shared_count,
                "union_articles": union_count,
                "jaccard": round(jaccard, 6),
                "overlap_coefficient": round(overlap_coefficient, 6),
            }
        )
        jaccard_values.append(jaccard)

    pattern_rows: list[dict[str, Any]] = []
    if not memberships.empty:
        pattern_counts = (
            memberships["membership"]
            .value_counts()
            .rename_axis("membership")
            .reset_index(name="article_count")
        )
        for _, row in pattern_counts.iterrows():
            members = list(row["membership"])
            pattern_rows.append(
                {
                    **group_key,
                    "membership_pattern": " + ".join(members) if members else "(none)",
                    "variant_count": len(members),
                    "article_count": int(row["article_count"]),
                }
            )

    variant_sizes = {variant: len(values) for variant, values in article_sets.items()}
    best_single_variant = ""
    best_single_articles = 0
    if variant_sizes:
        best_single_variant = max(
            variant_sizes.items(),
            key=lambda item: (item[1], -variants.index(item[0])),
        )[0]
        best_single_articles = int(variant_sizes[best_single_variant])

    duplicate_mentions = max(0, sum_variant_articles - len(all_articles))
    summary_row = {
        **group_key,
        "requested_variant_count": int(len(variants)),
        "variants_with_articles": int(sum(1 for values in article_sets.values() if values)),
        "sum_variant_articles": sum_variant_articles,
        "union_articles": int(len(all_articles)),
        "duplicate_mentions": duplicate_mentions,
        "duplicate_share": round(
            duplicate_mentions / sum_variant_articles, 6
        ) if sum_variant_articles else 0.0,
        "shared_all_variants_articles": int(len(intersection)) if variants else 0,
        "shared_any_variants_articles": shared_articles_any,
        "exclusive_articles": exclusive_articles,
        "best_single_variant": best_single_variant,
        "best_single_articles": best_single_articles,
        "incremental_articles_vs_best_single": int(len(all_articles) - best_single_articles),
        "incremental_ratio_vs_best_single": round(
            (len(all_articles) - best_single_articles) / best_single_articles,
            6,
        ) if best_single_articles else 0.0,
        "pairwise_mean_jaccard": round(
            sum(jaccard_values) / len(jaccard_values),
            6,
        ) if jaccard_values else 0.0,
        "pairwise_max_jaccard": round(max(jaccard_values), 6) if jaccard_values else 0.0,
        "pairwise_min_jaccard": round(min(jaccard_values), 6) if jaccard_values else 0.0,
    }
    return summary_row, pairwise_rows, pattern_rows


def build_summaries(
    *,
    runs_df: pd.DataFrame,
    article_df: pd.DataFrame,
    group_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []

    group_keys = (
        runs_df[group_columns]
        .drop_duplicates()
        .sort_values(group_columns, kind="stable")
    )

    for _, group_row in group_keys.iterrows():
        group_key = {column: group_row[column] for column in group_columns}
        mask = pd.Series(True, index=runs_df.index)
        article_mask = pd.Series(True, index=article_df.index)
        for column, value in group_key.items():
            mask &= runs_df[column] == value
            article_mask &= article_df[column] == value
        summary_row, group_pairwise_rows, group_pattern_rows = summarize_group(
            group_key=group_key,
            run_slice=runs_df.loc[mask].copy(),
            article_slice=article_df.loc[article_mask].copy(),
        )
        summary_rows.append(summary_row)
        pairwise_rows.extend(group_pairwise_rows)
        pattern_rows.extend(group_pattern_rows)

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(pairwise_rows),
        pd.DataFrame(pattern_rows),
    )


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    paths = project_paths(config)
    output_dir = (
        REPO_ROOT / args.output_dir
        if args.output_dir
        else paths.data_root / "experiments" / "language_variant_overlap"
    )
    ensure_dir(output_dir)
    logger = setup_logger(
        "language_variant_experiment",
        output_dir / "language_variant_experiment.log",
    )

    selected_languages = parse_csv_list(args.languages) or ["en", "fr", "es", "pt"]
    selected_topics = parse_csv_list(args.topics_filter)
    years = list(range(int(args.start_year), int(args.end_year) + 1))

    topics_path = Path(args.topics)
    if not topics_path.is_absolute():
        topics_path = (REPO_ROOT / topics_path).resolve()
    topics_df = load_topics(topics_path, selected_topics)
    variants = build_variant_specs(config, selected_languages)
    user_agent = str(config.get("rss", {}).get("user_agent", "Mozilla/5.0"))

    client_cache: dict[tuple[str, str, str], InstrumentedGoogleNews] = {}
    run_records: list[dict[str, Any]] = []
    article_frames: list[pd.DataFrame] = []

    total_runs = len(topics_df) * len(variants) * len(years)
    logger.info(
        "Starting language variant experiment | topics=%s | variants=%s | years=%s | runs=%s",
        len(topics_df),
        len(variants),
        len(years),
        total_runs,
    )

    for _, topic_row in topics_df.iterrows():
        for variant in variants:
            for year in years:
                cache_path = run_payload_cache_path(
                    output_dir=output_dir,
                    topic_id=str(topic_row["topic_id"]),
                    language_code=variant.language_code,
                    requested_variant=variant.requested_variant,
                    year=year,
                )
                ensure_dir(cache_path.parent)

                if cache_path.exists() and not args.force:
                    payload = json.loads(cache_path.read_text(encoding="utf-8"))
                else:
                    client = get_client(
                        client_cache=client_cache,
                        variant=variant,
                        timeout_seconds=int(args.timeout_seconds),
                        user_agent=user_agent,
                    )
                    fetched_payload = fetch_payload(
                        client=client,
                        topic_row=topic_row,
                        variant=variant,
                        year=year,
                        max_retries=int(args.max_retries),
                        retry_backoff_seconds=float(args.retry_backoff_seconds),
                        max_backoff_seconds=float(args.max_backoff_seconds),
                        logger=logger,
                    )
                    payload = {
                        "retrieved_at": utc_now_iso(),
                        **fetched_payload,
                    }
                    write_json_atomic(payload, cache_path)
                    if float(args.sleep_seconds) > 0:
                        time.sleep(float(args.sleep_seconds))

                run_records.append(
                    build_run_record(
                        topic_row=topic_row,
                        variant=variant,
                        year=year,
                        payload=payload,
                        cached_path=cache_path,
                    )
                )
                article_frame = parse_article_rows(
                    topic_row=topic_row,
                    variant=variant,
                    year=year,
                    payload=payload,
                    cached_path=cache_path,
                )
                if not article_frame.empty:
                    article_frames.append(article_frame)

    runs_df = pd.DataFrame(run_records).sort_values(
        ["topic_id", "language_code", "year", "variant_rank", "requested_variant"],
        kind="stable",
    )
    if article_frames:
        article_df = pd.concat(article_frames, ignore_index=True)
        article_df = article_df.drop_duplicates(
            subset=[
                "topic_id",
                "language_code",
                "year",
                "requested_variant",
                "article_key",
            ]
        ).reset_index(drop=True)
    else:
        article_df = pd.DataFrame()

    if article_df.empty:
        article_df = pd.DataFrame(
            columns=[
                "topic_id",
                "topic_label",
                "pathogen_domain",
                "year",
                "language_code",
                "language_label",
                "requested_variant",
                "variant_rank",
                "article_key",
            ]
        )

    variant_counts_df = (
        runs_df[
            [
                "topic_id",
                "topic_label",
                "pathogen_domain",
                "language_code",
                "language_label",
                "year",
                "requested_variant",
                "variant_rank",
                "response_hl",
                "response_gl",
                "response_ceid",
                "resolved_variant",
            ]
        ]
        .drop_duplicates()
        .merge(
            article_df.groupby(
                ["topic_id", "language_code", "year", "requested_variant"],
                as_index=False,
            ).size().rename(columns={"size": "article_count"}),
            on=["topic_id", "language_code", "year", "requested_variant"],
            how="left",
        )
        .fillna({"article_count": 0})
    )
    variant_counts_df["article_count"] = variant_counts_df["article_count"].astype(int)

    yearly_summary_df, yearly_pairwise_df, yearly_patterns_df = build_summaries(
        runs_df=runs_df,
        article_df=article_df,
        group_columns=["topic_id", "topic_label", "pathogen_domain", "language_code", "language_label", "year"],
    )
    overall_summary_df, overall_pairwise_df, overall_patterns_df = build_summaries(
        runs_df=runs_df,
        article_df=article_df,
        group_columns=["topic_id", "topic_label", "pathogen_domain", "language_code", "language_label"],
    )

    resolution_df = (
        runs_df.groupby(
            [
                "language_code",
                "language_label",
                "requested_variant",
                "response_hl",
                "response_gl",
                "response_ceid",
                "resolved_variant",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            years_queried=("year", "nunique"),
            total_entry_mentions=("entry_count", "sum"),
            success_runs=("status", lambda values: int(sum(value in {"success", "empty"} for value in values))),
        )
        .sort_values(
            ["language_code", "requested_variant", "response_hl", "response_gl"],
            kind="stable",
        )
    )

    metadata = {
        "generated_at": utc_now_iso(),
        "config_path": str(Path(config["_config_path"]).resolve()),
        "topics_path": str(topics_path),
        "output_dir": str(output_dir.resolve()),
        "start_year": int(args.start_year),
        "end_year": int(args.end_year),
        "languages": selected_languages,
        "topics": topics_df["topic_id"].tolist(),
        "notes": [
            "Overlap is computed on a stable article key derived from Google News article GUIDs or redirect URLs.",
            "Google News can canonicalize requested locale variants into different resolved hl/gl/ceid combinations; those resolved values are recorded separately.",
            "Base variants without an explicit country are issued with default countries: en->US, fr->FR, es->ES, pt->BR.",
            "The es-419 variant is issued with gl=MX because Google News RSS requires a concrete country code in the request URL.",
        ],
    }

    write_dataframe_atomic(
        runs_df,
        output_dir / "variant_runs.csv",
    )
    write_dataframe_atomic(
        article_df,
        output_dir / "variant_article_occurrences.csv",
    )
    write_dataframe_atomic(
        variant_counts_df,
        output_dir / "variant_counts_by_year.csv",
    )
    write_dataframe_atomic(
        yearly_summary_df.sort_values(
            ["topic_id", "language_code", "year"],
            kind="stable",
        ),
        output_dir / "overlap_yearly_summary.csv",
    )
    write_dataframe_atomic(
        yearly_pairwise_df.sort_values(
            ["topic_id", "language_code", "year", "variant_a", "variant_b"],
            kind="stable",
        ),
        output_dir / "overlap_yearly_pairwise.csv",
    )
    write_dataframe_atomic(
        yearly_patterns_df.sort_values(
            ["topic_id", "language_code", "year", "article_count", "membership_pattern"],
            ascending=[True, True, True, False, True],
            kind="stable",
        ),
        output_dir / "overlap_yearly_membership_patterns.csv",
    )
    write_dataframe_atomic(
        overall_summary_df.sort_values(
            ["topic_id", "language_code"],
            kind="stable",
        ),
        output_dir / "overlap_total_summary.csv",
    )
    write_dataframe_atomic(
        overall_pairwise_df.sort_values(
            ["topic_id", "language_code", "variant_a", "variant_b"],
            kind="stable",
        ),
        output_dir / "overlap_total_pairwise.csv",
    )
    write_dataframe_atomic(
        overall_patterns_df.sort_values(
            ["topic_id", "language_code", "article_count", "membership_pattern"],
            ascending=[True, True, False, True],
            kind="stable",
        ),
        output_dir / "overlap_total_membership_patterns.csv",
    )
    write_dataframe_atomic(
        resolution_df,
        output_dir / "variant_resolution_summary.csv",
    )
    write_json_atomic(metadata, output_dir / "experiment_metadata.json")

    logger.info(
        "Finished language variant experiment | runs=%s | article_rows=%s | output_dir=%s",
        len(runs_df),
        len(article_df),
        output_dir,
    )


if __name__ == "__main__":
    main()
