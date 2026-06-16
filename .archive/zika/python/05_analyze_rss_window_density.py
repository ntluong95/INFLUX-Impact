from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_WINDOWS = (1, 3, 7, 14, 30)


@dataclass(frozen=True)
class DailyCount:
    day: date
    count: int
    source_file: str


@dataclass(frozen=True)
class WindowSummary:
    window_days: int
    requests_if_used: int
    request_reduction_pct: float
    average_total: float
    median_total: float
    p90_total: float
    max_total: int
    avg_share_ge_warn: float
    best_share_ge_warn: float
    worst_share_ge_warn: float
    avg_share_ge_cap: float
    best_share_ge_cap: float
    worst_share_ge_cap: float
    recommendation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze cached daily Google RSS retrieval windows and estimate whether "
            "broader windows would likely reduce requests without triggering feed caps."
        )
    )
    parser.add_argument(
        "--raw-dir",
        default="zika/data/raw/rss",
        help="Directory containing cached RSS JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="zika/data/intermediate/rss_window_analysis",
        help="Directory for CSV summaries.",
    )
    parser.add_argument(
        "--candidate-windows",
        default="1,3,7,14,30",
        help="Comma-separated list of candidate window sizes in days.",
    )
    parser.add_argument(
        "--warn-threshold",
        type=int,
        default=80,
        help="Window totals at or above this value are treated as near-cap.",
    )
    parser.add_argument(
        "--cap-threshold",
        type=int,
        default=100,
        help="Window totals at or above this value are treated as cap-risk.",
    )
    return parser.parse_args()


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def parse_candidate_windows(raw_value: str) -> list[int]:
    windows = []
    for chunk in raw_value.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value < 1:
            raise ValueError("Candidate window sizes must be positive integers.")
        windows.append(value)
    if not windows:
        raise ValueError("At least one candidate window size is required.")
    return sorted(set(windows))


def load_daily_counts(raw_dir: Path) -> list[DailyCount]:
    counts_by_day: dict[date, DailyCount] = {}
    for path in sorted(raw_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        window_start = payload.get("window_start")
        window_end = payload.get("window_end")
        if not window_start or not window_end or window_start != window_end:
            continue

        day = date.fromisoformat(str(window_start))
        entry_count = len(payload.get("entries", []))
        counts_by_day[day] = DailyCount(
            day=day,
            count=entry_count,
            source_file=path.name,
        )

    if not counts_by_day:
        raise FileNotFoundError(f"No daily JSON files found in {raw_dir}")

    start_day = min(counts_by_day)
    end_day = max(counts_by_day)
    dense_counts: list[DailyCount] = []
    current = start_day
    while current <= end_day:
        dense_counts.append(
            counts_by_day.get(
                current,
                DailyCount(day=current, count=0, source_file=""),
            )
        )
        current += timedelta(days=1)
    return dense_counts


def block_totals(series: list[DailyCount], window_days: int, offset: int) -> list[int]:
    totals: list[int] = []
    cursor = offset
    while cursor < len(series):
        total = 0
        for step in range(window_days):
            index = cursor + step
            if index >= len(series):
                break
            total += series[index].count
        totals.append(total)
        cursor += window_days
    return totals


def classify_candidate(avg_share_ge_cap: float, avg_share_ge_warn: float) -> str:
    if avg_share_ge_cap <= 0.01 and avg_share_ge_warn <= 0.03:
        return "recommended"
    if avg_share_ge_cap <= 0.05 and avg_share_ge_warn <= 0.08:
        return "use_with_caution"
    return "not_recommended"


def summarize_window(
    series: list[DailyCount],
    window_days: int,
    warn_threshold: int,
    cap_threshold: int,
) -> WindowSummary:
    totals_by_offset: list[list[int]] = []
    warn_rates: list[float] = []
    cap_rates: list[float] = []

    for offset in range(window_days):
        totals = block_totals(series, window_days=window_days, offset=offset)
        if not totals:
            continue
        totals_by_offset.append(totals)
        warn_rates.append(sum(total >= warn_threshold for total in totals) / len(totals))
        cap_rates.append(sum(total >= cap_threshold for total in totals) / len(totals))

    flattened = [total for totals in totals_by_offset for total in totals]
    requests_if_used = math.ceil(len(series) / window_days)
    recommendation = classify_candidate(mean(cap_rates), mean(warn_rates))

    return WindowSummary(
        window_days=window_days,
        requests_if_used=requests_if_used,
        request_reduction_pct=100.0 * (1.0 - (requests_if_used / len(series))),
        average_total=mean(flattened),
        median_total=float(median(flattened)),
        p90_total=percentile(flattened, 0.90),
        max_total=max(flattened),
        avg_share_ge_warn=mean(warn_rates),
        best_share_ge_warn=min(warn_rates),
        worst_share_ge_warn=max(warn_rates),
        avg_share_ge_cap=mean(cap_rates),
        best_share_ge_cap=min(cap_rates),
        worst_share_ge_cap=max(cap_rates),
        recommendation=recommendation,
    )


def write_daily_csv(output_path: Path, series: Iterable[DailyCount]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["day", "retrieved_items", "source_file"],
        )
        writer.writeheader()
        for row in series:
            writer.writerow(
                {
                    "day": row.day.isoformat(),
                    "retrieved_items": row.count,
                    "source_file": row.source_file,
                }
            )


def write_summary_csv(output_path: Path, summaries: Iterable[WindowSummary]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "window_days",
                "requests_if_used",
                "request_reduction_pct",
                "average_total",
                "median_total",
                "p90_total",
                "max_total",
                "avg_share_ge_warn",
                "best_share_ge_warn",
                "worst_share_ge_warn",
                "avg_share_ge_cap",
                "best_share_ge_cap",
                "worst_share_ge_cap",
                "recommendation",
            ],
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow(
                {
                    "window_days": row.window_days,
                    "requests_if_used": row.requests_if_used,
                    "request_reduction_pct": round(row.request_reduction_pct, 2),
                    "average_total": round(row.average_total, 2),
                    "median_total": round(row.median_total, 2),
                    "p90_total": round(row.p90_total, 2),
                    "max_total": row.max_total,
                    "avg_share_ge_warn": round(row.avg_share_ge_warn, 4),
                    "best_share_ge_warn": round(row.best_share_ge_warn, 4),
                    "worst_share_ge_warn": round(row.worst_share_ge_warn, 4),
                    "avg_share_ge_cap": round(row.avg_share_ge_cap, 4),
                    "best_share_ge_cap": round(row.best_share_ge_cap, 4),
                    "worst_share_ge_cap": round(row.worst_share_ge_cap, 4),
                    "recommendation": row.recommendation,
                }
            )


def build_recommendation_text(summaries: list[WindowSummary]) -> str:
    recommended = [summary for summary in summaries if summary.recommendation == "recommended"]
    cautious = [summary for summary in summaries if summary.recommendation == "use_with_caution"]

    if recommended:
        chosen = max(recommended, key=lambda summary: summary.window_days)
        return (
            f"Recommendation: use {chosen.window_days}-day windows as the default compromise. "
            f"They would cut requests by about {chosen.request_reduction_pct:.1f}% while "
            f"keeping the estimated cap-risk at {chosen.avg_share_ge_cap:.2%} on average."
        )

    if cautious:
        chosen = min(cautious, key=lambda summary: summary.avg_share_ge_cap)
        return (
            f"Recommendation: keep 1-day windows as the safest default. "
            f"If request volume becomes a bottleneck, {chosen.window_days}-day windows are the "
            f"least risky broader option, but they still show an estimated cap-risk of "
            f"{chosen.avg_share_ge_cap:.2%}."
        )

    return (
        "Recommendation: keep 1-day windows. The broader candidates all show too much "
        "estimated cap-risk relative to the observed daily archive."
    )


def main() -> None:
    args = parse_args()
    raw_dir = (REPO_ROOT / args.raw_dir).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    candidate_windows = parse_candidate_windows(args.candidate_windows)

    series = load_daily_counts(raw_dir)
    summaries = [
        summarize_window(
            series=series,
            window_days=window_days,
            warn_threshold=args.warn_threshold,
            cap_threshold=args.cap_threshold,
        )
        for window_days in candidate_windows
    ]

    daily_counts = [row.count for row in series]
    daily_csv = output_dir / "daily_window_counts.csv"
    summary_csv = output_dir / "candidate_window_summary.csv"
    write_daily_csv(daily_csv, series)
    write_summary_csv(summary_csv, summaries)

    missing_days = sum(1 for row in series if not row.source_file)

    print("Zika RSS window-density analysis")
    print(f"Raw directory: {raw_dir}")
    print(f"Observed days: {len(series)}")
    print(f"Missing cached days backfilled as zero: {missing_days}")
    print(
        "Daily counts: "
        f"mean={mean(daily_counts):.2f}, median={median(daily_counts):.2f}, "
        f"p90={percentile(daily_counts, 0.90):.2f}, p99={percentile(daily_counts, 0.99):.2f}, "
        f"max={max(daily_counts)}"
    )
    print(
        "High-volume days: "
        f">=20={sum(count >= 20 for count in daily_counts)}, "
        f">=50={sum(count >= 50 for count in daily_counts)}, "
        f">=80={sum(count >= 80 for count in daily_counts)}, "
        f">=100={sum(count >= 100 for count in daily_counts)}"
    )
    print("")
    print("Candidate windows")
    for summary in summaries:
        print(
            f"- {summary.window_days:>2} day: "
            f"requests={summary.requests_if_used}, "
            f"reduction={summary.request_reduction_pct:>5.1f}%, "
            f"mean_total={summary.average_total:>6.2f}, "
            f"p90_total={summary.p90_total:>6.2f}, "
            f"share>={args.warn_threshold}={summary.avg_share_ge_warn:>6.2%}, "
            f"share>={args.cap_threshold}={summary.avg_share_ge_cap:>6.2%}, "
            f"status={summary.recommendation}"
        )
    print("")
    print(build_recommendation_text(summaries))
    print(f"Daily CSV: {daily_csv}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
