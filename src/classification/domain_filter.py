from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.common import domain_matches_blocklist, ensure_dir, extract_domain, setup_logger, write_dataframe_atomic
from src.utils.config import load_project_config, project_paths
from src.utils.project import DatasetKey, resolve_datasets, resolve_languages, resolve_pathogen_domains


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter RSS headlines using data/domains_removed.csv before OpenAI classification."
    )
    parser.add_argument("--config", default="src/config/pipeline.yaml")
    parser.add_argument("--pathogen-domains", default="all")
    parser.add_argument("--languages", default="all")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_blocked_domains(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing domain filter input: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    if "domain" not in df.columns or "is_news_outlet" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'domain' and 'is_news_outlet' columns.")
    blocked = df.loc[
        df["is_news_outlet"].fillna("").astype(str).str.strip().str.lower() == "no",
        "domain",
    ].fillna("")
    return {extract_domain(value) for value in blocked.tolist() if extract_domain(value)}


def filter_dataset(
    dataset: DatasetKey,
    config: dict[str, Any],
    force: bool,
    logger: Any,
) -> Path:
    paths = project_paths(config)
    input_csv = paths.rss_output_csv(dataset)
    output_csv = paths.prefiltered_headlines_csv(dataset)
    metrics_csv = paths.prefiltered_headlines_metrics_csv(dataset)

    if output_csv.exists() and not force:
        logger.info("Domain-filtered headlines already exist for %s: %s", dataset.stem, output_csv)
        return output_csv

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Missing RSS retrieval output for {dataset.stem}: {input_csv}"
        )

    blocked_domains = load_blocked_domains(
        REPO_ROOT / str(config["classification"]["domains_removed_path"])
    )
    rss_df = pd.read_csv(input_csv, low_memory=False)
    if "source_domain" not in rss_df.columns:
        rss_df["source_domain"] = rss_df.get("source_url", "").map(extract_domain)
    rss_df["domain_blocked"] = rss_df["source_domain"].map(
        lambda value: domain_matches_blocklist(str(value), blocked_domains)
    )
    filtered_df = rss_df.loc[~rss_df["domain_blocked"]].copy()
    filtered_df.drop(columns=["domain_blocked"], inplace=True)

    metrics_df = pd.DataFrame(
        [
            {"dataset_key": dataset.stem, "metric": "input_rows", "value": int(len(rss_df))},
            {
                "dataset_key": dataset.stem,
                "metric": "blocked_rows",
                "value": int(rss_df["domain_blocked"].sum()),
            },
            {"dataset_key": dataset.stem, "metric": "kept_rows", "value": int(len(filtered_df))},
        ]
    )
    ensure_dir(output_csv.parent)
    write_dataframe_atomic(filtered_df, output_csv)
    write_dataframe_atomic(metrics_df, metrics_csv)
    logger.info(
        "Domain filter complete for %s: input_rows=%s blocked_rows=%s kept_rows=%s",
        dataset.stem,
        len(rss_df),
        int(rss_df["domain_blocked"].sum()),
        len(filtered_df),
    )
    return output_csv


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    paths = project_paths(config)
    logger = setup_logger(
        "pathogen_domain_filter",
        paths.logs_dir() / "domain_filter.log",
    )
    datasets = resolve_datasets(
        resolve_pathogen_domains(args.pathogen_domains),
        resolve_languages(args.languages),
    )
    for dataset in datasets:
        filter_dataset(dataset, config, args.force, logger)


if __name__ == "__main__":
    main()
