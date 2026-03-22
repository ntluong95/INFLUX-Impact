from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.domain_filter import filter_dataset
from src.classification.headline_batch import run_headline_batch_filtering
from src.extraction.bertopic_fulltext import fit_dataset
from src.fulltext_retrieval.retrieve_fulltext import retrieve_dataset
from src.rss_retrieval.retrieve_google_rss import (
    retrieve_dataset as retrieve_rss_dataset,
)
from src.utils.common import setup_logger
from src.utils.config import load_project_config, project_paths
from src.utils.project import (
    resolve_datasets,
    resolve_languages,
    resolve_pathogen_domains,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the modular pathogen news pipeline inside src/."
    )
    parser.add_argument("--config", default="src/config/pipeline.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "rss", "classify", "fulltext", "bertopic"],
    )
    parser.add_argument("--pathogen-domains", default="all")
    parser.add_argument("--languages", default="all")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    paths = project_paths(config)
    logger = setup_logger(
        "pathogen_pipeline",
        paths.logs_dir() / "pipeline.log",
    )
    datasets = resolve_datasets(
        resolve_pathogen_domains(args.pathogen_domains),
        resolve_languages(args.languages),
    )

    for dataset in datasets:
        if args.stage in {"all", "rss"}:
            retrieve_rss_dataset(dataset, config, args.force, False, logger)
        if args.stage in {"all", "classify"}:
            filter_dataset(dataset, config, args.force, logger)
            headline_state = run_headline_batch_filtering(
                dataset, config, args.force, logger
            )
            if args.stage == "all" and not headline_state["ready"]:
                logger.info(
                    "Skipping downstream stages for %s because headline batch results are still pending. Re-run later to continue.",
                    dataset.stem,
                )
                continue
        if args.stage in {"all", "fulltext"}:
            retrieve_dataset(dataset, config, args.force, logger)
        if args.stage in {"all", "bertopic"}:
            fit_dataset(dataset, config, args.force, logger)


if __name__ == "__main__":
    main()
