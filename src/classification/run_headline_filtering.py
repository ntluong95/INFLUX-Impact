from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.domain_filter import filter_dataset
from src.classification.headline_batch import run_headline_batch_filtering
from src.utils.common import setup_logger
from src.utils.config import load_project_config, project_paths
from src.utils.project import resolve_datasets, resolve_languages, resolve_pathogen_domains


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run domain filtering plus batch headline relevance filtering (OpenAI or Anthropic)."
    )
    parser.add_argument("--config", default="src/config/pipeline.yaml")
    parser.add_argument("--pathogen-domains", default="all")
    parser.add_argument("--languages", default="all")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    paths = project_paths(config)
    logger = setup_logger(
        "pathogen_headline_filter",
        paths.logs_dir() / "headline_filter.log",
    )
    datasets = resolve_datasets(
        resolve_pathogen_domains(args.pathogen_domains),
        resolve_languages(args.languages),
    )
    for dataset in datasets:
        filter_dataset(dataset, config, args.force, logger)
        run_headline_batch_filtering(dataset, config, args.force, logger)


if __name__ == "__main__":
    main()
