"""Classification-only runner using existing extracted article content."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.classification.classify_ensemble import classify_articles_ensemble
from src.classification.llm_backends import OpenAIChatClient
from src.utils.common import ensure_dir, setup_logger

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_extracted_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Extracted dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)

    # Fallback by trying parquet then csv.
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run classification-only on existing extracted articles.",
    )
    parser.add_argument(
        "--input_extracted",
        default="data/processed/articles_extracted.parquet",
        help="Existing extracted dataset (parquet/csv)",
    )
    parser.add_argument(
        "--out",
        default="data/processed/articles_classified.csv",
        help="Classification output CSV",
    )
    parser.add_argument(
        "--config",
        default="src/classification/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all rows even if output already exists",
    )
    args = parser.parse_args()

    input_extracted_path = Path(args.input_extracted)
    output_path = Path(args.out)
    config_path = Path(args.config)

    cfg = _load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    logs_dir = Path(paths_cfg.get("logs_dir", "data/logs"))
    ensure_dir(logs_dir)
    ensure_dir(output_path.parent)

    logger = setup_logger(logs_dir / "classification_pipeline.log")

    logger.info("Classification-only run started")
    logger.info("Extracted input: %s", input_extracted_path)
    logger.info("Output CSV: %s", output_path)
    logger.info("Config: %s", config_path)

    load_dotenv()

    extracted_df = _load_extracted_dataset(input_extracted_path)
    logger.info("Loaded extracted rows: %s", len(extracted_df))

    # Keep consistent columns expected by classifier.
    for col in ["canonical_url", "extracted_title", "extracted_text"]:
        if col not in extracted_df.columns:
            extracted_df[col] = ""
            logger.warning(
                "Missing column '%s' in extracted dataset; filled with empty strings.",
                col,
            )

    cls_cfg = cfg.get("classification", {})
    if not bool(cls_cfg.get("enabled", True)):
        logger.info("Classification disabled in config; exiting.")
        return

    openai_cfg = cls_cfg.get("openai", {})
    base_url = openai_cfg.get("base_url") or os.getenv("OPENAI_BASE_URL")
    api_key_env = openai_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise ValueError(
            f"Missing {api_key_env}. Add it to .env before running classification-only mode."
        )

    client = OpenAIChatClient(
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=int(openai_cfg.get("timeout_seconds", 60)),
        max_retries=int(openai_cfg.get("max_retries", 3)),
        retry_backoff_seconds=float(openai_cfg.get("retry_backoff_seconds", 2.0)),
        rate_limit_per_minute=float(openai_cfg.get("rate_limit_per_minute", 0.0)),
    )

    classify_articles_ensemble(
        extracted_df=extracted_df,
        client=client,
        cls_cfg=cls_cfg,
        output_path=output_path,
        force=args.force,
        logger=logger,
    )
    logger.info("Classification output written: %s", output_path)
    logger.info("Classification-only run completed")


if __name__ == "__main__":
    main()
