"""End-to-end pipeline: inspect, dedupe, fetch, extract, classify."""

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
from src.classification.inspect_csv import inspect_csv
from src.classification.llm_backends import OpenAIChatClient
from src.cleaning.deduplicate_urls import deduplicate_feed_urls
from src.cleaning.extract_text import build_extracted_dataset, save_extracted_dataset
from src.cleaning.fetch_html import fetch_html_for_urls
from src.utils.common import ensure_dir, setup_logger


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_extracted_if_exists(outdir: Path) -> pd.DataFrame | None:
    parquet_path = outdir / "articles_extracted.parquet"
    csv_path = outdir / "articles_extracted.csv"

    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass

    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        # TODO: update
        description="Run Zika 2015 processing + classification pipeline."
    )
    parser.add_argument(
        "--input",
        default="data/Zika_News_2015.csv",
        help="Input RSS CSV path",
    )
    parser.add_argument(
        "--outdir",
        default="data/processed",
        help="Output directory for processed artifacts",
    )
    parser.add_argument(
        "--config",
        default="src/classification/config.yaml",
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even when output artifacts already exist",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    config_path = Path(args.config)

    cfg = _load_config(config_path)
    paths_cfg = cfg.get("paths", {})

    url_registry_db = Path(paths_cfg.get("url_registry_db", "data/url_registry.sqlite"))
    raw_html_dir = Path(paths_cfg.get("raw_html_dir", "data/raw_html"))
    logs_dir = Path(paths_cfg.get("logs_dir", "data/logs"))

    ensure_dir(outdir)
    ensure_dir(raw_html_dir)
    ensure_dir(logs_dir)

    logger = setup_logger(logs_dir / "classification_pipeline.log")

    logger.info("Pipeline started")
    logger.info("Input: %s", input_path)
    logger.info("Output directory: %s", outdir)
    logger.info("Config: %s", config_path)

    load_dotenv()

    # NOTE 1) Inspect CSV schema
    inspect_cfg = cfg.get("inspection", {})
    df, schema, _ = inspect_csv(
        input_path=input_path,
        sample_rows=int(inspect_cfg.get("sample_rows", 5)),
        logger=logger,
    )

    # NOTES 2) Canonicalize + dedupe + registry
    dedup_cfg = cfg.get("deduplication", {})
    unique_csv_path = outdir / "zika_2015_unique.csv"
    deduped_df = deduplicate_feed_urls(
        df=df,
        schema=schema,
        db_path=url_registry_db,
        output_path=unique_csv_path,
        source=dedup_cfg.get("source", "rss"),
        tracking_params=dedup_cfg.get("tracking_params"),
        resolve_google_news=bool(dedup_cfg.get("resolve_google_news", True)),
        resolver_timeout_seconds=int(dedup_cfg.get("resolver_timeout_seconds", 20)),
        resolver_workers=int(dedup_cfg.get("resolver_workers", 6)),
        logger=logger,
    )

    # NOTES 3) Fetch HTML (restartable by cached files)
    fetch_cfg = cfg.get("fetch", {})
    fetch_results_path = outdir / "fetch_results.csv"
    previous_fetch_df = None
    if (
        fetch_results_path.exists()
        and not args.force
        and bool(fetch_cfg.get("reuse_previous_results", True))
    ):
        try:
            previous_fetch_df = pd.read_csv(fetch_results_path, low_memory=False)
            logger.info(
                "Loaded previous fetch results for restart optimization: %s rows",
                len(previous_fetch_df),
            )
        except Exception as exc:
            logger.warning(
                "Could not load previous fetch results (%s): %s",
                fetch_results_path,
                exc,
            )

    fetch_df = fetch_html_for_urls(
        urls_df=deduped_df,
        raw_html_dir=raw_html_dir,
        fetch_cfg=fetch_cfg,
        force=args.force,
        previous_results_df=previous_fetch_df,
        logger=logger,
    )
    fetch_df.to_csv(fetch_results_path, index=False)
    logger.info("Fetch results written: %s", fetch_results_path)

    # NOTES 4) Extract text (skip if extracted output exists unless --force)
    extracted_df = None
    if args.force:
        logger.info("--force set: extraction will be recomputed")
    else:
        extracted_df = _load_extracted_if_exists(outdir)
        if extracted_df is not None:
            logger.info(
                "Extraction output exists; skipping extraction (use --force to recompute)"
            )

    if extracted_df is None:
        extraction_cfg = cfg.get("extraction", {})
        extracted_df = build_extracted_dataset(
            unique_df=deduped_df,
            fetch_df=fetch_df,
            raw_html_dir=raw_html_dir,
            extraction_cfg=extraction_cfg,
            logger=logger,
        )
        saved_path = save_extracted_dataset(
            extracted_df,
            outdir=outdir,
            prefer_parquet=bool(extraction_cfg.get("prefer_parquet", True)),
        )
        logger.info("Extraction output written: %s", saved_path)

    # 5) Ensemble classification (skip if output exists unless --force)
    cls_cfg = cfg.get("classification", {})
    classify_enabled = bool(cls_cfg.get("enabled", True))
    classified_path = outdir / "articles_classified.csv"

    if classify_enabled:
        resume_if_exists = bool(cls_cfg.get("resume_if_exists", True))
        if classified_path.exists() and not args.force and not resume_if_exists:
            logger.info(
                "Classification output exists; skipping classification: %s",
                classified_path,
            )
        else:
            if classified_path.exists() and not args.force and resume_if_exists:
                logger.info(
                    "Classification output exists; resuming from partial file: %s",
                    classified_path,
                )
            openai_cfg = cls_cfg.get("openai", {})
            base_url = openai_cfg.get("base_url") or os.getenv("OPENAI_BASE_URL")
            api_key_env = openai_cfg.get("api_key_env", "OPENAI_API_KEY")
            api_key = os.getenv(api_key_env, "")
            if not api_key:
                raise ValueError(
                    f"Missing {api_key_env}. Add it to .env (repo root) before classification."
                )

            client = OpenAIChatClient(
                api_key=api_key,
                base_url=base_url,
                timeout_seconds=int(openai_cfg.get("timeout_seconds", 60)),
                max_retries=int(openai_cfg.get("max_retries", 3)),
                retry_backoff_seconds=float(
                    openai_cfg.get("retry_backoff_seconds", 2.0)
                ),
                rate_limit_per_minute=float(
                    openai_cfg.get("rate_limit_per_minute", 0.0)
                ),
            )

            classify_articles_ensemble(
                extracted_df=extracted_df,
                client=client,
                cls_cfg=cls_cfg,
                output_path=classified_path,
                force=args.force,
                logger=logger,
            )
            logger.info("Classification output written: %s", classified_path)
    else:
        logger.info("Classification disabled in config.")

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
