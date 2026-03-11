from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from rss_utils import (
    RSS_OUTPUT_COLUMNS,
    dedupe_rss_records,
    load_config,
    manifest_path,
    parse_cached_payload,
    setup_logger,
    write_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse cached Google News RSS payloads into normalized stage-1 outputs."
    )
    parser.add_argument("--config", default="zika/config/zika.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()
    zika_root = config_path.parent.parent
    cfg = load_config(config_path)

    log_path = zika_root / "logs" / "01_parse_google_rss.log"
    logger = setup_logger("zika_rss_parse", log_path)

    query = str(cfg.get("query", {}).get("term", "zika"))
    manifest_csv = manifest_path(
        zika_root,
        str(cfg.get("rss", {}).get("manifest_filename", f"{query}_rss_manifest.csv")),
    )
    output_csv = zika_root / "data" / "intermediate" / "zika_rss_raw.csv"
    output_parquet = zika_root / "data" / "intermediate" / "zika_rss_raw.parquet"

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_csv}")

    manifest_df = pd.read_csv(manifest_csv, dtype=str).fillna("")
    done_mask = manifest_df["status"].isin(["success", "empty"])
    completed = manifest_df.loc[done_mask].copy()
    logger.info(
        "Parsing %s cached windows from manifest %s",
        len(completed),
        manifest_csv,
    )

    frames: list[pd.DataFrame] = []
    missing_cache = 0
    for _, row in completed.iterrows():
        cached_rel = str(row["cached_path"] or "")
        cached_path = zika_root / cached_rel
        if not cached_rel or not cached_path.exists():
            missing_cache += 1
            logger.warning(
                "Skipping manifest row %s..%s because cache is missing at %s",
                row["window_start"],
                row["window_end"],
                cached_rel,
            )
            continue

        with cached_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        frames.append(parse_cached_payload(payload, cached_rel))

    if frames:
        parsed_df = pd.concat(frames, ignore_index=True)
    else:
        parsed_df = pd.DataFrame(columns=RSS_OUTPUT_COLUMNS)

    deduped_df, counts_df = dedupe_rss_records(parsed_df)
    write_dataframe(deduped_df, output_csv, output_parquet)

    for _, metric_row in counts_df.iterrows():
        logger.info("Metric %s=%s", metric_row["metric"], metric_row["value"])
    logger.info(
        "Parsing summary parsed_rows=%s deduped_rows=%s missing_cache=%s outputs=%s",
        len(parsed_df),
        len(deduped_df),
        missing_cache,
        output_csv,
    )


if __name__ == "__main__":
    main()
