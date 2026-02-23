"""Inspect input RSS CSV and infer key schema columns."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

try:
    from .utils import detect_schema, safe_read_csv
except ImportError:  # pragma: no cover - script execution path
    from utils import detect_schema, safe_read_csv


def _emit(message: str, logger=None) -> None:
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def inspect_csv(
    input_path: Path,
    sample_rows: int = 5,
    logger=None,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], Dict[str, Any]]:
    """Load and inspect CSV, returning dataframe + inferred schema + report."""
    df = safe_read_csv(input_path)
    schema = detect_schema(df)

    null_rates = (df.isna().mean().sort_values(ascending=False) * 100.0).round(2)
    sample = df.head(sample_rows)

    report: Dict[str, Any] = {
        "input_path": str(input_path),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "null_rates_percent": null_rates.to_dict(),
        "schema": schema,
        "sample": sample.to_dict(orient="records"),
    }

    _emit(f"Input CSV: {input_path}", logger=logger)
    _emit(f"Rows: {len(df):,} | Columns: {len(df.columns)}", logger=logger)
    _emit("Columns:", logger=logger)
    for col in df.columns:
        _emit(f"  - {col}", logger=logger)

    _emit("Null rates (%):", logger=logger)
    for col, pct in null_rates.items():
        _emit(f"  - {col}: {pct:.2f}%", logger=logger)

    _emit("Inferred schema:", logger=logger)
    _emit(f"  - URL column: {schema.get('url_col')}", logger=logger)
    _emit(f"  - RSS title column: {schema.get('title_col')}", logger=logger)
    _emit(f"  - RSS description column: {schema.get('description_col')}", logger=logger)
    _emit(f"  - RSS date column: {schema.get('date_col')}", logger=logger)

    _emit(f"Sample rows (top {sample_rows}):", logger=logger)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        _emit(sample.to_string(index=False), logger=logger)

    if not schema.get("url_col"):
        raise ValueError(
            "Could not robustly infer a URL column. "
            "Please update detection logic or provide a CSV with explicit URL field."
        )

    return df, schema, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RSS CSV schema.")
    parser.add_argument("--input", required=True, help="Path to CSV input")
    parser.add_argument("--sample-rows", type=int, default=5, help="Number of sample rows")
    args = parser.parse_args()

    inspect_csv(Path(args.input), sample_rows=args.sample_rows)


if __name__ == "__main__":
    main()
