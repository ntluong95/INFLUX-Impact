#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from epp_emergence_analysis_batch import process_batches, resolve_run_dir, unique_stems
from epp_emergence_analysis_config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WORKBOOK,
    PROJECT_ROOT,
)
from epp_emergence_analysis_manifest import (
    build_run_manifest,
    print_batch_manifest,
    print_dataset_summary,
    write_manifest,
)
from epp_emergence_analysis_workbook import (
    country_context,
    dataframe_records,
    read_names_file,
    read_workbook,
    select_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query OpenAI for EPP emergence events and export JSON plus Excel."
    )
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dir", type=Path, help="Existing or new run folder for batch outputs.")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--review-model", help="Model for peer review. Defaults to --model.")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--names", nargs="*", default=[])
    parser.add_argument("--names-file", type=Path)
    parser.add_argument("--row-indices", nargs="*", type=int, default=[])
    parser.add_argument("--input-rows", "--excel-rows", nargs="*", type=int, default=[])
    parser.add_argument("--all", action="store_true", help="Process every Main list row.")
    parser.add_argument("--limit", type=int, help="Limit rows after selection.")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--case-sensitive", action="store_true")
    parser.add_argument("--year-window", default="2005-2025")
    parser.add_argument("--max-output-tokens", type=int, default=30000)
    parser.add_argument("--review-max-output-tokens", type=int, default=30000)
    parser.add_argument("--reasoning-effort", default="low", choices=["none", "low", "medium", "high", "xhigh"])
    parser.add_argument("--review-reasoning-effort", default="high", choices=["none", "low", "medium", "high", "xhigh"])
    parser.add_argument("--web-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--search-context-size", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--store", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true", help="Print selected rows and skip API calls.")
    parser.add_argument(
        "--show-batches",
        action="store_true",
        help="Print input dataset details and all batch memberships.",
    )
    parser.add_argument("--list-models", action="store_true", help="List model IDs available to your API key.")
    return parser.parse_args()


def build_payload(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any]]:
    main_df, country_df, country_sheet = read_workbook(args.workbook)
    names = list(args.names)
    if args.names_file:
        names.extend(read_names_file(args.names_file))
    selected_rows, warnings = select_rows(
        main_df=main_df,
        names=names,
        row_indices=args.row_indices,
        input_rows=args.input_rows,
        include_all=args.all,
        case_sensitive=args.case_sensitive,
        limit=args.limit,
    )
    for warning in warnings:
        print(f"Warning: {warning}")
    country_rows = dataframe_records(country_df)
    workbook_info = {
        "workbook_path": args.workbook,
        "country_sheet": country_sheet,
        "main_row_count": len(main_df),
        "country_row_count": len(country_df),
    }
    return selected_rows, country_rows, country_context(country_df), workbook_info


def print_selected_rows(rows: list[dict[str, Any]], max_rows: int = 50) -> None:
    print(f"Selected {len(rows)} row(s).")
    if len(rows) > max_rows:
        print(f"  Showing first {max_rows}; use --show-batches for the full batch plan.")
    for row in rows[:max_rows]:
        print(f"  input row {row.get('Input row')}: {row.get('Scientific name_standardize')}")


def main() -> int:
    args = parse_args()
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        if not args.dry_run:
            raise RuntimeError(
                "Missing API dependency. Install project dependencies or run: "
                "python3 -m pip install openai python-dotenv"
            ) from exc
        OpenAI = None
        load_dotenv = None

    if args.list_models:
        if load_dotenv is None or OpenAI is None:
            raise RuntimeError(
                "Missing API dependency. Install project dependencies or run: "
                "python3 -m pip install openai python-dotenv"
            )
        load_dotenv(PROJECT_ROOT / ".env")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"OPENAI_API_KEY not found. Add it to {PROJECT_ROOT / '.env'} or the environment.")
        print(f"Using API base URL: {args.base_url}")
        models = sorted(model.id for model in OpenAI(base_url=args.base_url).models.list().data)
        for model in models:
            print(model)
        return 0

    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be at least 1.")
    selected_rows, country_rows, country_rows_text, workbook_info = build_payload(args)
    if not args.show_batches:
        print_selected_rows(selected_rows)
    stems = unique_stems(selected_rows)
    manifest = build_run_manifest(
        selected_rows=selected_rows,
        batch_size=args.batch_size,
        stems=stems,
        args=args,
        **workbook_info,
    )
    print_dataset_summary(manifest)
    if args.show_batches:
        print_batch_manifest(manifest)
    if args.dry_run:
        return 0

    if load_dotenv is None or OpenAI is None:
        raise RuntimeError(
            "Missing API dependency. Install project dependencies or run: "
            "python3 -m pip install openai python-dotenv"
        )

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY not found. Add it to {PROJECT_ROOT / '.env'} or the environment.")

    args.run_dir = resolve_run_dir(args)
    manifest_path = write_manifest(manifest, args.run_dir / "batch_manifest.json")
    print(f"Batch manifest: {manifest_path}")
    client = OpenAI(base_url=args.base_url)
    curated_paths = process_batches(client, selected_rows, country_rows, country_rows_text, args)
    for path in curated_paths:
        print(f"Curated batch JSON: {path}")
        print(f"Curated batch Excel: {path.with_suffix('.xlsx')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
