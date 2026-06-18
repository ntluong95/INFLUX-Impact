#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from epp_emergence_analysis_config import DEFAULT_OPENAI_BASE_URL, DEFAULT_OUTPUT_DIR, DEFAULT_WORKBOOK
from epp_emergence_analysis_openai_batch_api_workflow import run_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit and hydrate OpenAI Batch API jobs for EPP analysis.")
    parser.add_argument(
        "action",
        choices=[
            "prepare-search",
            "submit-search",
            "poll-search",
            "submit-review",
            "poll-review",
            "finalize",
        ],
    )
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--openai-base-url", "--base-url", dest="openai_base_url", default=DEFAULT_OPENAI_BASE_URL)
    parser.add_argument("--names", nargs="*", default=[])
    parser.add_argument("--names-file", type=Path)
    parser.add_argument("--row-indices", nargs="*", type=int, default=[])
    parser.add_argument("--input-rows", "--excel-rows", nargs="*", type=int, default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--case-sensitive", action="store_true")
    parser.add_argument("--year-window", default="2005-2025")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--review-model", default="gpt-5.1")
    parser.add_argument("--max-output-tokens", type=int, default=50000)
    parser.add_argument("--review-max-output-tokens", type=int, default=50000)
    parser.add_argument("--reasoning-effort", default="low", choices=["none", "low", "medium", "high", "xhigh"])
    parser.add_argument("--review-reasoning-effort", default="high", choices=["none", "low", "medium", "high", "xhigh"])
    parser.add_argument("--web-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--search-context-size", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--store", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    run_action(parse_args())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
