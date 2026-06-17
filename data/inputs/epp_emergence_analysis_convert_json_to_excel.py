#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from epp_emergence_analysis_excel import load_json, write_excel_from_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert saved EPP emergence-event JSON to Excel."
    )
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--output", type=Path, help="Output .xlsx path. Defaults beside JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_json(args.json_path)
    output = args.output or args.json_path.with_suffix(".xlsx")
    write_excel_from_payload(payload, output)
    print(f"Excel output: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
