from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from epp_emergence_analysis_api import call_review_openai, call_search_openai
from epp_emergence_analysis_config import STANDARD_NAME_COLUMN, now_slug
from epp_emergence_analysis_excel import load_json, write_excel_from_payload, write_json


def safe_file_stem(row: dict[str, Any]) -> str:
    name = str(row.get(STANDARD_NAME_COLUMN) or row.get("Scientific name") or "epp")
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return stem or f"input-row-{row.get('Input row', 'unknown')}"


def unique_stems(rows: list[dict[str, Any]]) -> dict[int, str]:
    counts: dict[str, int] = {}
    stems: dict[int, str] = {}
    for row in rows:
        base = safe_file_stem(row)
        counts[base] = counts.get(base, 0) + 1
        stems[int(row["Input row"])] = base if counts[base] == 1 else f"{base}-row-{int(row['Input row']):04d}"
    return stems


def chunk_rows(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[index:index + batch_size] for index in range(0, len(rows), batch_size)]


def resolve_run_dir(args: Any) -> Path:
    return args.run_dir or args.output_dir / f"epp_emergence_run_{now_slug()}"


def process_batches(
    client: Any,
    selected_rows: list[dict[str, Any]],
    country_rows: list[dict[str, Any]],
    country_rows_text: str,
    args: Any,
) -> list[Path]:
    run_dir = resolve_run_dir(args)
    stems = unique_stems(selected_rows)
    outputs: list[Path] = []
    for batch_index, rows in enumerate(chunk_rows(selected_rows, args.batch_size), start=1):
        batch_dir = run_dir / f"batch_{batch_index:03d}"
        curated = process_batch(client, rows, country_rows, country_rows_text, args, batch_dir, stems)
        outputs.append(curated)
    return outputs


def process_batch(
    client: Any,
    rows: list[dict[str, Any]],
    country_rows: list[dict[str, Any]],
    country_rows_text: str,
    args: Any,
    batch_dir: Path,
    stems: dict[int, str],
) -> Path:
    search_dir = batch_dir / "search_json"
    review_dir = batch_dir / "reviewed_json"
    results = []
    for row in rows:
        row_id = int(row["Input row"])
        print(f"Batch {batch_dir.name}: search input row {row_id} - {row.get(STANDARD_NAME_COLUMN)}")
        search_path = search_dir / f"{stems[row_id]}.json"
        review_path = review_dir / f"{stems[row_id]}.json"
        search_result = load_json(search_path) if search_path.exists() else call_search_openai(
            client, row, country_rows_text, args
        )
        write_json(search_result, search_path)
        print(f"Batch {batch_dir.name}: review input row {row_id} - {row.get(STANDARD_NAME_COLUMN)}")
        review_result = load_json(review_path) if review_path.exists() else call_review_openai(
            client, row, country_rows_text, search_result, args
        )
        write_json(review_result, review_path)
        results.append(review_result)
    payload = {
        "metadata": batch_metadata(args, batch_dir),
        "selected_rows": rows,
        "country_rows": country_rows,
        "warnings": [],
        "results": results,
    }
    curated_json = write_json(payload, batch_dir / "curated_batch.json")
    write_excel_from_payload(payload, batch_dir / "curated_batch.xlsx")
    return curated_json


def batch_metadata(args: Any, batch_dir: Path) -> dict[str, Any]:
    return {
        "generated_at_utc": now_slug(),
        "batch_dir": str(batch_dir),
        "search_model": args.model,
        "review_model": args.review_model or args.model,
        "base_url": args.base_url,
        "batch_size": args.batch_size,
        "web_search": args.web_search,
    }
