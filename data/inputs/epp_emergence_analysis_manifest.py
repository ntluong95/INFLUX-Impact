from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from epp_emergence_analysis_config import MAIN_SHEET, STANDARD_NAME_COLUMN


def build_run_manifest(
    *,
    workbook_path: Path,
    country_sheet: str,
    main_row_count: int,
    country_row_count: int,
    selected_rows: list[dict[str, Any]],
    batch_size: int,
    stems: dict[int, str],
    args: Any,
) -> dict[str, Any]:
    return {
        "input_dataset": {
            "workbook": str(workbook_path),
            "main_sheet": MAIN_SHEET,
            "country_sheet": country_sheet,
            "main_list_data_rows": main_row_count,
            "country_sheet_rows": country_row_count,
        },
        "selection": {
            "all": bool(args.all),
            "names": list(args.names),
            "names_file": str(args.names_file) if args.names_file else "",
            "row_indices": list(args.row_indices),
            "input_rows": list(args.input_rows),
            "limit": args.limit,
            "case_sensitive": bool(args.case_sensitive),
            "selected_row_count": len(selected_rows),
        },
        "llm": {
            "search_provider": args.search_provider,
            "review_provider": args.review_provider,
            "search_model": args.search_model,
            "review_model": args.review_model_resolved,
            "web_search": bool(args.web_search),
        },
        "batching": {
            "batch_size": batch_size,
            "batch_count": (len(selected_rows) + batch_size - 1) // batch_size,
            "batches": build_batch_entries(selected_rows, batch_size, stems),
        },
    }


def build_batch_entries(
    selected_rows: list[dict[str, Any]],
    batch_size: int,
    stems: dict[int, str],
) -> list[dict[str, Any]]:
    batches = []
    for index in range(0, len(selected_rows), batch_size):
        batch_rows = selected_rows[index:index + batch_size]
        batch_number = len(batches) + 1
        batches.append(
            {
                "batch_id": f"batch_{batch_number:03d}",
                "row_count": len(batch_rows),
                "rows": [row_entry(row, stems[int(row["Input row"])]) for row in batch_rows],
            }
        )
    return batches


def row_entry(row: dict[str, Any], file_stem: str) -> dict[str, Any]:
    return {
        "input_row": int(row["Input row"]),
        "zero_based_row_index": int(row["Input row"]) - 2,
        "scientific_name_standardize": row.get(STANDARD_NAME_COLUMN, ""),
        "scientific_name": first_value(row, "Scientific name", "scientific_name"),
        "common_name": first_value(row, "Common name", "common_name", "English common name"),
        "file_stem": file_stem,
    }


def first_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def print_dataset_summary(manifest: dict[str, Any]) -> None:
    dataset = manifest["input_dataset"]
    selection = manifest["selection"]
    llm = manifest["llm"]
    batching = manifest["batching"]
    print(f"Input workbook: {dataset['workbook']}")
    print(
        f"Input sheets: {dataset['main_sheet']} "
        f"({dataset['main_list_data_rows']} data rows), "
        f"{dataset['country_sheet']} ({dataset['country_sheet_rows']} rows)"
    )
    print(
        f"Selected rows: {selection['selected_row_count']} | "
        f"Batch size: {batching['batch_size']} | "
        f"Batch count: {batching['batch_count']}"
    )
    print(
        f"Search: {llm['search_provider']} / {llm['search_model']} | "
        f"Review: {llm['review_provider']} / {llm['review_model']}"
    )


def print_batch_manifest(manifest: dict[str, Any]) -> None:
    for batch in manifest["batching"]["batches"]:
        print(f"{batch['batch_id']} ({batch['row_count']} EPPs)")
        for row in batch["rows"]:
            print(
                f"  input row {row['input_row']} "
                f"(index {row['zero_based_row_index']}): "
                f"{row['scientific_name_standardize']} -> {row['file_stem']}.json"
            )
