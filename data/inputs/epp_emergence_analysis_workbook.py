from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from epp_emergence_analysis_config import (
    COUNTRY_SHEET_CANDIDATES,
    MAIN_SHEET,
    STANDARD_NAME_COLUMN,
    normalize_text,
)


def read_workbook(workbook_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")
    try:
        xls = pd.ExcelFile(workbook_path)
    except (OSError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not read workbook: {workbook_path}. "
            "If this file is stored in OneDrive/iCloud, make it available offline "
            "or pass --workbook with a local copy."
        ) from exc
    if MAIN_SHEET not in xls.sheet_names:
        raise ValueError(f"Workbook must contain a '{MAIN_SHEET}' sheet.")
    country_sheet = next((s for s in COUNTRY_SHEET_CANDIDATES if s in xls.sheet_names), None)
    if country_sheet is None:
        raise ValueError("Workbook must contain a 'Country' or 'Country sheet' sheet.")
    try:
        main_df = pd.read_excel(workbook_path, sheet_name=MAIN_SHEET)
        country_df = pd.read_excel(workbook_path, sheet_name=country_sheet)
    except (OSError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not load workbook sheets from: {workbook_path}. "
            "Make the workbook available offline, then rerun the command."
        ) from exc
    if STANDARD_NAME_COLUMN not in main_df.columns:
        raise ValueError(f"'{MAIN_SHEET}' must contain '{STANDARD_NAME_COLUMN}'.")
    main_df = main_df.copy()
    main_df.insert(0, "Input row", range(2, len(main_df) + 2))
    return main_df, country_df, country_sheet


def read_names_file(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError("Names JSON file must contain a list of strings.")
        return [str(value) for value in values]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.where(pd.notna(df), "").to_dict(orient="records")
    return [{str(k): v for k, v in record.items()} for record in records]


def select_rows(
    main_df: pd.DataFrame,
    names: list[str] | None,
    row_indices: list[int] | None,
    input_rows: list[int] | None,
    include_all: bool,
    case_sensitive: bool,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    masks: list[pd.Series] = []
    warnings: list[str] = []
    if include_all:
        masks.append(pd.Series(True, index=main_df.index))
    if names:
        source = main_df[STANDARD_NAME_COLUMN].fillna("").astype(str)
        lookup = source if case_sensitive else source.map(normalize_text)
        for name in names:
            needle = name if case_sensitive else normalize_text(name)
            mask = lookup == needle
            if not mask.any():
                warnings.append(f"No row matched standardized scientific name: {name}")
            masks.append(mask)
    if row_indices:
        mask = pd.Series(False, index=main_df.index)
        for index in row_indices:
            if index < 0 or index >= len(main_df):
                warnings.append(f"Zero-based row index out of range: {index}")
                continue
            mask.iloc[index] = True
        masks.append(mask)
    if input_rows:
        mask = main_df["Input row"].isin(input_rows)
        found = set(main_df.loc[mask, "Input row"].astype(int))
        warnings.extend(f"Excel input row not found: {row}" for row in sorted(set(input_rows) - found))
        masks.append(mask)
    if not masks:
        raise ValueError("Provide --names, --names-file, --row-indices, --input-rows, or --all.")
    combined = masks[0].copy()
    for mask in masks[1:]:
        combined |= mask
    selected = main_df.loc[combined].drop_duplicates(subset=["Input row"])
    if limit is not None:
        selected = selected.head(limit)
    return dataframe_records(selected), warnings


def country_context(country_df: pd.DataFrame) -> str:
    parts = []
    for row in dataframe_records(country_df):
        parts.append(
            f"- {row.get('Continent', '')} | {row.get('Sub-regions', '')}: "
            f"{row.get('Countries / areas', '')}"
        )
    return "\n".join(parts)


def build_user_prompt(row: dict[str, Any], country_rows_text: str, year_window: str) -> str:
    row_text = json.dumps(row, ensure_ascii=False, indent=2)
    return (
        "Attached Excel context for this API request.\n\n"
        f"Selected row from sheet \"Main list\":\n{row_text}\n\n"
        f"Rows from sheet \"Country\":\n{country_rows_text}\n\n"
        f"Configured recent-emergence window for this run: {year_window}."
    )


def build_review_prompt(
    row: dict[str, Any],
    country_rows_text: str,
    search_result: dict[str, Any],
) -> str:
    row_text = json.dumps(row, ensure_ascii=False, indent=2)
    search_text = json.dumps(search_result.get("parsed_response", {}), ensure_ascii=False, indent=2)
    return (
        "Attached Excel context and ChatGPT work for peer review.\n\n"
        f"Selected row from sheet \"Main list\":\n{row_text}\n\n"
        f"Rows from sheet \"Country\":\n{country_rows_text}\n\n"
        f"ChatGPT emergence-event JSON to review:\n{search_text}\n\n"
        "Return a curated JSON dataset. Include all retained, added, and flagged "
        "event rows. Fill the Review field for every event row."
    )
