from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from epp_emergence_analysis_batch import chunk_rows, resolve_run_dir, unique_stems
from epp_emergence_analysis_excel import load_json, write_excel_from_payload, write_json
from epp_emergence_analysis_manifest import build_run_manifest, print_batch_manifest, print_dataset_summary, write_manifest
from epp_emergence_analysis_openai_batch_api_io import download_output, request_line, response_body, retrieve_batch, submit_batch
from epp_emergence_analysis_openai_provider import (
    REVIEW_PROMPT,
    SEARCH_PROMPT,
    build_structured_response_request,
    parse_response_json,
)
from epp_emergence_analysis_provider_utils import stage_result
from epp_emergence_analysis_workbook import (
    build_review_prompt,
    build_user_prompt,
    country_context,
    dataframe_records,
    read_names_file,
    read_workbook,
    select_rows,
)


def selected_payload(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any]]:
    main_df, country_df, country_sheet = read_workbook(args.workbook)
    names = list(args.names)
    if args.names_file:
        names.extend(read_names_file(args.names_file))
    rows, warnings = select_rows(main_df, names, args.row_indices, args.input_rows, args.all, args.case_sensitive, args.limit)
    for warning in warnings:
        print(f"Warning: {warning}")
    info = {
        "workbook_path": args.workbook,
        "country_sheet": country_sheet,
        "main_row_count": len(main_df),
        "country_row_count": len(country_df),
    }
    return rows, dataframe_records(country_df), country_context(country_df), info


def save_state(args: argparse.Namespace) -> dict[str, Any]:
    rows, country_rows, country_rows_text, info = selected_payload(args)
    args.run_dir = resolve_run_dir(args)
    args.search_provider = "openai"
    args.review_provider = "openai"
    args.search_model = args.model
    args.review_model_resolved = args.review_model
    stems = unique_stems(rows)
    manifest = build_run_manifest(
        selected_rows=rows,
        batch_size=args.batch_size,
        stems=stems,
        args=args,
        **info,
    )
    state = {
        "selected_rows": rows,
        "country_rows": country_rows,
        "country_rows_text": country_rows_text,
        "stems": {str(key): value for key, value in stems.items()},
        "manifest": manifest,
    }
    write_json(state, args.run_dir / "openai_batch" / "state.json")
    write_manifest(manifest, args.run_dir / "batch_manifest.json")
    print_dataset_summary(manifest)
    print_batch_manifest(manifest)
    return state


def load_state(args: argparse.Namespace) -> dict[str, Any]:
    path = args.run_dir / "openai_batch" / "state.json"
    if path.exists():
        return load_json(path)
    return save_state(args)


def write_search_jsonl(args: argparse.Namespace, state: dict[str, Any]) -> Path:
    path = args.run_dir / "openai_batch" / "search_requests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in state["selected_rows"]:
            body = build_structured_response_request(
                SEARCH_PROMPT,
                build_user_prompt(row, state["country_rows_text"], args.year_window),
                args.model,
                args.max_output_tokens,
                args.reasoning_effort,
                args.store,
                args.web_search,
                args.search_context_size,
                False,
            )
            custom_id = f"search-row-{int(row['Input row']):04d}"
            handle.write(json.dumps(request_line(custom_id, body), ensure_ascii=False) + "\n")
    return path


def write_review_jsonl(args: argparse.Namespace, state: dict[str, Any]) -> Path:
    path = args.run_dir / "openai_batch" / "review_requests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in state["selected_rows"]:
            row_id = int(row["Input row"])
            search_path = args.run_dir / "batch_search_json" / f"{state['stems'][str(row_id)]}.json"
            search_result = load_json(search_path)
            body = build_structured_response_request(
                REVIEW_PROMPT,
                build_review_prompt(row, state["country_rows_text"], search_result),
                args.review_model,
                args.review_max_output_tokens,
                args.review_reasoning_effort,
                args.store,
                args.web_search,
                args.search_context_size,
                True,
            )
            handle.write(json.dumps(request_line(f"review-row-{row_id:04d}", body), ensure_ascii=False) + "\n")
    return path


def hydrate(args: argparse.Namespace, state: dict[str, Any], label: str) -> None:
    batch = retrieve_batch(args.run_dir, args.openai_base_url, label)
    if batch["status"] != "completed":
        return
    out_dir = args.run_dir / ("batch_search_json" if label == "search" else "batch_reviewed_json")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {int(row["Input row"]): row for row in state["selected_rows"]}
    for record in download_output(args.openai_base_url, batch["output_file_id"]):
        row_id = int(str(record["custom_id"]).split("-")[-1])
        row = by_id[row_id]
        body = response_body(record)
        max_tokens = args.max_output_tokens if label == "search" else args.review_max_output_tokens
        parsed = parse_response_json(body, row, max_tokens)
        search_result = None
        if label == "review":
            search_result = load_json(args.run_dir / "batch_search_json" / f"{state['stems'][str(row_id)]}.json")
        result = stage_result(
            provider="openai",
            row=row,
            stage=label,
            response_id=str(body.get("id", "")),
            status=str(body.get("status", "")),
            model=str(body.get("model", args.model if label == "search" else args.review_model)),
            usage=body.get("usage") or {},
            parsed_response=parsed,
            search_result=search_result,
        )
        write_json(result, out_dir / f"{state['stems'][str(row_id)]}.json")
    print(f"{label} JSON hydrated: {out_dir}")


def finalize(args: argparse.Namespace, state: dict[str, Any]) -> None:
    stems = {int(key): value for key, value in state["stems"].items()}
    for batch_index, rows in enumerate(chunk_rows(state["selected_rows"], args.batch_size), start=1):
        batch_dir = args.run_dir / f"batch_{batch_index:03d}"
        results = [
            load_json(args.run_dir / "batch_reviewed_json" / f"{stems[int(row['Input row'])]}.json")
            for row in rows
        ]
        payload = {
            "metadata": {
                "search_provider": "openai",
                "review_provider": "openai",
                "search_model": args.model,
                "review_model": args.review_model,
                "batch_api": True,
            },
            "selected_rows": rows,
            "country_rows": state["country_rows"],
            "warnings": [],
            "results": results,
        }
        curated = write_json(payload, batch_dir / "curated_batch.json")
        write_excel_from_payload(payload, batch_dir / "curated_batch.xlsx")
        print(f"Curated batch JSON: {curated}")
        print(f"Curated batch Excel: {curated.with_suffix('.xlsx')}")


def run_action(args: argparse.Namespace) -> None:
    state = load_state(args)
    if args.action == "prepare-search":
        print(f"Search request JSONL: {write_search_jsonl(args, state)}")
    elif args.action == "submit-search":
        submit_batch(args.run_dir, args.openai_base_url, write_search_jsonl(args, state), "search")
    elif args.action == "poll-search":
        hydrate(args, state, "search")
    elif args.action == "submit-review":
        submit_batch(args.run_dir, args.openai_base_url, write_review_jsonl(args, state), "review")
    elif args.action == "poll-review":
        hydrate(args, state, "review")
    elif args.action == "finalize":
        finalize(args, state)
