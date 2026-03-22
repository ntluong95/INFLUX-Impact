from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.common import (
    estimate_prompt_tokens,
    normalize_whitespace,
    setup_logger,
    text_or_empty,
    utc_now_iso,
    write_dataframe_atomic,
)
from src.utils.config import project_paths, require_openai_api_key
from src.utils.openai_batch import (
    create_batch,
    download_file_content,
    extract_message_content,
    openai_max_tokens_param,
    openai_supports_temperature,
    parse_json_object,
    retrieve_batch,
    upload_batch_file,
)
from src.utils.project import DatasetKey


STATE_COLUMNS = [
    "record_id",
    "dataset_key",
    "pathogen_domain",
    "language_code",
    "search_string",
    "rss_title",
    "rss_description_text",
    "rss_pubdate",
    "source_hint",
    "source_url",
    "source_domain",
    "google_news_redirect_url",
    "openai_custom_id",
    "openai_batch_name",
    "openai_batch_id",
    "batch_state",
    "batch_submitted_at",
    "headline_label",
    "headline_confidence",
    "headline_rationale",
    "headline_parse_error",
    "classification_error",
    "classified_at",
    "final_action",
]

BATCH_REGISTRY_COLUMNS = [
    "dataset_key",
    "batch_name",
    "request_jsonl_path",
    "request_count",
    "estimated_prompt_tokens",
    "input_file_id",
    "batch_id",
    "status",
    "created_at",
    "completed_at",
    "hydrated_at",
    "output_file_id",
    "error_file_id",
]

TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}
ACTIVE_BATCH_STATUSES = {"validating", "in_progress", "finalizing", "submitted"}


def build_system_prompt(dataset: DatasetKey) -> str:
    domain_scope = {
        "human": "human pathogen or disease",
        "animal": "animal pathogen, pest, outbreak, or livestock/wildlife disease",
        "plant": "plant pathogen, pest, outbreak, or crop disease",
    }[dataset.pathogen_domain]
    return f"""You are a strict classifier for pathogen-news headline relevance.

Return JSON only with keys:
- label: relevant | irrelevant | unsure
- confidence: number from 0 to 1
- rationale: short rationale under 24 words

Rules:
- relevant: clearly about the searched {domain_scope}, including outbreaks, spread, control, surveillance, response, impacts, or consequences.
- irrelevant: homonym, metaphor, finance, sports, entertainment, spam, or a different topic.
- unsure: ambiguous from headline and summary alone.
- Use conservative confidence when the signal is weak.
"""


def build_user_prompt(row: pd.Series) -> str:
    return f"""Classify this Google News item.

Pathogen domain: {row.get("pathogen_domain", "[missing]")}
Language: {row.get("language_code", "[missing]")}
Matched search string: {row.get("search_string", "[missing]")}
Headline: {row.get("rss_title", "[missing]")}
Summary: {row.get("rss_description_text", "[missing]")}
Source: {row.get("source_hint", "[missing]")}
Source domain: {row.get("source_domain", "[missing]")}
Published: {row.get("rss_pubdate", "[missing]")}
"""


def parse_headline_label(raw_text: str) -> dict[str, Any]:
    parsed = parse_json_object(raw_text)
    if not parsed:
        return {
            "label": "unsure",
            "confidence": 0.0,
            "rationale": "",
            "parse_error": "invalid_json",
        }

    label = str(parsed.get("label", "unsure")).strip().lower()
    if label not in {"relevant", "irrelevant", "unsure"}:
        label = "unsure"
        parse_error = "invalid_label"
    else:
        parse_error = ""
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
        parse_error = parse_error or "invalid_confidence"

    confidence = max(0.0, min(1.0, confidence))
    rationale = normalize_whitespace(str(parsed.get("rationale", "")))[:280]
    return {
        "label": label,
        "confidence": confidence,
        "rationale": rationale,
        "parse_error": parse_error,
    }


def label_to_action(label: str) -> str:
    if label == "relevant":
        return "keep"
    if label == "irrelevant":
        return "drop"
    return "review"


def load_registry(registry_csv: Path) -> pd.DataFrame:
    if not registry_csv.exists():
        return pd.DataFrame(columns=BATCH_REGISTRY_COLUMNS)
    df = pd.read_csv(registry_csv, low_memory=False)
    for column in BATCH_REGISTRY_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[BATCH_REGISTRY_COLUMNS].fillna("")


def load_or_initialize_state(input_csv: Path, state_csv: Path, dataset: DatasetKey) -> pd.DataFrame:
    input_df = pd.read_csv(input_csv, low_memory=False)
    work = input_df.copy()
    work["dataset_key"] = dataset.stem
    work["pathogen_domain"] = dataset.pathogen_domain
    work["language_code"] = dataset.language_code
    if state_csv.exists():
        existing = pd.read_csv(state_csv, low_memory=False)
        merged = work.merge(
            existing,
            on="record_id",
            how="left",
            suffixes=("", "_existing"),
        )
        for column in STATE_COLUMNS:
            existing_column = f"{column}_existing"
            if column in work.columns and column != "record_id":
                continue
            if existing_column not in merged.columns:
                merged[column] = "" if column not in work.columns else merged[column]
                continue
            if column in work.columns:
                merged[column] = merged[column]
            else:
                merged[column] = merged[existing_column]
            merged.drop(columns=[existing_column], inplace=True, errors="ignore")
        work = merged

    for column in STATE_COLUMNS:
        if column not in work.columns:
            work[column] = ""
    return work[STATE_COLUMNS].fillna("")


def rows_needing_submission(state_df: pd.DataFrame) -> list[int]:
    pending_rows: list[int] = []
    for idx, row in state_df.iterrows():
        if text_or_empty(row.get("headline_label")):
            continue
        batch_state = text_or_empty(row.get("batch_state")).strip().lower()
        if batch_state in {"submitted", "validating", "in_progress", "finalizing"}:
            continue
        pending_rows.append(idx)
    return pending_rows


def split_request_chunks(
    request_rows: list[dict[str, Any]],
    *,
    max_requests: int,
    max_bytes: int,
    max_prompt_tokens: int,
) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    current_prompt_tokens = 0

    for row in request_rows:
        line = json.dumps(row["request"], ensure_ascii=False)
        line_bytes = len(line.encode("utf-8")) + 1
        line_tokens = int(row["estimated_prompt_tokens"])
        would_overflow = (
            current
            and (
                len(current) >= max_requests
                or current_bytes + line_bytes > max_bytes
                or (
                    max_prompt_tokens > 0
                    and current_prompt_tokens + line_tokens > max_prompt_tokens
                )
            )
        )
        if would_overflow:
            chunks.append(current)
            current = []
            current_bytes = 0
            current_prompt_tokens = 0

        current.append(row)
        current_bytes += line_bytes
        current_prompt_tokens += line_tokens

    if current:
        chunks.append(current)
    return chunks


def build_request_rows(
    state_df: pd.DataFrame,
    row_indexes: list[int],
    dataset: DatasetKey,
    openai_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    model = str(openai_cfg.get("model", "gpt-5-nano"))
    request_rows: list[dict[str, Any]] = []
    for idx in row_indexes:
        row = state_df.loc[idx]
        system_prompt = build_system_prompt(dataset)
        user_prompt = build_user_prompt(row)
        request_body: dict[str, Any] = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            openai_max_tokens_param(model): int(openai_cfg.get("max_tokens", 160)),
        }
        if openai_supports_temperature(model):
            request_body["temperature"] = float(openai_cfg.get("temperature", 0.0))
        custom_id = f"{dataset.stem}:{row['record_id']}"
        request_rows.append(
            {
                "idx": idx,
                "custom_id": custom_id,
                "estimated_prompt_tokens": estimate_prompt_tokens(
                    f"{system_prompt}\n{user_prompt}"
                ),
                "request": {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": request_body,
                },
            }
        )
    return request_rows


def write_request_jsonl(request_chunk: list[dict[str, Any]], jsonl_path: Path) -> None:
    lines = [json.dumps(item["request"], ensure_ascii=False) for item in request_chunk]
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def active_enqueued_prompt_tokens(registry_df: pd.DataFrame) -> int:
    if registry_df.empty:
        return 0
    active = registry_df.loc[
        ~registry_df["status"].fillna("").astype(str).str.lower().isin(TERMINAL_BATCH_STATUSES)
    ].copy()
    if active.empty:
        return 0
    return int(
        pd.to_numeric(active["estimated_prompt_tokens"], errors="coerce").fillna(0).sum()
    )


def hydrate_batches(
    dataset: DatasetKey,
    config: dict[str, Any],
    state_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    logger: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    openai_cfg = config["classification"]["openai"]
    api_key = require_openai_api_key(config)
    base_url = str(openai_cfg.get("base_url", "") or "")
    timeout_seconds = int(openai_cfg.get("timeout_seconds", 60))
    paths = project_paths(config)

    for registry_idx, registry_row in registry_df.iterrows():
        batch_id = text_or_empty(registry_row.get("batch_id"))
        if not batch_id:
            continue

        batch = retrieve_batch(
            batch_id=batch_id,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        status = text_or_empty(batch.get("status"))
        registry_df.loc[registry_idx, "status"] = status
        registry_df.loc[registry_idx, "output_file_id"] = text_or_empty(batch.get("output_file_id"))
        registry_df.loc[registry_idx, "error_file_id"] = text_or_empty(batch.get("error_file_id"))
        if status == "completed" and not text_or_empty(registry_df.loc[registry_idx, "completed_at"]):
            registry_df.loc[registry_idx, "completed_at"] = utc_now_iso()

        state_mask = state_df["openai_batch_id"].fillna("").astype(str) == batch_id
        if not state_mask.any():
            continue

        if status in {"submitted", "validating", "in_progress", "finalizing"}:
            state_df.loc[state_mask, "batch_state"] = status
            continue

        if status != "completed":
            state_df.loc[state_mask, "batch_state"] = status or "unknown"
            state_df.loc[state_mask, "classification_error"] = (
                state_df.loc[state_mask, "classification_error"]
                .replace("", f"batch_{status or 'unknown'}")
            )
            continue

        output_file_id = text_or_empty(batch.get("output_file_id"))
        error_file_id = text_or_empty(batch.get("error_file_id"))
        result_map: dict[str, dict[str, Any]] = {}

        if output_file_id:
            output_text = download_file_content(
                file_id=output_file_id,
                api_key=api_key,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
            result_path = paths.batch_results_dir(dataset) / f"{registry_row['batch_name']}_output.jsonl"
            result_path.write_text(output_text, encoding="utf-8")
            for raw_line in output_text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                except Exception:
                    continue
                custom_id = text_or_empty(result.get("custom_id"))
                if custom_id:
                    result_map[custom_id] = result

        if error_file_id:
            error_text = download_file_content(
                file_id=error_file_id,
                api_key=api_key,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
            error_path = paths.batch_results_dir(dataset) / f"{registry_row['batch_name']}_error.jsonl"
            error_path.write_text(error_text, encoding="utf-8")
            for raw_line in error_text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                except Exception:
                    continue
                custom_id = text_or_empty(result.get("custom_id"))
                if custom_id and custom_id not in result_map:
                    result_map[custom_id] = result

        for state_idx in state_df.index[state_mask].tolist():
            custom_id = text_or_empty(state_df.loc[state_idx, "openai_custom_id"])
            result = result_map.get(custom_id)
            if not result:
                continue
            response = result.get("response") or {}
            error = result.get("error")
            if error:
                state_df.loc[state_idx, "classification_error"] = normalize_whitespace(str(error))
                state_df.loc[state_idx, "batch_state"] = "error"
                continue

            status_code = int(response.get("status_code", 0) or 0)
            body = response.get("body") or {}
            if status_code >= 400:
                state_df.loc[state_idx, "classification_error"] = normalize_whitespace(
                    json.dumps(body, ensure_ascii=False)
                )[:500]
                state_df.loc[state_idx, "batch_state"] = "error"
                continue

            raw_text = extract_message_content(body)
            parsed = parse_headline_label(raw_text)
            state_df.loc[state_idx, "headline_label"] = parsed["label"]
            state_df.loc[state_idx, "headline_confidence"] = parsed["confidence"]
            state_df.loc[state_idx, "headline_rationale"] = parsed["rationale"]
            state_df.loc[state_idx, "headline_parse_error"] = parsed["parse_error"]
            state_df.loc[state_idx, "classification_error"] = ""
            state_df.loc[state_idx, "final_action"] = label_to_action(parsed["label"])
            state_df.loc[state_idx, "classified_at"] = utc_now_iso()
            state_df.loc[state_idx, "batch_state"] = "hydrated"

        registry_df.loc[registry_idx, "hydrated_at"] = utc_now_iso()
        logger.info(
            "Hydrated OpenAI batch %s for %s status=%s",
            registry_row["batch_name"],
            dataset.stem,
            status,
        )

    return state_df, registry_df


def submit_batches(
    dataset: DatasetKey,
    config: dict[str, Any],
    state_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    logger: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    openai_cfg = config["classification"]["openai"]
    batch_cfg = openai_cfg["batch"]
    pending_indexes = rows_needing_submission(state_df)
    if not pending_indexes:
        return state_df, registry_df

    request_rows = build_request_rows(state_df, pending_indexes, dataset, openai_cfg)
    request_chunks = split_request_chunks(
        request_rows,
        max_requests=int(batch_cfg.get("max_requests_per_batch", 50000)),
        max_bytes=int(batch_cfg.get("max_input_file_bytes", 190_000_000)),
        max_prompt_tokens=int(batch_cfg.get("max_estimated_prompt_tokens_per_batch", 0)),
    )
    if not request_chunks:
        return state_df, registry_df

    api_key = require_openai_api_key(config)
    base_url = str(openai_cfg.get("base_url", "") or "")
    timeout_seconds = int(openai_cfg.get("timeout_seconds", 60))
    paths = project_paths(config)
    paths.ensure_parent_dirs(dataset)

    max_pending_prompt_tokens = int(batch_cfg.get("max_estimated_enqueued_prompt_tokens", 0))
    max_pending_batches = int(batch_cfg.get("max_pending_batches_per_dataset", 0))
    pending_batch_count = int(
        (~registry_df["status"].fillna("").astype(str).str.lower().isin(TERMINAL_BATCH_STATUSES)).sum()
    ) if not registry_df.empty else 0

    next_batch_number = len(registry_df) + 1
    for offset, request_chunk in enumerate(request_chunks):
        estimated_chunk_tokens = int(sum(item["estimated_prompt_tokens"] for item in request_chunk))
        if max_pending_prompt_tokens > 0:
            existing_tokens = active_enqueued_prompt_tokens(registry_df)
            if existing_tokens + estimated_chunk_tokens > max_pending_prompt_tokens:
                logger.info(
                    "Stopped submitting new batches for %s because estimated enqueued prompt tokens would exceed the configured cap (%s).",
                    dataset.stem,
                    max_pending_prompt_tokens,
                )
                break
        if max_pending_batches > 0 and pending_batch_count >= max_pending_batches:
            logger.info(
                "Stopped submitting new batches for %s because pending batch count reached the configured cap (%s).",
                dataset.stem,
                max_pending_batches,
            )
            break

        batch_name = f"{dataset.stem}_headline_batch_{next_batch_number + offset:04d}"
        jsonl_path = paths.batch_requests_dir(dataset) / f"{batch_name}.jsonl"
        write_request_jsonl(request_chunk, jsonl_path)
        upload = upload_batch_file(
            jsonl_path,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        batch = create_batch(
            input_file_id=str(upload["id"]),
            api_key=api_key,
            base_url=base_url,
            endpoint="/v1/chat/completions",
            completion_window=str(batch_cfg.get("completion_window", "24h")),
            metadata={
                # The Batch API has metadata rather than a dedicated name field, so we make the
                # dataset and stage explicit here and in the saved JSONL filename.
                "batch_name": batch_name,
                "dataset_key": dataset.stem,
                "pathogen_domain": dataset.pathogen_domain,
                "language_code": dataset.language_code,
                "stage": "headline_filter",
            },
            timeout_seconds=timeout_seconds,
        )
        pending_batch_count += 1
        registry_row = {
            "dataset_key": dataset.stem,
            "batch_name": batch_name,
            "request_jsonl_path": str(jsonl_path.relative_to(paths.repo_root)),
            "request_count": len(request_chunk),
            "estimated_prompt_tokens": estimated_chunk_tokens,
            "input_file_id": str(upload["id"]),
            "batch_id": str(batch["id"]),
            "status": str(batch.get("status", "submitted")),
            "created_at": utc_now_iso(),
            "completed_at": "",
            "hydrated_at": "",
            "output_file_id": "",
            "error_file_id": "",
        }
        registry_df = pd.concat([registry_df, pd.DataFrame([registry_row])], ignore_index=True)
        for item in request_chunk:
            state_df.loc[item["idx"], "openai_custom_id"] = item["custom_id"]
            state_df.loc[item["idx"], "openai_batch_name"] = batch_name
            state_df.loc[item["idx"], "openai_batch_id"] = str(batch["id"])
            state_df.loc[item["idx"], "batch_state"] = str(batch.get("status", "submitted"))
            state_df.loc[item["idx"], "batch_submitted_at"] = utc_now_iso()
        logger.info(
            "Submitted OpenAI batch %s for %s request_count=%s batch_id=%s",
            batch_name,
            dataset.stem,
            len(request_chunk),
            batch["id"],
        )

    return state_df, registry_df


def run_headline_batch_filtering(
    dataset: DatasetKey,
    config: dict[str, Any],
    force: bool,
    logger: Any,
) -> dict[str, Any]:
    paths = project_paths(config)
    input_csv = paths.prefiltered_headlines_csv(dataset)
    state_csv = paths.classified_headlines_csv(dataset)
    registry_csv = paths.batch_registry_csv(dataset)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Missing prefiltered headline input for {dataset.stem}: {input_csv}"
        )

    if force and state_csv.exists():
        state_csv.unlink()
    if force and registry_csv.exists():
        registry_csv.unlink()

    state_df = load_or_initialize_state(input_csv, state_csv, dataset)
    registry_df = load_registry(registry_csv)
    state_df, registry_df = hydrate_batches(dataset, config, state_df, registry_df, logger)
    state_df, registry_df = submit_batches(dataset, config, state_df, registry_df, logger)

    write_dataframe_atomic(state_df[STATE_COLUMNS].fillna(""), state_csv)
    write_dataframe_atomic(registry_df[BATCH_REGISTRY_COLUMNS].fillna(""), registry_csv)

    pending = int(state_df["headline_label"].fillna("").astype(str).eq("").sum())
    ready = pending == 0 and not state_df.empty
    logger.info(
        "Headline filtering state for %s: ready=%s pending_rows=%s",
        dataset.stem,
        ready,
        pending,
    )
    return {
        "dataset_key": dataset.stem,
        "ready": ready,
        "pending_rows": pending,
        "state_csv": str(state_csv),
        "registry_csv": str(registry_csv),
    }
